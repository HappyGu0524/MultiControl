import torch
from tqdm import tqdm
class KCenters:
    def __init__(
        self,
        num_centers,
        latent_size,
        num_output_centers,
        device,
    ):
        '''
        num_centers:
            Number of clusters for searching.
        '''
        self.num_centers = num_centers
        self.num_output_centers = num_output_centers
        self.device = device
        self.latent_size = latent_size
        self.centers = None#self.init_cluster_center(self.num_centers, self.latent_size).to(device)
        self.score = None


    def init_cluster_center(self, num_centers, latent_size):
        '''
        '''
        if num_centers == 0:
            clusters = None
        else:
            clusters = torch.rand(num_centers, latent_size) * 2 - 1
        return clusters


    def Sparse_Distributed_Memory_Reinitalization(self, X, topk):
        length = len(X)
        self.centers = None
        for i in range(length):
            query_matrix = X[i]
            query_centers = torch.zeros_like(query_matrix).to(self.device)
            for j in range(length):
                if j !=i :
                    key_matrix = X[j]
                    query_centers += self.optim(query_matrix, key_matrix, topk, 'none')
            query_centers = (query_centers + query_matrix) / length

            query_score = torch.zeros(query_centers.shape[0]).to(self.device)
            for matrix in X:
                tmp_score = -self.distance(query_centers, matrix)
                tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
                query_score += torch.mean(tmp_values, dim=-1)
            query_score = query_score/length
            query_values, query_indices = torch.topk(query_score, k=self.num_centers)
            query_centers = torch.index_select(query_centers, 0, query_indices)

            if self.centers is None:
                self.centers = query_centers
            else:
                self.centers = torch.cat([self.centers, query_centers], dim=0)
        

        scores = torch.zeros(self.centers.shape[0]).to(self.device)
        for matrix in X:
            tmp_score = -self.distance(self.centers, matrix)
            tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
            scores += torch.mean(tmp_values, dim=-1)
        scores = scores/length
        out_values, out_indices = torch.topk(scores, k=self.num_centers)
        self.centers = torch.index_select(self.centers, 0, out_indices)
        


    def Kernel_Density_Estimation(self):
        return



    def train(
            self,
            X,
            weight=[1,1,1],
            topk=50,
            max_iter=1,
            strategy='none',
            SDM_reinit=False,
            tol=1e-10,
            temperature=50,
            num_output_centers=None
        ):
        '''
            X: [Tensor(batch, latent_size)]
                List of FloatTensors from different aspects
                example: X[0] from postive sentiment
                         X[1] from nontoxic
        '''

        assert strategy in {'none', 'weight'}
        length = sum(weight)

        if num_output_centers is not None:
            self.num_output_centers = num_output_centers

        if SDM_reinit:
            self.Sparse_Distributed_Memory_Reinitalization(X, topk)

        if strategy in {'none', 'weight'}:

            for i in tqdm(range(max_iter)):
                new_centers = torch.zeros_like(self.centers).to(self.device)
                
                for j in range(len(X)):
                    matrix = X[j]
                    w = weight[j]
                    new_centers += w * self.optim(self.centers, matrix, topk, strategy, temperature=temperature)
                new_centers = new_centers/length
                

                self.centers = new_centers
                        
                        

            
        
        new_score = torch.zeros(self.centers.shape[0]).to(self.device)
        for matrix in X:
            tmp_score = -self.distance(self.centers, matrix)
            tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
            new_score += torch.mean(tmp_values, dim=-1)
        self.score = new_score/length


        out_values, out_indices = torch.topk(self.score, k=self.num_output_centers)
        return torch.index_select(self.centers, 0, out_indices)

    def optim(self, centers, matrix, topk, strategy, batch=100, temperature=50):
        tmp_score = - self.distance(centers, matrix)
        tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
        
        tot_num = tmp_indices.shape[0]
        epoch = tot_num//batch + (1 if tot_num % batch != 0 else 0)

        new_centers = None
        for i in range(epoch):
            start = i * batch
            end = i * batch + batch
            if end > tot_num:
                end = tot_num
            if strategy == 'none':
                tmp_centers = torch.mean(torch.gather(matrix.unsqueeze(0).expand(end-start,-1,-1), 1, tmp_indices[start:end].unsqueeze(-1).expand(-1,-1,self.latent_size)),dim=1).squeeze()
            elif strategy == 'weight':
                #torch.gather -> [batch_size, topk, latent_size]
                #[batch_size, latent_size, topk] * [topk, 1]
                weight = torch.softmax(-torch.log(-tmp_values[start:end]) * temperature, dim=-1).unsqueeze(-1)
                tmp_c = torch.gather(matrix.unsqueeze(0).expand(end-start,-1,-1), 1, tmp_indices[start:end].unsqueeze(-1).expand(-1,-1,self.latent_size))
                tmp_centers = torch.matmul(tmp_c.permute(0,2,1),weight).squeeze()
            if new_centers is None:
                new_centers = tmp_centers
            else:
                new_centers = torch.cat([new_centers, tmp_centers], dim=0)
        return  new_centers
        


    def distance(self, matrix1, matrix2, batch=100):
        '''
        Input:
            matrix1: FloatTensor(i * m)
            matrix2: FloatTensor(j * m)
        Output:
            distance matrix: FloatTensor(i * j)
        '''

        assert len(matrix1.shape) == 2
        assert len(matrix2.shape) == 2

        dis = None
        tot_num = matrix1.shape[0]
        epoch = tot_num//batch + (1 if tot_num % batch != 0 else 0)
        matrix1 = matrix1.unsqueeze(dim=1)
        matrix2 = matrix2.unsqueeze(dim=0)
        for i in range(epoch):
            start = i * batch
            end = i * batch + batch
            if end > tot_num:
                end = tot_num
            tmp_matrix1 = matrix1[start:end]
            tmp_dis = (tmp_matrix1 - matrix2) ** 2.0
            tmp_dis = torch.sum(tmp_dis, dim=-1).squeeze()
            if dis is None:
                dis = tmp_dis
            else:
                dis = torch.cat([dis, tmp_dis], dim=0)

        
        return dis

