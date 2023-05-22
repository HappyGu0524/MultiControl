import torch
import torch.nn as nn
import wandb
import math
import random

from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


import logging
logger = logging.getLogger(__name__)


class Reshape(nn.Module):
    '''
    past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).
    '''
    def __init__(self, arg_dict):
        super(Reshape, self).__init__()
        self.seq_len = arg_dict['seq_len']
        self.num_layer = arg_dict['num_layer']
        self.hidden_size = arg_dict['hidden_size']
        self.num_head = arg_dict['num_head']
    def forward(self, x):
        batch_size = x.shape[0]
        assert self.hidden_size % self.num_head == 0
        embed_size_per_head = self.hidden_size//self.num_head
        x = x.view(batch_size, self.num_layer, 2, self.num_head, self.seq_len, embed_size_per_head).permute(1,2,0,3,4,5)
        past_key_values = []
        for i in range(self.num_head):
            past_key_values.append((x[i][0],x[i][1],))
        #assert past_key_values[0][0].requires_grad == True
        return tuple(past_key_values)



class AE(nn.Module):
    """
    AE with Decoder Fixed.
    We adopt connection method from prompt tuning.
    """
    #_keys_to_ignore_on_load_missing = [r"latent_classify_head\.\d+\.weight", r"latent_classify_head\.\d+\.bias"]
    def __init__(self, encoder, decoder, args): # 
        super(AE, self).__init__()
        self.encoder = encoder #BertModel
        self.decoder = decoder #GPT2LMHeadModel

        self.encoder_config = encoder.config
        self.encoder_hidden_size = self.encoder_config.hidden_size

        self.decoder_config = decoder.config
        self.decoder_num_layer = self.decoder_config.n_layer
        self.decoder_hidden_size = self.decoder_config.n_embd
        self.decoder_num_head = self.decoder_config.n_head

        self.losslist = None
        self.latent_classify_head = None


        

        self.args = args
        self.latent_size = args.latent_size
        self.seq_len_per_latent = args.seq_len_per_latent
        self.latent_num = args.latent_num
        self.seq_len = self.latent_num * self.seq_len_per_latent
        if 'variation' in args:
            self.variation = args.variation
        else:
            self.variation = 0

        self.mode = 'normal'



        ## connector: 
        # 1. from Bert hidden units to the latent space
        # 2. convert latent space to `past_key_values' in GPT
        # [batch_size, bert_hidden_size] -> [batch_size, latent_num * latent_size]
        # -> [batch_size, latent_num, decoder_layer * len([key,value]) * gpt_hidden_size]
        # -> (num_layer* (len([key,value])* tensor[batch_size, num_head, seq_len, embed_size_per_head]))
        self.trans1 = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_hidden_size, self.latent_num * self.latent_size),
            torch.nn.Tanh(),
            nn.Dropout(self.decoder_config.attn_pdrop)#added
        )
        self.trans2 = torch.nn.Sequential(
            torch.nn.Linear(self.latent_num * self.latent_size, self.seq_len * self.decoder_num_layer * 2 * self.decoder_hidden_size),
            nn.Dropout(self.decoder_config.attn_pdrop),#added
            Reshape({'seq_len':self.seq_len, 'num_layer':self.decoder_num_layer, 'hidden_size':self.decoder_hidden_size, 'num_head':self.decoder_num_head})
        )

        self.is_prior = False
        if args.prior is True:
            

            self.is_prior = True
            def subnet(dims_in, dims_out):
                return nn.Sequential(
                    nn.Linear(dims_in, 512),
                    nn.LeakyReLU(0.1),
                    nn.Linear(512, dims_out),
                )
            self.inv_flow = SequenceINN(self.latent_size * self.latent_num)
            for k in range(args.flow_num):
                self.inv_flow.append(AllInOneBlock, subnet_constructor=subnet, permute_soft=True)

            #learned mu and log_sigma
            self.priors = nn.ParameterList([nn.Parameter(torch.zeros(2, self.latent_size * self.latent_num)) for i in range(args.prior_num)])
            self.prior_num = args.prior_num

            self.prior_classify_head = None



    def set_mode(self, mode):
        #'normal': Train the AE structure
        #'prior': Fix encoder and train the Normalizing Flow Prior
        self.mode = mode
        if self.mode == 'normal':
            for struct in [self.encoder, self.trans1, self.trans2]:
                struct.train()
                for params in struct.parameters():
                    params.requires_grad = True
        
        elif self.mode == 'prior':
            if not self.is_prior:
                raise Exception('Need to initialize with a prior structure.')
            for struct in [self.encoder, self.trans1, self.trans2]:
                struct.eval()
                for params in struct.parameters():
                    params.requires_grad = False
        
        print('Trainable part are listed below:')
        for keys, params in self.named_parameters():
            if params.requires_grad is True:
                print(keys)


    def fix_decoder(self):
        '''
        Fix the decoder to work as prefix tuning.
        '''
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False


    def connect(self, encoder_output, variation=0):
        '''
        Connect encoder to decoder and get the latent representation.
        '''
        tmp_latent = self.trans1(encoder_output)
        eps = torch.zeros_like(tmp_latent).normal_(std=variation).to(tmp_latent.device)
        past_key_values = self.trans2(tmp_latent + eps)
        latent = tmp_latent.view(-1, self.latent_num, self.latent_size)
        return past_key_values, latent


    def sparse_loss(self, latent, dim=None):
        '''
        Increase the sparsity.
        '''
        if len(latent) == 3 and dim is None:
            raise Exception('Expect latent to be dim 2.')
        loss_func = nn.L1Loss(reduction='mean')
        batch_size = latent.shape[0]
        if dim is not None:
            tmp_latent = latent[:,dim,:].squeeze()
            average = torch.sum(tmp_latent, dim=0)/batch_size
            loss = loss_func(latent, average.expand(batch_size, -1))
        else:
            average = torch.sum(latent, dim=0)/batch_size
            loss = loss_func(latent, average.expand(batch_size, -1))
        return -loss

    def contrasitive_loss(self, latent1, latent2, loss_func=nn.SmoothL1Loss(reduction='mean'), dim=None):
        '''
        Increase the distance between latent1 and latent2.
        loss_func: nn.L1Loss, nn.SmoothL1Loss, nn.MSELoss, ...
        '''
        if dim is not None:
            loss = loss_func(latent1[:,dim,:].squeeze(), latent2[:,dim,:].squeeze())
        else:
            loss = loss_func(latent1, latent2)
        return -1 * loss
    
    def latent_classify_loss(self, latent, pos_label, neg_labels, head_index=None):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num*self.latent_size)
        if self.latent_classify_head_type == 'single':
            probs = torch.softmax(self.latent_classify_head(latent), dim=-1)
            batch_size, class_num = probs.shape
            loss = 0
            neg_len = neg_labels.shape[-1]
            
            for i in range(batch_size):
                pos_prob = probs[i, pos_label[i]]
                if pos_prob < 1/self.head_num:
                    loss += torch.log(pos_prob)
                loss += torch.log(1 - probs[i, neg_labels[i]]).sum()

            return -1 * loss / (batch_size * (neg_len+1))
        elif self.latent_classify_head_type == 'multiple':
            if head_index is None:
                print("UserWarning: head_index not set for multiple classifier head, default to 0")
                head_index = 0
            device = latent.device
            logits = self.latent_classify_head[head_index](latent)
            loss = torch.nn.functional.cross_entropy(logits, pos_label.to(device))
            return loss
        else:
            raise Exception('Wrong latent classifier head type.')

    def aspect_gap_loss(self, latent, head_index):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)

        mean_latent = torch.mean(latent, dim=0)
        loss = None
        for i in range(self.aspect_head_num):
            if i != head_index and self.aspect_gap_head[i] is not None:
                if loss is None:
                    loss = torch.nn.functional.mse_loss(mean_latent, self.aspect_gap_head[i]) * self.aspect_gap_loss_amplification
                else:
                    loss += torch.nn.functional.mse_loss(mean_latent, self.aspect_gap_head[i]) * self.aspect_gap_loss_amplification
        self.set_aspect_gap_head(mean_latent, head_index)
        return loss


    def prior_classify_loss(self, z, pos_label, head_index):
        device = z.device
        logits = self.prior_classify_head[head_index](z)
        loss = torch.nn.functional.cross_entropy(logits, pos_label.to(device))
        return loss
        

    def set_losslist(self, 
        losslist:dict,
        latent_classify_args={'head_num':1, 'class_num_per_head':2,'mid_size':128,'head_type':'single'},
        aspect_gap_args={'head_num':2, 'amplification':5},
        adv_z_prob_args={'grouping':[[0,1],[2,3,4,5],[6,7]]},
        prior_classify_args={'head_num':3, 'class_num_per_head':[2,2,4],'mid_size':128}
        ):
        '''
        losslist:
            Sample: {'contrasitive_loss': 0.001, 'sparse_loss': 0.001, 'latent_classify_loss':0.1, 'aspect_gap_loss':0.1}
        '''
        self.losslist = losslist
        ### AE training
        if 'latent_classify_loss' in losslist:
            self.head_num = 1
            class_num_per_head = 2
            mid_size = 128
            head_type = 'single'
            if latent_classify_args is not None:
                if 'head_num' in latent_classify_args:
                    self.head_num = latent_classify_args['head_num']
                if 'class_num_per_head' in latent_classify_args:
                    class_num_per_head = latent_classify_args['class_num_per_head']
                if 'mid_size' in latent_classify_args:
                    mid_size = latent_classify_args['mid_size']
                if 'head_type' in latent_classify_args:
                    head_type = latent_classify_args['head_type']
            
            self.set_latent_classify_head(head_num=self.head_num, class_num_per_head=class_num_per_head, mid_size=mid_size, head_type=head_type)
    
            self.latent_classify_head_type=head_type
        
        if 'aspect_gap_loss' in losslist:
            if 'latent_classify_loss' in losslist:
                if self.latent_classify_head_type == 'multiple':
                    self.aspect_head_num = self.head_num
                elif self.latent_classify_head_type == 'single':
                    print('set aspect head num to {aspect_head_num}.')
                    self.aspect_head_num = aspect_gap_args['head_num']
            else:
                print('set aspect head num to {aspect_head_num}.')
                self.aspect_head_num = aspect_gap_args['head_num']
            
            self.aspect_gap_loss_amplification = aspect_gap_args['amplification']

            self.aspect_gap_head = [None for i in range(self.aspect_head_num)]

        ### Normalizing Flow training
        
        if 'adv_z_prob_loss' in losslist:
            self.adv_prior_head_dict = {}
            for i in range(self.prior_num):
                for group in adv_z_prob_args['grouping']:
                    if i in group:
                        self.adv_prior_head_dict[i] = [k for k in group if k != i]
                        break

        if 'adv_x_prob_loss' in losslist:
            pass

        if 'prior_classify_loss' in losslist:
            self.head_num = 3
            class_num_per_head = [2,2,4]
            mid_size = 128
            if prior_classify_args is not None:
                if 'head_num' in prior_classify_args:
                    self.head_num = prior_classify_args['head_num']
                if 'class_num_per_head' in prior_classify_args:
                    class_num_per_head = prior_classify_args['class_num_per_head']
                if 'mid_size' in prior_classify_args:
                    mid_size = prior_classify_args['mid_size']
            self.set_prior_classify_head(head_num=self.head_num, class_num_per_head=class_num_per_head, mid_size=mid_size)
            


    def set_prior_classify_head(self, head_num, class_num_per_head, mid_size):
        if type(class_num_per_head) is list:
            self.prior_classify_head = nn.ModuleList([
                nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, head_n)
                ) for head_n in class_num_per_head]
            )
        else:
            self.prior_classify_head = nn.ModuleList([
                    nn.Sequential(
                    nn.Linear(self.latent_num * self.latent_size, mid_size),
                    nn.ReLU(),
                    nn.Linear(mid_size, class_num_per_head)
                ) for i in range(head_num)]
            )


    def set_latent_classify_head(self, head_num=1, class_num_per_head=2, mid_size=128, head_type='single'):
        if head_type == 'single':
            self.latent_classify_head = nn.Sequential(
                nn.Linear(self.latent_num * self.latent_size, mid_size),
                nn.ReLU(),
                nn.Linear(mid_size, class_num_per_head * head_num)
            )
        elif head_type == 'multiple':
            if type(class_num_per_head) is list:
                self.latent_classify_head = nn.ModuleList([
                        nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, head_num)
                    ) for head_num in class_num_per_head]
                )
            else:
                self.latent_classify_head = nn.ModuleList([
                        nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, class_num_per_head)
                    ) for i in range(head_num)]
                )
    
    def set_aspect_gap_head(self, latent, head_index):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)
        
        if len(latent.shape) == 2:
            mean_latent = torch.mean(latent.detach(), dim=0)
            if self.aspect_gap_head[head_index] is not None:
                assert self.aspect_gap_head[head_index].shape == mean_latent.shape
            self.aspect_gap_head[head_index] = mean_latent
        elif len(latent.shape) == 1:
            if self.aspect_gap_head[head_index] is not None:
                assert self.aspect_gap_head[head_index].shape == latent.shape
            self.aspect_gap_head[head_index] = latent.detach()


    def gaussian_log_prob(self, x, mu, log_sd):
        return -0.5 * math.log(2 * torch.pi) - log_sd - 0.5 * (x - mu) ** 2 / torch.exp(2 * log_sd)

    def gaussian_sample(self, eps, mu, log_sd):
        return mu + torch.exp(log_sd) * eps

    def cal_prior(self, latent, prior_head_index): #########################TO DO adv_prior prob########################
        #latent to normal distribution
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)
        batch_size = latent.shape[0]
        z, log_jac_det = self.inv_flow(latent)
        learnable_prior = self.priors[prior_head_index]

        log_prob = self.gaussian_log_prob(z, learnable_prior[0], learnable_prior[1]).view(batch_size,-1).sum(-1)

        loss = -log_prob - log_jac_det
        loss = loss.mean() / (self.latent_num * self.latent_size)
        return loss, z, log_prob, log_jac_det


    
    def forward(self,
        encoder_input_ids,
        encoder_attention_mask,
        encoder_token_type_ids,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        adv_input_ids=None,
        adv_attention_mask=None,
        adv_token_type_ids=None,
        pos_label=None,
        neg_labels=None,
        variation=None,
        variation_prob=0.2,
        head_index=None,
        prior_head_index=None
        ):
        '''
        Forward method for training which returns a reconstruction loss.
        Args:
            encoder_input_ids,
            encoder_atention_mask,
            encoder_token_type_ids:
                Outputs of BertTokenizer(List of Strings, return_tensors='pt', padding=True)
            decoder_input_ids,
            decoder_attention_mask:
                Outputs of GPT2Tokenizer(List of Strings, return_tensors='pt', padding=True)
            adv_input_ids,
            adv_attention_mask,
            adv_token_type_ids:
                Adversarial text

        '''
        if self.mode == 'normal':
            if len(encoder_input_ids.shape) == 3:
                encoder_input_ids = encoder_input_ids.view(encoder_input_ids.shape[1], encoder_input_ids.shape[2])
                encoder_attention_mask = encoder_attention_mask.view(encoder_attention_mask.shape[1], encoder_attention_mask.shape[2])
                encoder_token_type_ids = encoder_token_type_ids.view(encoder_token_type_ids.shape[1], encoder_token_type_ids.shape[2])
                assert decoder_input_ids is not None and decoder_attention_mask is not None
                decoder_input_ids = decoder_input_ids.view(decoder_input_ids.shape[1], decoder_input_ids.shape[2])
                decoder_attention_mask = decoder_attention_mask.view(decoder_attention_mask.shape[1], decoder_attention_mask.shape[2])
                if head_index is not None:
                    head_index = head_index.item()
                if adv_input_ids is not None:
                    adv_input_ids = adv_input_ids.view(adv_input_ids.shape[1], adv_input_ids.shape[2])
                if adv_attention_mask is not None:
                    adv_attention_mask = adv_attention_mask.view(adv_attention_mask.shape[1], adv_attention_mask.shape[2])
                if adv_token_type_ids is not None:
                    adv_token_type_ids = adv_token_type_ids.view(adv_token_type_ids.shape[1], adv_token_type_ids.shape[2])
                if pos_label is not None:
                    pos_label = pos_label.view(pos_label.shape[1])
                if neg_labels is not None:
                    neg_labels = neg_labels.view(neg_labels.shape[1], neg_labels.shape[2])

            if variation is None:
                variation = self.variation
            batch_size = decoder_input_ids.shape[0]
            infix_attn = torch.ones(batch_size, self.seq_len).bool().to(decoder_input_ids.device)
            decoder_attention_mask = torch.cat([infix_attn, decoder_attention_mask], dim=1)

            encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output
            past_key_values, latent = self.connect(encoder_output, variation)
            outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, labels=decoder_input_ids, past_key_values=past_key_values, return_dict=True)
            lm_loss = outputs.loss
            # lm_logits = outputs.logits
            
            #labels = decoder_input_ids
            # Shift so that tokens < n predict n
            #shift_logits = lm_logits[..., self.seq_len:-1, :].contiguous()#########lm_logits[..., :-1, :] -> lm_logits[..., self.seq_len:-1, :]
            #shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            #loss_fct = torch.nn.CrossEntropyLoss()
            #lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            #logits = outputs.lm_logits
            loss = 0
            loss_detail = {"lm_loss":lm_loss.detach()}
            w = 1
            if self.losslist is not None:
                if 'contrasitive_loss' in self.losslist:
                    if adv_input_ids is None:
                        raise Exception('Expect adversarial inputs for contrasitive loss.')
                    adv_encoder_output = self.encoder(input_ids=adv_input_ids, attention_mask=adv_attention_mask, token_type_ids=adv_token_type_ids, return_dict=True).pooler_output
                    adv_latent = self.trans1(adv_encoder_output)
                    adv_loss = self.contrasitive_loss(latent, adv_latent)
                    #TO DO: change arg `dim' in the future
                    loss += adv_loss * self.losslist['contrasitive_loss']
                    w -= self.losslist['contrasitive_loss']
                    loss_detail["contrasitive_loss"] = adv_loss.detach()
                
                if 'sparse_loss' in self.losslist:
                    spa_loss = self.sparse_loss(latent)
                    #TO DO: change arg `dim' in the future
                    loss += spa_loss * self.losslist['sparse_loss']
                    w -= self.losslist['sparse_loss']
                    loss_detail["sparse_loss"] = spa_loss.detach()
                
                if 'latent_classify_loss' in self.losslist:
                    lac_loss = self.latent_classify_loss(latent, pos_label, neg_labels, head_index)
                    if lac_loss.detach().item() < 0.1:
                        loss += lac_loss * 0.05
                        w -= 0.05
                    else:
                        loss += lac_loss * self.losslist['latent_classify_loss']
                        w -= self.losslist['latent_classify_loss']
                    loss_detail["latent_classify_loss"] = lac_loss.detach()


                if 'aspect_gap_loss' in self.losslist:
                    agp_loss = self.aspect_gap_loss(latent, head_index)
                    if agp_loss is not None:
                        loss += agp_loss * self.losslist['aspect_gap_loss']
                        w -= self.losslist['aspect_gap_loss']
                        loss_detail["aspect_gap_loss"] = agp_loss.detach()
                
                wandb.log(loss_detail)
            if w < 0:
                w = 1
            
            loss += w * lm_loss
            
            return loss, latent, loss_detail

        elif self.mode == 'prior':
            if len(encoder_input_ids.shape) == 3:
                encoder_input_ids = encoder_input_ids.view(encoder_input_ids.shape[1], encoder_input_ids.shape[2])
                encoder_attention_mask = encoder_attention_mask.view(encoder_attention_mask.shape[1], encoder_attention_mask.shape[2])
                encoder_token_type_ids = encoder_token_type_ids.view(encoder_token_type_ids.shape[1], encoder_token_type_ids.shape[2])
                if decoder_input_ids is not None:
                    print('Notion: Prior is guided Only by fixed encoder')
                if prior_head_index is not None:
                    prior_head_index = prior_head_index.item()
                else:
                    if self.prior_num > 1:
                        print('Warning: prior head index is required if `prior_num` is not 1. Default index is 0.')
                    prior_head_index = 0

            latent, _, _ = self.encode(encoder_input_ids=encoder_input_ids, encoder_attention_mask=encoder_attention_mask, encoder_token_type_ids=encoder_token_type_ids)
            if variation is None:
                variation = self.variation
            if variation > 0:
                choice = random.uniform(0,1) < variation_prob
                if choice:
                    eps = torch.zeros_like(latent).normal_(std=variation).to(latent.device)
                    latent = latent + eps
            loss, z, log_prob, log_jac_det = self.cal_prior(latent, prior_head_index)

            loss_detail = {"loss":loss.detach(), "log_prob":log_prob.detach().mean(), "log_det":log_jac_det.detach().mean()}

            if self.losslist is not None:

                if 'adv_z_prob_loss' in self.losslist:
                    adv_index_list = self.adv_prior_head_dict[prior_head_index]
                    adv_z_loss = 0
                    inds = 0
                    for adv_head_index in adv_index_list:
                        tmp_adv_z_loss, _, adv_log_prob, adv_log_det = self.cal_prior(latent, adv_head_index)
                        adv_z_loss += -1 * adv_log_prob.mean() / (self.latent_num * self.latent_size)
                        inds += 1

                    adv_z_loss = adv_z_loss / inds
                    
                    if adv_z_loss.item() < 10:
                        loss = loss - adv_z_loss * self.losslist['adv_z_prob_loss']

                        loss_detail["adv_z_loss"] = adv_z_loss.detach()
                    else:
                        loss_detail["adv_z_loss"] = 10

                if 'adv_x_prob_loss' in self.losslist:
                    if len(adv_input_ids.shape) == 3:
                        adv_input_ids = adv_input_ids.view(adv_input_ids.shape[1], adv_input_ids.shape[2])
                        adv_attention_mask = adv_attention_mask.view(adv_attention_mask.shape[1], adv_attention_mask.shape[2])
                        adv_token_type_ids = adv_token_type_ids.view(adv_token_type_ids.shape[1], adv_token_type_ids.shape[2])

                        adv_latent, _, _ = self.encode(encoder_input_ids=adv_input_ids, encoder_attention_mask=adv_attention_mask, encoder_token_type_ids=adv_token_type_ids)
                        if variation > 0:
                            choice = random.uniform(0,1) < variation_prob
                            if choice:
                                eps = torch.zeros_like(adv_latent).normal_(std=variation).to(adv_latent.device)
                                adv_latent = adv_latent + eps
                        
                        adv_x_loss, _, _, _ = self.cal_prior(adv_latent, prior_head_index)

                        if adv_x_loss.item() < 10:
                            loss = loss - adv_x_loss * self.losslist['adv_x_prob_loss']

                            loss_detail["adv_x_loss"] = adv_x_loss.detach()
                        else:
                            loss_detail["adv_x_loss"] = 10

                if 'prior_classify_loss' in self.losslist:
                    assert head_index is not None
                    head_index = head_index.item()

                    assert pos_label is not None
                    pos_label = pos_label.view(pos_label.shape[1])
                    
                    prior_cls_loss = self.prior_classify_loss(z, pos_label, head_index)
                    
                    loss = loss + prior_cls_loss

                    loss_detail["prior_cls_loss"] = prior_cls_loss.detach()






            wandb.log(loss_detail)
            return (loss, z)
            
    

    def encode(self,
        encoder_input_ids,
        encoder_attention_mask=None,
        encoder_token_type_ids=None,
        ):
        '''
        Encode the input text and get the latent representation
        '''
        device = next(self.parameters()).device
        encoder_input_ids = encoder_input_ids.to(device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)
        if encoder_token_type_ids is not None:
            encoder_token_type_ids = encoder_token_type_ids.to(device)
        encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output
        past_key_values, latent = self.connect(encoder_output)
        return latent, encoder_output, past_key_values


    def generate(
        self,
        input_latent,
        input_ids=None,
        attention_mask=None,
        batch_size=None,
        variation=None,
        min_len=30,
        max_len=50,
        do_sample=True,
        topk=5,
        topp=0.9,
        lp=1,
        rp=1.0,
        prior_head_index=None,
        classify_head_index=None,
        threshold_for_cls=None,
        mean=0,
        std=1,
        alpha=None,
        calculate=None,
        use_cache=True):
        '''
        Generate text with given latent represention.
        '''
        device = next(self.parameters()).device
        input_latent = input_latent.to(device)
        

        if len(input_latent.shape) == 3:
            tmp_batch_size, latent_num, latent_size = input_latent.shape
            input_latent = input_latent.view(tmp_batch_size, latent_num * latent_size)
        elif len(input_latent.shape) != 2:
            raise Exception('Shape of input_latent is expected to be [batch_size, latent_num, latent_size] \
                or [batch_size, latent_num * latent_size]')


        if batch_size is None:
            batch_size = input_latent.shape[0]
            if input_ids is not None:
                if input_ids.shape[0] > batch_size:
                    if batch_size == 1:
                        batch_size = input_ids.shape[0]
                        input_latent = input_latent.expand(batch_size, -1)
                    else:
                        raise Exception('Batch size of input_latent and input_ids mismatched')
                elif input_ids.shape[0] < batch_size and input_ids.shape[0] == 1:
                    input_ids = input_ids.expand(batch_size, -1)
                


        if input_latent.shape[0] < batch_size:
            input_latent.expand(batch_size, -1)

      
        if self.mode == 'prior':
            batch_size, latent_size = input_latent.shape
            tmp_input_latent = self.get_prior_sampling(batch_size, latent_size, mean=mean, std=std, prior_head_index=prior_head_index, alpha=alpha, calculate=calculate)

            #if prior_head_index is not None:
            #    learnable_prior = self.priors[prior_head_index]
            #    sampled_dis = self.gaussian_sample(input_latent, learnable_prior[0], learnable_prior[1])
            #    tmp_input_latent, _ = self.inv_flow(sampled_dis, rev=True)
            #else:
            #    tmp_input_latent, _ = self.inv_flow(input_latent, rev=True)
    
            if threshold_for_cls is not None and self.latent_classify_head is not None and classify_head_index is not None and prior_head_index is not None:
                prob = torch.softmax(self.latent_classify_head[classify_head_index[0]](tmp_input_latent), dim=-1)[:, classify_head_index[1]]
                mask = prob > threshold_for_cls
                candidate_latent = tmp_input_latent[mask,:]
                while candidate_latent.shape[0] < batch_size:
                    tmp_input_latent = self.get_prior_sampling(batch_size * 10, latent_size, mean=mean, std=std, prior_head_index=prior_head_index, alpha=alpha, calculate=calculate)
                    #ext_latent = torch.zeros_like(input_latent).repeat(10, 1).normal_(std=1).to(device)
                    #sampled_dis = self.gaussian_sample(ext_latent, learnable_prior[0], learnable_prior[1])
                    #tmp_input_latent, _ = self.inv_flow(sampled_dis, rev=True)
                    prob = torch.softmax(self.latent_classify_head[classify_head_index[0]](tmp_input_latent), dim=-1)[:, classify_head_index[1]]
                    mask = prob > threshold_for_cls
                    candidate_latent = torch.cat([candidate_latent, tmp_input_latent[mask,:]], dim=0)
                    del tmp_input_latent
                    del prob
                    del mask
                    torch.cuda.empty_cache()
                    

                candidate_latent = candidate_latent[:batch_size,:]
                #print(candidate_latent.shape)

                input_latent = candidate_latent
            
            else:
                input_latent = tmp_input_latent


        if variation is not None:
            eps = torch.zeros_like(input_latent).normal_(std=variation).to(device)
            input_latent = input_latent + eps  

                


            
        past_key_values = self.trans2(input_latent)

        if input_ids is None:
            input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]]*batch_size).to(device), max_length=3, do_sample=True)[:,1:]
            attention_mask = torch.ones(batch_size, 2).bool()
        else:
            input_ids = input_ids.to(device)
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, 2).bool()

        cur_len = input_ids.shape[1]
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(device)
        attention_mask = torch.cat([infix_attn, attention_mask.to(device)], dim=-1)

        if cur_len < 1:
            raise Exception('input length error')
        if cur_len == 1:
            result = self.decoder.generate(input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, repetition_penalty=rp,\
                do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp, max_length=max_len, min_length=min_len, use_cache=use_cache)
        else:
            past_key_values = self.decoder(input_ids=input_ids[:,:-1], attention_mask=attention_mask[:,:-1], past_key_values=past_key_values, return_dict=True, use_cache=True).past_key_values
            result = self.decoder.generate(input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, repetition_penalty=rp,\
                do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp, max_length=max_len, min_length=min_len, use_cache=use_cache)

        return result


    def get_prior_sampling(self,
        batch_size,
        latent_size,
        mean=0,
        std=1,
        prior_head_index=None,
        alpha=None,
        calculate=None
        ):
        device = next(self.parameters()).device
        eps = torch.normal(mean,std, (batch_size, latent_size)).to(device)
        if type(prior_head_index) is int:
            learnable_prior = self.priors[prior_head_index]
            sampled_dis = self.gaussian_sample(eps, learnable_prior[0], learnable_prior[1])
            input_latent, _ = self.inv_flow(sampled_dis, rev=True)
        elif type(prior_head_index) is list:
            learnable_prior_mean = [self.priors[p_index][0] for p_index in prior_head_index]
            learnable_prior_std = [torch.exp(self.priors[p_index][1]) for p_index in prior_head_index]
            if alpha is None:
                alpha = ([1/len(prior_head_index)] * len(prior_head_index))
            else:
                assert len(alpha) == len(prior_head_index)
            if calculate is not None:
                sampled_dis = calculate(alpha, learnable_prior_mean, learnable_prior_std, eps)

            else:
                def weighted_combine(alpha, learnable_prior_mean,learnable_prior_std, eps):
                    mu = 0
                    for w, mean in zip(alpha, learnable_prior_mean):
                        mu = mu + w * mean

                    sigma = 0
                    for w, std in zip(alpha, learnable_prior_std):
                        sigma = sigma + (w * std)**2
                    sigma = torch.sqrt(sigma)
                    
                    sampled_dis = sigma * eps + mu
                    return sampled_dis
                sampled_dis = weighted_combine(alpha, learnable_prior_mean, learnable_prior_std, eps)
            input_latent, _ = self.inv_flow(sampled_dis, rev=True)
        elif prior_head_index is None:
            input_latent, _ = self.inv_flow(eps, rev=True)


        return input_latent



    def reconstruct(self,
        encoder_input_ids,
        decoder_input_ids=None,
        encoder_attention_mask=None,
        encoder_token_type_ids=None,
        decoder_attention_mask=None,
        do_sample=True,
        max_len=50,
        min_len=30,
        topk=5,
        topp=0.9,
        lp=1.0,
        use_cache=True):
        '''
        Reconstruct input text.
        '''
        device = next(self.parameters()).device
        batch_size = encoder_input_ids.shape[0]
        encoder_input_ids = encoder_input_ids.to(device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)
        if encoder_token_type_ids is not None:
            encoder_token_type_ids = encoder_token_type_ids.to(device)
        encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output
        
        past_key_values, latent = self.connect(encoder_output)
        if decoder_input_ids is None:
            decoder_input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]]*batch_size).to(device), max_length=3, do_sample=True)[:,1:]
            decoder_attention_mask = torch.ones(batch_size, 2).bool()
        else:
            decoder_input_ids = decoder_input_ids.to(device)
            if decoder_attention_mask is None:
                decoder_attention_mask = torch.ones(batch_size, 2).bool()
        
        cur_len = decoder_input_ids.shape[1]
        
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(device)
        decoder_attention_mask = torch.cat([infix_attn, decoder_attention_mask.to(device)], dim=-1)

        if cur_len < 1:
            raise Exception('input length error')
        if cur_len == 1:
            result = self.decoder.generate(input_ids=decoder_input_ids, past=past_key_values, attention_mask=decoder_attention_mask,\
                do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp, max_length=max_len, min_length=min_len, use_cache=use_cache)
        else:
            past_key_values = self.decoder(input_ids=decoder_input_ids[:,:-1], attention_mask=decoder_attention_mask[:,:-1], past_key_values=past_key_values, return_dict=True, use_cache=True).past_key_values
            result = self.decoder.generate(input_ids=decoder_input_ids, past=past_key_values, attention_mask=decoder_attention_mask,\
                do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp, max_length=max_len, min_length=min_len, use_cache=use_cache)
        return result

















