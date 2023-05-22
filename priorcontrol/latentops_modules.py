import torch
import torch.nn as nn
import numpy as np
import math

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal


class DenseEmbedder(nn.Module):
    """Supposed to map small-scale features (e.g. labels) to some given late
    \nt dim"""

    def __init__(self, input_dim, up_dim, depth=4, num_classes=10):
        super().__init__()
        self.net = nn.ModuleList()
        dims = np.linspace(input_dim, up_dim, depth).astype(int)

        for l in range(len(dims) - 1):
            self.net.append(nn.Dropout(0.2))
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            # self.net.append(get_norm(dims[l + 1], norm))
            self.net.append(nn.LeakyReLU(0.2))
            # self.net.append(nn.Tanh())

        self.last_dim = up_dim
        self.linear = nn.Linear(up_dim, num_classes)
        self.energy_weight = 1
        # print('Using DenseEmbedder...')
        # print(f'{norm1} norm')

    def set_energy_weight(self, weight):
        self.energy_weight = weight
        # print('Energy Weight = ',weight)

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, :, None, None]

        for layer in self.net:
            x = layer(x)

        out = x.squeeze(-1).squeeze(-1)
        out = self.linear(out)
        logits = out
        return logits


class CCF(nn.Module):
    def __init__(self, classifier):
        super(CCF, self).__init__()
        self.f = nn.ModuleList()
        for cls in classifier:
            self.f.append(cls)

    def get_cond_energy(self, z, y_):
        energy_outs = []
        # for i, cls in enumerate(self.f):
        for i in range(y_.shape[1]):
            cls = self.f[i]
            logits = cls(z)
            # logits_list.append(logits)
            n_classes = logits.size(1)
            if n_classes > 1:
                y = y_[:, i].long()
                sigle_energy = torch.gather(logits, 1, y[:, None]).squeeze() - logits.logsumexp(1)
                energy_outs.append(cls.energy_weight * sigle_energy)
                # energy_outs.append((cls.energy_weight)*(torch.gather(logits, 1, y[:, None]).squeeze() - logits.logsumexp(1)))
            else:
                assert n_classes == 1, n_classes
                y = y_[:, i].float()
                sigma = 0.1  # this value works well
                sigle_energy = -torch.norm(logits - y[:, None], dim=1) ** 2 * 0.5 / (sigma ** 2)
                energy_outs.append(cls.energy_weight * sigle_energy)
        # print('dog:', round(energy_outs[0].sum().item(),2), '\tchild:', round(energy_outs[1].sum().item(),2), '\tball:',round(energy_outs[2].sum().item(),2))

        energy_output = torch.stack(energy_outs).sum(dim=0)
        return energy_output # - 0.03*torch.norm(z, dim=1) ** 2 * 0.5
    def get_cond_energy_single(self, z, y_):
        for i in range(y_.shape[1]):
            energy_outs = []
            # for i, cls in enumerate(self.f):
            cls = self.f[i]
            logits = cls(z)
            # logits_list.append(logits)
            n_classes = logits.size(1)
            if n_classes > 1:
                y = y_[:, i].long()
                sigle_energy = torch.gather(logits, 1, y[:, None]).squeeze() - logits.logsumexp(1)
                energy_outs.append(cls.energy_weight * sigle_energy)
                # energy_outs.append((cls.energy_weight)*(torch.gather(logits, 1, y[:, None]).squeeze() - logits.logsumexp(1)))
            else:
                assert n_classes == 1, n_classes
                y = y_[:, i].float()
                sigma = 0.1  # this value works well
                sigle_energy = -torch.norm(logits - y[:, None], dim=1) ** 2 * 0.5 / (sigma ** 2)
                energy_outs.append(cls.energy_weight * sigle_energy)
        # print('dog:', round(energy_outs[0].sum().item(),2), '\tchild:', round(energy_outs[1].sum().item(),2), '\tball:',round(energy_outs[2].sum().item(),2))

        energy_output = torch.stack(energy_outs).sum(dim=0)
        return energy_output

    def forward(self, z, y):
        energy_output = self.get_cond_energy(z, y) - torch.norm(z, dim=1) ** 2 * 0.5
        return energy_output


class DIS(nn.Module):
    def __init__(self, distributions, weights=None):
        super(DIS, self).__init__()
        self.f = nn.ParameterList()
        for dis in distributions:
            self.f.append(dis)

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [(1/len(distributions))] * len(distributions)

    def gaussian_log_prob(self, x, mu, log_sd):
        return -0.5 * math.log(2 * torch.pi) - log_sd - 0.5 * (x - mu) ** 2 / torch.exp(2 * log_sd)

    def get_cond_energy(self, z, y_):
        energy_outs = []
        # for i, cls in enumerate(self.f):
        batch_size, dim = z.shape[0], z.shape[1]
        for i in range(y_.shape[1]):
            dis = self.f[i]

            log_prob = self.gaussian_log_prob(z, dis[0], dis[1]).view(batch_size,-1).sum(-1) #/ dim

            #energy = torch.exp(log_prob) * self.weights[i]
            energy = log_prob * self.weights[i]
            energy_outs.append(energy)


        energy_output = torch.stack(energy_outs).sum(dim=0)
        return energy_output # - 0.03*torch.norm(z, dim=1) ** 2 * 0.5

    def forward(self, z, y):
        energy_output = self.get_cond_energy(z, y)
        return energy_output


class DIScons(nn.Module):
    def __init__(self, distributions, weights=None, eps = 8e-5):
        super(DIScons, self).__init__()
        self.f = nn.ParameterList()
        self.eps = eps
        for dis in distributions:
            self.f.append(dis)

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [(1/len(distributions))] * len(distributions)

    def gaussian_log_prob(self, x, mu, log_sd):
        return -0.5 * math.log(2 * torch.pi) - log_sd - 0.5 * (x - mu) ** 2 / torch.exp(2 * log_sd)

    def get_cond_energy(self, z, y_):
        energy_outs = []
        # for i, cls in enumerate(self.f):
        batch_size, dim = z.shape[0], z.shape[1]

        log_probs = []

        for i in range(y_.shape[1]):
            dis = self.f[i]

            log_prob = self.gaussian_log_prob(z, dis[0], dis[1]).view(batch_size,-1).sum(-1) #/ dim
            log_probs.append(log_prob)

            energy = log_prob * self.weights[i]
            energy_outs.append(energy)

        energy_output = torch.stack(energy_outs).sum(dim=0)

        for i in range(y_.shape[1]):
            for j in range(y_.shape[1]):
                if i != j:
                    left_prob, right_prob = log_probs[i], log_probs[j]

                    if torch.sum((left_prob.detach() - right_prob.detach()) / dim) > self.eps:
                        energy_output += -0.3 * (left_prob - right_prob)
                    
                    else:
                        energy_output += -0.01 * (left_prob - right_prob)

            

        return energy_output # - 0.03*torch.norm(z, dim=1) ** 2 * 0.5

    def forward(self, z, y):
        energy_output = self.get_cond_energy(z, y)
        return energy_output


class VPODE(nn.Module):
    def __init__(self, ccf, y, beta_min=0.1, beta_max=20, T=1.0):
        super().__init__()
        self.ccf = ccf
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.T = T
        self.y = y


    def forward(self, t_k, states):
        z = states[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
            cond_energy_neg = self.ccf.get_cond_energy(z, self.y)
            cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]
            dz_dt = -0.5 * beta_t * cond_f_prime
        return dz_dt,


#ode sampling
def sample_q_ode(ccf, y, device=torch.device('cuda'), **kwargs):
    """sampling in the z space"""
    ccf.eval()
    atol = kwargs['atol']
    rtol = kwargs['rtol']
    method = kwargs['method']
    use_adjoint = kwargs['use_adjoint']
    kwargs['device'] = device
    # generate initial samples
    z_k = kwargs['z_k']
    # z_k: batch x latent_dim,
    # y: batch
    # ODE function
    vpode = VPODE(ccf, y)
    states = (z_k,)
    if 'T' in kwargs:
        times = kwargs['T']
    else:
        times = vpode.T
    integration_times = torch.linspace(times, 0., 2).type(torch.float32).to(device)

    # ODE solver
    odeint = odeint_adjoint if use_adjoint else odeint_normal
    state_t = odeint(
        vpode,  # model
        states,  # (z,)
        integration_times,
        atol=atol,  # tolerance
        rtol=rtol,
        method=method)

    ccf.train()
    z_t0 = state_t[0][-1]
    # print(f'#ODE steps : {vpode.n_evals}')
    return z_t0.detach()


#langevin dynamics sampling
def sample_q_sgld(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    sgld_lr = kwargs['sgld_lr']
    sgld_std = kwargs['sgld_std']
    n_steps = kwargs['n_steps']

    # generate initial samples
    init_sample = torch.randn(y.size(0), latent_dim).to(device)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)

    # sgld
    for k in range(n_steps):
        energy_neg = ccf(x_k, y=y)
        f_prime = torch.autograd.grad(energy_neg.sum(), [x_k])[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)

    ccf.train()
    final_samples = x_k.detach()

    return final_samples,k


#sde sampling
def sample_q_vpsde(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5,
                   beta_min=0.1, beta_max=20, T=1, eps=1e-3, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    N = kwargs['N']
    correct_nsteps = kwargs['correct_nsteps']
    target_snr = kwargs['target_snr']

    # generate initial samples
    z_init = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
    z_k = torch.autograd.Variable(z_init, requires_grad=True)

    discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    alphas = 1. - discrete_betas
    timesteps = torch.linspace(T, eps, N, device=device)

    # vpsde
    for k in range(N):
        energy_neg = ccf(z_k, y=y)

        # predictor
        t_k = timesteps[k]
        timestep = (t_k * (N - 1) / T).long()
        beta_t = discrete_betas[timestep]
        alpha_t = alphas[timestep]

        score_t = torch.autograd.grad(energy_neg.sum(), [z_k])[0]

        z_k = (2 - torch.sqrt(alpha_t)) * z_k + beta_t * score_t
        noise = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
        z_k = z_k + torch.sqrt(beta_t) * noise

        # corrector
        for j in range(correct_nsteps):
            noise = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)

            grad_norm = torch.norm(score_t.reshape(score_t.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha_t

            assert step_size.ndim == 0, step_size.ndim

            z_k_mean = z_k + step_size * score_t
            z_k = z_k_mean + torch.sqrt(step_size * 2) * noise

    ccf.train()
    final_samples = z_k.detach()

    return final_samples, k