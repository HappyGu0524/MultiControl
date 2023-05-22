import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
from tqdm import tqdm
import json
import random
import numpy as np
from functools import partial
import math

from model import AE
from latentops_modules import DIS, DIScons, sample_q_ode

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--latent_size", type=int, default=768)
parser.add_argument("--latent_num",type=int, default=1)
parser.add_argument("--seq_len_per_latent",type=int, default=20)
parser.add_argument("--model_path", type=str, default='../model/priorcontrol/All_notricks_checkpoint-300000/pytorch_model.bin')
parser.add_argument("--output_dir", type=str, default="../res/priorcontrol/generate_combination_optim.txt")
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--pre_tokens",
    type=str,
    default=json.dumps(
        ['In summary','This essay discusses','Views on','The connection','Foundational to this is',
        'To review,','In brief,','An illustration of','Furthermore,','The central theme',
        'To conclude,','The key aspect','Prior to this','Emphasised are','To summarise',
        'The relationship','More importantly,','It has been shown','The issue focused on','In this essay',
        'Once upon a time','The book','The chicken','The city','The country',
        'The horse','The lake','The last time','The movie','The painting',
        'The pizza','The potato','The president of the country','The road','The year is 1910']
    )
)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--variation", type=float, default=0)

#Parameters for Prior
parser.add_argument("--prior", type=bool, default=True)
parser.add_argument("--flow_num", type=int, default=8)
parser.add_argument("--prior_num", type=int, default=8)


#Generation
parser.add_argument("--std", type=float, default=1)

#Attribute Combination
parser.add_argument("--weight",
    type=str, 
    default=json.dumps(
        [1,5,1]
    )
)
parser.add_argument("--config", type=str, default="generate_config_combine.json")

parser.add_argument("--optim_weight",
    type=str, 
    default=json.dumps(
        [1,1,1]
    )
)
parser.add_argument("--optim_config", type=str, default="generate_config_combine_optim.json")

#Optimizaiton with or without constraints
parser.add_argument("--is_constrained", action="store_true")
args = parser.parse_args()


weight = json.loads(args.weight)
optim_weight = json.loads(args.optim_weight)
std = args.std

if args.config is not None:
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
        for keys in config:
            if keys == 'weight':
                weight = config['weight']
            
            if keys == 'std':
                std = config['std']

if args.optim_config is not None:
    with open(args.optim_config, 'r') as f:
        optim_config = json.loads(f.read())
        for keys in config:
            if keys == 'weight':
                optim_weight = optim_config['weight']


if isinstance(weight, dict):
    default_weight = weight['default']
    weight_dict = [[default_weight for jt in range(4)]for it in range(2)]
    for keys in weight:
        if keys != 'default':
            tmp_i = int(keys[0])
            tmp_j = int(keys[1])
            weight_dict[tmp_i][tmp_j] = weight[keys]
else:
    weight_dict = [[weight for jt in range(4)]for it in range(2)]


if isinstance(optim_weight, dict):
    default_weight = optim_weight['default']
    optim_weight_dict = [[default_weight for jt in range(4)]for it in range(2)]
    for keys in optim_weight:
        if keys != 'default':
            tmp_i = int(keys[0])
            tmp_j = int(keys[1])
            optim_weight_dict[tmp_i][tmp_j] = optim_weight[keys]
else:
    optim_weight_dict = [[optim_weight for jt in range(4)]for it in range(2)]




encoder = BertModel.from_pretrained(args.pretrained_encoder)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

model = AE(encoder=encoder, decoder=decoder, args=args)


model.load_state_dict(torch.load(args.model_path), strict=False)
model.eval()
model.fix_decoder()
model.set_mode('prior')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.no_cuda:
    device='cpu'
else:
    device='cuda'

model.to(device)


def calculate(alpha, mean, std, eps):
    total = sum(alpha)
    alpha = [num/total for num in alpha]
    mu = 0
    for w, mean in zip(alpha, mean):
        mu = mu + w * mean

    sigma = 0
    for w, std in zip(alpha, std):
        if w < 0:
            w = 0
        if w > 1:
            w = 1
        sigma = sigma + (w * std)**2
    sigma = torch.sqrt(sigma)
    
    sampled_dis = sigma * eps + mu
    return sampled_dis


def gaussian_log_prob(x, mu, log_sd):
    return -0.5 * math.log(2 * torch.pi) - log_sd - 0.5 * (x - mu) ** 2 / torch.exp(2 * log_sd)

priors = model.priors
#disdict = []
#for prior in priors:
#    disdict.append(DIS([prior], [10]).to(device))


ode_kwargs = {'atol': 1e-3, 'rtol': 1e-3, 'method': 'dopri5', 'use_adjoint': True, 'latent_dim': args.latent_size}
sampler = partial(sample_q_ode, device=device, **ode_kwargs)

model.set_mode('normal')

output_text = []
labels = []
for i in range(2):
    for j in range(4):
        weight = weight_dict[i][j]
        optim_weight = optim_weight_dict[i][j]

        if isinstance(std, list):
            tmp_std = std[i*4+j]
        else:
            tmp_std = std

        if args.is_constrained:
            dismodel = DIScons([priors[i], priors[2+j], priors[7]], [optim_weight[0], optim_weight[1], optim_weight[2]]).to(device)
        else:
            dismodel = DIS([priors[i], priors[2+j], priors[7]], [optim_weight[0], optim_weight[1], optim_weight[2]]).to(device)

        probs_raw = None
        probs_dis = None


        for prompts in tqdm(json.loads(args.pre_tokens)):
            tokens = decoder_tokenizer(prompts, return_tensors='pt')
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
            input_ids = input_ids.expand(args.batch_size, -1)
            attention_mask = attention_mask.expand(args.batch_size, -1)

            #latents = torch.normal(0,1, (args.batch_size, args.latent_num * args.latent_size))
            #latents = torch.zeros(args.batch_size, args.latent_num * args.latent_size)
            eps = torch.normal(0,tmp_std, (args.batch_size, args.latent_size)).to(device)
            
            prior_head_index = [i,j+2,7]

            learnable_prior_mean = [priors[p_index][0] for p_index in prior_head_index]
            learnable_prior_std = [torch.exp(priors[p_index][1]) for p_index in prior_head_index]

            z_k = calculate(weight, learnable_prior_mean, learnable_prior_std, eps)

            if probs_raw is None:
                probs_raw = [gaussian_log_prob(z_k, priors[i][0], priors[i][1]).view(args.batch_size,-1).sum(-1) / args.latent_size,
                            gaussian_log_prob(z_k, priors[j+2][0], priors[j+2][1]).view(args.batch_size,-1).sum(-1) / args.latent_size,
                            gaussian_log_prob(z_k, priors[7][0], priors[7][1]).view(args.batch_size,-1).sum(-1) / args.latent_size
                        ]
            else:
                probs_raw = [torch.concat([probs_raw[0], gaussian_log_prob(z_k, priors[i][0], priors[i][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0),
                            torch.concat([probs_raw[1], gaussian_log_prob(z_k, priors[j+2][0], priors[j+2][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0),
                            torch.concat([probs_raw[2], gaussian_log_prob(z_k, priors[7][0], priors[7][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0)
                        ]

            y = torch.tensor([prior_head_index] * args.batch_size).to(device)
            latent = sampler(ccf=dismodel, y=y, z_k=z_k.clone())

            if probs_dis is None:
                probs_dis = [gaussian_log_prob(latent, priors[i][0], priors[i][1]).view(args.batch_size,-1).sum(-1) / args.latent_size,
                            gaussian_log_prob(latent, priors[j+2][0], priors[j+2][1]).view(args.batch_size,-1).sum(-1) / args.latent_size,
                            gaussian_log_prob(latent, priors[7][0], priors[7][1]).view(args.batch_size,-1).sum(-1) / args.latent_size
                        ]
            else:
                probs_dis = [torch.concat([probs_dis[0], gaussian_log_prob(latent, priors[i][0], priors[i][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0),
                            torch.concat([probs_dis[1], gaussian_log_prob(latent, priors[j+2][0], priors[j+2][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0),
                            torch.concat([probs_dis[2], gaussian_log_prob(latent, priors[7][0], priors[7][1]).view(args.batch_size,-1).sum(-1) / args.latent_size], dim=0)
                        ]            
            
            input_latent, _ = model.inv_flow(latent, rev=True)
            output = model.generate(
                input_latent=input_latent,
                input_ids=input_ids,
                attention_mask=attention_mask,
                variation=args.variation,
                max_len=50,
                rp=1.2
            )

            output_text.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
            labels.extend([[i,j,1]] * args.batch_size)
            assert len(labels) == len(output_text)
        
        if args.is_constrained:
            log_dir = 'logs/logs_combine_optimcons.txt'
        else:
            log_dir = 'logs/logs_combine_optim.txt'
        with open(log_dir, 'a') as f:
            f.write(str(i) + '\n')
            f.write('RAW:' + str([res.mean().item() for res in probs_raw]) + '\n')
            f.write('ODE:' + str([res.mean().item() for res in probs_dis]) + '\n')


with open(args.output_dir, 'w') as f:
    for i in tqdm(range(len(output_text))):
        f.write(json.dumps([labels[i], output_text[i]])+'\n')
