import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
from tqdm import tqdm
import json
import random
import numpy as np

from model import AE


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--latent_size", type=int, default=768)
parser.add_argument("--latent_num",type=int, default=1)
parser.add_argument("--seq_len_per_latent",type=int, default=20)
parser.add_argument("--model_path", type=str, default='../model/priorcontrol/All_notricks_checkpoint-300000/pytorch_model.bin')
parser.add_argument("--output_dir", type=str, default="../res/priorcontrol/generate_combination.txt")
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

args = parser.parse_args()


weight = json.loads(args.weight)
std = args.std

if args.config is not None:
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
        for keys in config:
            if keys == 'weight':
                weight = config['weight']
            
            if keys == 'std':
                std = config['std']


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







encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
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





output_text = []
labels = []
for i in range(2):
    for j in range(4):
        weight = weight_dict[i][j]

        if isinstance(std, list):
            tmp_std = std[i*4+j]
        else:
            tmp_std = std


        for prompts in tqdm(json.loads(args.pre_tokens)):
            tokens = decoder_tokenizer(prompts, return_tensors='pt')
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
            input_ids = input_ids.expand(args.batch_size, -1)
            attention_mask = attention_mask.expand(args.batch_size, -1)

            #latents = torch.normal(0,1, (args.batch_size, args.latent_num * args.latent_size))
            latents = torch.zeros(args.batch_size, args.latent_num * args.latent_size)

            output = model.generate(
                input_latent=latents,
                input_ids=input_ids,
                attention_mask=attention_mask,
                variation=args.variation,
                max_len=50,
                rp=1.2,
                prior_head_index=[i,j+2,7],
                alpha=weight,
                calculate=calculate,
                std=tmp_std
            )

            output_text.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
            labels.extend([[i,j,1]] * args.batch_size)
            assert len(labels) == len(output_text)



with open(args.output_dir, 'w') as f:
    for i in tqdm(range(len(output_text))):
        f.write(json.dumps([labels[i], output_text[i]])+'\n')
