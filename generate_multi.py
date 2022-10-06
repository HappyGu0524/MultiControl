import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from tqdm import tqdm
import json
from sklearn.cluster import KMeans
import random
import numpy as np

from generation_utils import KCenters
from model import AE


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--latent_size", type=int, default=768)
parser.add_argument("--latent_num",type=int, default=1)
parser.add_argument("--seq_len_per_latent",type=int, default=20)
parser.add_argument("--model_path", type=str, default='./model/AE/checkpoint-30000/pytorch_model.bin')
parser.add_argument("--output_dir", type=str, default="./res/AE/predict_final.txt")
parser.add_argument("--batch_size", type=int, default=60)
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
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--variation", type=float, default=1e-3)

#Parameters for KCenters
parser.add_argument("--num_centers", type=int, default=1000)
parser.add_argument("--num_output_centers", type=int, default=10)
parser.add_argument("--topk", type=int, default=200)
parser.add_argument("--batch", type=int, default=5)
parser.add_argument("--max_iter", type=int, default=15)
parser.add_argument("--strategy", type=str, default='none', choices=('none', 'weight'))
parser.add_argument("--temperature", type=float, default=50)
parser.add_argument("--SDM_reinit", type=bool, default=True)
parser.add_argument("--weight",
    type=str, 
    default=json.dumps(
        [1,5,1]
    )
)
parser.add_argument("--config", type=str, default=None)

args = parser.parse_args()

weight = json.loads(args.weight)

if args.config is not None:
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
        for keys in config:
            if keys == 'weight':
                weight = config['weight']
            if keys == 'num_output_centers':
                args.num_output_centers = config['num_output_centers']


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


if isinstance(args.num_output_centers, int):
    args.num_output_centers = [[args.num_output_centers]*4]*2





encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
encoder = BertModel.from_pretrained(args.pretrained_encoder)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

model = AE(encoder=encoder, decoder=decoder, args=args)
model.load_state_dict(torch.load(args.model_path), strict=False)
model.eval()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.no_cuda:
    device='cpu'
else:
    device='cuda'

model.to(device)

imdb_dataset = [{'sent':[]} for i in range(2)]
ag_dataset = [{'sent':[]} for i in range(4)]
toxic_dataset = [{'sent':[]} for i in range(2)]

with open('data/IMDb/IMDb.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        imdb_dataset[label]['sent'].append(line[1].strip())

with open('data/ToxicComment/Toxic.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        toxic_dataset[label]['sent'].append(line[1].strip())

with open('data/AGnews/AG-data.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        ag_dataset[label]['sent'].append(line[1].strip())
        label = int(line[0])
        ag_dataset[label]['sent'].append(line[1].strip())

#with open('data/SST/pos.txt', 'r') as f:
#    for line in f.readlines():
#        pos_dataset['sent'].append(line.strip())
#        pos_dataset['type'].append(1)

#with open('data/SST/neg.txt', 'r') as f:
#    for line in f.readlines():
#        neg_dataset['sent'].append(line.strip())
#        neg_dataset['type'].append(0)

#with open('data/toxic/s_notoxic.json', 'r') as f:
#    for line in json.loads(f.read()):
#        not_dataset['sent'].append(line.strip())
#        not_dataset['type'].append(3)

#with open('data/toxic/s_toxic.json', 'r') as f:
#    for line in json.loads(f.read()):
#        tox_dataset['sent'].append(line.strip())
#        tox_dataset['type'].append(2)

imdb_dataset = [Dataset.from_dict(i) for i in imdb_dataset]
ag_dataset = [Dataset.from_dict(i) for i in ag_dataset]
toxic_dataset = [Dataset.from_dict(i) for i in toxic_dataset]


#if args.pre_tokens is not None and args.pre_tokens > 1:
#    pos_train_dataset = pos_train_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.pre_tokens, padding='max_length', truncation=True), batched=True)
#    pos_train_dataset = pos_train_dataset.rename_columns({'input_ids':'decoder_input_ids', 'attention_mask':'decoder_attention_mask'})
#    pos_train_dataset = pos_train_dataset.map(lambda e: encoder_tokenizer(e['sent'], padding=True, truncation=True), batched=True, batch_size=args.batch_size)
#    pos_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'decoder_input_ids', 'decoder_attention_mask', 'type'])

#    neg_train_dataset = neg_train_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.pre_tokens, padding='max_length', truncation=True), batched=True)
#    neg_train_dataset = neg_train_dataset.rename_columns({'input_ids':'decoder_input_ids', 'attention_mask':'decoder_attention_mask'})
#    neg_train_dataset = neg_train_dataset.map(lambda e: encoder_tokenizer(e['sent'], padding=True, truncation=True), batched=True, batch_size=args.batch_size)
#    neg_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'decoder_input_ids', 'decoder_attention_mask', 'type'])

#    not_train_dataset = not_train_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.pre_tokens, padding='max_length', truncation=True), batched=True)
#    not_train_dataset = not_train_dataset.rename_columns({'input_ids':'decoder_input_ids', 'attention_mask':'decoder_attention_mask'})
#    not_train_dataset = not_train_dataset.map(lambda e: encoder_tokenizer(e['sent'], padding=True, truncation=True), batched=True, batch_size=args.batch_size)
#    not_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'decoder_input_ids', 'decoder_attention_mask', 'type'])

#    tox_train_dataset = tox_train_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.pre_tokens, padding='max_length', truncation=True), batched=True)
#    tox_train_dataset = tox_train_dataset.rename_columns({'input_ids':'decoder_input_ids', 'attention_mask':'decoder_attention_mask'})
#    tox_train_dataset = tox_train_dataset.map(lambda e: encoder_tokenizer(e['sent'], padding=True, truncation=True), batched=True, batch_size=args.batch_size)
#    tox_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'decoder_input_ids', 'decoder_attention_mask', 'type'])
#else:
#    raise Exception('args.pre_tokens not in required ranges')

imdb_dataloader = []
for dataset in imdb_dataset:
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    imdb_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))

ag_dataloader = []
for dataset in ag_dataset:
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    ag_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))

toxic_dataloader = []
for dataset in toxic_dataset:
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    toxic_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))



#pos_latents = None
#sports_latents = None
not_latents = None
sentiment_latents = {0:None, 1:None}
topic_latents = {0:None, 1:None, 2:None, 3:None}

for i in range(2):
    for cnt in tqdm(iter(imdb_dataloader[i])):
        encoder_input_ids = cnt['input_ids']
        encoder_attention_mask = cnt['attention_mask']
        encoder_token_type_ids = cnt['token_type_ids']
        
        latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask, encoder_token_type_ids)
        if sentiment_latents[i] is None:
            sentiment_latents[i] = latent.squeeze().detach()
        else:
            sentiment_latents[i] = torch.cat((sentiment_latents[i], latent.squeeze().detach()), dim=0)

for i in range(4):
    for cnt in tqdm(iter(ag_dataloader[i])):
        encoder_input_ids = cnt['input_ids']
        encoder_attention_mask = cnt['attention_mask']
        encoder_token_type_ids = cnt['token_type_ids']
        
        latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask, encoder_token_type_ids)
        if topic_latents[i] is None:
            topic_latents[i] = latent.squeeze().detach()
        else:
            topic_latents[i] = torch.cat((topic_latents[i], latent.squeeze().detach()), dim=0)


for cnt in tqdm(iter(toxic_dataloader[1])):
    encoder_input_ids = cnt['input_ids']
    encoder_attention_mask = cnt['attention_mask']
    encoder_token_type_ids = cnt['token_type_ids']
    
    latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask, encoder_token_type_ids)
    if not_latents is None:
        not_latents = latent.squeeze().detach()
    else:
        not_latents = torch.cat((not_latents, latent.squeeze().detach()), dim=0)




#pos_latents = pos_latents.to('cpu').numpy()
#not_latents = not_latents.to('cpu').numpy()
#neg_latents = neg_latents.to('cpu').numpy()

kcmodel = KCenters(num_centers=args.num_centers, latent_size=args.latent_size, num_output_centers=args.num_output_centers, device='cuda')

output_text = []
labels = []



for i in range(2):
    for j in range(4):
        weight = weight_dict[i][j]
        num_output_centers = args.num_output_centers[i][j]
        print(weight)
        print(num_output_centers)
        centers = kcmodel.train(
            [sentiment_latents[i].to('cuda'), topic_latents[j].to('cuda'), not_latents.to('cuda')],
            weight=weight,
            topk=args.topk,
            SDM_reinit=args.SDM_reinit,
            max_iter=args.max_iter,
            strategy=args.strategy,
            temperature=args.temperature,
            num_output_centers=num_output_centers
            ).cpu().numpy()
        centers = [torch.FloatTensor(k).unsqueeze(0) for k in centers]


        for prompts in tqdm(json.loads(args.pre_tokens)):
            tokens = decoder_tokenizer(prompts, return_tensors='pt')
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
            input_ids = input_ids.expand(args.batch_size, -1)
            attention_mask = attention_mask.expand(args.batch_size, -1)

            output = model.generate(
                input_latent=random.choice(centers),
                input_ids=input_ids,
                attention_mask=attention_mask,
                variation=args.variation,
                max_len=50,
                rp=1.2
            )

            output_text.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
            labels.extend([[i,j,1]] * args.batch_size)
            assert len(labels) == len(output_text)


#for cnt in tqdm(iter(pos_dataloader)):
#    encoder_input_ids = cnt['input_ids']
#    encoder_attention_mask = cnt['attention_mask']
#    encoder_token_type_ids = cnt['token_type_ids']
#    decoder_input_ids = cnt['decoder_input_ids']
#    decoder_attention_mask = cnt['decoder_attention_mask']

    
#    output = model.generate(
#        input_latent=random.choice(centers),
#        input_ids=decoder_input_ids,
#        attention_mask=decoder_attention_mask,
#        variation=args.variation,
#        max_len=50,
#        rp=1.2
#        )

#    raw.extend(encoder_tokenizer.batch_decode(encoder_input_ids.cpu(), skip_special_tokens=True))
#    gen.extend([i.strip().split('\n\n')[0].strip() for i in decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)])
#    gen.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
#    label.extend(cnt['type'].tolist())
#    assert len(label) == len(gen)

with open(args.output_dir, 'w') as f:
    for i in tqdm(range(len(output_text))):
        f.write(json.dumps([labels[i], output_text[i]])+'\n')
