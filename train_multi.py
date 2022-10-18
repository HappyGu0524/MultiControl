import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import json
import wandb

import random

from model import AE


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
parser.add_argument('--model_dir', default='./model/AE/')
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--latent_size", type=int, default=768)
parser.add_argument("--latent_num",type=int, default=1)
parser.add_argument("--seq_len_per_latent",type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epoch",type=int, default=200)
parser.add_argument("--lr",type=float, default=1e-4)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--no_fix", action="store_true")
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--contrasitive_loss", type=float, default=None)
parser.add_argument("--sparse_loss", type=float, default=None)
parser.add_argument("--latent_classify_loss", type=float, default=None)
parser.add_argument("--aspect_gap_loss", type=float, default=None)
parser.add_argument("--variation", type=float, default=0)

parser.add_argument("--classifier_head_num", type=int, default=3)
parser.add_argument("--classifier_class_num_per_head", type=str, default='[2,2,4]')
parser.add_argument("--classifier_mid_size", type=int, default=128)
parser.add_argument("--classifier_head_type", type=str, default='multiple', choices=('single', 'multiple'))

parser.add_argument("--aspect_gap_head_num", type=int, default=3)
parser.add_argument("--aspect_gap_amplification", type=int, default=10)

args = parser.parse_args()

if args.wandb:
    wandb.login()
    wandb.init(project="", entity="")#your account

encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
encoder = BertModel.from_pretrained(args.pretrained_encoder)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

model = AE(encoder=encoder, decoder=decoder, args=args)

loss_list = {}
latent_classify_args = None
aspect_gap_args = None
if args.contrasitive_loss is not None:
    loss_list['contrasitive_loss'] = args.contrasitive_loss
if args.sparse_loss is not None:
    loss_list['sparse_loss'] = args.sparse_loss
if args.latent_classify_loss is not None:
    loss_list['latent_classify_loss'] = args.latent_classify_loss
    latent_classify_args = {
        'head_num':args.classifier_head_num,
        'class_num_per_head':[2,2,4],#json.loads(args.classifier_class_num_per_head),
        'mid_size':args.classifier_mid_size,
        'head_type':args.classifier_head_type
    }
if args.aspect_gap_loss is not None:
    loss_list['aspect_gap_loss'] = args.aspect_gap_loss
    aspect_gap_args = {
        'head_num':args.aspect_gap_head_num,
        'amplification':args.aspect_gap_amplification
    }



if len(loss_list) == 0:
    loss_list = None

model.set_losslist(loss_list, latent_classify_args, aspect_gap_args)

if not args.no_fix:
    model.fix_decoder()


if args.classifier_head_type == 'multiple':
    dataset = [{'sent':[], 'type':[]} for i in range(3)]
    if 'contrasitive_loss' in loss_list:
        for i in range(len(dataset)):
            dataset[i]['adv_sent'] = []
    
    with open('data/IMDb/IMDb.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            dataset[0]['sent'].append(line[1].strip())
            dataset[0]['type'].append(int(line[0]))

    with open('data/ToxicComment/Toxic.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            dataset[1]['sent'].append(line[1].strip())
            dataset[1]['type'].append(int(line[0]))
    
    with open('data/AGnews/AG-data.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            dataset[2]['sent'].append(line[1].strip())
            dataset[2]['type'].append(int(line[0]))


    
    columns = ['encoder_input_ids', 'encoder_attention_mask', 'encoder_token_type_ids', 'type', 'decoder_input_ids', 'decoder_attention_mask']
    if 'contrasitive_loss' in loss_list:
        columns.extend(['adv_input_ids', 'adv_attention_mask', 'adv_token_type_ids'])
    if 'latent_classify_loss' in loss_list:
        columns.extend(['pos_label', 'neg_labels'])
    train_dataset = {i:[] for i in columns}
    train_dataset['head_index']=[]
    for i in range(3):
        tmp_dataset = Dataset.from_dict(dataset[i])
        tmp_dataset = tmp_dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
        tmp_dataset = tmp_dataset.rename_columns({'input_ids':'encoder_input_ids', 'attention_mask':'encoder_attention_mask', 'token_type_ids':'encoder_token_type_ids'})
        tmp_dataset = tmp_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
        tmp_dataset = tmp_dataset.rename_columns({'input_ids':'decoder_input_ids', 'attention_mask':'decoder_attention_mask'})
        if 'contrasitive_loss' in loss_list:
            tmp_dataset = tmp_dataset.map(lambda e: encoder_tokenizer(e['adv_sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
            tmp_dataset = tmp_dataset.rename_columns({'input_ids':'adv_input_ids', 'attention_mask':'adv_attention_mask', 'token_type_ids':'adv_token_type_ids'})
        if 'latent_classify_loss' in loss_list:
            tmp_dataset = tmp_dataset.map(lambda e: {'pos_label': e['type'], 'neg_labels': [1 - e['type']]})
        
        tmp_dataset.set_format(type='torch', columns=columns)

        tmp_dataloader = torch.utils.data.DataLoader(tmp_dataset, batch_size=args.batch_size)
        for cnt in iter(tmp_dataloader):
            for k in columns:
                train_dataset[k].append(cnt[k])
            train_dataset['head_index'].append(i)
    train_dataset = Dataset.from_dict(train_dataset)
    train_dataset.set_format(columns=columns+['head_index'])

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        #gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        logging_dir='./logs',
        logging_steps=100,
        do_train=True,
        do_eval=False,
        no_cuda=args.no_cuda,
        save_strategy="steps",
        save_steps=2000,
        fp16=args.fp16,
        report_to='wandb' if args.wandb else 'none'
    )





elif args.classifier_head_type == 'single':
    #need implemention
    dataset = {'sent':[], 'type':[], 'neg_types':[]}
    if 'contrasitive_loss' in loss_list:
        dataset['adv_sent'] = []


    with open('data/SST/pos.txt', 'r') as f:
        for line in f.readlines():
            dataset['sent'].append(line.strip())
            dataset['type'].append(1)
            dataset['neg_types'].append([0])

    with open('data/SST/neg.txt', 'r') as f:
        for line in f.readlines():
            dataset['sent'].append(line.strip())
            dataset['type'].append(0)
            dataset['neg_types'].append([1])


    with open('data/topic/military/s_military.json', 'r') as f:
        for line in json.loads(f.read()):
            dataset['sent'].append(line.strip().split('<|endoftext|>')[0].strip())
            dataset['type'].append(3)
            dataset['neg_types'].append([2])

    with open('data/topic/military/s_common.json', 'r') as f:
        for line in json.loads(f.read()):
            dataset['sent'].append(line.strip().split('<|endoftext|>')[0].strip())
            dataset['type'].append(2)
            dataset['neg_types'].append([3])

    train_dataset = Dataset.from_dict(dataset)

    train_dataset = train_dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    train_dataset = train_dataset.rename_columns({'input_ids':'encoder_input_ids', 'attention_mask':'encoder_attention_mask', 'token_type_ids':'encoder_token_type_ids'})
    train_dataset = train_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    train_dataset = train_dataset.rename_columns({'input_ids':'decoder_input_ids', 'attention_mask':'decoder_attention_mask'})
    columns = ['encoder_input_ids', 'encoder_attention_mask', 'encoder_token_type_ids', 'type', 'decoder_input_ids', 'decoder_attention_mask']
    if 'contrasitive_loss' in loss_list:
        train_dataset = train_dataset.map(lambda e: encoder_tokenizer(e['adv_sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
        train_dataset = train_dataset.rename_columns({'input_ids':'adv_input_ids', 'attention_mask':'adv_attention_mask', 'token_type_ids':'adv_token_type_ids'})
        columns.extend(['adv_input_ids', 'adv_attention_mask', 'adv_token_type_ids'])

    if 'latent_classify_loss' in loss_list:
        train_dataset = train_dataset.map(lambda e: {'pos_label': e['type'], 'neg_labels': e['neg_types']})
        columns.extend(['pos_label', 'neg_labels'])


    train_dataset.set_format(type='torch', columns=columns)
    
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        logging_dir='./logs',
        logging_steps=100,
        do_train=True,
        do_eval=False,
        no_cuda=args.no_cuda,
        save_strategy="steps",
        save_steps=2000,
        fp16=args.fp16,
        report_to='wandb' if args.wandb else 'none'
    )



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
train_out = trainer.train()