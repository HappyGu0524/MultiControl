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
parser.add_argument('--model_dir', default='../model/priorcontrol/')
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--latent_size", type=int, default=768)
parser.add_argument("--latent_num",type=int, default=1)
parser.add_argument("--seq_len_per_latent",type=int, default=20)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epoch",type=int, default=200)
parser.add_argument("--lr",type=float, default=1e-4)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--no_fix", action="store_true")
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--model_path", type=str, default='../model/multicontrol/checkpoint-30000/pytorch_model.bin')
parser.add_argument("--variation", type=float, default=1e-3)

#prior
parser.add_argument("--not_prior", action="store_true")
parser.add_argument("--flow_num", type=int, default=8)
parser.add_argument("--prior_num", type=int, default=8)

#adv_z_prior
parser.add_argument("--adv_z_prob_loss", type=float, default=None)
parser.add_argument("--adv_z_prob_grouping", default=json.dumps([[0,1],[2,3,4,5],[6,7]]))

#adv_x_prior
parser.add_argument("--adv_x_prob_loss", type=float, default=None)

#prior_classify_loss

parser.add_argument("--prior_classify_loss", type=float, default=0.3)
parser.add_argument("--prior_classifier_head_num", type=int, default=3)
parser.add_argument("--prior_classifier_class_num_per_head", type=str, default=json.dumps([2,2,4]))
parser.add_argument("--prior_classifier_mid_size", type=int, default=128)

args = parser.parse_args()

if not args.not_prior:
    args.prior = True
else:
    args.prior = False

if args.wandb:
    wandb.login()
    wandb.init(project="", entity="")#your account




adv_z_prob_args = None
prior_classify_args = None

loss_list = {}
if args.adv_z_prob_loss is not None:
    loss_list['adv_z_prob_loss'] = args.adv_z_prob_loss
    adv_z_prob_args = {
        "grouping": json.loads(args.adv_z_prob_grouping)
    }

if args.adv_x_prob_loss is not None:
    loss_list['adv_x_prob_loss'] = args.adv_x_prob_loss

if args.prior_classify_loss is not None:
    loss_list['prior_classify_loss'] = args.prior_classify_loss
    prior_classify_args = {
        'head_num':args.prior_classifier_head_num, 
        'class_num_per_head':json.loads(args.prior_classifier_class_num_per_head),
        'mid_size':args.prior_classifier_mid_size
    }




encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
encoder = BertModel.from_pretrained(args.pretrained_encoder)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token




model = AE(encoder=encoder, decoder=decoder, args=args)
model.load_state_dict(torch.load(args.model_path), strict=False)


model.set_losslist(loss_list, adv_z_prob_args=adv_z_prob_args, prior_classify_args=prior_classify_args)


if not args.no_fix:
    model.fix_decoder()

model.set_mode('prior')


dataset = [{'sent':[]} for i in range(8)]

with open('../data/IMDb/IMDb.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        dataset[0+label]['sent'].append(line[1].strip())
        #dataset[0]['type'].append(int(line[0]))

with open('../data/AGnews/AG-data.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        dataset[2+label]['sent'].append(line[1].strip())
        #dataset[1]['type'].append(int(line[0]))

with open('../data/ToxicComment/Toxic.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        dataset[6+label]['sent'].append(line[1].strip())
        #dataset[2]['type'].append(int(line[0]))



columns = ['encoder_input_ids', 'encoder_attention_mask', 'encoder_token_type_ids']
adv_columns = ['adv_input_ids', 'adv_attention_mask', 'adv_token_type_ids']

if args.adv_x_prob_loss is not None:
    train_dataset = {i:[] for i in (columns + adv_columns)}
else:
    train_dataset = {i:[] for i in columns}

if args.prior_classify_loss is not None:
    train_dataset['head_index'] = []
    train_dataset['pos_label'] = []
train_dataset['prior_head_index']=[]

#for i in range(8):

if args.adv_x_prob_loss is not None:
    #####adv_data
    adv_sent = dataset[3]['sent'] + dataset[4]['sent'] + dataset[5]['sent']
    random.shuffle(adv_sent)
    adv_dataset = {'sent':adv_sent}

    tmp_dataset = Dataset.from_dict(adv_dataset)
    tmp_dataset = tmp_dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset = tmp_dataset.rename_columns({'input_ids':'adv_input_ids', 'attention_mask':'adv_attention_mask', 'token_type_ids':'adv_token_type_ids'})
    tmp_dataset.set_format(type='torch', columns=['adv_input_ids', 'adv_token_type_ids', 'adv_attention_mask'])
    adv_dataloader = torch.utils.data.DataLoader(tmp_dataset, batch_size=(args.batch_size * 3))
    

if args.prior_classify_loss is not None:
    label_dict=[[0,0],[0,1],[2,0],[2,1],[2,2],[2,3],[1,0],[1,1]]


#for i in [2,3,4,5]:
for i in range(8):
    tmp_dataset = Dataset.from_dict(dataset[i])
    tmp_dataset = tmp_dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset = tmp_dataset.rename_columns({'input_ids':'encoder_input_ids', 'attention_mask':'encoder_attention_mask', 'token_type_ids':'encoder_token_type_ids'})
    tmp_dataset.set_format(type='torch', columns=['encoder_input_ids', 'encoder_token_type_ids', 'encoder_attention_mask'])
    tmp_dataloader = torch.utils.data.DataLoader(tmp_dataset, batch_size=args.batch_size)
    for cnt in iter(tmp_dataloader):
        for k in columns:
            train_dataset[k].append(cnt[k].tolist())
        train_dataset['prior_head_index'].append(i)
        if args.prior_classify_loss is not None:
            train_dataset['head_index'].append(label_dict[i][0])
            train_dataset['pos_label'].append([label_dict[i][1]]*args.batch_size)
    
    if args.adv_x_prob_loss is not None:
        for adv_cnt in iter(adv_dataloader):
            for k in adv_columns:
                train_dataset[k].append(adv_cnt[k].tolist())

    


train_dataset = Dataset.from_dict(train_dataset)

ult_columns = columns + ['prior_head_index']
if args.adv_x_prob_loss is not None:
    ult_columns = ult_columns + adv_columns
    #train_dataset.set_format(columns=columns+adv_columns+['prior_head_index'])
if args.prior_classify_loss is not None:
    ult_columns = ult_columns + ['head_index', 'pos_label']
    #train_dataset.set_format(columns=columns+['prior_head_index'])

train_dataset.set_format(columns=ult_columns)






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
    save_steps=5000,
    fp16=args.fp16,
    report_to='wandb' if args.wandb else 'none'
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
train_out = trainer.train()
