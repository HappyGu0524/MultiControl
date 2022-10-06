import json
import torch
import transformers
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import argparse
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json


parser = argparse.ArgumentParser()
parser.add_argument("--file_loc", type=str, default="../res/AE/predict.txt")
parser.add_argument(
    "--specify",
    type=str,
    default=None
    #json.dumps([1,1,1])
)

args = parser.parse_args()

def compute_metrics(pred):
    labels = torch.tensor(pred.label_ids).long()
    preds = torch.softmax(torch.tensor(pred.predictions),dim=-1)
    probs = torch.gather(preds, 1,labels.view(-1, 1))
    acc = torch.mean(probs).item()
    #print(labels)
    #print(preds)
    #print(probs)
    #print(acc)
    #raise Exception('test')
    #preds = pred.predictions.argmax(-1)

    #precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    #acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        #'f1': f1,
        #'precision': precision,
        #'recall': recall
    }

dataset = {'label':[], 'sent':[]}
with open(args.file_loc, 'r') as f:
    for line in f.readlines():
        label, sent = json.loads(line.strip())
        dataset['label'].append(label)
        dataset['sent'].append(sent.strip())


dataset = Dataset.from_dict(dataset)

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')


model_list = ['model/Yelp2-checkpoint-64000', 'model/AGnews-checkpoint-6000', 'model/Toxic-checkpoint-3000']
topics = ["world","sports","business","science"]
task_list = ['sentiment', 'topic', 'detoxification']

test_args = TrainingArguments(
    output_dir='logs',
    do_train = False,
    do_predict = True,
    no_cuda = False,
    per_device_eval_batch_size=64,
    dataloader_drop_last = False,
    report_to='none'
)

train_out = {}

if args.specify is not None:
    specify = json.loads(args.specify)
    dataset = dataset.filter(lambda e: e['label'] == specify)

for i in range(3):
    if args.specify is not None:
        if specify[i] == -1:
            continue
    model = DebertaV2ForSequenceClassification.from_pretrained(model_list[i], num_labels=2)
    eval_dataset = None
    if 'AGnews' in model_list[i]:
        eval_dataset = dataset.map(lambda e: tokenizer(topics[e['label'][i]]+'[SEP]'+e['sent'], truncation=True, padding='max_length', max_length=100))
        eval_dataset = eval_dataset.map(lambda e: {'labels': 1})
    else:
        eval_dataset = dataset.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=100), batched=True)
        eval_dataset = eval_dataset.map(lambda e: {'labels': e['label'][i]})
    eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    trainer = Trainer(
        model=model,
        args = test_args,
        compute_metrics=compute_metrics, 
    )
    train_out[task_list[i]] = trainer.evaluate(eval_dataset)['eval_accuracy']



print(train_out)