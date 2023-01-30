# Install
! pip install transformers
! pip install ijson
# Import
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import ijson
import numpy as np

class i2b2Dataset(Dataset):
  def __init__(self, fpath):
    # Set file path
    self.fpath = fpath
    # Read data into array
    self.ids, self.contexts, self.triplets = self.readfile(fpath)
  
  def readfile(self, fpath):
    ids = []
    contexts = {}
    triplets = {}
    with open(fpath, 'r') as f:
      for dic in ijson.items(f, '', multiple_values = True):
        id = dic['id']
        ids.append(id)
        contexts[id] = dic['context']['text'].strip()
        triplets[id] = dic['triplet'].strip()
    return ids, contexts, triplets

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    max_length = 1024
    # given a problem index idx, recover instance
    id = self.ids[idx]
    context = self.contexts[id]
    triplet = self.triplets[id]
    model_inputs = tokenizer(text=context, max_length=max_length, padding='max_length', truncation=True)
    model_inputs['labels'] = tokenizer(text=triplet, max_length=max_length, padding='max_length', truncation=True)["input_ids"]
    return model_inputs

# Set file paths
dataDir = "/content/drive/MyDrive/Colab Notebooks/Thesis Model/data/"
modelDir = "/content/drive/MyDrive/Colab Notebooks/Thesis Model/model/"
trainJsonPath = dataDir + "TrainJson_MergedTLINK.json"
testJsonPath = dataDir + "TestJson_MergedTLINK.json"

legacy_tokens = ['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>']
model_name = "Babelscape/rebel-large"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add I2B2 entity marker tokens to tokenizer
new_tokens = ['<prob>','<test>','<tret>','<dept>','<evid>','<occr>','<date>','<time>','<durt>','<freq>']
tokenizer.add_tokens(new_tokens, special_tokens = True)
# Load tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

import tensorflow as tf
text = 'Admission Date : 2014-11-29 Discharge Date : 2014-12-05 Service : MEDICINE History of Present Illness : 47 yo F w/ h/o steroid-induced hyperglycemia , SLE w/ h/o pericarditis , transverse myelitis w/ paraplegia and neurogenic bladder s/p urostomy w/ ileal conduit , h/o ureteropelvic stone and urosepsis , and h/o RLE DVT a/w F 2014-11-29 transferred to CMED 2014-11-30 for hypotn resistant to IVFs and stress steroids . Patient initially p/w c/o sudden onset N/abd pain/chills w/ T 103 at NH . Rigors progressed so she was brought to Robert . She reported h/o fatigue and anorexia for the past few days and had noticed foul smelling urine and some abdominal distension , similar to prior episodes of pyelo . She also c/o LLQ and groin pain which responded to tylenol . She denies V or D . No AMS . No c/o CP . On arrival to Shirley , temperature was 101.2 . CT abd showed an 8 mm right proximal ureteral stone with right-sided hydronephrosis and inflammatory stranding , in addition to pyelonephritis of the left kidney wi'
print(len(tokenizer(text)['input_ids']))
print(tokenizer.batch_decode(tokenizer(text)['input_ids']))

trainSet = i2b2Dataset(trainJsonPath)
testSet = i2b2Dataset(testJsonPath)

batch_size = 1
args = Seq2SeqTrainingArguments(
    output_dir = modelDir,
    evaluation_strategy = "no",
    save_strategy = "no",
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    learning_rate = 0.00005,
    warmup_steps=1000,
    weight_decay = 0.01,
    num_train_epochs = 10,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model = model,
    args = args,
    data_collator = data_collator,
    train_dataset = trainSet,
    eval_dataset = testSet,
    tokenizer = tokenizer,
)

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

import gc

gc.collect()

torch.cuda.empty_cache()

trainer.train()

trainer.save_model(modelDir)
