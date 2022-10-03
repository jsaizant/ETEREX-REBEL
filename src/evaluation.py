# Install
!pip install torch
!pip install transformers
!pip install ijson
# Import
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import ijson
import numpy as np
from lxml import etree
import xml.etree.ElementTree as ET
import re

# Set file paths
dataDir = "/content/drive/MyDrive/Colab Notebooks/timexes_thesis/data/"
modelDir = "/content/drive/MyDrive/Colab Notebooks/timexes_thesis/model/"
trainPath = dataDir + "train_Merged/"
testPath = dataDir + "test_Merged/"

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(modelDir)
model = AutoModelForSeq2SeqLM.from_pretrained(modelDir)

type2abbr = {# EVENT
             "PROBLEM": "<prob>",
             "TEST": "<test>",
             "TREATMENT": "<tret>",
             "CLINICAL_DEPT": "<dept>",
             "EVIDENTIAL": "<evid>",
             "OCCURRENCE": "<occr>",
             # TIMEX3
             "DATE": "<date>",
             "TIME": "<time>",
             "DURATION": "<durt>",
             "FREQUENCY": "<freq>",
             # TLINK
             "AFTER": "follows",
             "BEFORE": "followed by"
             "OVERLAP": "said to be the same as"} # "partially coincident with"

def loadXML(filePath):
  """A function to parse XML files.
  """
  # Set Error Recovery parser because default XMLParser gives error
  parser = etree.XMLParser(recover=True)
  # Create XML tree from file
  tree = ET.parse(filePath, parser = parser)
  return tree

def readXML(filePath): 
  # Open and load XML file
  docm = loadXML(filePath)
  # ROOT contains TEXT (Index 0) and TAGS (Index 1) nodes
  root = docm.getroot()
  text = root[0].text
  tags = root[1]
  # Get sentences
  #sentenceList = tokenize.sent_tokenize(text)
  sents = text.split("\n")
  sentList = []
  for idx, sent in enumerate(sents):
    if len(sent) == 0: continue
    if ':' in sent: sentList.append(sent+' '+sents[idx+1])
    elif sentList != [] and ':' not in sents[idx-1]: sentList.append(sent)
  sentenceList = []
  for idx, sent in enumerate(sentList):
    sentenceList.append(sent)
    if idx != len(sentList)-1:
      sentenceList.append(sent+' '+sentList[idx+1])
  return sentenceList

# For conditional beam search: Forces one of the list words to appear // Gives corrupt relation types
# force_flexible = ["follows", "followed by", "said to be the same as"]
# force_words_ids = [tokenizer(force_flexible, add_special_tokens=False).input_ids]

# For grouped beam search use: // Predicts bad types on short context sentences
# num_beam_groups = 2 (equal or less to output sentences)
# diverse_penalty = float(10)

# https://huggingface.co/transformers/v4.8.0/main_classes/configuration.html

def predict(text):
  model_inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
  outputs = model.generate(
      model_inputs["input_ids"].to(model.device),
      attention_mask=model_inputs["attention_mask"].to(model.device),
      max_new_tokens=1024,
      length_penalty=4.0,
      num_beam_groups=4,
      early_stopping=8,
      diversity_penalty=4.0,
      num_beams=12,
      num_return_sequences=4)
  return tokenizer.batch_decode(outputs)

def saveJSONL(fPath, f):
  with open(fPath, 'w') as outfile:
    for entry in f:
        json.dump(entry, outfile)
        outfile.write('\n')

def readJSONL(fpath):
    l = []
    with open(fpath, 'r') as f:
      for d in ijson.items(f, '', multiple_values = True):
        l.append(d)
    return l

def extract_triplets_typed(text, mapping_types= {'<prob>':'PROBLEM', '<test>':'TEST', '<tret>':'TREATMENT', '<dept>':'CLINICAL_DEPT', '<evid>':'EVIDENTIAL', '<occr>':'OCCURRENCE', '<date>':'DATE', '<time>':'TIME', '<durt>':'DURATION', '<freq>':'FREQUENCY'}):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token in mapping_types:
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                object_ = ''
                subject_type = mapping_types[token]
            else:
                current = 'o'
                object_type = mapping_types[token]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
    return triplets

# Set devie to CUDA and check if connected
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)
next(model.parameters()).is_cuda

# Generate predictions, decode and put into list
testpred = []
testList = os.listdir(testPath)
for idx, f in enumerate(testList):
  filePath = testPath + f
  sentenceList = readXML(filePath)
  # Get document ID   
  docID = f.replace(".xml","")  
  # Iterate over sentences
  for sent in sentenceList:
    # Iterate over predictions of the same sentence
    tripletList = [extract_triplets_typed(pred) for pred in predict(sent)]
    if tripletList != []:
      # Iterate over the triplets of the same sentence
      for triplet in tripletList:
        preddict = {'docidx':docID, 'triplet':triplet}
        testpred.append(preddict)

# Remove repeated triplets
seenSet = set()
testpred_trimmed = []
for dct in testpred:
  for trl in dct['triplet']:
    t = tuple(trl.items())
    if t not in seenSet:
        seenSet.add(t)
        testpred_trimmed.append({'docidx':dct['docidx'], 'triplet':trl})

# Save predictions
saveJSONL(predjsonPath, predList)
