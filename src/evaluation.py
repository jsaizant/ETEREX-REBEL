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
import tensorflow as tf

# Set file paths
dataDir = "/content/drive/MyDrive/Colab Notebooks/Thesis Model/data/"
modelDir = "/content/drive/MyDrive/Colab Notebooks/Thesis Model/model/"
trainJsonPath = dataDir + "TrainJson_MergedTLINK.json"
testJsonPath = dataDir + "TestJson_MergedTLINK.json"
predJsonPath = dataDir + "TestPreds.json"
trainPath = dataDir + "TrainSet_MergedTLINK/"
testPath = dataDir + "TestSet_MergedTLINK/"

# Local model
tokenizer = AutoTokenizer.from_pretrained(modelDir)
model = AutoModelForSeq2SeqLM.from_pretrained(modelDir)

token2label = {
    # EVENT
    '<prob>':'PROBLEM', '<test>':'TEST', '<tret>':'TREATMENT', '<dept>':'CLINICAL_DEPT', '<evid>':'EVIDENTIAL', '<occr>':'OCCURRENCE', 
    # TIMEX3
    '<date>':'DATE', '<time>':'TIME', '<durt>':'DURATION', '<freq>':'FREQUENCY'}
type2label = {
    # EVENT
    'PROBLEM':'EVENT', 'TEST':'EVENT', 'TREATMENT':'EVENT', 'CLINICAL_DEPT':'EVENT', 'EVIDENTIAL':'EVENT', 'OCCURRENCE':'EVENT', 
    # TIMEX3
    'DATE':'TIMEX3', 'TIME':'TIMEX3', 'DURATION':'TIMEX3', 'FREQUENCY':'TIMEX3',
    # TLINK
    'follows':'AFTER', 'followed by':'BEFORE', 'same as':'OVERLAP'}

def loadXML(filePath):
    """A function to parse XML files.
    """
    # Set Error Recovery parser because default XMLParser gives error
    parser = etree.XMLParser(recover=True)
    # Create XML tree from file
    tree = ET.parse(filePath, parser = parser)
    # ROOT contains TEXT (Index 0) and TAGS (Index 1) nodes
    root = tree.getroot()
    text = root[0].text
    tags = root[1]
    return text, tags

def add_header(header, string):
  # Create a new string to store the modified string
  modified_string = ''
  # Check if the header is not already in the string
  if header not in string:
    # If the header is not in the string, add it to the beginning of the string
    modified_string = header + ' ' + string
  else:
    # If the header is already in the string, leave the string unchanged
    modified_string = string
  return modified_string

def readXML(filePath): 
  # Open and load XML file
  text, tags = loadXML(filePath)
  # Identify header dates
  header = ' '.join(text.strip().splitlines()[0:4])
  # Get sentences
  text = text.strip().splitlines()[4:]
  textSplit = []
  # Merge lines with ':' together in textSplit
  for id, sent in enumerate(text):
    if ':' in sent and sent != text[-1]: textSplit.append(" ".join([sent+" "+text[id+1]]).strip())
    elif textSplit != [] and ":" not in text[id-1]: textSplit.append(sent)
    else: continue 

  # THE FOLLOWING ARE DIFFERENT METHODS OF SPLITTING EACH DOCUMENT IN THE
  # TEST SET: CHOOSE ONLY ONE IN RETURN BY CHANGING THE LIST
  
  # 1. Merge sentence list into the following patron: 1, 1-2, 2, 2-3, 3, 3-4, etc.
  # singleList = []
  # for id, sent in enumerate(textSplit):
  #   singleList.append(sent)
  #   if id != len(textSplit)-1:
  #     singleList.append(sent+' '+textSplit[id+1])
  #   else: continue

  # 2. Merge sentence list into the following patron: 1-2, 3-4, 5-6, 7-8, etc.
  #doubleList = [ " ".join(sent) for sent in zip(textSplit[0::2], textSplit[1::2]) ]

  # 3. Merge sentence list into the following patron: 1-2-3, 2-3-4, 3-4-5, etc.
  tripleList = []
  for id, sent in enumerate(textSplit):
    threesent = ""
    if id < len(textSplit) - 2:
      for n in range(3): # Sent. 0 + Sent. 1 + Sent. 2 + Sent. 3 
        threesent += " " + textSplit[id+n]
      tripleList.append(add_header(header, threesent))
    elif id == len(textSplit) - 2: 
      for n in range(2): # Sent. 0 + Sent. 1 + Sent. 2
        threesent += " " + textSplit[id+n]
      tripleList.append(add_header(header, threesent))
    else: break

  # 4. By character count
  # minLength = 12
  # maxLength = 128
  # # Remove breaklines for accurate character splitting
  # text = text.replace("\n", " ")
  # # Get context list from file
  # contextList = [add_header(header, context.group(0)).strip() for context in re.finditer(f".{{{minLength},{maxLength}}}", text)]
  return tripleList

def predict(text): # https://huggingface.co/transformers/v4.8.0/main_classes/configuration.html
  gen_kwargs = {
    "max_length": 1024,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 1}
  model_inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
  outputs = model.generate(
    model_inputs["input_ids"].to(model.device),
    attention_mask=model_inputs["attention_mask"].to(model.device),
    **gen_kwargs)
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

def extract_triplets_typed(text, mapping_types= token2label):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '' and relation.strip() in type2label:
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': type2label[relation.strip()],'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token in mapping_types:
            if current == 't' or current == 'o':
                current = 's'
                if relation != '' and relation.strip() in type2label:
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': type2label[relation.strip()],'tail': object_.strip(), 'tail_type': object_type})
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
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '' and relation.strip() in type2label:
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': type2label[relation.strip()],'tail': object_.strip(), 'tail_type': object_type})
    return triplets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)
next(model.parameters()).is_cuda

# Generate predictions for one file:
count = 0
testpred = []
fil = testPath + "206.xml"
sentenceList = readXML(fil)
# Iterate over sentences
for sent in sentenceList:
  print(sent)
  # Iterate over predictions of the same sentence
  prediction = [extract_triplets_typed(pred) for pred in predict(sent)]
  # Iterate over the triplets of the same sentence
  for tripletList in prediction:
    for triplet in tripletList:
      print(triplet)
      count += 1
print(f"Amount of triplets predicted:{count}")

# Generate predictions, decode and put into list
testpred = []
testList = os.listdir(testPath)
for idx, f in enumerate(testList):
  pairList = []
  filePath = testPath + f
  sentenceList = readXML(filePath)
  # Get document ID   
  docID = f.replace(".xml","")  
  # Iterate over sentences
  for sent in sentenceList:
    # Iterate over predictions of the same sentence
    prediction = [extract_triplets_typed(pred) for pred in predict(sent)]
    if prediction != []:
      # Iterate over the triplets of the same sentence
      for tripletList in prediction:
        for triplet in tripletList:
          pair = (triplet['head'], triplet['tail'])
          if pair not in pairList:
            preddict = {'docidx':docID, 'triplet':triplet}
            testpred.append(preddict)
            pairList.append(pair)
          else: continue
  print(f'Document {idx+1}/{len(testList)} ({f}) tested. Prediction example for "{sent}" are:\n{tripletList}')

saveJSONL(predJsonPath, testpred)
