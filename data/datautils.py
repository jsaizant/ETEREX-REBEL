# Install
!pip install torch
!pip install transformers
!pip install datasets
# Import data libraries
import os
import xml.etree.ElementTree as ET
from lxml import etree
import numpy as np
import re
import json, ijson
from operator import itemgetter
# Pytorch/ Transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
from datasets import Dataset, load_metric
import torch
from torch.utils.data import DataLoader

class i2b2Dataset(object):
  """A class to handle the i2b2 2012 Temporal Relations dataset.

  Args:
    dirPath (str): The directory location of the dataset.
  
  Returns:
    A list of string triplets encoding the relations and entities in each document.
  """

  def __init__(self, dirPath: str, *args, **kwargs) -> None:
    """Initialize the dataset.
    """
    # Initialize parent
    super(i2b2Dataset, self).__init__(*args, **kwargs)
    # Set directory path
    self.dirPath = dirPath
    # Get file list
    self.fileList = os.listdir(dirPath)
    # Load labels and triplets
    self.dictList = self.loadTriplets()
  
  def __len__(self):
    """Returns the length of the dataset.
    """
    return len(self.triplets)

  def __getitem__(self, index):
    """Returns the item at the given index.
    """
    return self.dictList[index]

  def loadXML(self, filePath):
    """A function to parse XML files.
    """
    # Set Error Recovery parser because default XMLParser gives error
    parser = etree.XMLParser(recover=True)
    # Create XML tree from file
    tree = ET.parse(filePath, parser = parser)
    return tree

  def loadCorpus(self, f):
    """A function to obtain a list of entities and relations from the i2b2 2012 Temporal Relations dataset.      
    """
    # Set dictionary abbreviations
    type2abbr = {# EVENT
                 "PROBLEM": "prob",
                 "TEST": "test",
                 "TREATMENT": "tret",
                 "CLINICAL_DEPT": "dept",
                 "EVIDENTIAL": "evid",
                 "OCCURRENCE": "occr",
                 # TIMEX3
                 "DATE": "date",
                 "TIME": "time",
                 "DURATION": "durt",
                 "FREQUENCY": "freq",
                 # TLINK
                 "AFTER": "follows",
                 "BEFORE": "followed by",
                 "OVERLAP": "said to be the same as"} # "partially coincident with"

    # Get document ID
    docID = f.replace(".xml","")
    # Open and load XML file
    filePath = self.dirPath + f
    docm = self.loadXML(filePath)
    # Create list to store dictionaries with relation instances
    instanceList = []
    # ROOT contains TEXT (Index 0) and TAGS (Index 1) nodes
    root = docm.getroot()
    text = root[0].text
    tags = root[1]
    # Get entities and relations separately
    entityList = []
    relationList = []
    for element in tags:
      if element.attrib["type"] == "": continue # Ommit if type attribute is blank
      elif element.tag != "TLINK" and element.tag != "SECTIME": # Add if tag is EVENT, TIMEX3
        entityList.append({"id":element.attrib["id"],
                           "text":element.attrib["text"],
                           "start":int(element.attrib["start"]), 
                           "end":int(element.attrib["end"]),
                           "tag":element.tag, 
                           "type":type2abbr[element.attrib["type"]]})
      elif element.tag == "TLINK" and "TL" in element.attrib["id"]: # Add if tag is TLINK but is not related with SECTIME
        relationList.append({"id":element.attrib["id"], 
                             "fromID":element.attrib["fromID"], 
                             "fromText":element.attrib["fromText"],
                             "toText":element.attrib["toText"],
                             "toID":element.attrib["toID"],
                             "type":type2abbr[element.attrib["type"]]})
      else: continue
      # Sort list by appearance order
      entityList = sorted(entityList, key=lambda ent: ent['start']) 
    return entityList, relationList
  
  def extractTriplet(self, headEnt, tailEntList):
    """A function to parse a given text and extract the triplets.
    """
    # Start triplet
    triplets = f"<triplet> {headEnt['text']} "
    # Go through the tail entity list
    for tailEnt in tailEntList:
      triplets += f"<{headEnt['type']}> {tailEnt['text']} <{tailEnt['enttype']}> {tailEnt['reltype']} " # triplets += <subj> + o + <obj> + r

    return triplets

  def loadTriplets(self):
    # Iterate over input directory files
    dictList = []
    for f in self.fileList:
      text = self.loadXML(self.dirPath + f).getroot()[0].text
      entityList, relationList = self.loadCorpus(f)
      sentenceList = [{"text":sent.group(0), "start":sent.start(), "end":sent.end()} for sent in re.finditer("[^\r\n]+", text)]
      headIDList = list(set([relation['fromID'] for relation in relationList]))
      headEntList = [ent for id in headIDList for ent in entityList if ent['id'] in headIDList]
      # Iterate over head entities
      for headEnt in headEntList:
        limitList = [int(headEnt['start'])]
        tailEntList = []
        # Get head and tail sentence positions
        for relation in relationList:
          if relation['fromID'] == headEnt['id']:
            for entity in entityList:
              if relation['toID'] == entity['id']:
                tailEnt = entity
                limitList.append(tailEnt['end'])
                tailEntList.append({'id':tailEnt['id'], 'text':tailEnt['text'], 'start':tailEnt['start'], 'end':tailEnt['end'], 'tag':tailEnt["tag"], 'enttype':tailEnt["type"], 'reltype':relation['type']})
              else: continue
          else: continue
        # Sort list by appearance order
        tailEntList = sorted(tailEntList, key=lambda ent:ent['start']) 
        # Get sentence character boundaries
        sentIndList = []
        for index, sent in enumerate(sentenceList):
          if sent["start"] <= min(limitList) <= sent["end"]:
            sentIndList.append(index)
          elif sent["start"] <= max(limitList) <= sent["end"]:
            sentIndList.append(index)
          else: continue
        # Get context and triplets
        context = [sent["text"] for sent in sentenceList[min(sentIndList):max(sentIndList)+1]]
        context = " ".join(context)
        triplet = self.extractTriplet(headEnt, tailEntList)
        # Merge
        dictList.append({'text':context, 'triplet':triplet})
    print(dictList)
    return dictList
  
  def getDict(self):
    """A function to get the dictionary of the dataset.
    """
    return {'data':self.dictList}
 
# Set file paths
dataDir = "/content/drive/MyDrive/Colab Notebooks/timexes_thesis/data/i2b2/"
trainPath = dataDir + "trainset/"
testPath = dataDir + "testset/"  

# Load model and tokenizer from HuggingFace or from checkpoint
modelPath = "Babelscape/rebel-large"
config = AutoConfig.from_pretrained(modelPath,
                                    decoder_start_token_id = 0,
                                    early_stopping = False,
                                    no_repeat_ngram_size = 0,
                                    dropout=0.1,
                                    forced_bos_token_id=None)
additional_special_tokens = ['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>']
tokenizer = AutoTokenizer.from_pretrained(modelPath, use_fast = True, additional_special_tokens = additional_special_tokens)
config.vocab_size = tokenizer.vocab_size  # setting Tokenizer and Model to have same vocab size
model = AutoModelForSeq2SeqLM.from_pretrained(modelPath, config=config, ignore_mismatched_sizes=True)
  
# Set function to tokenize text and labels
max_length = 1024
def tokenize(instance):
  tokenized_instance = tokenizer(text = instance["text"], max_length=max_length, padding=True, truncation=True)
  tokenized_instance["labels"] = tokenizer(text = instance["triplet"], max_length=max_length, padding=True, truncation=True)["input_ids"]
  return tokenized_instance

# Load dataset
trainDS = i2b2Dataset(trainPath).loadTriplets()
testDS = i2b2Dataset(testPath).loadTriplets()

# Tokenize dataset
trainDSTK = trainDS.map(lambda instance: tokenize(instance), batched=True)
testDSTK = testDS.map(lambda instance: tokenize(instance), batched=True)

# Save encoded set
#trainDSTK.save_to_disk(encodedTrainPath)
#testDSTK.save_to_disk(encodedTestPath)
