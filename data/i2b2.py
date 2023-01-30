# Import data libraries
import os
import json
import xml.etree.ElementTree as ET
from lxml import etree
import numpy as np
import re
from operator import itemgetter

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
    # Initialize parent function
    super(i2b2Dataset, self).__init__(*args, **kwargs)
    # Set global directory path
    self.dirPath = dirPath
    # Get global file list
    self.fileList = os.listdir(dirPath)

  def loadXML(self, filePath):
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

  def loadFile(self, filePath):
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
                 "OVERLAP": "same as"} # "partially coincident with"
    # Open and load XML file
    text, tags = self.loadXML(filePath)
    # Create list to store dictionaries with relation instances
    instanceList = []
    # Get entities and relations separately
    eventList = []
    tlinkList = []
    for element in tags:
      if element.attrib["type"] == "": continue # Ommit if type attribute is blank
      if element.tag == "EVENT" or element.tag == "TIMEX3": # Add if tag is EVENT or TIMEX3
        eventList.append({"id":element.attrib["id"],
                          "text":element.attrib["text"],
                          "start":int(element.attrib["start"]), 
                          "end":int(element.attrib["end"]),
                          "tag":element.tag, 
                          "type":type2abbr[element.attrib["type"]]})
      if element.tag == "TLINK": # and "TL" in element.attrib["id"] (If tag is TLINK but is not related with SECTIME)
        tlinkList.append({"id":element.attrib["id"],
                          "tlinktype": ''.join(filter(lambda x: not x.isdigit(), element.attrib["id"])).strip(), # Capture if its normal TL or SECTIME 
                          "fromID":element.attrib["fromID"], 
                          "fromText":element.attrib["fromText"],
                          "toText":element.attrib["toText"],
                          "toID":element.attrib["toID"],
                          "reltype":type2abbr[element.attrib["type"]]})
      else: continue
    # Sort list by appearance order
    eventList = sorted(eventList, key=lambda ent: ent['start']) 
    return eventList, tlinkList
  
  def extractTripletSequence(self, dataList):
    """A function to parse a triplet sequence given a a list of head entities and tail entities.
       For example: '<triplet> head <subj> tail <obj> relation <subj> tail <obj> relation'
    """
    tripletSeq = ""
    for triplet in dataList:
      # Start triplet
      head = triplet['head']
      tripletSeq += f"<triplet> {head['text']} "
      # Go through the tail entity list
      for tail in triplet['tail']:
        tripletSeq += f"<{head['type']}> {tail['text']} <{tail['enttype']}> {tail['reltype']} " # triplets += <subj> + o + <obj> + r
    
    return tripletSeq

  def add_header(self, header, string):
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

  def loadDict(self):
    # Set minimum and maximum length for context instances
    minLength = 0
    maxLength = 512
    dataDict = []
    lengthList = []

    # Iterate over input directory files
    for n, fil in enumerate(self.fileList):
      print(f"File {n}/{len(self.fileList)} complete ({fil})")
      # Get document ID
      docID = fil.replace(".xml","")
      # Get file path
      filePath = self.dirPath + fil
      # Read XML
      text, tags = self.loadXML(filePath)
      # Identify header dates
      header = ' '.join(text.strip().splitlines()[0:4])
      # Remove breaklines for accurate character splitting
      text = text.replace("\n", " ")
      # Get event, tlink and context lists from file
      eventList, tlinkList = self.loadFile(filePath)
      contextList = [
          {"text":self.add_header(header, context.group(0)).strip(),
           "start":context.start(),
           "end":context.end()}
          for context in re.finditer(f".{{{minLength},{maxLength}}}", text)]
      # Classify events into head and tail lists
      headIDList = list(set([tlink['fromID'] for tlink in tlinkList]))
      tailIDList = list(set([tlink['toID'] for tlink in tlinkList]))
      headEventList = [event for id in headIDList for event in eventList if event['id'] == id]
      # Sort list by appearance order
      headEventList = sorted(headEventList, key=lambda event:event['start'])
      tripletList = []
      
      # Get tail for each head and compile into dictionary
      for head in headEventList:
        tailEventList = []
        limitList = []
        limitList.append(head['start'])
        limitList.append(head['end'])
        # Get tail(s) for each head
        for tlink in tlinkList:
          if tlink['fromID'] == head['id']:
            if tlink['tlinktype'] == 'TL': # Count limits for TLINKRELs
              for event in eventList:
                if event['id'] == tlink['toID']:
                  tail = event
                  limitList.append(tail['start'])
                  limitList.append(tail['end'])
                  tailEventList.append({'id':tail['id'], 'text':tail['text'], 'start':tail['start'], 'end':tail['end'], 'tag':tail['tag'], 'enttype':tail['type'], 'reltype':tlink['reltype']})
                else: continue
            if tlink['tlinktype'] == 'SECTIME': # Do not count limits for SECTIMERELs
              for event in eventList:
                if event['id'] == tlink['toID']:
                  tail = event
                  tailEventList.append({'id':tail['id'], 'text':tail['text'], 'start':tail['start'], 'end':tail['end'], 'tag':tail['tag'], 'enttype':tail['type'], 'reltype':tlink['reltype']})
                else: continue
            else: continue
          else: continue
        # Sort list by appearance order
        tailEventList = sorted(tailEventList, key=lambda event:event['start'])
        # Compile dictionary
        tripletList.append({'start':min(limitList), 'end':max(limitList), 'head':head, 'tail':tailEventList})
      
      # Get triplets within each context boundaries
      for i, context in enumerate(contextList):
        dataList = []
        for triplet in tripletList:
          if context["start"] <= triplet['start'] <= context["end"] and context["start"] <= triplet['end'] <= context["end"]:
            dataList.append(triplet)
          else:continue

        # Get triplet sequence
        tripletSeq = self.extractTripletSequence(dataList)
        # Merge
        if tripletSeq != "":
          id = f"{docID}-{i}"
          print(f"\nFROM {context['start']} TO {context['end']}\nTEXT: {context['text']}\nTRIPLET:{tripletSeq}")
          dataDict.append({'id':id, 'context':context, 'triplet':tripletSeq})
          lengthList.append(len(context['text']))
          lengthList.append(len(tripletSeq))
        else: continue
    return dataDict

# Set file paths
dataDir = "/content/drive/MyDrive/Colab Notebooks/Thesis Model/data/"
trainPath = dataDir + "TrainSet_MergedTLINK/"
testPath = dataDir + "TestSet_MergedTLINK/"
trialPath = dataDir + "TrialSet/"
trainJsonPath = dataDir + "TrainJson_MergedTLINK.json"
testJsonPath = dataDir + "TestJson_MergedTLINK.json"

# Create dataset dictionary
trainSet = i2b2Dataset(trainPath).loadDict()

with open(trainJsonPath, 'w') as outfile:
    for entry in trainSet:
        json.dump(entry, outfile)
        outfile.write('\n')

# Create dataset dictionary
testSet = i2b2Dataset(testPath).loadDict()

with open(testJsonPath, 'w') as outfile:
    for entry in testSet:
        json.dump(entry, outfile)
        outfile.write('\n')
