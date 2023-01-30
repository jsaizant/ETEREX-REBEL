# Install
!pip install ijson
# Import
import os
import json, ijson
from lxml import etree
import xml.etree.ElementTree as ET
import numpy as np
import re
import random

def loadXML(filePath):
  """A function to parse XML files.
  """
  # Set Error Recovery parser because default XMLParser gives error
  parser = etree.XMLParser(recover=True)
  # Create XML tree from file
  tree = ET.parse(filePath, parser = parser)
  # Get the root
  root = tree.getroot()
  return root

def readXML(filePath):
  root = loadXML(filePath)
  # Get the text of the root element and the tags
  text = root[0].text
  tags = root[1]
  return text, tags

def emptyXML(filePath, elementList):
  # Open and load XML file
  root = loadXML(filePath)
  # Remove the specified elements from the tree
  etree.strip_elements(root, elementList)
  return root

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

# Set file paths
dataDir = "/content/drive/MyDrive/Colab/ClinicalREBEL/data/"
modelDir = "/content/drive/MyDrive/Colab/ClinicalREBEL/model/"
predJsonPath = dataDir + "TestPreds.json"
testPath = dataDir + "TestSet_MergedTLINK/"
predPath = dataDir + "system_dir/"
evalPath= dataDir + "TemporalEvaluationScripts/"

predList = readJSONL(predJsonPath)
testList = os.listdir(testPath)

type2label = {
    # EVENT
    'PROBLEM':'EVENT', 'TEST':'EVENT', 'TREATMENT':'EVENT', 'CLINICAL_DEPT':'EVENT', 'EVIDENTIAL':'EVENT', 'OCCURRENCE':'EVENT', 
    # TIMEX3
    'DATE':'TIMEX3', 'TIME':'TIMEX3', 'DURATION':'TIMEX3', 'FREQUENCY':'TIMEX3'}

gold_count, pred_count = 0, 0
for fileName in testList:
    inPath, outPath = testPath + fileName, predPath + fileName
    # Read XMLs
    print(fileName)
    textGold, tagsGold = readXML(inPath)
    rootPred = emptyXML(inPath, ['EVENT', 'TIMEX3', 'TLINK', 'SECTIME'])
    tagsPred = rootPred[1]
    gold_count += len(tagsGold)


    # Iterate over predictions that match the document
    entity_list = set()
    element_list = []
    for prediction in predList:
        if prediction['docidx']+'.xml' == fileName:
            triplet = prediction['triplet']
            if triplet['head'] in textGold and triplet['head'] != '' and triplet['tail'] in textGold and triplet['tail'] != '':
              # compile head entity into XML element
              hlabel = type2label[triplet['head_type']]
              hstart = textGold.find(triplet['head'])
              hend = textGold.find(triplet['head']) + len(triplet['head'])
              htext = triplet['head'].replace('&','&amp;').replace('<','&lt;')
              htype = triplet['head_type']
              if hlabel == 'EVENT':
                helement = etree.Element(hlabel, id="", start=str(hstart), end=str(hend), text=htext, modality="", polarity="", **{'type': htype})
                helement.tail = '\n'
              if hlabel == 'TIMEX3':
                helement = etree.Element(hlabel, id="", start=str(hstart), end=str(hend), text=htext, **{'type': ttype}, val="", mod="")
                helement.tail = '\n'

              # compile tail entity into XML element
              tlabel = type2label[triplet['tail_type']]
              tstart = textGold.find(triplet['tail'])
              tend = textGold.find(triplet['tail']) + len(triplet['tail'])
              ttext = triplet['tail'].replace('&', '&amp;').replace('<','&lt;')
              ttype = triplet['tail_type']
              if tlabel == 'EVENT':
                telement = etree.Element(tlabel, id="", start=str(tstart), end=str(tend), text=ttext, modality="", polarity="", **{'type': ttype})
                telement.tail = '\n'
              if tlabel == 'TIMEX3':
                telement = etree.Element(tlabel, id="", start=str(tstart), end=str(tend), text=ttext, **{'type': ttype}, val="", mod="")
                telement.tail = '\n'
                
              # compile relation into XML element
              rtype = triplet['type']
              relement = etree.Element("TLINK", id="", fromID="", fromText=htext, toID="", toText=ttext)
              relement.set("type", rtype)
              relement.tail = '\n'

              # append if not already included
              if htext not in entity_list:
                  entity_list.add(htext)
                  tagsPred.append(helement)
              if ttext not in entity_list:
                  entity_list.add(ttext)
                  tagsPred.append(telement)
              
              tagsPred.append(relement)

            else:continue
        else:continue
    
    pred_count += len(tagsPred)
    
    # Sort EVENT and TIMEX3 tags by start character
    tagsPred[:] = sorted(tagsPred, key = lambda x : int(x.get('start')) if x.tag != 'TLINK' else False)
    # Assign IDs
    ide, idt, idr = 0, 0, 0
    for tag in tagsPred:
        if tag.attrib['id'] == '':
          if tag.tag == 'EVENT':
            tag.attrib['id'] = f'E{ide}'
            ide += 1
          if tag.tag == 'TIMEX3':
            tag.attrib['id'] = f'T{idt}'
            idt += 1 
          else:continue
        else: continue          
    for tag in tagsPred:
      if tag.attrib['id'] == '':
        if tag.tag == 'TLINK':
          tag.attrib['id'] = f'TL{idr}'
          tag.attrib['fromID'] = tagsPred.xpath('.//*[contains(@text, "{}")]'.format(tag.attrib['fromText']))[0].attrib['id']
          tag.attrib['toID'] = tagsPred.xpath('.//*[contains(@text, "{}")]'.format(tag.attrib['toText']))[0].attrib['id']
          idr += 1
        else:continue
      else: continue

    # Find admission and discharge dates
    admission_date = tagsPred.xpath('//TIMEX3[@id="T0"]/@text')[0]
    discharge_date = tagsPred.xpath('//TIMEX3[@id="T1"]/@text')[0]
    idr = 0
    for tag in tagsPred:
      if tag.tag == 'TLINK':
        if tag.attrib['toID'] in ["T1", "T0"] and tag.attrib['fromText'].lower() not in ["admission", "discharge"]:
          tag.attrib['id'] = f'SECTIME{idr}'
          idr += 1
        else: continue
      else: continue
    
    # Separate EVENT and TIMEX3 tags by name
    tagsPred[:] = sorted(tagsPred, key = lambda x : x.get('id'))
    #etree.dump(tagsPred)

    # Write XML
    treeOut = etree.tostring(rootPred, encoding='UTF-8', pretty_print=True, xml_declaration=True)
    with open(outPath, 'w') as f:
        f.write(treeOut.decode())

print(f'Total count: {gold_count} standard tags. {pred_count} predicted tags.')
