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
  parser = etree.XMLParser(recover=True, strip_cdata=False)
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

token2label = {
    # EVENT
    '<prob>':'PROBLEM', '<test>':'TEST', '<tret>':'TREATMENT', '<dept>':'CLINICAL_DEPT', '<evid>':'EVIDENTIAL', '<occr>':'OCCURRENCE', 
    # TIMEX3
    '<date>':'DATE', '<time>':'TIME', '<durt>':'DURATION', '<freq>':'FREQUENCY'}
    # TLINK
    #'follows':'AFTER', 'followed by':'BEFORE', 'said to be the same as':'OVERLAP'}

type2label = {
    # EVENT
    'PROBLEM':'EVENT', 'TEST':'EVENT', 'TREATMENT':'EVENT', 'CLINICAL_DEPT':'EVENT', 'EVIDENTIAL':'EVENT', 'OCCURRENCE':'EVENT', 
    # TIMEX3
    'DATE':'TIMEX3', 'TIME':'TIMEX3', 'DURATION':'TIMEX3', 'FREQUENCY':'TIMEX3',
    # TLINK
    'follows':'AFTER', 'followed by':'BEFORE', 'said to be the same as':'OVERLAP'}

def extract_triplets_typed(text, mapping_types=token2label):
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

jsonlPath = '/Users/josejaviersaizanton/Documents/TFM_local/data/raw_preds/testpreds_beam123.json'
testPath = '/Users/josejaviersaizanton/Documents/TFM_local/data/test_Merged/'
predPath = '/Users/josejaviersaizanton/Documents/TFM_local/data/xml_preds/'
predList = readJSONL(jsonlPath)
testList = [fileName for fileName in os.listdir(testPath) if '.xml' in fileName]
tagGoldcount, tagPredcount = 0, 0

for fileName in testList:
    # Read XMLs
    print(fileName)
    inPath, outPath = testPath + fileName, predPath + fileName
    rootTruth, rootPred = readXML(inPath), readXML(inPath)
    textGold, tagsGold, tagsPred = rootTruth[0].text, rootTruth[1], rootPred[1]
    etree.strip_elements(rootPred, ['EVENT', 'TIMEX3', 'TLINK', 'SECTIME'])
    parser = etree.XMLParser(encoding='UTF-8', strip_cdata=False)
    tagGoldcount += len(tagsGold)
    # Iterate over predictions that match the document
    entityList = []
    for pred in predList:
        if pred['docidx']+'.xml' == fileName:
            triplet = pred['triplet']
            if triplet['head'] in textGold and triplet['head'] != '' and triplet['tail'] in textGold and triplet['tail'] != '' and triplet['type'] in type2label:
                # head
                hlabel, hspan, htext, htype, hpol = type2label[triplet['head_type']], next(re.compile(re.escape(triplet['head'])).finditer(textGold)).span(), triplet['head'].replace('&','&amp;').replace('<','&lt;'), triplet['head_type'], random.choice(['NEG', 'POS'])
                # tail
                tlabel, tspan, ttext, ttype, tpol = type2label[triplet['tail_type']], next(re.compile(re.escape(triplet['tail'])).finditer(textGold)).span(), triplet['tail'].replace('&', '&amp;').replace('<','&lt;'), triplet['tail_type'], random.choice(['NEG', 'POS'])        
                if triplet['head'] in entityList: continue
                elif triplet['tail'] not in entityList:
                    entityList.append(triplet['head'])
                    if hlabel == 'EVENT':
                        hstring = f'<{hlabel} id="" start="{hspan[0]}" end="{hspan[1]}" text="{htext}" modality="FACTUAL" polarity="{hpol}" type="{htype}" />'
                        helement = etree.fromstring(hstring, parser)
                        helement.tail = '\n'
                        tagsPred.append(helement)
                    if hlabel == 'TIMEX3':
                        hstring = f'<{hlabel} id="" start="{hspan[0]}" end="{hspan[1]}" text="{htext}" type="{htype}" val="" mod="NA" />'
                        helement = etree.fromstring(hstring, parser)
                        helement.tail = '\n'
                        tagsPred.append(helement)
                if triplet['tail'] in entityList: continue
                elif triplet['tail'] not in entityList:
                    entityList.append(triplet['tail'])
                    if tlabel == 'EVENT':
                        tstring = f'<{tlabel} id="" start="{tspan[0]}" end="{tspan[1]}" text="{ttext}" modality="FACTUAL" polarity="{tpol}" type="{ttype}" />'
                        telement = etree.fromstring(tstring, parser)
                        telement.tail = '\n'
                        tagsPred.append(telement)
                    if tlabel == 'TIMEX3':
                        tstring = f'<{tlabel} id="" start="{tspan[0]}" end="{tspan[1]}" text="{ttext}" type="{ttype}" val="" mod="NA" />'
                        telement = etree.fromstring(tstring, parser)
                        telement.tail = '\n'
                        tagsPred.append(telement)
                # relation
                rtype = type2label[triplet['type']]
                rstring = f'<TLINK id="" fromID="" fromText="{htext}" toID="" toText="{ttext}" type="{rtype}" />'
                relement = etree.fromstring(rstring, parser)
                relement.tail = '\n'
                tagsPred.append(relement)
            else:continue
        else:continue
    # Modify file
    tagsPred[:] = sorted(tagsPred, key = lambda x : int(x.get('start')) if x.tag != 'TLINK' else False)
    tagsPred[:] = sorted(tagsPred, key = lambda x : x.tag)
    ide, idt, idr = 0, 0, 0
    for tagP in tagsPred:
        # if tagP.tag != 'TLINK':
        #     for tagG in tagsGold:
        #         if tagG.tag == 'EVENT' or tagG.tag == 'TIMEX3':
        #             if int(tagP.attrib['start']) == int(tagG.attrib['start']) and int(tagP.attrib['end']) == int(tagG.attrib['end']):
        #                 tagP.attrib['id'] = tagG.attrib['id']       
        #             else: continue
        if tagP.attrib['id'] == '':
            if tagP.tag == 'EVENT':
                tagP.attrib['id'] = f'E{ide}'
                ide += 1
            if tagP.tag == 'TIMEX3':
                tagP.attrib['id'] = f'T{idt}'
                idt += 1                
        if tagP.tag == 'TLINK':
            tagP.attrib['id'] = f'TL{idr}'
            tagP.attrib['fromID'] = tagsPred.xpath('.//*[contains(@text, "{}")]'.format(tagP.attrib['fromText']))[0].attrib['id']
            tagP.attrib['toID'] = tagsPred.xpath('.//*[contains(@text, "{}")]'.format(tagP.attrib['toText']))[0].attrib['id']
            idr += 1
    tagsPred[:] = sorted(tagsPred, key = lambda x : int(x.get('id').replace('TL', '')) if x.tag == 'TLINK' else False)
    tagPredcount += len(tagsPred)
    # Write file
    rootPred = etree.ElementTree(rootPred, parser=parser)
    with open(outPath, "wb") as file:
        rootPred.write(file)

if '.DS_Store' in os.listdir(testPath):
    os.remove(testPath + '.DS_Store')
if '.DS_Store' in os.listdir(predPath):
    os.remove(predPath + '.DS_Store')

print(f'Number of predictions: {tagPredcount}. Number of original tags: {tagGoldcount}')
