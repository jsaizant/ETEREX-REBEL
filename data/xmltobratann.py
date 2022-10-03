 HOW TO OPEN BRAT: TERMINAL > CD/BRAT-MASTER > python3 standalone.py

# Import
import os
from lxml import etree
import xml.etree.ElementTree as ET
import numpy as np
import re

class i2b2TemporalRelationDataset(object):

    def __init__(self, inputPath: str):
        self.xmltobrat(inputPath)

    def loadxml(self, filePath):
        """A function to parse XML files.

        Args:
        filePath (str): The XML file location.

        Returns:
        A parsed tree from the XML file.
        """
        # Set parser with error recovery because default XMLParser does not work
        parser = etree.XMLParser(recover=True)
        # Create XML tree from file
        tree = ET.parse(filePath, parser = parser)
        return tree
    
    def writetxt(self, filePath, text):
        """A function to write text to a file.

        Args:
        filePath (str): The file location.
        text (str): The text to write to the file.
        """
        filePath = filePath.replace(".xml", ".ann")
        with open(filePath, "w+") as f:
            f.write(text)

    def xmltobrat(self, inputPath):
        """A function to convert XML files to brat files.
        """

        # Create a list of file names in the input directory
        fileList = os.listdir(inputPath)
        print(fileList)
        # Iterate over input directory files
        print("Processing files...")
        for f in fileList:
            # Create file path
            filePath = inputPath + f
            if filePath.endswith(".xml"):
                # Open and load XML file
                docm = self.loadxml(filePath)
                # ROOT contains TAGS (Index 1)
                root = docm.getroot()
                tags = root[1]
                text = root[0].text
                # Get entities and relations separately
                string = ""
                entList = []
                idList = []
                for index, element in enumerate(tags):
                    if element.tag != "TLINK" and element.attrib['text'] in text:
                        idList.append(element.attrib['id'])
                        entList.append({"index": f"T{index}", "id": element.attrib["id"], "text": element.attrib["text"]})
                        entspan = re.search(element.attrib['text'], text).span()
                        string += f"T{index}\t{element.attrib['type']} {int(entspan[0])-1} {int(entspan[1])-1}\t{element.attrib['text']}\n"
                    else:continue
                print(idList)
                for index, element in enumerate(tags):    
                    if element.tag == "TLINK" and element.attrib['fromID'] in idList and element.attrib['toID'] in idList and 'SECTIME' not in element.attrib['id']:
                        print(element.attrib)
                        for entity in entList:
                            if element.attrib["fromID"] == entity["id"]:
                                fromIndex = entity["index"]
                            if element.attrib["toID"] == entity["id"]:
                                toIndex = entity["index"]
                            else:continue
                    else:continue
                    string += f"R{index}\t{element.attrib['type']} Arg1:{fromIndex} Arg2:{toIndex}\n"
                self.writetxt(filePath, string)

dirPath = "/Users/josejaviersaizanton/Documents/brat-master/data/i2b2/xmldir/"

i2b2TemporalRelationDataset(dirPath)
