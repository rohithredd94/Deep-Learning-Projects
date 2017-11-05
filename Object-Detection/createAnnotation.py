import xml.etree.cElementTree as ET
import sys
import re
import os
'''
annotation = ET.Element("annotation")

folder = ET.SubElement(annotation, "folder")
folder.text="find_phone"

filename = ET.SubElement(annotation, "filename")
filename.text="test.jpg"

path = ET.SubElement(annotation, "path")
path.text="/home/rohith/"

tree = ET.ElementTree(annotation)
tree.write("filename.xml")
'''
def main():
    labels = "./find_phone/labels.txt"
    f = open(labels,'r')
    

    for line in f:
        #line = f.readline()
        annotation = ET.Element("annotation")

        folder = ET.SubElement(annotation, "folder")
        folder.text="find_phone"

        line = line.strip('\n')
        s = re.split(' ',line)
        #print(s[0])
        output = s[0][:(len(s[0])-3)] + "xml"
        xmin = float(s[1])*490 - 20
        xmax = float(s[1])*490 + 20
        ymin = float(s[2])*326 - 20
        ymax = float(s[2])*326 + 20
        filename = ET.SubElement(annotation, "filename")
        filename.text=s[0]

        ET.SubElement(annotation, "path").text = os.path.dirname(os.path.realpath(__file__)) +"/find_phone/"+s[0]
        
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = "490"
        ET.SubElement(size, "height").text = "326"
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(annotation, "segmented").text = "0"

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "phone" 
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))
        tree = ET.ElementTree(annotation)
        tree.write("./annotations/"+output)
    '''
    for line in f:
        line = line.strip('\n')
        s = re.split(' ',line)
        print(s)
    '''
if __name__ == '__main__':
    main()