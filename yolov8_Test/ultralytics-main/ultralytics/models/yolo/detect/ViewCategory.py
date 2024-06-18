# coding=gbk
import xml.dom.minidom as xmldom
import os

# voc���ݼ���ȡ���б�ǩ�����������"
annotation_path = r"C:\SoftWare\DevelopSoft\Pycharm2023_3_3\Project\yolov8_Test_1\ultralytics-main\ultralytics\models\yolo\detect\mydata_tuomin\Annotations"

annotation_names = [os.path.join(annotation_path, i) for i in os.listdir(annotation_path)]

labels = list()
for names in annotation_names:
    xmlfilepath = names
    domobj = xmldom.parse(xmlfilepath)
    # �õ�Ԫ�ض���
    elementobj = domobj.documentElement
    # ����ӱ�ǩ
    subElementObj = elementobj.getElementsByTagName("object")
    for s in subElementObj:
        label = s.getElementsByTagName("name")[0].firstChild.data
        # print(label)
        if label not in labels:
            labels.append(label)
print(labels)