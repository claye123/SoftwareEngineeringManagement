
from ultralytics import YOLO

import sys
import os

sys.path.append('/root/ultralytics/')
print(sys.path)
img_files=os.listdir(r'D:\yolo\测试集图片')


lens=len(img_files)

model = YOLO(r'D:\yolo\yolov8_Test\runs\detect\train20\weights\best.pt')

results=model.predict(source=r'D:\yolo\测试集图片',conf=0.2,imgsz=640,save=True)
#print(results)
with open('result.txt','w') as f:
    for i in results:
        for res in i:
            box=res.boxes
            cls=box.cls
            conf=box.conf
            xyxy=box.xyxy
            print('xyxy',xyxy)
            xyxy=xyxy.squeeze()
            print('xyxy',xyxy)
            xyxy=[int(y.item()) for y in xyxy]
            path = i.path
            print('path:', path)
            path = path.split("\\")[-1]
            f.write(path+' '+str(int(cls.item()))+' '+str(conf.item())+' '+' '.join(str(i) for i in xyxy)+'\n')
