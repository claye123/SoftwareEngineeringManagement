from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(data=r"F:\yolo\yolov8_Test_1\ultralytics-main\datasets\well\txt\my_data.yaml", imgsz=640, epochs=300, batch=16, device=0, workers=4)