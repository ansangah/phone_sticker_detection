1st try code
!pip install ultralytics
---
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="rCA4UBFXGUuPha7kmyfE")
project = rf.workspace("-k8kkf").project("phone_sticker_detection")
dataset = project.version(1).download("yolov8")
---
from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")  # 모델 업그레이드 (n → s)

model.train(
    data="/content/phone_sticker_detection-1/data.yaml",
    epochs=100,   # 학습 더 오래
    imgsz=640
)
