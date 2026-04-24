from ultralytics import YOLO
from roboflow import Roboflow
import os

rf = Roboflow(api_key="fvxjBIcJXiPzXMrb5Nqt")
project = rf.workspace("pasus-workspace").project("medicine_name_detection")
version = project.version(1)

version.deploy(
    "yolov8-obb",
    "./obb/roboflow_training_2/train",
)