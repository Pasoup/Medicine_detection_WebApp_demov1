from ultralytics import YOLO
from roboflow import Roboflow

def prepare_data():
    rf = Roboflow(api_key="fvxjBIcJXiPzXMrb5Nqt")
    project = rf.workspace("pasus-workspace").project("medicine_label_obb")
    version = project.version(6)
    dataset = version.download("yolov8-obb")
    return dataset

if __name__ == '__main__':
   
    dataset = prepare_data()


    model = YOLO("yolo11n-obb.pt") 

    
    results = model.train(
        data=f"{dataset.location}/data.yaml", 
        epochs=100, 
        imgsz=960, 
        device=0,
        batch=8, 
        task='obb',
        project="roboflow_training",
        degrees=180.0,
        flipud=0.5
    )

    print("Training Complete! Your model is saved in 'runs/detect/train/weights/best.pt'")