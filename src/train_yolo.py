from ultralytics import YOLO

def train_yolo(data_yaml, epochs=10, img_size=640):
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size
    )
