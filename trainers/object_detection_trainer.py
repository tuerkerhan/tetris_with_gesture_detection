'''
This script uses the YOLO (You Only Look Once) object detection model
from the Ultralytics library. It simply specifies the model
configuration file and the dataset file, and trains the model 
for a specified number of epochs.
'''

from ultralytics import YOLO


if __name__ == '__main__':
    # Instantiate the YOLOv8 model
    model = YOLO("yolov8n.yaml")

    # Train the model
    model.train(data="data.yaml", epochs=10, workers=4)
