from ultralytics import YOLO


################# TRAIN ###################

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")

    model.train(data="data.yaml", epochs=10, workers=4)





################# Test ####################

"""
model = YOLO("runs/detect/train12/weights/best.pt")

results = model.predict("2918043.jpg", device = 0)
print(model.device)
print(results)
"""