from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")

    model.train(data="data.yaml", epochs=1)

# model = YOLO('runs/train2/weights/best.pt')

# results = model(['im1.jpg', 'im2.jpg'])  # return a list of Results objects

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Class probabilities for classification outputs

# results = model.predict("dog.jpg")

# result = results[0]

# len(result.boxes) # number of detected objects

# box = result.boxes[0]

# print(len(result.boxes))

# # print("Object type:", box.cls)
# # print("Coordinates:", box.xyxy)
# # print("Probability:", box.conf)

# # print("Object type:",box.cls[0])
# # print("Coordinates:",box.xyxy[0])
# # print("Probability:",box.conf[0])

# # cords = box.xyxy[0].tolist()
# # class_id = box.cls[0].item()
# # conf = box.conf[0].item()
# # print("Object type:", class_id)
# # print("Coordinates:", cords)
# # print("Probability:", conf)

# # Show dict of detectable objects
# # print(result.names)

# cords = box.xyxy[0].tolist()
# cords = [round(x) for x in cords]
# class_id = result.names[box.cls[0].item()]
# conf = round(box.conf[0].item(), 2)
# print("Object type:", class_id)
# print("Coordinates:", cords)
# print("Probability:", conf)

# # Show every detected object
# # for box in result.boxes:
# #   class_id = result.names[box.cls[0].item()]
# #   cords = box.xyxy[0].tolist()
# #   cords = [round(x) for x in cords]
# #   conf = round(box.conf[0].item(), 2)
# #   print("Object type:", class_id)
# #   print("Coordinates:", cords)
# #   print("Probability:", conf)
# #   print("---")

# Image.fromarray(result.plot()[:,:,::-1])


from ultralytics import YOLO

model = YOLO("runs/detect/train10/weights/best.pt")

results = model.predict("2918043.jpg", max_det=2, show=True, save_crop=True, )
print(model.device)
print(results)