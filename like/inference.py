import os
import cv2
import datetime
import json
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
from cut_picture import cut_picture
from model2_train import LandmarkRegressionNet
from model3_train import GestureClassifier
import copy
import time
import cvzone

# ==================== Models ================================

gestures = {
    0: "like",
    1: "dislike",
    2: "thumb_left",
    3: "thumb_right",
    4: "no_gesture"
}
"""
detector_model_path   = "runs/detect/model_1/weights/best.pt"
regressor_model_path  = "models/landmark_regression/model_2.pt"
classifier_model_path = "models/gesture_regression/20230719_1518193368_model.pt"

my_detector = YOLO(detector_model_path)
my_regressor = LandmarkRegressionNet()
my_classifier = GestureClassifier(num_landmarks=21, num_classes=5)

my_regressor.load_state_dict(torch.load(regressor_model_path))
my_classifier.load_state_dict(torch.load(classifier_model_path))

my_image = cv2.imread('image.png')
my_image_2 = cv2.imread(r'subsamples\images\dislike\001c6f56-85cf-4e45-bfc1-1af53c0e501b.jpg')
"""


def get_boxes(object):
    as_pixel   = []
    normalized = []
    for r in object:
        boxes = r.boxes
        for box in boxes:
            x1, y1, w, h = box.xywh.tolist()[0]
            as_pixel.append([int(x1), int(y1), int(w), int(h)])
            x1, y1, w, h = box.xywhn.tolist()[0]
            normalized.append([int(x1), int(y1), int(w), int(h)])
    return as_pixel, normalized

# gets list of landmarks [(333,234),(232, 536)...] prints them into image
def set_dots(landm, image):
    new_image = image
    for x, y in landm:
        # Check if the point is within the image boundaries
        if 0 <= x < new_image.shape[1] and 0 <= y < new_image.shape[0]:
            cv2.circle(new_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the point
    return new_image

"""
landmarks: tensor([[0.2655, 0.0891, 0.6242, 0.1681, 0.7823, 0.3407, 0.8392, 0.5255, 0.8882, 0.6653, 0.4032, 0.4865, 0.5309, 0.7183, 0.6882, 0.7872, 0.8048, 0.8228, 0.2121, 0.4959, 0.3719, 0.7307, 0.5752, 0.7912, 0.7215, 0.8114, 0.1139, 0.4875, 0.2640, 0.7025, 0.4580, 0.7580, 0.6008, 0.7738, 0.0802, 0.4783, 0.1915, 0.6439, 0.3409,
         0.6976, 0.4620, 0.7145]])
"""
def get_landmarks(landmark, w, h, bboxes):
    landmark_list = landmark
    result = []
    for i in range(len(landmark_list)):
        edge_x, edge_y = bboxes[0][i][0] - bboxes[0][i][2]/2, bboxes[0][i][1] - bboxes[0][i][3]/2 # as_pixel coords
        x_s = landmark_list[i][0::2] # local x normalized
        y_s = landmark_list[i][1::2] # local y normalized
        x_s = [int(edge_x + x*bboxes[0][i][2]) for x in x_s] # global x pixel corrdinate
        y_s = [int(edge_y + y*bboxes[0][i][3]) for y in y_s] # global y pixel corrdinate

        result.append(list(zip(x_s, y_s)))
    return result





# ==================== Model 1 ================================


def infer_action(image, detector, regressor, classifier, image_name):

    detector_results = detector(image, stream=True, device='cpu')

    bboxes = get_boxes(detector_results)


    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128,128)),
                transforms.ToTensor(),
            ])
    
    hands = []
    for bbox in bboxes[0]:
        cropped_image = cut_picture(image, bbox[0]-int(bbox[2]/2), bbox[1]-int(bbox[3]/2), bbox[2], bbox[3])
        resized_image = transform(cropped_image)
        hands.append(resized_image)
        


    predicted_gestures = []
    
    landmarks_tovisualize = []
    landmarks = []
    with torch.no_grad():
        for hand in hands:
            hand = hand.unsqueeze(0)
            landmark = regressor(hand)
            landmarks += landmark
            landmarks_tovisualize += landmark.tolist()

            gesture_softmax = classifier(landmark)
            predicted_class = torch.argmax(gesture_softmax, dim=1)
            predicted_class = predicted_class.item()

            predicted_gestures.append(gestures[predicted_class])
    
    

    
    h, w, _ = image.shape
    landmarks_tovisualize = get_landmarks(landmarks_tovisualize, w, h, bboxes)



    counter=0
    dotted_image = copy.copy(image)
    for bbox in bboxes[0]:
        start_point, end_point, color, thickness = (int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[3]/2)), (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)), (255 if (counter+2) % 3 == 0 else 0, 255 if (counter+3) % 3 == 0 else 0, 255 if (counter+1) % 3 ==0 else 0), 2
        dotted_image = cv2.rectangle(dotted_image, start_point, end_point, color, thickness)
        
        for landmark in landmarks_tovisualize:
            dotted_image = set_dots(landmark, dotted_image)
        counter+=1


    return predicted_gestures, dotted_image


