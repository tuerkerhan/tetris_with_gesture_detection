from ultralytics import YOLO
import cv2
import torch
import time
from trainers.landmark_regression_trainer import LandmarkRegressionNet
from trainers.gesture_classifier_trainer import GestureClassifier
from inference import infer_action

DETECTOR_MODEL_PATH = "models/detector/best.pt"
REGRESSOR_MODEL_PATH = "models/landmark_regressor/model_2.pt"
CLASSIFIER_MODEL_PATH = "models/gesture_classifier/20230719_1518193368_model.pt"

detector = YOLO(DETECTOR_MODEL_PATH)
regressor = LandmarkRegressionNet()
classifier = GestureClassifier(num_landmarks=21, num_classes=5)

regressor.load_state_dict(torch.load(REGRESSOR_MODEL_PATH))
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH))

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)



# game_time = time.time()
# last_move = []
predicted_gestures_window = []
#counter = 0

while True:
    success, img = cap.read()

    predicted_gestures, dotted_image = infer_action(img, detector, regressor, classifier)
    predicted_gestures_window.append(predicted_gestures)

    # TODO: Remove bounding boxes intersecting, remove both of the bounding boxes if one is inside of another
    # sub_lists = [['like', 'like', 'like', 'like'], 
    #              ['dislike', 'dislike', 'dislike', 'dislike'], 
    #              ['thumb_left', 'thumb_left', 'thumb_left', 'thumb_left'], 
    #              ['thumb_right', 'thumb_right', 'thumb_right', 'thumb_right']]

    # for sub_list in sub_lists:
    #     for idx in range(len(predicted_gestures_window) - len(sub_list) + 1):
    #         if predicted_gestures_window[idx: idx + len(sub_list)] == sub_list:
    #             print(sub_list[0])
    #             predicted_gestures_window = []
    #             break 
    
    if len(predicted_gestures_window) == 2:
        print('============',predicted_gestures_window[0], '================') 
        predicted_gestures_window = []


    filtered_gestures = [gesture for gesture in predicted_gestures if gesture != 'no_gesture']
    if predicted_gestures_window == [] and filtered_gestures != []:
        predicted_gestures_window.append(filtered_gestures[0])
    elif predicted_gestures_window != [] and filtered_gestures != []:
        if predicted_gestures_window[0] == filtered_gestures[0]:
            predicted_gestures_window.append(filtered_gestures[0])
        else:
            predicted_gestures_window = []
    #elif predicted_gestures_window != [] and filtered_gestures == []:
    #    predicted_gestures_window == []

        

    
    
    # if game_time % 200:
    #     last_move = predicted_gestures_window[:-1]
    #     send(last_move)
    #     game_time = 0
    #     game_time = time.start()
        


    cv2.imshow("Image", dotted_image)
    cv2.waitKey(1)
