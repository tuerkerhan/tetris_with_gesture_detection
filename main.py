'''
This script provides a Main class that, when instantiated, 
sets up the necessary models for gesture detection, 
and provides a predict method for performing the inference 
in a loop on images captured from a video capture device. 
The results are displayed on the screen in real time.
'''


from ultralytics import YOLO
import cv2
import torch
import time
import os
from trainers.landmark_regression_trainer import LandmarkRegressionNet
from trainers.gesture_classifier_trainer import GestureClassifier
from inference import infer_action


class Main:
    def __init__(self):
        # Directories for trained model files
        DETECTOR_MODEL_PATH = "models/detector/best.pt"
        REGRESSOR_MODEL_PATH = "models/landmark_regressor/model_2.pt"
        CLASSIFIER_MODEL_PATH = "models/gesture_classifier/20230719_1518193368_model.pt"

        # Initialize the models
        self.detector = YOLO(DETECTOR_MODEL_PATH)
        self.regressor = LandmarkRegressionNet()
        self.classifier = GestureClassifier(num_landmarks=21, num_classes=5)

        # Load weights into models
        self.regressor.load_state_dict(torch.load(REGRESSOR_MODEL_PATH))
        self.classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH))

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # Start measuring game time and initialize variables
        self.start_time = time.time()
        self.predicted_gestures_window = []
        self.last_button = None
                
    def predict(self):
        i = 0
        
        # Game Loop
        while True:
            i += 1
            time.sleep(0.05)
            _, img = self.cap.read()

            # Run inference on the image
            predicted_gestures, dotted_image = infer_action(img, self.detector, self.regressor, self.classifier)
            
            # If we have 2 predictions, print the first and clear the list
            if len(self.predicted_gestures_window) == 4:
                self.last_button = self.predicted_gestures_window[0]
                self.predicted_gestures_window = []

            # Filter out 'no_gesture' results
            filtered_gestures = [gesture for gesture in predicted_gestures if gesture != 'no_gesture']
            if self.predicted_gestures_window == [] and filtered_gestures != []:
                self.predicted_gestures_window.append(filtered_gestures[0])
            elif self.predicted_gestures_window != [] and filtered_gestures != []:
                if self.predicted_gestures_window[0] == filtered_gestures[0]:
                    self.predicted_gestures_window.append(filtered_gestures[0])
                else:
                    self.predicted_gestures_window = []
            elif self.predicted_gestures_window != [] and filtered_gestures == []:
                self.predicted_gestures_window == []
                self.last_button = None

            os.system('cls')
            # Show the image with landmarks
            cv2.imshow("Image", dotted_image)
            cv2.waitKey(1)
            
            current_time = time.time()
            if current_time - self.start_time > 1 and self.last_button != None:
                print('============',self.last_button, '================')
                yield self.last_button
