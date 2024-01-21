'''
This script handles image inference. It imports various 
modules, sets up the gesture detection model, 
and defines several helper functions for processing the 
images and performing the inference. 
The main function infer_action() is what the other scripts
call to perform the inference.
'''


import cv2
import torch
from torchvision import transforms
from preprocessing_scripts.cut_picture import cut_picture
import copy
import time


gestures = {
    0: "like",
    1: "dislike",
    3: "thumb_left",
    2: "thumb_right",
    4: "no_gesture"
}

# Function to get bounding boxes
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


# Function to add landmarks to an image
def set_dots(landm, image):
    new_image = image
    for x, y in landm:
        # Check if the point is within the image boundaries
        if 0 <= x < new_image.shape[1] and 0 <= y < new_image.shape[0]:
            cv2.circle(new_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the point
    return new_image


# Function to get landmarks from data
def get_landmarks(landmark, bboxes):
    landmark_list = landmark
    result = []
    for i, value in enumerate(landmark_list, 0):
        edge_x, edge_y = bboxes[0][i][0] - bboxes[0][i][2]/2, bboxes[0][i][1] - bboxes[0][i][3]/2 # as_pixel coords
        x_s = value[0::2] # local x normalized
        y_s = value[1::2] # local y normalized
        x_s = [int(edge_x + x*bboxes[0][i][2]) for x in x_s] # global x pixel corrdinate
        y_s = [int(edge_y + y*bboxes[0][i][3]) for y in y_s] # global y pixel corrdinate

        result.append(list(zip(x_s, y_s)))
    return result


# Function to run inference on an image using all three models
def infer_action(image, detector, regressor, classifier):
    # Run object detection
    detector_results = detector(image, stream=True, device='cpu', conf=0.6)
    # Get bounding boxes
    bboxes = get_boxes(detector_results)

    # Define image transformation
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128,128)),
                transforms.ToTensor(),
            ])
    
    # Prepare hands images
    hands = []
    for bbox in bboxes[0]:
        cropped_image = cut_picture(image, bbox[0]-int(bbox[2]/2), bbox[1]-int(bbox[3]/2), bbox[2], bbox[3])
        resized_image = transform(cropped_image)
        hands.append(resized_image)

    # Predict gestures
    predicted_gestures = []
    landmarks_to_visualize = []
    landmarks = []
    with torch.no_grad():
        for hand in hands:
            hand = hand.unsqueeze(0)
            landmark = regressor(hand)
            landmarks += landmark
            landmarks_to_visualize += landmark.tolist()

            gesture_softmax = classifier(landmark)
            predicted_class = torch.argmax(gesture_softmax, dim=1)
            predicted_class = predicted_class.item()

            predicted_gestures.append(gestures[predicted_class])
    
    # Visualize landmarks
    landmarks_to_visualize = get_landmarks(landmarks_to_visualize, bboxes)

    counter=0
    dotted_image = copy.copy(image)
    for bbox in bboxes[0]:
        start_point, end_point, color, thickness = (
            int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)),
            (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)),
            (255 if (counter + 2) % 3 == 0 else 0,
            255 if (counter + 3) % 3 == 0 else 0,
            255 if (counter + 1) % 3 == 0 else 0),
            2       
        dotted_image = cv2.rectangle(dotted_image, start_point, end_point, color, thickness)
        
        for landmark in landmarks_to_visualize:
            dotted_image = set_dots(landmark, dotted_image)
        counter+=1


    return predicted_gestures, dotted_image


