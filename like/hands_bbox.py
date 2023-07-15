import cv2
import json
import shutil
from tqdm import tqdm
import copy

def cut_picture(img, x, y, width, height):
    
    # Calculate the coordinates of the bottom-right corner
    x2 = x + width
    y2 = y + height

    # Crop the image based on the coordinates
    cropped_image = img[y:y2, x:x2]
    
    return cropped_image



datasets = [('train_val_like_left', 'train_val_like_left.json'), ('train_val_like_right', 'train_val_like_right.json')]

hands_json = {}

def set_landmarks(object, i, height, width):
    new_object = copy.copy(object)
    new_object["landmarks"] = new_object["landmarks"][i]
    new_object["bboxes"] = new_object["bboxes"][i]
    left_top    = new_object['bboxes'][0], new_object['bboxes'][1]
    bbox_width  = int(new_object['bboxes'][2]*width)
    bbox_height = int(new_object['bboxes'][3]*height)
    for j in range(len(new_object["landmarks"])):
        if new_object["landmarks"][j][0] - left_top[0] >=0:
            new_object["landmarks"][j][0] = ((new_object["landmarks"][j][0] - left_top[0])*width) /bbox_width
        else:
            new_object["landmarks"][j][0] = 0
        if new_object["landmarks"][j][1] - left_top[1] >=0:
            new_object["landmarks"][j][1] = ((new_object["landmarks"][j][1] - left_top[1])*height) /bbox_height
        else:
            new_object["landmarks"][j][1] = 0
    return new_object

for d in tqdm(range(len(datasets))):
    # Open the JSON file and load the data
    source      = datasets[d][0]
    source_json = datasets[d][1]

    
    with open(source_json, 'r') as file:
        data = json.load(file)
    

    for key, value in tqdm(list(data.items())):
        hand = value["leading_hand"]
        
        object = data[key]

        image_path = f'{source}/{key}.jpg'
        image = cv2.imread(image_path)
        h, w, channels = image.shape

        bboxes = object['bboxes']
        landmarks = object['landmarks']

        new_object = copy.copy(object)
        new_bbox = []
        new_landmark = []
        for i in range(len(bboxes)):
            hand_image_id = f'{key}_{d}-{i}'
            #hand_image = cut_picture(image, int(bboxes[i][0]*w), int(bboxes[i][1]*h), int(bboxes[i][2]*w), int(bboxes[i][3]*h))
            #cv2.imwrite(f'hands/{hand_image_id}.jpg', hand_image)
            new_object = set_landmarks(object, i, h, w)
            hands_json[hand_image_id] = new_object
        
hands_json = json.dumps(hands_json)
with open('hands.json', "w") as file:
    file.write(hands_json)