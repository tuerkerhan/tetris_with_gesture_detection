import cv2
import json
import shutil
from tqdm import tqdm


def right_hand(inp, out):
    image = cv2.imread(inp)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    output_path = out
    cv2.imwrite(output_path, rotated_image)
    #cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def left_hand(inp, out):
    image = cv2.imread(inp)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    flipped_image = cv2.flip(rotated_image, 1)
    output_path = out
    cv2.imwrite(output_path, flipped_image)
    #cv2.imshow('Rotated Image', flipped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def identity(inp, out):
    image = cv2.imread(inp)
    output_path = out
    cv2.imwrite(output_path, image)
    #cv2.imshow('Rotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Open the JSON file and load the data
with open('like_val.json', 'r') as file:
    data = json.load(file)

# bbox: [top left X position, top left Y position, width, height]
def bbox_transformation_left(bbox):
    new_top_left = (bbox[0],bbox[1])
    result = [1-new_top_left[1], new_top_left[0], bbox[3], bbox[2]]
    result[0] = 1- result[0]
    return result

# bbox: [top left X position, top left Y position, width, height]
def bbox_transformation_right(bbox):
    new_top_left = (bbox[0]+bbox[2],bbox[1])
    result = [new_top_left[1], 1-new_top_left[0], bbox[3], bbox[2]]
    return result

# bbox: [top left X position, top left Y position, width, height]
def landmark_transformation_left(landmark):
    result = []
    for point in landmark:
        result.append([point[1], point[0]])
    return result

# bbox: [top left X position, top left Y position, width, height]
def landmark_transformation_right(landmark):
    result = []
    for point in landmark:
        result.append([point[1],1-point[0]])
    return result

for key, value in tqdm(list(data.items())):
    hand = value["leading_hand"]
    
    object = data[key]
    image_name = f'train_val_like/{key}.jpg'

    bboxes = object['bboxes']
    landmarks = object['landmarks']
    

    
    
    if hand == 'left':
        left_hand(f'train_val_like/{key}.jpg', f'train_val_like_right/{key}.jpg')
        data[key]['bboxes'] = list(map(bbox_transformation_left, bboxes))
        data[key]['landmarks'] = list(map(landmark_transformation_left, landmarks))
    if hand == 'right':
        right_hand(f'train_val_like/{key}.jpg', f'train_val_like_right/{key}.jpg')
        data[key]['bboxes'] = list(map(bbox_transformation_right, bboxes))
        data[key]['landmarks'] = list(map(landmark_transformation_right, landmarks))



