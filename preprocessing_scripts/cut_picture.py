import cv2
import json
import shutil
from tqdm import tqdm
import copy

def cut_picture(image, x, y, width, height):
    # Calculate the coordinates of the bottom-right corner
    x2 = x + width
    y2 = y + height

    # Crop the image based on the coordinates
    cropped_image = image[y:y2, x:x2]
    
    return cropped_image