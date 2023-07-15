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

image = cv2.imread('test/0001a82e-14af-4db8-bfec-8b09f67612d2.jpg')

new_image = cut_picture(image, 100, 200, 300, 400)
cv2.imshow('Hand Image', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()