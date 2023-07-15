import cv2
import json
import numpy as np

# Open the JSON file and load the data
with open('hands.json', 'r') as file:
    data = json.load(file)




def set_polygon(image_id):
    object = data[image_id]
    image_name = f'test_hands/{image_id}.jpg'

    bboxes = object['bboxes']
    landmark = object['landmarks']


    #bbox = object['bboxes'][0]
    #landmarks = object['landmarks'][0]
    #print('bboxes:', bbox)
    #print('landmarks',landmarks)
    # List of polygon coordinates (x, y)
    # Create a blank image to draw the polygon on
    image = cv2.imread(image_name)

    
    image = set_dots(landmark, image)


    cv2.imshow("Polygon", image)
    cv2.imwrite(f'test/{image_id}.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def set_dots(landmarks, image):
    new_image = image
    h, w, channels = new_image.shape
    points = list(map(lambda x : (int(w*x[0]), int(h*x[1])), landmarks))
    for point in points:
        x, y = point
        # Check if the point is within the image boundaries
        if 0 <= x < new_image.shape[1] and 0 <= y < new_image.shape[0]:
            cv2.circle(new_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the point
    return new_image

"""
'train_val_like/0001a82e-14af-4db8-bfec-8b09f67612d2.jpg'

bboxes:
                0.31371071,
                0.4016502,
                0.14489726,
                0.14797368

                
                [0.31371071,0.4016502,0.14489726,0.14797368]
                """
test= ['0a0a079c-1467-4498-b37c-e5ae87638211_0-0', '0a0a079c-1467-4498-b37c-e5ae87638211_1-0', '0a0ad196-8084-4e26-8cb1-c279b8aea4d8_0-0']

for key in test:
    set_polygon(key)