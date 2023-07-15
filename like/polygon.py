import cv2
import json
import numpy as np

# Open the JSON file and load the data
with open('hands.json', 'r') as file:
    data = json.load(file)






def set_rotate(image, flip=False, rotate=0):
    new_image = cv2.imread(image)
    if flip:
        new_image = flipped_image = cv2.flip(new_image, 1)
    
    for i in range(rotate):
        new_image = cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #output_path = out
    #cv2.imwrite(output_path, rotated_image)
    return new_image


# bbox: [top left X position, top left Y position, width, height]
def bbox_to_coordinates(bbox, image):
    h, w, channels = image.shape
    centerX = bbox[0]*w
    centerY = bbox[1]*h
    width   = bbox[2]*w
    height  = bbox[3]*h
    top_left     = int(centerX),       int(centerY)
    top_right    = int(centerX+width), int(centerY)
    bottom_left  = int(centerX),       int(centerY+height)
    bottom_right = int(centerX+width), int(centerY+height)
    result = [top_left, top_right,bottom_right, bottom_left]
    return result



def set_polygon(image_id):
    object = data[image_id]
    image_name = f'test_hands/{image_id}.jpg'

    bboxes = object['bboxes']
    landmarks = object['landmarks']


    #bbox = object['bboxes'][0]
    #landmarks = object['landmarks'][0]
    #print('bboxes:', bbox)
    #print('landmarks',landmarks)
    # List of polygon coordinates (x, y)
    # Create a blank image to draw the polygon on
    image = cv2.imread(image_name)
    image = set_rotate(image_name, flip=False, rotate=0)

    for bbox in bboxes:
        polygon_coordinates = bbox_to_coordinates(bbox, image)
        # Convert the list of coordinates to a NumPy array
        pts = np.array(polygon_coordinates, np.int32)

        # Reshape the array into the required shape for cv2.polylines() function
        pts = pts.reshape((-1, 1, 2))

        # Draw the polygon on the image
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    for landmark in landmarks:
        image = set_dots(landmark, image)
    
    polygon_coordinates = bbox_to_coordinates(bbox, image)
    # Convert the list of coordinates to a NumPy array
    pts = np.array(polygon_coordinates, np.int32)

    # Reshape the array into the required shape for cv2.polylines() function
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon on the image
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    # Display the image with the polygon

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