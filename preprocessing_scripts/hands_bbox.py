'''
This script loads the datasets with the images and their annotations 
(bounding boxes and landmarks). It iterates over all images and for 
each one, it processes the bounding boxes and landmarks for the hands 
detected in the image. The script normalizes the landmark coordinates
relative to their bounding boxes and creates a new JSON file (hands.json) 
with the processed data.
'''


def set_landmarks(object, i, height, width):
    # Creates a copy of the object
    new_object = copy.copy(object)

    # Sets the landmarks and bounding boxes of the new object
    new_object["landmarks"] = new_object["landmarks"][i]
    new_object["bboxes"] = new_object["bboxes"][i]

    # Calculates the top left corner and dimensions of the bounding box
    left_top    = new_object['bboxes'][0], new_object['bboxes'][1]
    bbox_width  = int(new_object['bboxes'][2]*width)
    bbox_height = int(new_object['bboxes'][3]*height)

    # Iterates over all the landmarks
    for j in range(len(new_object["landmarks"])):

        # Normalizes the x coordinate of the landmark
        if new_object["landmarks"][j][0] - left_top[0] >=0:
            new_object["landmarks"][j][0] = ((new_object["landmarks"][j][0] - left_top[0])*width) /bbox_width
        else:
            new_object["landmarks"][j][0] = 0

        # Normalizes the y coordinate of the landmark
        if new_object["landmarks"][j][1] - left_top[1] >=0:
            new_object["landmarks"][j][1] = ((new_object["landmarks"][j][1] - left_top[1])*height) /bbox_height
        else:
            new_object["landmarks"][j][1] = 0

    return new_object
