'''
This script loads the processed data from the hands.json file
and visualizes the landmarks on the images. It does this by 
drawing green dots on the landmarks' coordinates.
'''
import cv2
import json

# The 'hands.json' file, which presumably contains annotations of 
#hand landmarks in a certain format, is opened and its contents are 
#loaded into the 'data' variable

if __name__ == '__main__':
    with open('hands.json', 'r') as file:
        data = json.load(file)

    def set_polygon(image_id):
        # For a given image ID, this function extracts the corresponding 
        #bounding boxes and landmarks from the data, loads the corresponding image,
        # draws the landmarks on the image, and then displays and saves the image
        object = data[image_id]
        image_name = f'test_hands/{image_id}.jpg'
        bboxes = object['bboxes']
        landmark = object['landmarks']

        image = cv2.imread(image_name)
        image = set_dots(landmark, image)

        cv2.imshow("Polygon", image)
        cv2.imwrite(f'test/{image_id}.jpg', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def set_dots(landmarks, image):
        # This helper function takes a list of landmarks and an image, 
        #and draws a green dot on the image at the location of each landmark
        new_image = image
        h, w, channels = new_image.shape
        points = list(map(lambda x : (int(w*x[0]), int(h*x[1])), landmarks))
        for point in points:
            x, y = point
            if 0 <= x < new_image.shape[1] and 0 <= y < new_image.shape[0]:
                cv2.circle(new_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the point
        return new_image

    # This list contains some test image IDs. The script will process each image ID in this list.
    test = ['0a0a079c-1467-4498-b37c-e5ae87638211_0-0', '0a0a079c-1467-4498-b37c-e5ae87638211_1-0', '0a0ad196-8084-4e26-8cb1-c279b8aea4d8_0-0']

    # Call the set_polygon function for each image ID in the test list
    for key in test:
        set_polygon(key)
