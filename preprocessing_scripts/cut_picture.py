#This script defines a function, cut_picture(), 
#which takes an image and coordinates as input. 
#The function crops the input image according to 
#the given coordinates (bounding box) and returns the cropped image.

def cut_picture(image, x, y, width, height):
    # Calculate the coordinates of the bottom-right corner
    x2 = x + width
    y2 = y + height

    # Crop the image based on the coordinates
    # Here, it uses numpy slicing to crop the image
    # y:y2 is the range on the height (row) axis
    # x:x2 is the range on the width (column) axis
    cropped_image = image[y:y2, x:x2]
    
    return cropped_image
