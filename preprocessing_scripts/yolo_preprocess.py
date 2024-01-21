'''
Pre-processing needed for yolo training.
Creates .txt files for each image with the same
name as the image and includes bboxes of objects
in the image.
'''


import json

with open("like.json", "r") as file:
    data = json.load(file)

ids = list(data.keys())
print(ids[0])

print(data[ids[0]]["labels"][0])

result = []

for id in ids:
    my_box = ""
    bboxes = data[id]["bboxes"]
    for bbox in bboxes:
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        width = bbox[2]
        height = bbox[3]
        my_box = f"0 {x_center} {y_center} {width} {height}"
        result.append(my_box)
    #print(result)
    with open(f"dataset/val/labels/{id}.txt", "w") as file:
        for box in result:
            file.write(box + '\n')

    result = []