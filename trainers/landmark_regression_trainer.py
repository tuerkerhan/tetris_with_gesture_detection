import os
import cv2
import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def cut_picture(image, x, y, width, height):
    # Calculate the coordinates of the bottom-right corner
    h, w, _ = image.shape
    x2 = int(w * (x + width))
    y2 = int(h * (y + height))

    # Crop the image based on the coordinates
    cropped_image = image[int(y * h): y2, int(x * w): x2]
    
    return cropped_image


class LandmarkRegressionDataset(Dataset):
    def __init__(self, root_folder, annotations_file, target_size):
        self.root_folder = root_folder
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        self.resized_images_list = []
        self.normalized_landmarks_list = []
        for idx in tqdm(range(len(list(self.annotations.keys()))), desc='Data Loading'):
            image_name = list(self.annotations.keys())[idx]  # Get the image name from the keys
            image_path = os.path.join(self.root_folder,f'{image_name}.jpg')
            image = cv2.imread(image_path)

            annotation = self.annotations[image_name]  


            for i in range(len(annotation['bboxes'])):
                if annotation['landmarks'][i] == []:
                    continue
                x, y, width, height = annotation['bboxes'][i]
                
                # Extract landmark coordinates
                landmarks = torch.tensor(annotation['landmarks'][i])
                # Crop the image based on bounding box with padding
                cropped_image = cut_picture(image, x, y, width, height)
                # Resize the padded image
                resized_image = self.transform(cropped_image)
                self.resized_images_list.append(resized_image)

                # Calculate relative coordinates to the bounding box
                # (landmark_x - x) / width
                # (landmark_y - y) / height
                normalized_landmarks = landmarks.clone()
                normalized_landmarks[:, 0] = (landmarks[:, 0] - x) / width
                normalized_landmarks[:, 1] = (landmarks[:, 1] - y) / height

                new_normalized = []
                # normalized_landmarks = [[x,y],[x,y]...]
                for i in range(21):
                    new_normalized.append(normalized_landmarks[:, 0][i])
                    new_normalized.append(normalized_landmarks[:, 1][i])
                new_normalized = torch.tensor(new_normalized)
                self.normalized_landmarks_list.append(new_normalized)
                

                #self.normalized_landmarks_list.append(normalized_landmarks)
                # Normalized = [[x,y]]
                # self.normalized_landmarks_list  = [[x,y,x,y,. ..], ..., ]


    def __len__(self):
        return len(list(self.annotations.keys()))

    def __getitem__(self, idx):
        return self.resized_images_list[idx], self.normalized_landmarks_list[idx]


class LandmarkRegressionNet(nn.Module):
    def __init__(self):
        super(LandmarkRegressionNet, self).__init__()

        self.date_string = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%f')}"

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Landmark regression head
        self.regression_head = nn.Sequential(
            nn.Linear(16384, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 42)  # 21 landmarks, each with x and y coordinates
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Landmark regression
        landmarks = self.regression_head(x)

        # Split the output into x and y coordinates for each landmark
        # x_coords = landmarks[:, 0::2]  # Even indices represent x-coordinates
        # y_coords = landmarks[:, 1::2]  # Odd indices represent y-coordinates

        return landmarks
        
    
    def save_model(self):
        os.makedirs("models/landmark_regression", exist_ok=True)
        save_path = os.path.join("models/landmark_regression", self.date_string + "_model.pt")
        torch.save(self.state_dict(), save_path)



if __name__ == '__main__':
    
    # Set your desired target size for the resized images (e.g., 128x128)
    target_size = (128, 128)

    # Path to your images folder and annotations JSON file
    images_folder = 'dataset/train/images'
    annotations_file = 'dataset.json'

    # Create the custom dataset
    dataset = LandmarkRegressionDataset(images_folder, annotations_file, target_size)

    # Set batch size according to your memory capacity
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = LandmarkRegressionNet()

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
        
    
    print('training')
    for epoch in tqdm(range(6), desc="Epoch"):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    net.save_model()