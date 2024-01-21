'''
This script trains a gesture classifier using a custom dataset. 
The gestures it recognizes are: "like", "dislike", "thumb_left", 
"thumb_right", and "no_gesture". It uses a simple neural network 
with two fully connected layers.

The GestureDataset class loads and preprocesses the gesture data.
It normalizes the coordinates of the hand landmarks and labels 
the gestures based on a predefined dictionary. This 
class implements __getitem__() and __len__() methods 
required by PyTorch Dataset.

The GestureClassifier class defines the architecture of the 
neural network. The forward method is how the network computes 
outputs from the inputs. save_model saves the trained model to disk.

The script then creates an instance of the dataset, the dataloader,
the model, the loss function (Cross Entropy Loss), and 
the optimizer (Adam). It trains the model over a specified number
of epochs, calculating the loss, backpropagating the gradients, 
and updating the model parameters.
'''


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
import os


# Define gesture labels
gestures = {
    "like": 1,
    "dislike": 2,
    "thumb_left": 3,
    "thumb_right": 4,
    "no_gesture": 5
}


# Custom PyTorch Dataset
class GestureDataset(Dataset):
    def __init__(self, annotations_file):
        # Load data from the JSON file
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

        self.normalized_landmarks_list = []
        self.gesture_list = []

        # Iterate through each item in the dataset
        for idx in tqdm(range(len(list(self.data.keys()))), desc='Data Loading'):
            key, annotation = list(self.data.items())[idx]

            # Iterate through each annotation
            for i in range(len(annotation['landmarks'])):
                # Ignore if there's no landmarks
                if annotation['landmarks'][i] == []:
                    continue
                # Ignore if the label is not recognized
                if not annotation['labels'][i] in ['no_gesture', 'like', 'dislike', 'thumb_left', 'thumb_right']:
                    continue

                # Extract bounding box
                x, y, width, height = annotation['bboxes'][i]

                # Normalize landmark coordinates
                landmarks = torch.tensor(annotation['landmarks'][i])
                normalized_landmarks = landmarks.clone()
                normalized_landmarks[:, 0] = (landmarks[:, 0] - x) / width
                normalized_landmarks[:, 1] = (landmarks[:, 1] - y) / height

                # Flatten the normalized landmarks into a 1D tensor
                new_normalized = normalized_landmarks.view(-1)
                self.normalized_landmarks_list.append(new_normalized)
                
                # Convert gesture label to a tensor
                gesture_label = gestures[annotation['labels'][i]] - 1
                single_gesture_list = [gesture_label]
                single_gesture_tensor = torch.tensor(single_gesture_list)
                self.gesture_list.append(single_gesture_tensor)
        
        self.gesture_list = torch.cat(self.gesture_list, dim=0)

    # Return number of samples
    def __len__(self):
        return len(self.normalized_landmarks_list)

    # Return a sample
    def __getitem__(self, idx):
        return self.normalized_landmarks_list[idx], self.gesture_list[idx]


# The neural network architecture
class GestureClassifier(nn.Module):
    def __init__(self, num_landmarks, num_classes):
        super(GestureClassifier, self).__init__()

        self.date_string = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%f')}"
        
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes

        # Define network layers
        self.fc1 = nn.Linear(num_landmarks * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    # Forward pass
    def forward(self, x):
        # Flatten the input tensor (21 landmarks * 2 for x and y coordinates)
        x = x.view(-1, self.num_landmarks * 2)

        # Pass the input through the layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply softmax to the output for classification
        x = F.softmax(x, dim=1)

        return x
    
    # Save the model
    def save_model(self):
        os.makedirs("models/gesture_regression", exist_ok=True)
        save_path = os.path.join("models/gesture_regression", self.date_string + "_model.pt")
        torch.save(self.state_dict(), save_path)


# Training code
if __name__ == '__main__':

    # Create the custom dataset
    gesture_dataset = GestureDataset('dataset.json')

    # Create DataLoader with batch size and shuffling
    batch_size = 4
    shuffle = True
    gesture_dataloader = DataLoader(gesture_dataset, batch_size=batch_size, shuffle=shuffle)

    # Instantiate the Gesture Classifier Model
    num_landmarks = 21
    num_classes = 5  # Replace with the number of gesture classes in your dataset
    net = GestureClassifier(num_landmarks, num_classes)

    # Define the Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 5  # Replace with the desired number of training epochs

    for epoch in tqdm(range(num_epochs), desc='Training'):
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(gesture_dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Running loss
            running_loss += loss.item()

        # Print loss for each epoch
        avg_loss = running_loss / len(gesture_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print('Training Finished!')
    net.save_model()