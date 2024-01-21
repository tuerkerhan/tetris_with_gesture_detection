# Tetris with Hand - Gestures
Welcome to the Tetris with Hand Gestures repository! Our project's goal is to accurately detect and recognize various hand gestures using machine learning and computer vision techniques. Detected hand gestures are then passed to the game logic of Tetris to be processed as game moves. You can play the game with the following gestures:

- `like` (thumbs up): rotate
- `dislike` (thumbs down): drop
- `thumb_left` (pointing the thumb to the left): move left
- `thumb_right` (pointing the thumb to the right): move right

# Project-Structure
## Dataset Used
We used the following dataset: [HaGRID](https://www.kaggle.com/datasets/kapitanov/hagrid) - HAnd Gesture Recognition Image Dataset. We used two labels from the dataset (`like` and `dislike`) and we generated two more gesture labels by modifying the labels `like` and `dislike`.

## Trainers
Includes training scripts for each of the three models. Trained models for Landmark Regression and Gesture Classifier are saved in models/landmark_regressor and models/gesture_classifier respectively. Our Object Detection model `yolov8n` creates its own directory  after the training with its training results and trained model. The trained model can be copied to model/detector so that it can be loaded properly.

## Models
Includes trained models for each network. These are then loaded to empty network instances in `main.py`

## Templates
Template used by the `Flask` server which also includes Tetris game logic.

## Pre-processing Scripts
Scripts which were used to pre-process the data.

# Dependencies
Make sure to install these dependencies before running the app: `flask`, `torch`, `OpenCV`, `ultralytics`, `tqdm`

# Usage
- Run the `app.py`: `python app.py`
- Copy the url to the local server `Flask` creates

# Contributors

- Oguzhan Cesur
- Ozan Ilgin Tanriverdi
- Sarp Cagin Erdogan
- Turker Han

# Sources

- https://www.kaggle.com/datasets/kapitanov/hagrid
- https://docs.ultralytics.com/modes/
- https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/
- https://pytorch.org/docs/stable/index.html
- https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
- https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/

