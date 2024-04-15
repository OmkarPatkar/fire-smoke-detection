# Fire-smoke Detection using Convolutional Neural Networks (CNN)

This repository contains a Python script to build and train a Convolutional Neural Network (CNN) for fire detection using TensorFlow and OpenCV. The model is designed to classify images as either containing fire or not containing fire.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/OmkarPatkar/fire-smoke-detection.git
cd fire-smoke-detection
pip install -r requirements.txt
```

## Dataset
The dataset used for this project should be organized in the following directory structure:
```
Image_Dataset/
|-- Not_Fire/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|-- Fire/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
```

## Code Explanation
Importing Libraries
- os: For interacting with the operating system.
- cv2: OpenCV library for image processing.
- numpy: For numerical operations.
- glob: For file matching.
- tensorflow: For building and training the model.
- matplotlib: For plotting.
- sklearn: For data splitting.
- ImageDataGenerator: For data augmentation.

## Data Preprocessing
- Iterate through the dataset folders to load and preprocess images.
- Resize images to 128x128 pixels.
- Normalize image pixel values to the range [0, 1].
- Split the dataset into training and testing sets.

## Model Architecture
- The CNN model architecture consists of:
- Separable Convolutional layers with ReLU activation.
- Batch Normalization layers.
- MaxPooling layers.
- Fully connected layers with ReLU activation and Dropout.

## Training
- Data augmentation using ImageDataGenerator.
- Class weight calculation to handle class imbalance.
- Compile the model with SGD optimizer and binary crossentropy loss.
- Train the model for 50 epochs with batch size of 64.

## Results
- Save the trained model.
- Plot training and validation loss/accuracy curves.
- Perform inference on test images and visualize predictions.
