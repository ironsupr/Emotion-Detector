# Emotion Recognition System

An advanced system for recognizing human emotions from facial expressions using a Convolutional Neural Network (CNN). This project involves training a deep learning model and deploying it for real-time emotion detection via a webcam.

## Features

- **Emotion Categories:** Detects seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Model:** Built using TensorFlow/Keras with a CNN architecture.
- **Data Preprocessing:** Converts images to grayscale, resizes them to 48x48 pixels, and normalizes pixel values.
- **Real-Time Recognition:** Uses Haar Cascade for face detection and a trained CNN model for emotion prediction.
- **Visualization:** Includes functions to plot training progress, validation accuracy, and loss.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/emotion-recognition-system.git
   cd emotion-recognition-system
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the Haar Cascade XML file for face detection:

   ```bash
   curl -O https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
   ```

## Usage

### Training the Model

1. Organize your dataset in the following format:

   ```
   Dataset/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── sad/
   │   ├── surprise/
   │   └── neutral/
   └── test/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── sad/
       ├── surprise/
       └── neutral/
   ```

2. Run the training script:

   ```bash
   python model.py
   ```

   This will train the CNN and save the best-performing model as **emotion_cnn_model.h5**.

### Testing the Model in Real-Time

1. Ensure a webcam is connected to your system.
2. Run the real-time detection script:

   ```bash
   python test.py
   ```

3. Press **Q** to exit the real-time feed.

## Project Structure

- **model.py:** Contains the code to train and evaluate the CNN model.
- **test.py:** Implements real-time emotion recognition using a webcam.
- **requirements.txt:** Lists the required Python packages for the project.
- **Dataset/**: Directory structure for organizing the dataset.

## Results

- The model achieves an accuracy of over **96%** (replace with actual accuracy) on the validation set.
- Real-time predictions are rendered efficiently, with detected emotions displayed alongside the detected face.

## Dependencies

- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib

## Contributing

Feel free to contribute to this project by submitting a pull request. For significant changes, please open an issue first to discuss the planned modifications.
