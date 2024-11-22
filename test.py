import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained emotion recognition model (ensure it's in the same directory or provide full path)
model = load_model('emotion_cnn_model.h5')  # or 'emotion_cnn_model.keras'

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load the pre-trained Haar Cascade for face detection (you can also use a DNN model for more accurate detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess face for the model
def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_img = cv2.resize(face_img, (48, 48))  # Resize to 48x48 pixels
    face_img = img_to_array(face_img)  # Convert to array
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    face_img = face_img.astype('float32') / 255.0  # Normalize pixel values
    return face_img

# Start the video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # For each detected face, predict the emotion and draw a green square around it
    for (x, y, w, h) in faces:
        # Draw green square around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face region of interest (ROI)
        face_roi = frame[y:y + h, x:x + w]
        
        # Preprocess the face image and make a prediction
        preprocessed_face = preprocess_face(face_roi)
        emotion_probs = model.predict(preprocessed_face)
        
        # Get the predicted emotion label
        predicted_emotion = emotion_labels[np.argmax(emotion_probs)]
        
        # Put the emotion text near the face square
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Recognition - Live Feed', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
