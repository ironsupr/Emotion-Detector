import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Emotion categories
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Map emotion labels to numeric indices
label_map = {emotion: index for index, emotion in enumerate(emotion_labels)}

# Function to load and preprocess data
def load_data(directory):
    X, y = [], []
    emotion_counts = {emotion: 0 for emotion in emotion_labels}
    
    # Loop through each emotion folder
    for label in emotion_labels:
        folder_path = os.path.join(directory, label)
        
        # Loop through each image in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):  # Assuming images are in JPG format
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                image = cv2.resize(image, (48, 48))  # Resize to 48x48 (standard size)
                X.append(image)
                y.append(label_map[label])  # Map emotion label to an index
                
                # Update the emotion count
                emotion_counts[label] += 1
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Normalize pixel values to [0, 1] range
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)  # Add channel dimension for grayscale images

    # One-hot encode the labels
    y = to_categorical(y, num_classes=7)
    
    return X, y, emotion_counts

# Function to build the CNN model
def build_cnn_model(input_shape=(48, 48, 1)):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotions
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    # Callback to save the best model during training (with the .keras extension)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
        verbose=1
    )
    
    return history


# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Function to plot training results
def plot_training_results(history):
    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Example usage in another script
if __name__ == "__main__":
    # Define paths for train and test folders
    train_dir = ''# Path to the training data
    test_dir = ''# Path to the test data
    
    # Load the training, validation, and test data
    X_train, y_train, emotion_counts_train = load_data(train_dir)
    X_test, y_test, emotion_counts_test = load_data(test_dir)
    
    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Build the CNN model
    cnn_model = build_cnn_model(input_shape=(48, 48, 1))
    
    # Train the model
    history = train_model(cnn_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Evaluate the model on the test set
    evaluate_model(cnn_model, X_test, y_test)
    
    # Plot the training results
    plot_training_results(history)
    
    # Save the model after training
    cnn_model.save('emotion_cnn_model.h5')
    print("Model saved as 'emotion_cnn_model.h5'")
