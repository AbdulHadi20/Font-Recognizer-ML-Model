# main.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# Initialize global variables
dataset_path = "Font Dataset Large"
img_size = (128, 128)
label_encoder = LabelEncoder()
model = None

# Create the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(48, activation='softmax')  # 48 classes for 48 fonts
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Load and preprocess all images
images = []
labels = []

# Walk through the dataset directory
for font_name in os.listdir(dataset_path):
    font_path = os.path.join(dataset_path, font_name)
    if os.path.isdir(font_path):
        for img_name in os.listdir(font_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(font_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_size)
                    img_array = np.array(img) / 255.0
                    
                    images.append(img_array)
                    labels.append(font_name)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

# Convert lists to numpy arrays
X = np.array(images)
label_encoder.fit(labels)
y = label_encoder.transform(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Generate predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Calculate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
acc_score = accuracy_score(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(15, 15))
sns.heatmap(conf_matrix, annot=True, fmt='d',
           xticklabels=label_encoder.classes_,
           yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Save the trained model
model.save('font_classifier_model.h5')

# Gradio interface for predictions
def predict_font(image):
    try:
        # Preprocess the uploaded image
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        
        img = Image.open(temp_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = label_encoder.classes_[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Clean up
        os.remove(temp_path)
        
        return f"Predicted Font: {predicted_class}\nConfidence: {confidence:.2f}%"
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_font,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Font Classifier",
    description="Upload an image to identify its font"
)

if __name__ == "__main__":
    interface.launch()