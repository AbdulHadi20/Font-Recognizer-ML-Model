"""
Machine Learning Module: Assessment 2 (2025) 

AbdulHadi Aamir
Creative Computing Year 3
Bath Spa University, RAK
STUDENT ID: 581756

FONT RECOGNIZER MODEL
"""

############ START OF THE PROGRAM ############

# importing all the required moules and libraries to help run the model

import os                                                              # used for systeme file operations
import numpy as np                                                     # used for performing mathematical operations
import pandas as pd                                                    # used for data manipulation and analysis
import gradio as gr                                                    # used for creating the GUI
import seaborn as sns                                                  # used for data visualization    
import tensorflow as tf                                                # used for building and training the model (deep learning, neural networks)
import matplotlib.pyplot as plt                                        # used for plotting visual graphs

from PIL import Image                                                  # used for image processing  
from sklearn.model_selection import train_test_split                   # used for splitting the data into training and testing sets
from sklearn.preprocessing import LabelEncoder                         # used for encoding the labels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # used for evaluating the model
from tensorflow.keras import layers, models                            # used for building the layers of the model

############ DATA ANALYSIS ############

# initializing the global variables

pathOfDataset = "Font Dataset Large"                                   # path of the dataset
sizeOfImage = (128, 128)                                               # size of the image
label_encoder = LabelEncoder()                                         # initializing the label encoder

# counting the number of fonts in the dataset (subfolders of fonts with 500 images for each font)
print('\n Scanning The DataSet...')

fonts = []  # list to store fonts

# creating a for loop to run trough the dataset folder
for nameofFont in os.listdir(pathOfDataset):
    fontPath = os.path.join(pathOfDataset, nameofFont)

    if os.path.isdir(fontPath): 
        if any(f.endswith('.jpg') for f in os.listdir(fontPath)):
            fonts.append(nameofFont)

# saving the number of font files to display the number of fonts we're working with(+1 as the index number starts with 0)
numFonts = len(fonts)
print(f'\n {numFonts + 1} fonts were found in the dataset.')

############ DATA LOADING AND PREPROCESSING ############

# Creating The CNN Model (Sequential Model)

fontModel = models.Sequential([
   
   # first layer, 3x3 kernel, 32 
   layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (*sizeOfImage, 3)),
   layers.BatchNormalization(),
   layers.MaxPooling2D((2, 2)), 

   # second layer, 3x3 kernel, 64
   layers.Conv2D(64, (3, 3), activation = 'relu'),
   layers.BatchNormalization(),
   layers.MaxPooling2D((2, 2)),

   # third layer, 3x3 kernel, 128
   layers.Conv2D(128, (3, 3), activation ='relu'),
   layers.BatchNormalization(),
   layers.MaxPooling2D((2, 2)), 

   # dense layers
   layers.Flatten(), 
   layers.Dense(256, activation = 'relu'), 
   layers.BatchNormalization(), 
   layers.Dropout(0.5), 

   layers.Dense(numFonts, activation = 'softmax')

])

# compiling the images using the adam optimizer & sparse categorical crossentropy loss function
fontModel.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# loading & preprocessing all images: then saving them in empty list as follows repectively

fontImages = []
fontLabels = []

# creating a for loop to go through the dataset folder, then adding the images and font names in their respective lists

for nameofFont in fonts:
    fontPath = os.path.join(pathOfDataset, nameofFont)
    numImages = 0

    # nested for loop to iterate through subfolders
    for fontImage in os.listdir(fontPath):
        if fontImage.endswith('.jpg'):
            pathOfImage = os.path.join(fontPath, fontImage)

            # using try except blocks for error hamdling
            try: 
                img = Image.open(pathOfImage).convert('RGB')
                img = img.resize(sizeOfImage)
                imgArray = np.array(img) / 255.0

                # appending the images and fonts into their respective lists
                fontImages.append(imgArray)
                fontLabels.append(nameofFont)
                numImages +=1

            except Exception as e: 
                print(f'\n Error Loading {pathOfImage}: {e}')

    print(f'\n {numImages} loaded from {nameofFont} subfloder.')      

                
############### TRAINING & TESTING THE MODEL ####################

# converting the lists into numpy arrays
pot1 = np.array(fontImages)
label_encoder.fit(fontLabels)
pot2 = label_encoder.transform(fontLabels)

# splitting the data into training & testing folders
pot1_train, pot1_test, pot2_train, pot2_test = train_test_split(pot1, pot2, test_size = 0.2, random_state = 42)

# training the model
print('\n TRAINING THE MODEL...')
modelHistory = fontModel.fit(
    pot1_train, pot2_train, 
    epochs = 15, 
    batch_size = 32, 
    validation_data = (pot1_test, pot2_test)
)

# making predictions
modelPrediction = fontModel.predict(pot1_test)
pot2_pred = np.argmax(modelPrediction, axis = 1)

################### EVALUATING THE MODEL ###########################

# generating the accuracy report 
accScore = accuracy_score(pot2_test, pot2_pred)
print(f'\n Accuracy Score: {accScore}')

# generating the classification report
classReport = classification_report(pot2_test, pot2_pred, target_names = label_encoder.classes_)
print(f'\n Classification Report: \n {classReport}')

# generating the confusion matrix
conMax = confusion_matrix(pot2_test, pot2_pred)
print(f'\n Confusion MAtrix: \n {conMax}')

################# GENERATING THE GRAPHS ###########################

# plotting graph for the confusion matrix
plt.figure(figsize = (15, 5))
sns.heatmap(conMax, annot = True, fmt = 'd', xticklabels = label_encoder.classes_, yticklabels = label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Font Names")
plt.ylabel("Actual Font Names")
plt.tight_layout()
plt.savefig('confusion-matrix.png')
plt.close()

# plotting the graph for the training history
plt.figure(figsize = (12, 4))
plt.subplot(1, 2, 1)
plt.plot(modelHistory.history['accuracy'], label = 'Training Accuracy')
plt.plot(modelHistory.history['val_accuracy'], label = 'Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy ")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(modelHistory.history['loss'], label = 'Training Loss')
plt.plot(modelHistory.history['val_loss'], label = 'Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('Training-History.png')
plt.close()

# saving the trained model in an .h5 file
fontModel.save('font-recognition-model.h5')
np.save('label-encoder-classes.npy', label_encoder.classes_)

######################## WORKING WITH THE USER UPLOADED IMAGE ###################

# creating a function to initialize the gradio interface to predict fonts in images uploaded by the user
def fontPredictor(fontImage):
    # creating a try except blocks for enhanced error handling
    try: 
        img = fontImage.convert('RGB')
        img = img.resize(sizeOfImage)
        imgArray = np.array(img) / 255.0
        imgArray = np.expand_dims(imgArray, axis = 0)

        # making the predictions on the uploaded images
        modelPrediction = fontModel.predict(imgArray)
        predictedClass = label_encoder.classes_[np.argmax(modelPrediction)]
        confidence = np.max(modelPrediction) * 100

        # generating the results
        return f"\n Font Predicted: {predictedClass} \n Model Confidence: {confidence:.2f}%"
    
    except Exception as e:
        return f"Error: {str(e)}"

################# GUI INITIALIZATION #######################################

# generating the interface
modelInterface = gr.Interface(
    fn = fontPredictor, 
    inputs = gr.Image(type = 'pil'), 
    outputs = gr.Textbox(label = 'Font Prediction Result'), 
    title = "Font Recognition Model",
    description = "Kindly upload and image to recognize the font!"
)

# launching the model interface
if __name__ == "__main__":
    modelInterface.launch()

########################## END OF PROGRAM ###############################