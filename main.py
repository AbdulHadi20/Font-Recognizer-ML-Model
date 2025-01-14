"""
Machine Learning Module: Assessment 2 (2025) 

AbdulHadi Aamir
Creative Computing Year 3
Bath Spa University, RAK
STUDENT ID: 581756
FONT RECOGNIZER MODEL

"""

############ START OF THE PROGRAM ############

# importing all of the required modules and libraries

import os                                          # used for systeme file operations
import numpy as np                                 # used for performing mathematical operations
import pandas as pd                                # used for data manipulation and analysis
import seaborn as sns                              # used for data visualization
import matplotlib.pyplot as plt                    # used for plotting visual graphs  
import tensorflow as tf                            # used for building and training the model (deep learning, neural networks)

from tensorflow.keras import layers, models                                               # used for building the layers of the model
from PIL import Image                                                                     # used for image processing

from sklearn.model_selection import train_test_split                                      # used for splitting the data into training and testing sets
from sklearn.preprocessing import LabelEncoder                                            # used for encoding the labels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix       # used for evaluating the model

############ DATA LOADING AND PREPROCESSING ############

# creating a class 
class FontRecognizerModel:

    # creating a constructor
    def __init__(self, pathOfDataset, imgSize = (128, 128)):
        self.pathOfDataset = pathOfDataset
        self.imgSize = imgSize
        self.model = None
        self.LabelEncoder = LabelEncoder()
        self.history = None

    # creating a mehtod to load and preproccess the dataset
    def loadDataset(self):
        processed_images = []       # creating an empty list to store the processed images
        font_labels = []            # creating an empty list to store the font labels

        # creating a loop to iterate through the dataset folder
        for nameOfFont in os.listdir(self.pathOfDataset):                           # os.listdir returns the files/folders in the main directory
            fontPath = os.path.join(self.pathOfDataset, nameOfFont)                 # os.join combines the path of the main directory with the name of the font

            # creating a condition to make sure that only the subfodlers are processed
            if os.path.isdir(fontPath):

                # creating a nested loop to make sure to iterate through the subfolders accurately
                for fontImage in os.listdir(fontPath):
                    # checking if each image file is a .jpg file, then processing the images
                    
                    if fontImage.endswith('.jpg'):
                        imgPath = os.path.join(fontPath, fontImage)

                        # using try except block to handle any exceptions that may occur
                        try: 
                            img = Image.open(imgPath)                # opening the image file
                            img = img.resize(self.imgSize)           # resizing the image to the specified size
                            imgArray = np.array(img) / 255.0         # converting the image into an array

                            processed_images.append(imgArray)        # appending the processed image to the list
                            font_labels.append(nameOfFont)           # appending the font label to the list

                        except Exception as e:
                            print(f"Error processing image: {imgPath}: {e}")

            # converting the lists into numpy arrays
            pot1 = np.array(processed_images)
            self.LabelEncoder.fit(font_labels)
            pot2 = self.LabelEncoder.transform(font_labels)

            # splitting the data into training and testing sets
            return train_test_split(pot1, pot2, test_size = 0.2, random_state = 42)
            
 ############ BUILDING THE CNN MODEL ############

    # creating a function to build the CNN model
    def modelBuild(self, numFonts):
        
        # using the Sequential API to build the model, a stack of layers in a linear pipeline
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (*self.imgSize + 3,)),              # fist convolutional layer, 32 filters, 3x3 kernel size
                layers.MaxPooling2D((2, 2)),                                                                     # max pooling layer, 2x2 pool size
                layers.Conv2D(64, (3, 3), activation = 'relu'),                                                  # second convolutional layer, 64 filters, 3x3 kernel size
                layers.MaxPooling2D((2, 2)),                                                                     # max pooling layer, 2x2 pool size
                layers.Conv2D(64, (3, 3), activation = 'relu'),                                                  # third convolutional layer, 128 filters, 3x3 kernel size
                layers.Flatten(),                                                                                # flattening the input
                layers.Dense(64, activation = 'relu'),                                                           # first dense layer, 64 neurons
                layers.Dropout(0.5),                                                                             # dropout layer, 0.5 dropout rate
                layers.Dense(numFonts, activation = 'softmax')                                                   # output layer, number of neurons equal to the number of fonts
                ]
        )              
        
        # compiling the model using the adam optimizer, sparse categorical crossentropy loss function
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])     
        return model 
        
        
 ############ TRAINING THE MODEL ############
    
    # creating a function to train the model and save training history
    def modelTrain(self, epochs = 10, batch_size = 32):
        
        # splitting the data into training and testing sets
        pot1_train, pot1_test, pot2_train, pot2_test = self.loadDataset()



    
    
    

############ BUILDING THE CNN MODEL ############

    
