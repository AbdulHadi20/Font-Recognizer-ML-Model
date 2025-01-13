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


