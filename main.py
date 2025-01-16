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

        # creating a condition to check if the model has been built
        if self.model is None:
            self.modelBuild(len(self.LabelEncoder.classes_))


        # training the model using the training data
        self.history = self.model.fit(
            pot1_train, pot2_train, 
            epochs = epochs, 
            batch_size = batch_size, 
            validation_data = (pot1_test, pot2_test))
            
        # returns the evaluation of the model    
        return self.evaluateModel(pot1_test, pot2_test)

############ EVALUATING THE MODEL ############

    # creating a function to evaluate the model
    def evaluateModel(self, pot1_test, pot2_test):

        pred = self.model.predict(pot1_test)                    # takes the images from test and gives the predictions from the model
        pot2_pred = np.argmax(pred, axis = 1)                   # takes the prediction and returns the font with the highest probability

        # calculating the accuracy of the model
        accuracyScore = accuracy_score(pot2_test, pot2_pred)    
        print(f"Accuracy: {accuracyScore}")

        # creating a classification report
        classReport = classification_report(pot2_test, pot2_pred, target_names = self.LabelEncoder.classes_)
        print(classReport)

        # creating a confusion matrix
        conMax = confusion_matrix(pot2_test, pot2_pred)
        print(conMax)

        ############# PLOTTING GRAPHS FOR THE MODEL ##################

        # generating the plot for the confusion matrix of the model
        plt.figure(figsize=(20, 20))
        sns.heatmap(conMax, annot=True, fmt='d', cmap='Blues', xticklabels=self.LabelEncoder.classes_, yticklabels=self.LabelEncoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Fonts')
        plt.ylabel('Actual Fonts')
        plt.tight_layout()
        plt.savefig('Confusion-Matrix.png')
        plt.close()



    # creating a function to make predictions for the model
    def predict(self, imgPath):

        # using try except blocks to handle any exceptions that may occur
        try: 

            # opening the image file
            img = Image.open(imgPath).convert('RGB')
            img = img.resize(self.imgSize)
            imgArray = np.array(img) / 255.0
            imgArray = np.expand_dims(imgArray, axis = 0)         # adding an extra dimension to the array to make it compatible with the model

            # making the predictions
            pred = self.model.predict(imgArray)
            predClass = self.LabelEncoder.classes_[np.argmax(pred, axis = 1)]
            modelConficence = np.max(pred) * 100

            return predClass, modelConficence

        except Exception as e:
            return f"Error processing image: {imgPath}: {e}"

    # creating a function to save the trained model
    def saveTrainedModel(self, modelPathTrain = 'Font-Recognizer-Model.h5'):
        self.model.save(modelPathTrain)

    # creating a function to load the trained model
    def loadTrainedModel(self, modelPathTrain = 'Font-Recognizer-Model.h5'):
        self.model = models.loadModel(modelPathTrain)
 


    
    
    

############ BUILDING THE CNN MODEL ############

    
