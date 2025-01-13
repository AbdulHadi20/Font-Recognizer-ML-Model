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
import requests                                    # used for handling urls (HTTPS requests)
import numpy as np                                 # used for performing mathematical operations
import matplotlib.pyplot as plt                    # used for plotting visual graphs  
  
from PIL import Image                              # (Python Imaging Library) used for image processing
from io import BytesIO                             # used to handle binary data   
from model import FontRecognizerModel              # to import the class from the model.py file

# creating a function for processing the image uploaded
def image_processor(input_image):

    # using the try and except block to catch any exceptions that may occur
    try:
        if isinstance(input_image, str): # image is fetched from url and loaded via Image.open if the input image is a string
            user_response = requests.get(input_image) 
            user_image = Image.open(BytesIO(user_response.content)) 

        else:  # image is loaded via Image.open if the input image is a file object
            user_image = Image.open(input_image)
            
        # image preprocessing,
        user_image = user_image.convert('RGB')            # image conversion to RGB  
        user_image = user_image.resize((224, 224))        # image resizing to 224x224 pixels
        user_image_array = np.array(user_image)           # image conversion to array
        return user_image_array

    except:          # error handling incase of exceptions, will return None if true
        return None

# creating a function to predict the font in the image
def img_font_prediction(input_image, input_url, model):
    # creating a condition to check if the input image is a url
    if input_url and input_url.strip():                          # using the .strip() method to remove any whitespace in the url
        input_image = input_url.strip()

    # processing the image by calling the image processor function
    processed_image = image_processor(input_image)

    # condition if the processed image is none, to display an error message to the user
    if processed_image is None:
        return "\n Unexpected Error: Please Try Again!"
    
    #
    return model.img_font_prediction(processed_image)

# creating a function to initialize the model
def main():
    # inializing the model
    model = FontRecognizerModel()

    # condition to check if the trained model exists
    if not os.path.exits('font_rec_model.h5'):
        print('\n Training the model...') 
        model.buildtrain_model('data') # path of the dataset file
        model.save_model()
        print('\n Model training completed!')

    else:
        print('\n Loading the trained model...')
        model.load_model()

if __name__ == '__main__':
    main()
