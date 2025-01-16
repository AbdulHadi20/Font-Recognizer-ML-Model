"""
This is the main file for the GUI of the application. 

"""

###################### START OF THE PROGRAM ######################\

# importing all of the required modules and libraries
import os                                         # used for systeme file operations
import gradio as gr                                # used for creating the GUI

from main import FontRecognizerModel               # importing the FontRecognizerModel class from the main file


# initializing the Font Recognizer Model
fontModel = FontRecognizerModel('Font Dataset Large')

# creating a model to train the model on the dataset
def trainModel():

    # using try except block to handle any exceptions that may occur
    try: 
        classReport, accuracyScore = fontModel.modelTrain(epochs=10)     # training the model on the dataset
        fontModel.saveTrainedModel()                                           # saving the trained model

        # saving the results in a variable
        results = f'Training Completed Successfully! \n Accuracy Score: {accuracyScore:.2%} \n'
        results += f'Classification Report: \n {classReport}'

        return (results, 'Confusion-Matrix.png')
    
    except Exception as e:
        print(f'An Error Occurred: {e}')

# creating a model to predict the font of the uploaded image
def predFont(img):
    
    # using try except block to handle any exceptions that may occur
    try:
        
        # saving the uploaded image temporarily
        tempImg = 'tempImg.jpg'
        img.save(tempImg)


        # making predictions on the uploaded image
        fontName, fontConfidence = fontModel.predict(tempImg)

        # removing the temporary image
        os.remove(tempImg)

        return f'The font of the uploaded image is: {fontName} with a confidence of: {fontConfidence:.2f}%'
    
    except Exception as e:
        return f'An Error Occurred: {e}'
    

###################### CREATING THE GUI ######################

# creating the interface for the model
with gr.Blocks(title='Font Recognizer Model') as modelInterface:
    gr.Markdown(' # Font Recognizer Model')

    with gr.Tab('Train Model'):
        trainBtn = gr.Button('Train Model')
        outputTxt = gr.Textbox(label='Training Results')
        conMatrix = gr.Image(label='Confusion Matrix')


        trainBtn.click(trainModel, outputs= [outputTxt, conMatrix])

    with gr.Tab('Predict Font'):

        with gr.Row():
            imageInput = gr.Image(type='pil', label='Upload Image')
            outputTxt = gr.Textbox(label='Prediction Results')

        predBtn = gr.Button('Predict Font')
        predBtn.click(predFont, inputs=[imageInput], outputs=[outputTxt])


if __name__ == '__main__':
    modelInterface.launch()
