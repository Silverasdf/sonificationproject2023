# Config file for front seat part - See pl_trainevaltestsave.py for more details

import os
#Mode - used when naming the perf curve png files and on the json files
MODE = 'Sonification EfficientNetB7'

#Location of the ground truth images - used for redisperse
DISPERSE_LOCATION = '/root/SonificationProject/Data/All_Data_off' #For old data

#Training Data - must have subdirectories for Training, Validation, and Testing, The testing subdirectory can just have dummy images, since it isn't meant to be used
OLD_DIR = '/root/SonificationProject/Data/Added_Data_off'

#Testing Data - must have subdirectories for Training, Validation, and Testing. The training and validation subdirectories can just have dummy images, since they aren't meant to be used
NEW_DIR = '/root/SonificationProject/Data/Added_Data_off'

#Where models are both saved and loaded from
MODEL_DIR = '/root/SonificationProject/Experiments/EfficientNetB7_off_Models'

#Where results are saved - these are saved into JSON files and also saved as png files
SAVE_DIR = '/root/SonificationProject/Experiments/EfficientNetB7_off_Added_Results'

#Model type - must be one of the following: "ResNet18", "EfficientNetB7", "ViT"
MODEL_TYPE = 'EfficientNetB7'

#Flag for whether to train the model - set to false to skip directly to testing
TRAIN = False

#Flag for whether to save the model - set to false to not save the model nor the results
SAVE = True

#Flag for whether to redisperse the data - set to false to skip redisperse
REDISPERSE = False
REDISPERSE_FREQUENCY = 10 #After how many models does the data redisperse? Can be ignored if REDISPERSE is set to false

#Number of models to train/test (depending on whether the TRAIN flag is set to true or false)
NUM_OF_MODELS = 1

#Hyperparameters
BATCH_SIZE = 4
EPOCHS = 1000
PATIENCE = 10
lr = 0.01