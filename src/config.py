# Config file for back seat part - See pl_trainevaltestsave.py for more details
import os
MODE = 'Sonification EfficientNetB7'
DISPERSE_LOCATION = '/root/SonificationProject/Data/All_Data_off' #For old data
OLD_DIR = '/root/SonificationProject/Data/Added_Data_off'
NEW_DIR = '/root/SonificationProject/Data/Added_Data_off'
MODEL_DIR = '/root/SonificationProject/Experiments/EfficientNetB7_off_Models'
SAVE_DIR = '/root/SonificationProject/Experiments/EfficientNetB7_off_Added_Results'
MODEL_TYPE = "EfficientNetB7"
TRAIN = False
SAVE = True
REDISPERSE = False
REDISPERSE_FREQUENCY = 10
NUM_OF_MODELS = 1
BATCH_SIZE = 4
EPOCHS = 1000
PATIENCE = 10
lr = 0.01