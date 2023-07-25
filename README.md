# Sonification Project

This project makes spectrograms of data. Then, it uses an Image Classifier to train and classify those different spectrograms.

The spectrograms I used were from the dataset of .mat files here: <https://doi.ccs.ornl.gov/ui/doi/350>. I took
these .mat files and turned them into spectrograms using the code in the sonify folder. The spectrograms were then used to train and test an Image Classifier. The Image Classifier was trained and tested using the code in the src folder.

Later, I took different classes of spectrograms and added them together (sonify/add_spectrograms.py). This was done to see if the classifier could still classify the spectrograms.

This project was done as part of a 10-week internship at ORNL

## Folders

Analysis - Shows some plots
sonify - Turns .mat data into both spectrograms and wav files
src - Main code for the Image Classification
config - Config file for the Image Classification

## src and config

All of these files are modifications of my other repo: <https://github.com/Silverasdf/blurimageproject2023>

## sonify

These are all of the files used to sonify files.

add_spectrograms.py - Takes every permutation of 2 .mat files from a directory and combines them.
This requires each .mat file to have the same number of rows and columns in "data" and "samp_rate"

sonify_everything.py - Given a directory, sonifies each sample and puts it in a new directory

sonifyfn.py - The function to include that does the sonification. This outputs a .wav file and a .png file

## Analysis

view_added_data.ipynb - Checks how a classifier does in classifying two spectrograms that have been added together

perf_curves.ipynb - Plots the performance curves for the classifier

make_data.ipynb - Takes sonified data and puts it in a format that can be used by the classifier
