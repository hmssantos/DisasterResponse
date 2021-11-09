# DisasterResponse

# Project Overview
This project aims to develop an application that is able, using a Machine Learning model, to categorize received messages.
The repository contains the code used to train the model, as well as making it possible to receive new datasets (in the same structure as the dataset used).

# File Descriptions

**process_data.py:** This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data.

**train_classifier.py:** This code takes the SQLite database produced by process_data.py as an input and uses the data to train and tune a ML model for categorizing messages. 
The output is a pickle file containing the fitted model.

**data:** This folder contains sample messages and categories datasets in csv format.

**app:** This folder contains all of the files necessary to run and render the web app.
