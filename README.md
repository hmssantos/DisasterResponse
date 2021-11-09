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

# Instructions

1. On your terminal, run the installation of the packages you will use throughout the project:

python -m pip install pandas sqlalchemy nltk sklearn plotly flask

2. Run the following commands in the project's root directory to set up your database and model.

     - To run ETL pipeline that cleans data and stores in database
         `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
     - To run ML pipeline that trains classifier and saves
         `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
     `python app/run.py`

4. Go to http:localhost:3001

# Heads up

The web app will only be available (http:localhost:3001) while your terminal is open.

If you close it, the web app will "crash".

# Sample access

When accessing the web app, you should be able to see this screen:

![image](https://user-images.githubusercontent.com/91185275/140841761-3d82145b-bd91-45f3-a1a7-3191df1c9489.png)

To check how the machine learning algorithm classifies a particular message, just type and click on the button.

![image](https://user-images.githubusercontent.com/91185275/140842074-7a6a6d55-a023-45f9-8d5e-c5d1be0a5000.png)

