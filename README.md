# DisasterResponse

# Project Overview

This project aims to develop an application that is able, using a Machine Learning model, to categorize received messages.

During a disaster, the algorithm can help companies and individuals in directing a series of messages to the responsible agencies, enabling an efficient targeting and therefore faster and more accurate actions.

The repository contains the code used to train the model, in addition to making it possible to receive new datasets (in the same structure as the dataset used).

# File Descriptions

**DisasterResponse/app/templates/go.html:** Classification result page of web app.

**DisasterResponse/app/templates/master.html:** Main page of web app.

**DisasterResponse/app/run.py:** Flask file that runs app.

**DisasterResponse/data/disaster_categories.csv:** Categories datasets in csv format.

**DisasterResponse/data/disaster_messages.csv:** Categories datasets in csv format.

**DisasterResponse/data/process_data.py:** ETL process. Here, I read the dataset, clean the data, and then store it in a SQLite database. 

**DisasterResponse/data/DisasterResponse.db:** Database to save clean data to.

**DisasterResponse/models/train_classifier.py:** Machine learning portion. I split the data into a training set and a test set. Then, I create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, I export the model to a pickle file.

**DisasterResponse/models/classifier.pkl:** Saved model.

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

![image](https://user-images.githubusercontent.com/91185275/141108322-cf5f3210-720a-4959-b6a5-684adcf447d8.png)

![image](https://user-images.githubusercontent.com/91185275/141108366-c4f9d2a3-2a3a-4d42-bd65-9407cab13f86.png)

To check how the machine learning algorithm classifies a particular message, just type and click on the button.

![image](https://user-images.githubusercontent.com/91185275/140842074-7a6a6d55-a023-45f9-8d5e-c5d1be0a5000.png)

