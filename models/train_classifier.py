from datetime import datetime, timedelta
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import *

import sys
import nltk
import numpy as np
import pandas as pd
import time
import re
import pickle
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    '''
    Data Loading
    Arguments:
        database_filepath: path to SQLite db
    Output:
        X: input feature
        Y: labels
        category_names: labels names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('projectTable', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenization
    Arguments:
        text: text messages
    Output:
        clean_tokens: tokenized and cleaned text
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return clean_tokens

def build_model():
    '''
    Building Model
    Output:
        cv: model architecture
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[10, 25],
                  'clf__estimator__min_samples_split':[2, 5, 10]}
    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the trained model and show classification report
    Arguments:
    model: trained model
    X_test: test data
    Y_test: test target
    category_names: labels
    '''
    # predict X_test
    Y_pred = model.predict(X_test)
    for i in range(0,len(Y_test.columns)):
        print('Informações sobre a variável: ',Y_test.columns[i],'\n')
        print(classification_report(Y_test.iloc[:,i], pd.DataFrame(Y_pred)[i]))

def save_model(model, model_filepath):
    '''
    Function that save the final model
    Arguments:
    model: result of the GridSearchCV
    model_filepath: file path of final model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
