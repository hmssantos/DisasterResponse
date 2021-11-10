# import libraries
import sys
import pandas as pd
from sqlalchemy import *

def load_data(messages_filepath, categories_filepath):
    '''
    Data Loading function
    Arguments:
        messages_filepath: path to messages csv file
        categories_filepath: path to categories csv file
    Output:
        df: Loaded DataFrame
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on=['id'])
    return df

def clean_data(df):
    '''
    Data Cleaning function
    Arguments:
        df: Uncleaned DataFrame
    Outputs:
        df: Cleaned DataFrame
    '''
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.partition("-")[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.partition("-")[2])
        categories[column] = categories[column].astype(int)
    del df['categories']
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    my_list = []
    for i in range(4,df.shape[1]):
      df = df[(df.iloc[:,i] == 1) | (df.iloc[:,i] == 0)]
      if df.iloc[:,i].nunique() == 1:
        my_list.append(df.columns[i])
    df = df.drop(columns=my_list)
    return df

def save_data(df, database_filename):
    '''
    Save Data
    Arguments:
        df: Clean DataFrame
        database_filename: Database destination
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('projectTable', engine, index=False, if_exists='replace')


def main():
    '''
    Main function executes:
        1) Data Loading
        2) Data Cleaning
        3) Data Saving
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
