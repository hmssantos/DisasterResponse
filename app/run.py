import json
import plotly
import pandas as pd
import re
import nltk
import string
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('projectTable', engine)

# load model
model = joblib.load("models/classifier.pkl")

# extract data needed for visuals
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

df_categories = df.drop(['id'], axis=1)._get_numeric_data()
top_categories_pcts1 = df_categories.sum().sort_values(ascending=False).head(10)
top_categories_names1 = list(top_categories_pcts1.index)

top_categories_pcts2 = df_categories.sum().sort_values(ascending=True).head(10)
top_categories_names2 = list(top_categories_pcts2.index)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_categories_names1,
                    y=top_categories_pcts1
                )
            ],

            'layout': {
                'title': 'Top 10 most recidivist categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_categories_names2,
                    y=top_categories_pcts2
                )
            ],

            'layout': {
                'title': 'Top 10 less recidivist categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
