import json
import plotly
import pandas as pd
<<<<<<< HEAD

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
=======
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
>>>>>>> 970826dbceee0da8c079c11b3e7620115c7ca9ec
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
<<<<<<< HEAD
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")
=======
engine = create_engine('sqlite:////Applications/workspace/data/disaster_messages.db')
df = pd.read_sql_table("disasterMessages", engine)

# load model
model = joblib.load('/Applications/workspace/models/model.pkl')
>>>>>>> 970826dbceee0da8c079c11b3e7620115c7ca9ec


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
<<<<<<< HEAD
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df1 = df.drop(['id','message','original','genre'], axis=1)
    category_counts=df1.sum(axis=0)
    category_names = df1.columns
    
=======
    # TODO: Below is an example - modify to extract data for your own visuals

    # Vis1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Vis2
    columns_count = df.astype(bool).sum(axis=0).iloc[4:] / len(df)
    columns = list(df.astype(bool).sum(axis=0).iloc[4:].index)

    # Vis3
    df['text length'] = df['message'].apply(lambda x: len(x.split()))
    histogram = df[df['text length'] < 100].groupby('text length').count()['id']

>>>>>>> 970826dbceee0da8c079c11b3e7620115c7ca9ec
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
<<<<<<< HEAD
                    x=category_names,
                    y=category_counts
=======
                    x=columns,
                    y=columns_count
>>>>>>> 970826dbceee0da8c079c11b3e7620115c7ca9ec
                )
            ],

            'layout': {
<<<<<<< HEAD
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
=======
                'title': 'Percentage of each Category in the Dataset',
                'yaxis': {
                    'title': "Percentage"
>>>>>>> 970826dbceee0da8c079c11b3e7620115c7ca9ec
                },
                'xaxis': {
                    'title': "Category"
                }
            }
<<<<<<< HEAD
=======
        },
        {
            'data': [
                Bar(
                    x=histogram.index,
                    y=histogram.values
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Length',
                'yaxis': {
                    'title': "Total Messages"
                },
                'xaxis': {
                    'title': "Message Length"
                }
            }
>>>>>>> 970826dbceee0da8c079c11b3e7620115c7ca9ec
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