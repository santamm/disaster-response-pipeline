import json
import plotly
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string



app = Flask(__name__)


def tokenize(text):
    """
    Tokenize a sentence into lemmas, convert to lowercase and remove excess spaces
    Args:
        text (string): sentence to tokenize
    Returns:
        clean_tokens: list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

"""
Loads a model named "classifier.pkl" from the "models" directory
Load a database "DisasterResponse.db" from the "data" directory
Genetates a wordmap of the text in the training data and saves it
in a file named "haiti-mask.png" in the "static" directory 

"""


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Response', engine)
X = df.message.values
y = df.iloc[:,4:].values
categories = df.columns[4:].values


# load model
model = joblib.load("../models/classifier.pkl")



def generate_wordcloud(messages, mask_file=False):
    """
    generate a wordcloud object from the messages dataset

    Args:
        messages (numpy array of text): datasst of all text messages
        mask_file (string, optional): filepath of a mask file to be used as a base
            for the wordmap
    Returns:
        wordmap object

    """
    # drop rows without reviews
    text = " ".join(msg for msg in messages)
    # remove punctation
    nopunc = [char for char in text if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # remove stopwords
    stopwords = set(STOPWORDS)
    # generating wordcloud
    if mask_file != False:
        mask = np.array(Image.open(mask_file))
        wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white",
                     mask=mask, margin=3, scale=2).generate(nopunc)
    else:
        wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white",
                          margin=3, scale=2).generate(nopunc)

    return wordcloud

def return_graphs():
    """
    Generated visual representations of the data:
        - a Bar of the genres from the 'genres' columns
        - a bar of the categories of the messages, from the 36 category columns
        generated while loading the data
    Args:
    Returns:
        graphs (list): list of dictionaries (data and layout) of the Plotly
            graphs to be visualized

    """
    genres = df.genre.value_counts()
    graph_one = []
    graph_one.append(
      Bar(
      x = list(genres.index),
      y = genres.values,
      name='genres'
      ))

    layout_one = dict(title = 'Message Genres',
                xaxis = dict(title = 'Genre'),
                yaxis = dict(title = 'Count'),
                width=300, height=300
                )

    graph_two = []
    graph_two.append(
      Bar(
      x = categories,
      y = y.sum(axis=0),
      name='categs'
      ))

    layout_two = dict(title = 'Message Categories',
                xaxis = dict(title = 'Categories',),
                yaxis = dict(title = 'Count'),
                width=600, height=300
                )

    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    #figures.append(dict(data=graph_three, layout=layout_three))
    return graphs


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)
    """
    - Generate visualization of the home page: plotly data visualizations and wordmap
    - Save wordmap on fiesystem as a static png image
    - encode plotly graphs using JSON

    Returns: home page template with visualizations

    """

    # create visuals

    graphs = return_graphs()
    # generate wordcloud
    wc = generate_wordcloud(X, "static/haiti-mask.png")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("static/haiti-wp.png")


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Visualize classification if the user query: shows only classes that produced
    a prediction == 1

    Returns:
        page template for the query reply page

    """
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
    """
    Main app
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
