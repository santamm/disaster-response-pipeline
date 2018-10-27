from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import pickle

def tokenize(text, keep_stopwords=True):
    """
    Tokenization function to process data
    remove urls, special symbols, stopwords, lemmatize tokens, convert all to
    lowercase and remove excess whitespaces
    Args:
        text (numpy array):  text sentences
        keep_stopwords (boolean): whether to keep or remove stopwords
    Returns:
        clean_tokens (list): list of tokens
    """
    # Replace urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text))
    # remove stopwords
    if keep_stopwords == False:
        tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        #clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok = lemmatizer.lemmatize(tok.lower().strip(), pos='v')
        clean_tokens.append(clean_tok)

    return clean_tokens

def generate_forms(lemmas):
    """
    Generates a list of synonyms from a list of words and return all in same list
    Args:
        lemmas (list): list of lemmas
    Returns:
        list of all synonyms of the given lemmas
    """
    forms = []
    for lemma in lemmas:
        synonyms = wordnet.synsets(lemma)
        forms = forms + [word.lemma_names() for word in synonyms]
        flattened_forms = [item for sublist in forms for item in sublist]

    return list(set(flattened_forms))

def load_data(database_filepath):
    """
    Load data from sqllite database into a pandas dataframe
    Args:
            database_filepath (string): filepath of the sqllit database
    Returns:
            X: np array of predictors (text sentences)
            y: np array of labels (, 36)
            column names of 36 the categories
    """
    table = 'Response'
    database = "sqlite:///" + database_filepath
    engine = create_engine(database)
    df = pd.read_sql_table(table, engine)
    X = df.message.values
    y = df.iloc[:,4:].values
    return X, y, df.columns[4:].values

def save_model(model, model_filepath):
    """
    Dumps model into pickle file
    Args:
        model (model object): model to save
        model_filepath (string): filepath to save model to
    """
    pickle.dump(model, open(model_filepath, 'wb'))
