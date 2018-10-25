import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
