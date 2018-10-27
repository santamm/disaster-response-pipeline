import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from shared_utils import tokenize
from nltk.tokenize import sent_tokenize


class NeedExtractor(BaseEstimator, TransformerMixin):
    """
    Custom tranformer to implement an alternative model with an additional feature
    We check if the sentence include verbs asking for supplies like "we need water",
    or "food required", etc., starting from a list of lemmas and genarating all
    synonyms. The list of synonyms is passed as an initialization parameter for
    the class. The new feature is True if any words in the list of forms is part
    of the text, False otherwise


    """

    def __init__(self, listofforms):
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self.forms = listofforms


    def is_need(self, text):
        """
        Checks if there are tokens included in the list of forms
        """
        # tokenize by sentences
        sentence_list = sent_tokenize(text)

        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            tokens = tokenize(sentence, keep_stopwords=True)
            if any({*self.forms} & {*tokens}):
                return True

        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        Apply is_need function to all values in X
        """

        X_tagged = pd.Series(X).apply(lambda x : self.is_need(x)).values

        return pd.DataFrame(X_tagged)
