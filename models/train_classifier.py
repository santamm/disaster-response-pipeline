import sys
import argparse
import numpy as np
import pandas as pd

import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from need_extractor import NeedExtractor
from shared_utils import tokenize, load_data, generate_forms, save_model

def build_model(gridsearch=False):
    """
    Build and train pipeline model to classify text sentences. Apply BOW and
    TfidfTransformer. Classify using a RandomForestClassifier. As we have 36
    categories as target this is actually a Multi target classification, where
    we fit one classifier per target.

    Args:
        gridsearch (boolean, default False): whether to apply gridsearch to
        optimize the model
    Returns: pipeline: trained model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multiclf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    if gridsearch == True:
        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000),
            'tfidf__use_idf': (True, False),
            'multiclf__estimator__n_estimators': [50, 100, 200],
            'multiclf__estimator__min_samples_split': [2, 3, 4]
            }
            # create grid search object
        print("Looking for best paramaters with GridSearch, please patient.....")
        cv = GridSearchCV(pipeline, parameters)
        return cv
    else:
        return pipeline



def build_alt_model():
    """
    Build an alternative model with a different classifier and a custom Transformer
    The alternative model uses a Linear Support Vector Machine. Since the LinearSVC
    does not imoplement a predict_proba method we cannot pass it directly to the
    Multioutputclassifier, but we must encapsulate it with a OneVsRestClassifier,
    that fits a classifier against all other classes
    The custom transformer defined in needextractor.py searches the text for lemmas
    generated as synonyms of a given list of words that express a peculiar need

    Returns: (pipeline) model

    """
    lemmas = ['need', 'require', 'urgent', 'scarce', 'food', 'water']
    forms = generate_forms(lemmas)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text-pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('needs', NeedExtractor(forms))
        ])),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    return pipeline


def print_results(y_true, y_pred):
    """
    Prints a mean of the classification report of all 36 classifiers predictions
    Args:
        y_true (numpy array of size (,36)): groundtruth values array
        y_pred (numpy array of size (,36)): predicted labels


    Returns:

    """
    precisions = []
    recalls = []
    fscores = []
    for label in range(y_pred.shape[1]):
        precision, recall, fscore, _ = score(y_true[:, label], y_pred[:, label], average='weighted')
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    print('Precision : {}'.format(np.mean(precisions)))
    print('Recall    : {}'.format(np.mean(recalls)))
    print('F-score   : {}'.format(np.mean(fscores)))
    print("Accuracy  : {}".format((y_true == y_pred).mean()) )


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate model and prints a classification report for each classifier,
    so overall it prints 36 classification reports. Then prints overall
    means of all reports

    Args:
        model: trained classifier model
        X_test: test dataset predictors (numpy array of text sentences)
        y_test: test values (numpy array of size (,36))
        category_names (nump§y array of size (1,36)): category (labels) names
    """
    y_pred = model.predict(X_test)
    for label in range(y_pred.shape[1]):
        print(classification_report(y_test[:,label], y_pred[:,label], labels=[0,1],
                target_names=[category_names[label]+'-0', category_names[label]+'-1']))
    print("Overall statistics for the model:")
    print_results(y_test, y_pred)





def main():
    """
    Main loop: parse command line arguments
    usage: train_classifier.py [-h] [--gridsearch] [--no-gridsearch]
                           database_filepath model_filepath

    Load data from saved sqllite database, build and train model, evaluate against
    test dataset, print classification reports, save model on a pickle file
    Args:
        database_filepath: filepath to load database from
        model_filepath: filepath to save model to
        --gridsearch :  if specified, performs a gridsearch on hyperparameters
        to optimizize model
        --alternative: if specified, train classifier on alternative model

    """
    parser = argparse.ArgumentParser(
    description='Optional model optimization with GridSearchCV',
    )
    parser.add_argument('database_filepath', action="store")
    parser.add_argument('model_filepath', action="store")
    parser.add_argument('--gridsearch', dest='gridsearch', action='store_true')
    parser.add_argument('--no-gridsearch', dest='gridsearch', action='store_false')
    parser.set_defaults(gridsearch=False)
    parser.add_argument('--alternative', dest='altmodel', action='store_true')
    parser.add_argument('--no-alternative', dest='altmodel', action='store_false')
    parser.set_defaults(alternative=False)
    #parser.add_argument('--gridsearch', action="store_true", dest="gridsearch", default=False)
    try:
        args = parser.parse_args()
    except:
        return
    database_filepath = args.database_filepath
    model_filepath = args.model_filepath
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, y, category_names = load_data(database_filepath)

    # split into training and test dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    if args.altmodel==True:
        print("Building alternative model with custom Transformer and SVC classifier...")
        model = build_alt_model()
    else:
        print('Building default model...')
        model = build_model(args.gridsearch)

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

    """
    print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
    """

if __name__ == '__main__':
    main()
