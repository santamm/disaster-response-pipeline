# disaster-response-pipeline
Disaster Response Pipeline based on Figure8 messages

### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Analysis](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code in this project is written in Python 3.6.6 :: Anaconda custom (64-bit).
The following additional libraries have been used:
* pandas
* numpy
* matplotlib
* sqlalchemy
* pickle
* nltk
* re
* sklearn
* Flask (for the webapp only)
* plotly (for the webapp only)
* wordcloud (for the webapp only)

## Usage  <a name="usage"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database, from the data directory:
    
        usage: process_data.py messages_filepath categories_filepath database_filepath
        
        example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
        
    - To run ML pipeline that trains classifier and saves the model into a pickle file
    
        usage: train_classifier.py [-h] [--gridsearch] [--no-gridsearch]
                           [--alternative] [--no-alternative]
                           database_filepath model_filepath
                           
        where --gridsearch activates a gridsearchCV for optimal hyperparameters
        and --alternative loads an alternative model (LinearSVC) with a custom Transformer
        
        Please use this parameters of you want to test the webapp too:
        
        python train_classifier.py ../data/DisasterResponse.db classifier.pkl

2. Webapplication: run the following command in the app's directory to run your web app.

    `python run.py`
    and go to http://0.0.0.0:3001/
    Otherwise you can access the webapp deployed on heroku [here](https://disaster-response-webapp.herokuapp.com/)

## Project Analysis<a name="motivation"></a>
In this project we analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. We created machine learning pipeline to classify these events so that messages can be sent to an appropriate disaster relief agency. The text messages in the training dataset are classified into 36 categories, as shown ![below](labels.png)

We can see that the distribution of the categories shows a imbalanced dataset, as some categories like hospitals and even water have very few examples. This can be a problem as standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. Then  while working in an imbalanced domain accuracy is not an appropriate measure to evaluate model performance, we should probably look more to the weighted average of the F1 score.

A basic model has been generated using a RandomForestClassifier that gave us an overall accuracy of 94.19% and a weighted F1 score of 0.93; however when launching the train_classifier.py you can choose to pass a --gridsearch parameter that will activate a grisearchCV to look for best hyperparameters to train the model on. Please be aware that this can be very slow. As it was running for more tha 12 hours on a dual core CPU, in the Juyter notebook you will can also run a RandomizedSearchCV on a sample dataset (for example 2000 datapoints), that will run in few minutes.

Also it is possible to pass the -- alternative parameter, that will build an alternative model with a LinearSVC classifier and a custom Tranformer that adds an additional feature. We check if the sentence include verbs asking for supplies like "we need water",
 or "food required", etc., starting from a list of lemmas and genarating all synonyms. The list of synonyms is passed as an initialization parameter for the class. The new feature is True if any words in the list of forms is part of the text, False otherwise. With this alternative model we achieved 95.05% overall accuracy and weighted F1 score of 0.9435.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data and a wordmap generated from the text messages receoved.

## File Descriptions <a name="files"></a>
The Jupyter notebooks included in this project are:
- ETL Pipeline Preparation.ipynb
- ML Pipeline Preparation.ipynb

Python Files:
- process_data.py:      load, clean, and save data to a sqllite db
- train_classifier.py:  train and test classifier
- train_utils.py:       utilities functions for training
- need_extractor.py:    custom Transformer
- run.py:               start webapp

Data files (under data directory):
- disaster_categories.csv       categories to classify nessages to
- message-categories.csv        messages received
- requirements.txt              list of python libraries to install to run webapp


## Results<a name="results"></a>
The webapp has been deployed on Heroku and can be accessed [here](https://disaster-response-webapp.herokuapp.com/)
The webapps shows three visualizations of the training dataset:
- a wordmap of the text messages using a png mask of Haiti
- a barchart of the counts of the messages by genre
- a barchart of the counts of the message categories

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
For licensing see LICENSE file.
