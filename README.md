# disaster-response-pipeline
Disaster Response Pipeline based on Figure8 messages

### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code in this project is written in Python 3.6.6 :: Anaconda custom (64-bit).
The following additional libraries have been used:
* sqlalchemy
* picke
* nltk
* re
* sklearn


## Usage  <a name="usage"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database, from the data directory:
        usage: process_data.py messages_fiepath categories_filepath database_filepath
    - To run ML pipeline that trains classifier and saves the model into a pickle file
        usage: train_classifier.py [-h] [--gridsearch] [--no-gridsearch]
                           [--alternative] [--no-alternative]
                           database_filepath model_filepath
        where --gridsearch activates a gridsearchCV for optimal hyoerparameters
        and --alternative loads an alternative model (LinearSVC) with a custom Transformer

2. Webapplication: Rrun the following command in the app's directory to run your web app.
    `python run.py`
    and go to http://0.0.0.0:3001/
    

## Project Motivation<a name="motivation"></a>
In this project we analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. We created machine learning pipeline to categorize these events so that messages can be sent to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

The data has been made available by ...., and the original source can be found [here](...)

## File Descriptions <a name="files"></a>
The Jupyter notebooks included in this project are:

- Python Files:

- webapp files...



## Results<a name="results"></a>
The following results are showed in the notebooks:
- 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
For licensing see LICENSE file.
