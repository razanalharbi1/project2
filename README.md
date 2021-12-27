# Disaster Response Pipeline Project
- Dependencies needed. Use pip install
- Machine Learning Libraries: Numpy, Pandas, Sklearn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly

#About the Project
- This project is about analyzing message data for disaster response. The data gotten from Figure Eight is used to build a model that classifies disaster -  messages and web app where an respondent can input a new message and get classification results in several categories

#Files structure

- data: folder contains sample messages and categories datasets in csv format

|-  disaster_categories.csv # data to process
|-  disaster_messages.csv # data to process
|-  process_data.py # python code takes as input csv files(message data and message categories datasets), clean it, and then creates a SQL database
|-  disasterMessagesDatabase.db # database to save clean data to

- app: contains the run.py to deploy the web app.
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app

- models
|- train_classifier.py # code to train the ML model with the SQL data base
|- Classifier.pkl # saved model
|- ETLPipelinePreparation.ipynb: process_data.py development process
|- MLPipelinePreparation.ipynb: train_classifier.py development process
- README.md

### Instructions:
- Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv  data/DisasterResponse.db

- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

- Run the following command in the app's directory to run your web app. python run.py

- Go to http://0.0.0.0:3001/
