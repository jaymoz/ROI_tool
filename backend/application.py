''' Functions - Import CSV API, Preprocessing Textual Data Api, ML Model Selection API, '''

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
import os
import datetime
import threading
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics.cluster import entropy
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE 
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict
from imblearn.over_sampling import ADASYN
from sklearn.naive_bayes import GaussianNB
from scipy.stats import sem
from numpy import mean
from numpy import std
import numpy as np
import time
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string, re
import pickle


application = Flask(__name__, static_folder='static')
CORS(application)

'''Global variables'''
df_train = None
df_test = None
training_size = None
trim_rows = None
df_train_trimmed = None
selected_columns = None
df_filtered_trimmed = None
text_columns = []
f1_score_lg = []
f1_score_nb = []
f1_score_rf = []
f1_score_svc = []
f1_score_dt = []

recall_score_lg = []
recall_score_nb = []
recall_score_rf = []
recall_score_svc = []
recall_score_dt = []

precision_score_lg = []
precision_score_nb = []
precision_score_rf = []
precision_score_svc = []
precision_score_dt = []
global f1score

'''ROI Variables'''
global fp_cost
global fn_cost
global tp_cost
global resources_cost
global preprocessing_cost
global product_value
global tp_lg
global fn_lg
global fp_lg
global tp_nb
global fn_nb
global fp_nb
global tp_rf
global fn_rf
global fp_rf
global tp_svc
global fn_svc
global fp_svc
global tp_dt
global fn_dt
global fp_dt

fp_lg_list = []
fn_lg_list = []
tp_lg_list = []

fp_nb_list = []
fn_nb_list = []
tp_nb_list = []

fp_rf_list = []
fn_rf_list = []
tp_rf_list = []

fp_svc_list = []
fn_svc_list = []
tp_svc_list = []

fp_dt_list = []
fn_dt_list = []
tp_dt_list = []

roi_lg = []
roi_nb = []
roi_rf = []
roi_svc = []
roi_dt = []

'''Active Learning'''
# lables are converted as 
# {0: 'incorporates', 1: 'independent', 2: 'relates to'}
iteration = 0
args_dict = {}
args1 = []
splitratio = 0
maxIterations = 0
resamplingTechnique = ""
logFilePath = ""
df_rqmts = None
df_resultTracker = None
df_training = pd.DataFrame()
df_testing = pd.DataFrame()
df_LM_training = pd.DataFrame()
df_LM_testing = pd.DataFrame()
path = os.getcwd()+"/static/"
fields = ['summary1','summary2','dependency','id1', 'id2','label']
fullFile = path+"data.csv"
depType =  {'incorporates':0, 'independent':1, 'relates to':2}
label = 'label'
req1 = 'summary1'
req2 = 'summary2'
req1Id = 'id1'
req2Id = 'id2'
annStatus = 'AnnotationStatus'
classifierx = ''
projectName = "Solr"

'''Preprocessing functions'''
def preprocess_text(text):
  if isinstance(text, str):

    """Remove URLs from a sample string"""
    text =  re.sub(r"http\S+", "", text)
    text = re.sub(r"[\[\],@\'?\.$%_:()\-\"&;<>{}|+!*#]", " ", text, flags=re.I)
    text = ' '.join(w for w in text.split() if not any(x.isdigit() for x in w)) 
    text = text.lower()
    
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = ' '.join([lemmatizer.lemmatize(w) for w in text])
    
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    text = text.split() 
    text = [t for t in text if not t.isdigit()] 
    text = " ".join(text) 

    '''Tokenize the text'''
    tokens = word_tokenize(text.lower())

    # '''Remove punctuation'''
    tokens = [token for token in tokens if token not in string.punctuation]

    # '''Remove extra white spaces'''
    tokens = [token.strip() for token in tokens if token.strip()]


    return text
  else:
        return str(text)



def delete(file_paths, delay):
    """
    Delete the specified image files after a certain delay.
    """
    def delete_files():
        time.sleep(delay)
        for file_path in file_paths:
            os.remove(file_path)

    deletion_thread = threading.Thread(target=delete_files)
    deletion_thread.start()

    
''' Function - Handles the upload of the train data CSV file
        Pass In: train data (.csv)
        Pass Out: train data, graph, column and rows count, features
    Endfunction '''
@application.route('/upload/train_data', methods=['POST'])
def upload_train_data():
    global df_train
    global training_size
    file = request.files['file']
    training_size = float(request.form.get('training_size'))
    if not file:
        return jsonify({'error': 'No file provided'})

    try:
        df_train = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        csv_path = os.path.join(application.static_folder, 'data.csv')
        df_train.to_csv(csv_path, index=False)
        rows, columns = df_train.shape
        
        return jsonify({'success': True,
                        'df_train':df_train.to_dict(), 
                        'rows': rows, 
                        'columns': columns,
                        'training_size': training_size
                        })
    except Exception as e:
        return jsonify({'error': str(e)}) 
    

''' Function - Handles the upload of the test data CSV file
        Pass In: test data (.csv)
        Pass Out: test data, graph, column and rows count, features
    Endfunction '''  
@application.route('/upload/test_data', methods=['POST'])
def upload_test_data():
    global df_test
    file1 = request.files['file']
    if not file1:
        return jsonify({'error': 'No file provided'})

    try:
        df_test = pd.read_csv(io.StringIO(file1.read().decode('utf-8')))
        rows, columns = df_test.shape
        return jsonify({'success': True, 
                        'rows': rows, 
                        'columns': columns, 
                        })
    except Exception as e:
        return jsonify({'error': str(e)})


''' Function -  serves static files from the server's static folder.
        Pass In: any data
        Pass Out: new data added in static folder with path
    Endfunction '''
@application.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(application.static_folder, path)


''' Function -  filters columns based on user input in Multiselection Bar
        Pass In: required columns
        Pass Out: selected columns array
    Endfunction '''
@application.route('/filter_columns', methods=['POST'])
def filter_columns():
    global df_train
    global selected_columns
    global df_test

    try:
        selected_columns = request.json['columns']
        selected_columns = [column for column in selected_columns]
        df_train = df_train[selected_columns]
        return jsonify({'success': True, 'selected_columns': selected_columns})

    except Exception as e:
        return jsonify({'error': str(e)})


''' Function - trims data as per training size
        Pass In: dataframe
        Pass Out: preprocessed and trimmed dataframe
    Endfunction '''
@application.route('/trim_data', methods=['POST'])
def trim_data():
    global df_train
    global trim_rows
    global df_train_trimmed
    global training_size
    global selected_columns
    global df_filtered_trimmed

    rows, columns = df_train.shape
    trim_rows = int(training_size * rows)
    df_train['req1'] = df_train['req1'].apply(preprocess_text)
    df_train['req2'] = df_train['req2'].apply(preprocess_text)

    df_train_trimmed = df_train[:trim_rows]
    df_filtered_trimmed = df_train_trimmed[selected_columns]

    for column in df_filtered_trimmed.columns:
        if df_filtered_trimmed[column].dtype == object or isinstance(df_filtered_trimmed[column].dtype, pd.StringDtype):
            text_columns.append(column)
    return jsonify({'success': True,
                    'df_filtered_trimmed' : df_filtered_trimmed, 'text_columns': text_columns
                     })


# ML models - STEP 2
''' Following pseudocode remains same for all ML models
    Function - Logistic Regression / Naive Bayes / Support Vector Machine / Random Forest / Decision Tree
            Pass In: dataframe (both train + test)
            Pass Out: accuracy, graphs, f1 scores, recall scores, precision score, predicted values in JSON
    Endfunction '''


@application.route('/logistic-regression', methods=['POST'])
def perform_logistic_regression():
    global df_test
    global df_filtered_trimmed
    global f1_score_lg
    global recall_score_lg
    global precision_score_lg
    recall_score_lg.clear()
    precision_score_lg.clear()
    f1_score_lg.clear()
    global tp_lg
    global fn_lg
    global fp_lg
    global tp_lg_list
    global fn_lg_list
    global fp_lg_list
    tp_lg_list.clear()
    fn_lg_list.clear()
    fp_lg_list.clear()
    

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('logistic_regression.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)

        # Predict on the test set
        y_pred = model.predict(X_test)
        
        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test,labels=[0,1])
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            cm=confusion_matrix(y_pred,y_test,labels=[0,1])
            tn, fp_lg, fn_lg, tp_lg = cm.ravel()
            fp_lg_list.append(int(fp_lg))
            fn_lg_list.append(int(fn_lg))
            tp_lg_list.append(int(tp_lg))
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_lg.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_lg.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_lg.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(application.static_folder,'lg_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(4.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        lg_f1_score = os.path.join(application.static_folder, 'lg.png')
        plt.show()
        plt.savefig(lg_f1_score)
        plt.close()

        image_files_to_delete = [cm, lg_f1_score]
        delete(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/lg.png','cm':'/static/lg_cm.png','f1':f1,'f1score':f1_score_lg,'recallscore':recall_score_lg,'precisionscore':precision_score_lg,'tp':tp_lg_list,'fp':fp_lg_list,'fn':fn_lg_list})
    
    except Exception as e:
        return jsonify({'error': str(e)})

    

@application.route('/naive-bayes', methods=['POST'])
def perform_naive_bayes():
    global df_test
    global df_filtered_trimmed
    global f1_score_nb
    global recall_score_nb
    global precision_score_nb
    recall_score_nb.clear()
    precision_score_nb.clear()
    f1_score_nb.clear()
    global tp_nb
    global fn_nb
    global fp_nb
    global tp_nb_list
    global fn_nb_list
    global fp_nb_list
    tp_nb_list.clear()
    fn_nb_list.clear()
    fp_nb_list.clear()

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('naive_bayes.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)
        y_pred = model.predict(X_test)

        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test,labels=[0,1])
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            cm=confusion_matrix(y_pred,y_test,labels=[0,1])
            tn, fp_nb, fn_nb, tp_nb = cm.ravel()
            fp_nb_list.append(int(fp_nb))
            fn_nb_list.append(int(fn_nb))
            tp_nb_list.append(int(tp_nb))
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_nb.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_nb.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_nb.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(application.static_folder,'nb_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(4.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        nb_f1_score = os.path.join(application.static_folder, 'nb.png')
        plt.show()
        plt.savefig(nb_f1_score)
        plt.close()

        image_files_to_delete = [cm, nb_f1_score]
        delete(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/nb.png','cm':'/static/nb_cm.png','f1':f1,'f1score':f1_score_nb,'recallscore':recall_score_nb,'precisionscore':precision_score_nb,'tp':tp_nb_list,'fp':fp_nb_list,'fn':fn_nb_list})
    
    except Exception as e:
        return jsonify({'error': str(e)})

    

@application.route('/random-forest', methods=['POST'])
def perform_random_forest():
    global df_test
    global df_filtered_trimmed
    global f1_score_rf
    global recall_score_rf
    global precision_score_rf
    recall_score_rf.clear()
    precision_score_rf.clear()
    f1_score_rf.clear()
    global tp_rf
    global fn_rf
    global fp_rf
    global tp_rf_list
    global fn_rf_list
    global fp_rf_list
    tp_rf_list.clear()
    fn_rf_list.clear()
    fp_rf_list.clear()

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('random_forest.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)
        y_pred = model.predict(X_test)

        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test,labels=[0,1])
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            cm=confusion_matrix(y_pred,y_test,labels=[0,1])
            tn, fp_rf, fn_rf, tp_rf = cm.ravel()
            fp_rf_list.append(int(fp_rf))
            fn_rf_list.append(int(fn_rf))
            tp_rf_list.append(int(tp_rf))
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_rf.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_rf.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_rf.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(application.static_folder,'rf_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(4.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        rf_f1_score = os.path.join(application.static_folder, 'rf.png')
        plt.show()
        plt.savefig(rf_f1_score)
        plt.close()

        image_files_to_delete = [cm, rf_f1_score]
        delete(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/rf.png','cm':'/static/rf_cm.png','f1':f1,'f1score':f1_score_rf,'recallscore':recall_score_rf,'precisionscore':precision_score_rf,'tp':tp_rf_list,'fp':fp_rf_list,'fn':fn_rf_list})
    
    except Exception as e:
        return jsonify({'error': str(e)})



@application.route('/support-vector-machine', methods=['POST'])
def perform_support_vector_machine():
    global df_test
    global df_filtered_trimmed
    global f1_score_svc
    global recall_score_svc
    global precision_score_svc
    recall_score_svc.clear()
    precision_score_svc.clear()
    f1_score_svc.clear()
    global tp_svc
    global fn_svc
    global fp_svc
    global tp_svc_list
    global fn_svc_list
    global fp_svc_list
    tp_svc_list.clear()
    fn_svc_list.clear()
    fp_svc_list.clear()

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('svc.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)
        y_pred = model.predict(X_test)

        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test,labels=[0,1])
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            cm=confusion_matrix(y_pred,y_test,labels=[0,1])
            tn, fp_svc, fn_svc, tp_svc = cm.ravel()
            fp_svc_list.append(int(fp_svc))
            fn_svc_list.append(int(fn_svc))
            tp_svc_list.append(int(tp_svc))
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_svc.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_svc.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_svc.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(application.static_folder,'svc_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(4.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        svc_f1_score = os.path.join(application.static_folder, 'svc.png')
        plt.show()
        plt.savefig(svc_f1_score)
        plt.close()

        image_files_to_delete = [cm, svc_f1_score]
        delete(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/svc.png','cm':'/static/svc_cm.png','f1':f1,'f1score':f1_score_svc, 'recallscore':recall_score_svc,'precisionscore':precision_score_svc,'tp':tp_svc_list,'fp':fp_svc_list,'fn':fn_svc_list})
    
    except Exception as e:
        return jsonify({'error': str(e)})

   

@application.route('/decision-tree', methods=['POST'])
def perform_decision_tree():
    global df_test
    global df_filtered_trimmed
    global f1_score_dt
    global recall_score_dt
    global precision_score_dt
    recall_score_dt.clear()
    precision_score_dt.clear()
    f1_score_dt.clear()
    global tp_dt
    global fn_dt
    global fp_dt
    global tp_dt_list
    global fn_dt_list
    global fp_dt_list
    tp_dt_list.clear()
    fn_dt_list.clear()
    fp_dt_list.clear()

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('decision_tree.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)
        y_pred = model.predict(X_test)

        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test,labels=[0,1])
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            cm=confusion_matrix(y_pred,y_test,labels=[0,1])
            tn, fp_dt, fn_dt, tp_dt = cm.ravel()
            fp_dt_list.append(int(fp_dt))
            fn_dt_list.append(int(fn_dt))
            tp_dt_list.append(int(tp_dt))
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_dt.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_dt.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_dt.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(application.static_folder,'dt_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(4.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        dt_f1_score = os.path.join(application.static_folder, 'dt.png')
        plt.show()
        plt.savefig(dt_f1_score)
        plt.close()

        image_files_to_delete = [cm, dt_f1_score]
        delete(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/dt.png','cm':'/static/dt_cm.png','f1':f1,'f1score':f1_score_dt, 'recallscore':recall_score_dt,'precisionscore':precision_score_dt,'tp':tp_dt_list,'fp':fp_dt_list,'fn':fn_dt_list})
    
    except Exception as e:
        return jsonify({'error': str(e)})


@application.route('/weekly-supervised', methods=['POST'])
def weekly_supervised():
    global df_test
    global df_filtered_trimmed

    try:
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        testChcek = "testing active-learning working"
        return jsonify({'success': True, 'testChcek' : testChcek})
    except Exception as e:
        return jsonify({'error': str(e)})
    

@application.route('/active-learning', methods=['POST'])
def active_learning():
    global df_test
    global df_filtered_trimmed

    try:
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        testChcek = "testing weekly supervised model working"
        return jsonify({'success': True, 'testChcek' : testChcek})
    except Exception as e:
        return jsonify({'error': str(e)})


# evaluation metrics
@application.route('/f1score', methods=['POST'])
def f1score():
    global f1_score_lg
    global f1_score_nb
    global f1_score_rf
    global f1_score_svc
    global f1_score_dt

    global recall_score_lg
    global recall_score_nb
    global recall_score_rf
    global recall_score_svc
    global recall_score_dt

    global precision_score_lg
    global precision_score_nb
    global precision_score_rf
    global precision_score_svc
    global precision_score_dt

    size = [2,3,4,5,6,7,8,9]
   
    return jsonify({'success':True,
                    'f1_score_lg': f1_score_lg,
                    'f1_score_nb': f1_score_nb,
                    'f1_score_rf': f1_score_rf,
                    'f1_score_svc': f1_score_svc,
                    'f1_score_dt': f1_score_dt,
                    'recall_score_lg': recall_score_lg,
                    'recall_score_nb': recall_score_nb,
                    'recall_score_rf': recall_score_rf,
                    'recall_score_svc': recall_score_svc,
                    'recall_score_dt': recall_score_dt,
                    'precision_score_lg': precision_score_lg,
                    'precision_score_nb': precision_score_nb,
                    'precision_score_rf': precision_score_rf,
                    'precision_score_svc': precision_score_svc,
                    'precision_score_dt': precision_score_dt})

''' ROI Analysis '''
@application.route('/roi-parameters', methods=['POST'])
def set_roi_parameters():
    global fp_cost
    global fn_cost
    global tp_cost
    global resources_cost
    global preprocessing_cost
    global product_value

    fp_cost = float(request.form.get('fp_cost'))
    fn_cost = float(request.form.get('fn_cost'))
    tp_cost = float(request.form.get('tp_cost'))
    resources_cost = float(request.form.get('resources_cost'))
    preprocessing_cost = float(request.form.get('preprocessing_cost'))
    product_value = float(request.form.get('product_value'))

    return jsonify({'success':True, 
                    'fp_cost' : fp_cost,
                    'fn_cost' : fn_cost,
                    'tp_cost' : tp_cost,
                    'resources_cost' : resources_cost,
                    'preprocessing_cost' : preprocessing_cost,
                    'product_value' : product_value})


'''Benefit = TP*tp_cost - FP*fp_cost - FN*fn_cost'''
@application.route('/roi-graphs', methods=['POST'])
def set_roi_graphs():
    global fp_cost
    global fn_cost
    global tp_cost
    global resources_cost
    global preprocessing_cost
    global product_value

    global fp_lg_list 
    global fn_lg_list 
    global tp_lg_list 
    global fp_nb_list 
    global fn_nb_list 
    global tp_nb_list 
    global fp_rf_list 
    global fn_rf_list 
    global tp_rf_list 
    global fp_svc_list
    global fn_svc_list
    global tp_svc_list
    global fp_dt_list 
    global fn_dt_list 
    global tp_dt_list 

    global roi_lg 
    global roi_nb 
    global roi_rf 
    global roi_svc
    global roi_dt

    global f1_score_lg
    global f1_score_nb
    global f1_score_rf
    global f1_score_svc
    global f1_score_dt

    for k in range(8):
        cost = (int(resources_cost) + int(preprocessing_cost) + int(product_value))
        if len(tp_lg_list)>0:
            benefit_lg = tp_lg_list[k]*tp_cost - fn_lg_list[k]*fn_cost - fp_lg_list[k]*fp_cost
            roi_lg.append((benefit_lg - cost)/cost)
        if len(tp_nb_list)>0:
            benefit_nb = tp_nb_list[k]*tp_cost - fn_nb_list[k]*fn_cost - fp_nb_list[k]*fp_cost
            roi_nb.append((benefit_nb - cost)/cost)
        if len(tp_rf_list)>0:
            benefit_rf = tp_rf_list[k]*tp_cost - fn_rf_list[k]*fn_cost - fp_rf_list[k]*fp_cost
            roi_rf.append((benefit_rf - cost)/cost)
        if len(tp_svc_list)>0:
            benefit_svc = tp_svc_list[k]*tp_cost - fn_svc_list[k]*fn_cost - fp_svc_list[k]*fp_cost
            roi_svc.append((benefit_svc - cost)/cost)
        if len(tp_dt_list)>0:
            benefit_dt = tp_dt_list[k]*tp_cost - fn_dt_list[k]*fn_cost - fp_dt_list[k]*fp_cost
            roi_dt.append((benefit_dt - cost)/cost)

    return jsonify({'success':True, 
                    'roi_lg':roi_lg,
                    'roi_nb':roi_nb,
                    'roi_rf':roi_rf,
                    'roi_svc':roi_svc,
                    'roi_dt':roi_dt,
                    'f1_score_lg': f1_score_lg,
                    'f1_score_nb': f1_score_nb,
                    'f1_score_rf': f1_score_rf,
                    'f1_score_svc': f1_score_svc,
                    'f1_score_dt': f1_score_dt,})


@application.route('/activeLearning1', methods=['POST'])
def active_learning1():
    global classifierx
    global args_dict
    global args1
    global logFilePath
    
    #Ignore Future warnings if any occur. 
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    pd.set_option('display.max_columns', 500)   #To make sure all the columns are visible in the 
    pd.set_option('display.width', 1000)

    # Get the current working directory
    currentFileDir = os.getcwd()

    # Parse the JSON data from the client
    data = request.get_json()

    # Retrieve the parameters from the JSON data
    comments = data.get('comments')
    threshold = data.get('threshold')
    max_iterations = data.get('max_iterations')
    resampling = data.get('resampling')
    classifier = data.get('classifier')
    sampling_type = data.get('sampling_type')
    test_size = data.get('test_size')
    manual_annotations_count = data.get('manual_annotations_count')

    # Use the parameters as needed
    args_dict['comments'] = [comments]
    args_dict['threshold'] = [threshold]
    args_dict['max_iterations'] = [max_iterations]
    args_dict['resampling'] = [resampling]
    args_dict['classifier'] = [classifier]
    args_dict['sampling_type'] = [sampling_type]
    args_dict['test_size'] = [test_size]
    args_dict['manual_annotations_count'] = [manual_annotations_count]

    # Convert the dictionary to a pandas DataFrame
    args1 = pd.DataFrame(args_dict)

    
    classifierx = args1.loc[0,'classifier']
    #Creates Logs folder structure
    logFilePath,OFilePath = createLogs(currentFileDir+"/Logs",args1)
    learnTargetLabel(args1)

    # Check if the file exists
    if os.path.isfile(logFilePath):
        # Open the file in read mode
        with open(logFilePath, 'r') as file:
            # Read the file content
            fileContent = file.read()
        # Return the file content
        return jsonify({'fileContent': fileContent})

    # In case the file does not exist
    return jsonify({'error': 'File not found'}), 404

    


def learnTargetLabel(args):
    
    '''
    Active Learning iterative process
    1. Prepare Data
    2. Create Classifier
    3. Evaluate Classifier
    4. Select Uncertain Samples and get them annotated by Oracle
    5. Update Data Set (Merge newly annotated samples to original dataset) 
    6. Repeat steps 1-5 until stopping condition is reached.

    Parameters : 
    args (dataframe) : Run-time arguments in a dataframe.

    Returns :
    df_rqmts (dataframe) : Updated / Final requirements dataset, included the prediction values at the last iteration of Active learning process. 
    df_resultTracker (dataframe) : Results for tracking purpose

    '''    

    global args1, iteration
    global df_training, df_testing, df_LM_training, df_LM_testing, df_rqmts, df_resultTracker  # Declare them as global inside the function
    iteration = 0
    #Read run time arguments
    splitratio = float(args.loc[0,'test_size']) 
    maxIterations = int(args.loc[0,'max_iterations'])
    resamplingTechnique = args.loc[0,'resampling']
    print(splitratio,maxIterations,resamplingTechnique)
    #input("hit enter")
    
    writeLog("Fetching data from the input directory.")
    #Read To be Annotated, Training, Test and Validation Sets generated after executing splitData.py
    try:
        df_rqmts = pd.read_csv(fullFile) #this has training data with Annotated = 'M' and rest with Nothing ''
        print(df_rqmts[annStatus].value_counts())

    except FileNotFoundError as err:
        writeLog ("File Not Found! Please provide correct path of the directory containing Training, Test, Validation, ToBeAnnotated and Manually Annotated DataSet.")
        print (err)
        exit()

    
    #Create a dataframe to track the results
    df_resultTracker = pd.DataFrame()
    df_rqmts=df_rqmts.sample(frac=1) #shuffulles
    df_rqmts[label] = df_rqmts[label].astype('int')
  
    df_training = df_rqmts[df_rqmts[annStatus]=='M']
    df_testing = df_rqmts[df_rqmts[annStatus]!='M']
    
    #these are two df's for local model(LM) training for first iteration
    df_LM_training = df_training
    df_LM_testing = df_testing

@application.route('/next', methods=['POST'])
def next():
    global iteration
    global splitratio
    global maxIterations
    global resamplingTechnique
    global df_rqmts
    global df_resultTracker
    global df_rqmts
    global df_training
    global df_testing
    global df_LM_training
    global df_LM_testing
    global args1
    global logFilePath
    f1_score_nb.clear()
    f1_score_rf.clear()
    f1_score_svc.clear()

    recall_score_nb.clear()
    recall_score_rf.clear()
    recall_score_svc.clear()

    precision_score_nb.clear()
    precision_score_rf.clear()
    precision_score_svc.clear()

    tp_nb_list.clear()
    fn_nb_list.clear()
    fp_nb_list.clear()

    tp_rf_list.clear()
    fn_rf_list.clear()
    fp_rf_list.clear()

    tp_svc_list.clear()
    fn_svc_list.clear()
    fp_svc_list.clear()


    iteration+=1
    if iteration < maxIterations:
        return logFilePath,outputFilePath
    writeLog("\n"+100*"-")
    writeLog("\n\nIteration : "+str(iteration)+"\n")
    #####run it multiple times say 10 and accumulate average results
    V_f1=[]
    V_prec = []
    V_rcl=[]
    V_indPrec = []
    V_indRcl=[]
    V_indF1=[]
    V_ReqPre=[]
    V_ReqRcl=[]
    V_ReqF1=[]
    V_SimPrec=[]
    V_SimRcl=[]
    V_SimF1=[]
    V_confusioMatrix=[]

    LM_f1=[]
    LM_prec=[]
    LM_rcl=[]
    LM_indPrec=[]
    LM_indRcl=[]
    LM_indF1=[]
    LM_ReqPre=[]
    LM_ReqRcl=[]
    LM_ReqF1=[]
    LM_SimPrec=[]
    LM_SimRcl=[]
    LM_SimF1=[]
    LM_confusionMatrix = []

    print(df_LM_testing.columns)

    for i in range(9):
        #-----------------------------------------AL model -------------------------------------#
        writeLog("\nCreating Classifier...")
        #just pass the ones with AnnotationStatus = 'M'
        countVectorizer, tfidfTransformer, classifier, classifierTestScore = createClassifier(args1.loc[0,'classifier'],df_training,resamplingTechnique)
        writeLog("\n\n5 fold Cross Validation Score : "+str(classifierTestScore))
        writeLog ("\n\nValidating Classifier...")

        #pass the rest as testing data annStatus]!='M'
        classifierValidationScore,v_f1Score,v_precisionScore,v_recallScore,v_precisionArr,v_recallArr,v_fscoreArr,v_supportArr,v_confusionMatrix = validateClassifier(countVectorizer,tfidfTransformer,classifier,df_testing)
        writeLog("\n\nClassifier Validation Set Score : "+str(classifierValidationScore))

        #Update Analysis DataFrame (For tracking purpose)
        df_training[label] = df_training[label].astype('int')
        independentCount = len(df_training[df_training[label]==depType['independent']])
        requiresCount = len(df_training[df_training[label]==depType['relates to']])
        similarCount = len(df_training[df_training[label]==depType['incorporates']])
        #----------------------------------------------AL ends-----------------------------------------


        #-----------------------------------------LM starts-------------------------------------------
        LM_countVectorizer, LM_tfidfTransformer, LM_classifier, LM_classifierTestScore= createClassifier(args1.loc[0,'classifier'],df_LM_training,resamplingTechnique)

        #pass the rest as testing data annStatus]!='M'
        LM_classifierValidationScore,LM_f1Score,LM_precisionScore,LM_recallScore,LM_precisionArr,LM_recallArr,LM_fscoreArr,LM_supportArr,lm_confusionMatrix = validateClassifier(LM_countVectorizer,LM_tfidfTransformer,LM_classifier,df_LM_testing)
        writeLog("\n\nClassifier Validation Set Score for LM: "+str(LM_classifierValidationScore))

        #Update Analysis DataFrame (For tracking purpose)
        df_LM_training[label] = df_LM_training[label].astype('int')
        LM_independentCount = len(df_LM_training[df_LM_training[label]==depType['independent']])
        LM_requiresCount = len(df_LM_training[df_LM_training[label]==depType['relates to']])
        LM_similarCount = len(df_LM_training[df_LM_training[label]==depType['incorporates']])

        #-----------------------------------------LM Ends--------------------------------------------
        #store results to average later
        V_f1.append(v_f1Score)
        V_prec.append(v_precisionScore)
        V_rcl.append(v_recallScore)
        V_indPrec.append(v_precisionArr[0])
        V_indRcl.append(v_recallArr[0])
        V_indF1.append(v_fscoreArr[0])
        V_ReqPre.append(v_precisionArr[1])
        V_ReqRcl.append(v_recallArr[1])
        V_ReqF1.append(v_fscoreArr[1])
        V_SimPrec.append(v_precisionArr[2])
        V_SimRcl.append(v_recallArr[2])
        V_SimF1.append(v_fscoreArr[2])
        V_confusioMatrix.append(v_confusionMatrix)

        LM_f1.append(LM_f1Score)
        LM_prec.append(LM_precisionScore)
        LM_rcl.append(LM_recallScore)
        LM_indPrec.append(LM_precisionArr[0])
        LM_indRcl.append(LM_recallArr[0])
        LM_indF1.append(LM_fscoreArr[0])
        LM_ReqPre.append(LM_precisionArr[1])
        LM_ReqRcl.append(LM_recallArr[1])
        LM_ReqF1.append(LM_fscoreArr[1])
        LM_SimPrec.append(LM_precisionArr[2])
        LM_SimRcl.append(LM_recallArr[2])
        LM_SimF1.append(LM_fscoreArr[2])
        #

    tempList = (lm_confusionMatrix.tolist())
    LM_confusionMatrix.append(tempList)
    #print(tempList[-1],"\n", tempList)
    #input("hit enter")
    df_resultTracker = df_resultTracker.append({'Iteration':iteration,
                                                'Total data':len(df_rqmts),
                                                'TraiAKA_ManlyAntd':len(df_training),
                                                'Testing':len(df_testing),
                                                'CV':classifierTestScore,
                                                '#Ind':independentCount,
                                                '#Req':requiresCount,
                                                '#Sim':similarCount,
                                                'f1':"{:.2f}".format(np.average(V_f1)),
                                                'prec':"{:.2f}".format(np.average(V_prec)),
                                                'rcl':"{:.2f}".format(np.average(V_rcl)),
                                                'indPrec': "{:.2f}".format(np.average(V_indPrec)),
                                                'indRcl':"{:.2f}".format(np.average(V_indRcl)),
                                                'indF1':"{:.2f}".format(np.average(V_indF1)),
                                                'indSup':v_supportArr[0],
                                                'ReqPre': "{:.2f}".format(np.average(V_ReqPre)),
                                                'ReqRcl':"{:.2f}".format(np.average(V_ReqRcl)),
                                                'ReqF1':"{:.2f}".format(np.average(V_ReqF1)),
                                                'ReqSup':v_supportArr[1],
                                                'SimPrec': "{:.2f}".format(np.average(V_SimPrec)),
                                                'SimRcl':"{:.2f}".format(np.average(V_SimRcl)),
                                                'SimF1':"{:.2f}".format(np.average(V_SimF1)),
                                                'SimSup':v_supportArr[2],
                                                'ConfusionM':v_confusionMatrix,

                                                '#LM Training':len(df_LM_training),
                                                '#LM Testing':len(df_LM_testing),
                                                'LM CV':LM_classifierTestScore,
                                                '#LM Ind':LM_independentCount,
                                                '#LM Req':LM_requiresCount,
                                                '#LM Sim':LM_similarCount,
                                                'LM f1':"{:.2f}".format(np.average(LM_f1)),
                                                'LM prec':"{:.2f}".format(np.average(LM_prec)),
                                                'LM rcl':"{:.2f}".format(np.average(LM_rcl)),
                                                'LM indPre': "{:.2f}".format(np.average(LM_indPrec)),
                                                'LM indRcl':"{:.2f}".format(np.average(LM_indRcl)),
                                                'LM indF1':"{:.2f}".format(np.average(LM_indF1)),
                                                'LM indSup':LM_supportArr[0],
                                                'LM ReqPre': "{:.2f}".format(np.average(LM_ReqPre)),
                                                'LM ReqRcl':"{:.2f}".format(np.average(LM_ReqRcl)),
                                                'LM ReqF1':"{:.2f}".format(np.average(LM_ReqF1)),
                                                'LM ReqSup':LM_supportArr[1],
                                                'LM SimPrec': "{:.2f}".format(np.average(LM_SimPrec)),
                                                'LM SimRcl':"{:.2f}".format(np.average(LM_SimRcl)),
                                                'LM SimF1':"{:.2f}".format(np.average(LM_SimF1)),
                                                'LM SimSup':LM_supportArr[2],
                                                'LM ConfusionM':LM_confusionMatrix[-1]


                                                },ignore_index=True)


    writeLog("\n\nAnalysis DataFrame : \n"+str(df_resultTracker))
    print("-----------Before-----------")
    print(len(df_rqmts),"=",len(df_training),"+", len(df_testing))

    writeLog ("\n\nPredicting Labels....")
    df_predictionResults = predictLabels(countVectorizer,tfidfTransformer,classifier,df_testing)   #operate on the df_rqmts only

    writeLog("\n\nFinding Uncertain Samples and Annotating them.....")
    df_finalPredictions, df_remaining_Testing = analyzePredictions(args1,df_predictionResults)
    writeLog("\n\nMerging Newly Labelled Data Samples....")
    df_rqmts = pd.concat([df_training,df_finalPredictions],axis=0,ignore_index=True)
    df_rqmts = pd.concat([df_rqmts,df_remaining_Testing],axis=0,ignore_index=True)
    print("After")
    print(len(df_rqmts),"=", len(df_training),"+", len(df_remaining_Testing),"+", len(df_finalPredictions))
    print(df_rqmts[annStatus].value_counts())
    print("-"*100)
    #Remove unwanted columns
    df_rqmts = df_rqmts[[req1,req2,label,annStatus]]#df_rqmts[['req_2','req_1',label,annStatus]]#df_rqmts[['comboId','req1Id','req1','req_1','req2Id','req2','req_2',label,annStatus]]

    #input("hit enter to proceed")


    #if iteration >=maxIterations:
    #new stopping condition is if the validation set is more than equal to or less than 30% of training size
    #df_validation = df_rqmts[df_rqmts[annStatus]!='M']
    if int(len(df_testing)) <= int(0.3*(int(len(df_training)))) or iteration >=maxIterations:
        writeLog("\n\nStopping Condition Reached... Exiting the program."+str(len(df_testing))+str(len(df_training)))


    #----------------for next iteration-------------------------------------#
    df_rqmts = df_rqmts.sample(frac=1)
    df_training = df_rqmts[df_rqmts[annStatus]=='M']
    df_testing = df_rqmts[df_rqmts[annStatus]!='M']

    #add equal amount of randomly selected labels to LM models
    #for this extract the number from df_finalPredictions
    stats=df_finalPredictions[label].value_counts()

    print("-------------Before LM---------------")
    print(len(df_rqmts),len(df_LM_training), len(df_LM_testing))
    print(stats)
    for key in stats.keys():
        #fetch the sample of size values for each class randomly and add tp LM training set
        #and remove from testing set
        sampleCount = int(stats[key])
        print(type(sampleCount), sampleCount)
        #df_temp = df_LM_testing[df_LM_testing["Label"]==key].sample(sampleCount)
        #df_temp= df_LM_testing[df_LM_testing[label]==key].sample(sampleCount)
        df_temp = df_LM_testing[df_LM_testing[label]==key]
        if df_temp.shape[0] >= sampleCount:
            df_temp = df_temp.sample(sampleCount)
        else:
            df_temp = df_temp.sample(df_temp.shape[0])  # or df_temp.sample(df_temp.shape[0], replace=True)

        df_temp[annStatus] == 'M'
        df_LM_training = pd.concat([df_LM_training,df_temp],axis=0,ignore_index=True)
        #print(sampleCount, len(df_temp), len(df_LM_testing), len(df_LM_training))
        df_LM_testing = df_LM_testing[~df_LM_testing.isin(df_temp)].dropna(how='any', subset=[req1, req2]) #df_LM_testing[~df_LM_testing.isin(df_temp)].dropna(how='any', subset=['req1', 'req2'])
        df_LM_testing[annStatus] = ""
        #print(sampleCount, len(df_temp), len(df_LM_testing))

    print("-------------After LM---------------")
    print(len(df_rqmts),len(df_LM_training), len(df_LM_testing))
    #input ("Hit enter")
    #house keeping
    df_LM_training = df_LM_training.sample(frac=1)
    df_LM_training[label] = df_LM_training[label].astype('int')
    df_LM_testing[label] = df_LM_testing[label].astype('int')
    #---------------------------------------Next iteration ends ---------------- #
    if os.path.isfile(logFilePath):
        # Open the file in read mode
        with open(logFilePath, 'r') as file:
            # Read the file content
            fileContent = file.read()
        # Return the file content
        return jsonify({'fileContent': fileContent})

    # In case the file does not exist
    return jsonify({'error': 'File not found'}), 404

def leastConfidenceSampling(df_uncertain):
    
    df_uncertain['lconf']=1-df_uncertain['maxProb']
    df_uncertain = df_uncertain.sort_values(by=['lconf'],ascending=False)
    #writeLog("\n\nLeast Confidence Calculations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]

    return sampleIndex

def minMarginSampling(df_uncertain):
    
    df_uncertain['sorted'] = df_uncertain['predictedProb'].sort_values().apply(lambda x:sorted(x,reverse=True))
    df_uncertain['first'] = [x[0] for x in df_uncertain['sorted']]
    df_uncertain['second'] = [x[1] for x in df_uncertain['sorted']] 
    df_uncertain['Margin'] = df_uncertain['first']-df_uncertain['second']
    
    df_uncertain = df_uncertain.sort_values(by=['Margin'],ascending=True)
    writeLog("\n\nMin Margin Calcuations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]
    return sampleIndex

def entropySampling(df_uncertain):
    df_uncertain['entropy'] = [entropy(x) for x in df_uncertain['predictedProb']]
    df_uncertain = df_uncertain.sort_values(by=['entropy'],ascending=False)
    #writeLog("\n\nEntropy Calculations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]
    return sampleIndex

def createLogs(fPath,args):
    '''
    Creates the structure for saving the log file, output file, results file and annotations file
    1. Log file - saves all the details are printed on the command line
    2. Output File - saves all outputs (Analysis Dataframes in this case)
    3. Results File - saves all predicted labels of unlabelled dataset
    4. Annotations File - saves all manually annotated data points (by oracle)
    '''
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    
    if not os.path.exists(fPath+"/"+current_date):
        os.makedirs(fPath+"/"+current_date)
    global logFilePath,outputFilePath,resultsPath,annotationsPath
    logFilePath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-"+args.loc[0,'resampling']+"_"+args.loc[0,'comments']+".txt"
    outputFilePath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-"+args.loc[0,'resampling']+"_"+args.loc[0,'comments']+".csv"
    resultsPath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-RESULTS-"+args.loc[0,'resampling']+"_"+args.loc[0,'comments']+".csv"
    annotationsPath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-ANNOTATIONS-"+args.loc[0,'resampling']+"_"+args.loc[0,'comments']+".csv"
    for fPath in [logFilePath,outputFilePath]:
        file = open(fPath,'a')
        file.write("\n"+100*"-"+"\nArguments :- \n")
        for col in args.columns:
            file.write(str(col)+" : "+str(args.loc[0,str(col)])+"\n")
        file.write("\n"+100*"-"+"\n")
        file.close()

    
    return logFilePath,outputFilePath

def getArguments(fName):
    '''
    Reads the arguments available in the file and converts them into a data frame.
    '''
    file = open(fName,'r')
    df_args = pd.DataFrame()
    print ("\n"+100*"-"+"\nArguments :- \n")
    for line in file:
        print (line.strip())
        kv_pair = line.split(":")
        df_args.loc[0,str(kv_pair[0]).strip()] = str(kv_pair[1]).strip()
    print (100*"-")
    #validateArguments(df_args)
    return df_args

def validateArguments(df_args):
    '''
    Validates the arguments.
    '''
    try:
        if not os.path.exists(os.getcwd()+df_args.loc[0,'input']):
            raise("")
        elif ((df_args.loc[0,'classifier'] not in ['RF','NB','SVM','ensemble']) or (df_args.loc[0,'resampling'] not in ['under_sampling','over_sampling'])or (df_args.loc[0,'sampling_type'] not in ['leastConfidence','minMargin','entropy']) ):
            raise ("")
        elif (float(df_args.loc[0,'test_size']) not in [x/10 for x in range(0,11)]):
            raise ("")
        elif ((int(df_args.loc[0,'manual_annotations_count']))or (int(df_args.loc[0,'max_iterations']))):
            pass
    except :
        print ("\nERROR! Input Arguments are invalid....\nPlease verify your values with following reference.\n")
        showExpectedArguments()
        exit()
    return None

def showExpectedArguments():
    '''
    prints the expected arguments, stored at ALParams_Desc.txt 
    '''
    file = open(os.getcwd()+"/ALParams_Desc.txt")
    for line in file:
        print (line)

def writeLog(content):
    '''
    Dumps the content into Log file
    '''
    file = open(logFilePath,"a", encoding='utf-8')
    file.write(content)
    file.close()
    #print (content)#.encode('utf-8'))
    return None

def createAnnotationsFile(df_rqmts):
    '''
    Dumps the manuall Annotations data into a csv file.
    '''
    if not os.path.exists(annotationsPath):
        df_rqmts.to_csv(annotationsPath,mode="a",index=False,header=True)
    else:
        df_rqmts.to_csv(annotationsPath,mode="a",index=False,header=False)
    return resultsPath

def addOutputToExcel(df,comment):
    '''
    Appends the dataframe df and corresponding comment to the output file.
    '''
    file = open(outputFilePath,"a", encoding='utf-8')
    file.write(comment)
    file.close()
    print (comment)
    print (str(df))
    df.to_csv(outputFilePath,mode='a',index=False)
    return None

def updateResults(df_results,args):
    '''
    Merges the Results data frame with arguments dataframe and stores the results in a csv file. 
    '''
    df_results.reset_index(inplace=True,drop=True)
    args.reset_index(inplace=True,drop=True)
    combined_df = pd.concat([df_results,args],axis=1)
    combined_df.to_csv(resultsPath,mode="a",index=False)
    
    return resultsPath

def analyzePredictions(args,df_predictions):
    '''
    Analyzis the predictions, samples the most uncertain data points and queries it from the oracle (original database/file) and updates dataframe accordingly.
    '''
    #df_manuallyAnnotated = pd.DataFrame(columns=['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus])#Create an empty Dataframe to store the manually annotated Results


    print("\n\nZeeshan")
    print(df_predictions)
    print("\n\nZeeshan")

    """Intelligently Annotate""" 

    threshold = float(args.loc[0, 'threshold'])


    # Filter rows based on the condition
    confident_predictions = df_predictions[df_predictions['maxProb'] > threshold]

    # Get the index values of the filtered rows
    confident_indexes = confident_predictions.index

    # Remove the filtered rows from df_predictions
    df_predictions.drop(index=confident_indexes, inplace=True)
    df_predictions.reset_index(drop=True, inplace=True)
    confident_predictions['annStatus'] = 'I'  # Mark all rows as intelligently annotated
    confident_predictions = confident_predictions[[req1,req2,label,annStatus]]

    print("Amaan")
    print(len(confident_predictions))
    print("Amaan")

    """Annotate with Active Learning (Manual)""" 

    queryType = args.loc[0,'sampling_type']
    df_userAnnot = pd.DataFrame(columns = fields)
    
    for field in [0,1,2,3,4,5]: # there are 6 labels in this csv from 0-5
        iteration = 0
        writeLog("\n\nIteration for field: "+str(field))
        #input("hit enter to proceed")
        print (queryType)
        while iteration<int(args.loc[0,'manual_annotations_count']):  #while iteration is less than number of annotations that need to be done.
            if (len(df_predictions[df_predictions[label]==field ])>0):
                writeLog("\n\nIteration  : "+str(iteration+1))
                if queryType == 'leastConfidence':
                    indexValue = leastConfidenceSampling(df_predictions[df_predictions[label]==field ])
                elif queryType == 'minMargin':
                    indexValue = minMarginSampling(df_predictions[df_predictions[label]==field ])
                elif queryType == 'entropy':
                    indexValue =entropySampling(df_predictions[df_predictions[label]==field ])
            
                sample = df_predictions.loc[indexValue,:]
                writeLog("\n\nMost Uncertain Sample : \n"+str(sample))
                df_userAnnot = df_userAnnot.append({req1:sample[req1],req2:sample[req2],label:sample[label],annStatus:'M'},ignore_index=True)#df_userAnnot.append({'comboId':sample['comboId'],'req1Id':sample['req1Id'],'req1':sample['req1'],req1:sample[req1],'req2Id':sample['req2Id'],'req2':sample['req2'],req2:sample[req2],label:sample[label],annStatus:'M'},ignore_index=True)  #Added AnnotationStatus as M 
                #createAnnotationsFile(df_userAnnot)
                
                #Remove the selected sample from the original dataframe
                df_predictions.drop(index=indexValue,inplace=True)   
                df_predictions.reset_index(inplace=True,drop=True)
            else:
                print("All of unlabelled data is over")            
                    
                #df_manuallyAnnotated = pd.concat([df_manuallyAnnotated,df_userAnnot])
                
            iteration+=1


    #Remove all the extra columns. df now contains only combinations marked 'A'
    df_predictions=df_predictions[[req1,req2,label,annStatus]]#df_predictions[['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus]]
    df_remaining = df_predictions
    df_remaining[annStatus] = 'M'
    #df_manuallyAnnotated=df_manuallyAnnotated[['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus]]
    writeLog("\n\nManually Annotated Combinations... "+str(len(df_predictions))+"Rows \n"+str(df_predictions[:10]))

    combined_df = pd.concat([df_userAnnot, confident_predictions], ignore_index=True) # I added this (Zeeshan)
    return combined_df, df_remaining

def computeGridnget(X,y):
    print("Grid searching")
    rf = RandomForestClassifier()
    #weights = np.linspace(0.005, 0.05, 10)
    params = {'class_weight':{2:10}}
    gsc = GridSearchCV(param_grid = params, estimator=rf)
    grid_result = gsc.fit(X, y)

    print("Best parameters : %s" % grid_result.best_params_)
    return

def createClassifier(clf,df_trainSet,resampling_type):
    df_trainSet= df_trainSet.sample(frac=1)
    '''
    Passes the dataset via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Performs Synthetic Monitoring Over-Sampling after performing TFIDF transformation (ONLY when resampling_type is over_sampling)
    Trains the classifier (Random Forest / Naive Bayes / SVM / Ensemble using Voting Classifier)

    Parameters : 
    clf (str) : Name of classifier (options - RF, NB, SVM , ensemble)
    df_trainSet (DataFrame) : Training Data
    df_testSet (DataFrame) : Test Data

    Returns : 
    count_vect : Count Vectorizer Model
    tfidf_transformer : TFIDF Transformer Model
    clf_model : Trained Model 
    clf_test_score (float) : Accuracy achieved on Test Set 
    f1/precision/recall (float) : F1, Precision and Recall scores (macro average)
    '''

    #df_trainSet = shuffle(df_trainSet)
    #df_testSet = shuffle(df_testSet)

    #Convert dataframes to numpy array's
    X_train = df_trainSet.loc[:,[req1,req2]]  #Using req_1,req_2 rather than req1,req2 because req_1,req_2 have been cleaned - lower case+punctuations
    y_train = df_trainSet.loc[:,label].astype("int")

    writeLog("\nTraining Set Size : "+str(len(X_train)))
    writeLog("\nTrain Set Value Count : \n"+str(df_trainSet[label].value_counts()))

    writeLog("\n\nTraining Model....")
    
    #Perform Bag of Words
    count_vect = CountVectorizer(tokenizer=my_tokenizer,lowercase=False)
    X_train_counts = count_vect.fit_transform(np.array(X_train))
    
    #Transform a count matrix to a normalized tf or tf-idf representation.
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
    
    #######################################################################################
    # if resampling_type == "over_sampling":
    #     writeLog("\n\nValue Count for each class in training set."+str(Counter(y_train)))
        
    #     writeLog("\n\nPerforming Over Sampling")
    #     sm = SMOTE()#k_neighbors=3) #ADASYN(sampling_strategy="minority")
    #     X_train_tfidf, y_train = sm.fit_resample(X_train_tfidf, y_train)
    #     writeLog("\n\nValue Count for each class in training set."+str(Counter(y_train)))
    ######################################################################################

    #Initiate Classifiers
    rf_model = RandomForestClassifier(random_state=0, class_weight={0:10,1:20,2:100})
    nb_model = MultinomialNB()
    svm_model = SVC(random_state = 0, probability=True)  #predict_proba not available if probability = False

    #Random Forest Classifier Creation
    if clf == "RF" :
        clf_model = rf_model.fit(X_train_tfidf, np.array(y_train).astype('int'))
        
    #Naive Bayes Classifier Creation
    elif clf == "NB":
        clf_model = nb_model.fit(X_train_tfidf, np.array(y_train).astype('int'))

    #Support Vector Machine Classifier Creation.
    elif clf == "SVM":
        clf_model = svm_model.fit(X_train_tfidf,np.array(y_train).astype('int'))
    
    #Ensemble Creation
    elif clf == "ensemble":
        #Predict_proba works only when Voting = 'soft'
        #n_jobs = -1 makes allows models to be created in parallel (using all the cores, else we can mention 2 for using 2 cores)
        #clf_model = VotingClassifier(estimators=[('RF', rf_model), ('NB', nb_model),('SVM',svm_model)], voting='soft',n_jobs=1)  
        clf_model = VotingClassifier(estimators=[('RF', rf_model), ('NB', nb_model), ('SVM',svm_model) ], voting='soft',n_jobs=1)#,weights=[30,10])  
        clf_model.fit(X_train_tfidf,np.array(y_train).astype('int'))
 
    
    #perform cross validation instead of train and test split
    clf_test_score=evaluate_model(X_train_tfidf,np.array(y_train).astype('int'),clf_model)
    print(str(round(clf_test_score.mean(),2)) +"(+/- "+str(round(clf_test_score.std()*2,2))+")")
    #input("hit enter")
    return count_vect, tfidf_transformer, clf_model, clf_test_score

# evaluate a model

def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # summarize
    #print(mean(scores), sem(scores))
    #input("hit enter")
    #y_pred = cross_val_predict(model, X, y, cv=cv)
    #conf_mat = confusion_matrix(y, y_pred)
	return scores


def predictLabels(cv,tfidf,clf,df_toBePredictedData):
    '''
    Passes the to be predicted dataset via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Predicts and returns the labels for the input data in a form of DataFrame.

    Parameters : 
    cv : Count Vectorizer Model
    tfidf : TFIDF Transformer Model
    clf : Trained Model 
    df_toBePredictedData (DataFrame) : To Be Predicted Data (Unlabelled Data)

    Returns : 
    df_toBePredictedData (DataFrame) : Updated To Be Predicted Data (Unlabelled Data), including prediction probabilities for different labels
    '''
    predictData = np.array(df_toBePredictedData.loc[:,[req1,req2]])
    #writeLog(str(df_toBePredictedData))
    
    predict_counts = cv.transform(predictData)
    predict_tfidf = tfidf.transform(predict_counts)
    predict_labels = clf.predict(predict_tfidf)
    predict_prob = clf.predict_proba(predict_tfidf)
    
    writeLog ("\nTotal Labels Predicted : "+ str(len(predict_labels)))

    df_toBePredictedData['predictedProb'] = predict_prob.tolist() 
    df_toBePredictedData['maxProb'] = np.amax(predict_prob,axis=1)

    return df_toBePredictedData    

def validateClassifier(cv,tfidf,clf_model,df_validationSet):
    '''
    Passes the validation dataset (Unseen data) via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Calculate the accuracy and other metrics to evaluate the performance of the model on validation set (unseen data)
    
    Parameters : 
    cv : Count Vectorizer Model
    tfidf : TFIDF Transformer Model
    clf : Trained Model 
    df_validationSet (DataFrame) : Validation Data (Unseen Data)

    Returns : 
    clf_val_score/f1/precision/recall (float) : Accuracy Value on Validation Data / F1 score / Precision / Recall
    '''
    
    global fp_rf_list 
    global fn_rf_list 
    global tp_rf_list 

    global fp_svc_list 
    global fn_svc_list 
    global tp_svc_list 

    global fp_nb_list 
    global fn_nb_list 
    global tp_nb_list 


    global f1_score_nb
    global f1_score_rf
    global f1_score_svc

    global recall_score_nb
    global recall_score_rf
    global recall_score_svc

    global precision_score_nb
    global precision_score_rf
    global precision_score_svc


    predictData = np.array(df_validationSet.loc[:,[req1,req2]])
    actualLabels = np.array(df_validationSet.loc[:,label]).astype('int')
    predict_counts = cv.transform(predictData)
    predict_tfidf = tfidf.transform(predict_counts)
    
    predict_labels = clf_model.predict(predict_tfidf)
    clf_val_score = clf_model.score(predict_tfidf,actualLabels)
    precisionArr,recallArr,fscoreArr,supportArr=score(actualLabels,predict_labels,average=None)

    f1 = round(f1_score(actualLabels, predict_labels,average='macro'),2)
    precision = round(precision_score(actualLabels, predict_labels,average='macro'),2)
    recall = round(recall_score(actualLabels, predict_labels,average='macro'),2)
    labelClasses = list(set(actualLabels))   #np.array(y_train).astype('int')
    writeLog ("\n\nClassification Report On Validation Set: \n\n"+str(classification_report(actualLabels,predict_labels)))
    cm = confusion_matrix(actualLabels,predict_labels,labels=[0,1])
#     pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
#     fig, ax = plt.subplots(figsize=(4.5, 3.8))


    if(classifierx == "RF"):
        precision_score_rf.append(precision)
        recall_score_rf.append(recall)
        f1_score_rf.append(f1)
        tn, fp_rf, fn_rf, tp_rf = cm.ravel()
        fp_rf_list.append(int(fp_rf))
        fn_rf_list.append(int(fn_rf))
        tp_rf_list.append(int(tp_rf))
#         sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
#         ax.set_xlabel('Predicted Labels')
#         ax.set_ylabel('True Labels')
#         ax.set_title('Confusion Matrix')
#         cm = os.path.join(application.static_folder,'rf_cm.png')
#         plt.show()
#         plt.savefig(cm)
#         plt.close()
#
#         fig = plt.figure(figsize=(4.5, 4))
#         fig.patch.set_facecolor('white')
#         plt.plot(size,acc, color='#AFD5F0')
#         plt.xlabel("Training Size",color='black')
#         plt.ylabel("Validation Accuracy",color='black')
#         dt_f1_score = os.path.join(application.static_folder, 'rf.png')
#         plt.show()
#         plt.savefig(dt_f1_score)
#         plt.close()
#
#         image_files_to_delete = [cm, dt_f1_score]
#         delete(image_files_to_delete, delay=7)

    elif(classifierx == "SVM"):
        precision_score_svc.append(precision)
        recall_score_svc.append(recall)
        f1_score_svc.append(f1)
        tn, fp_svc, fn_svc, tp_svc = cm.ravel()
        fp_svc_list.append(int(fp_svc))
        fn_svc_list.append(int(fn_svc))
        tp_svc_list.append(int(tp_svc))
#         sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
#         ax.set_xlabel('Predicted Labels')
#         ax.set_ylabel('True Labels')
#         ax.set_title('Confusion Matrix')
#         cm = os.path.join(application.static_folder,'svm_cm.png')
#         plt.show()
#         plt.savefig(cm)
#         plt.close()
#
#         fig = plt.figure(figsize=(4.5, 4))
#         fig.patch.set_facecolor('white')
#         plt.plot(size,acc, color='#AFD5F0')
#         plt.xlabel("Training Size",color='black')
#         plt.ylabel("Validation Accuracy",color='black')
#         dt_f1_score = os.path.join(application.static_folder, 'svm.png')
#         plt.show()
#         plt.savefig(dt_f1_score)
#         plt.close()
#
#         image_files_to_delete = [cm, dt_f1_score]
#         delete(image_files_to_delete, delay=7)


    elif(classifierx == "NB"):
        precision_score_nb.append(precision)
        recall_score_nb.append(recall)
        f1_score_nb.append(f1)
        tn, fp_nb, fn_nb, tp_nb = cm.ravel()
        fp_nb_list.append(int(fp_nb))
        fn_nb_list.append(int(fn_nb))
        tp_nb_list.append(int(tp_nb))
#         sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
#         ax.set_xlabel('Predicted Labels')
#         ax.set_ylabel('True Labels')
#         ax.set_title('Confusion Matrix')
#         cm = os.path.join(application.static_folder,'nb_cm.png')
#         plt.show()
#         plt.savefig(cm)
#         plt.close()
#
#         fig = plt.figure(figsize=(4.5, 4))
#         fig.patch.set_facecolor('white')
#         plt.plot(size,acc, color='#AFD5F0')
#         plt.xlabel("Training Size",color='black')
#         plt.ylabel("Validation Accuracy",color='black')
#         dt_f1_score = os.path.join(application.static_folder, 'nb.png')
#         plt.show()
#         plt.savefig(dt_f1_score)
#         plt.close()
#
#         image_files_to_delete = [cm, dt_f1_score]
#         delete(image_files_to_delete, delay=7)
    
    #writeLog(str(tp_rf_list[0]) + str(fp_rf_list[0]) + str(fn_rf_list[0]) + "hello" + str(f1_score_rf))
    writeLog ("\n\nConfusion Matrix : \n"+str(cm)+"\n")
    
    return clf_val_score,f1,precision,recall, precisionArr,recallArr,fscoreArr,supportArr, cm

def my_tokenizer(arr):
    '''
    Returns a tokenized version of input array, used in Count Vectorizer
    '''
    return (arr[0]+" "+arr[1]).split()

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)
CORS(application)