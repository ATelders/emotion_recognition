# Import libraries

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# ------------------------------------------------------------------------------------------------

# NLP spacy model

nlp = spacy.blank("en")



# ------------------------------------------------------------------------------------------------

# Define functions 

@st.cache
def load_data():
    '''
    Load data based on the user's choice
    '''
    if data_input == 'Kaggle':
        DATA_URL = ('../data/raw/Emotion_final.csv')
        data = pd.read_csv(DATA_URL)
        labels = ['happy', 'sadness', 'love', 'anger', 'fear', 'surprise'] 
    elif data_input == 'data.world':
        DATA_URL = ('../data/raw/text_emotion.csv')
        data = pd.read_csv(DATA_URL)
        data['emotion'] = data['sentiment']
        data['text'] = data['content']
        data = data[['text', 'emotion']]
        labels = ['happy', 'sadness', 'love', 'anger', 'fear', 'surprise'] 
    else:
        st.write('No data')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data, labels

@st.cache(allow_output_mutation=True)
def create_model(data, model):
    '''
    Create a model that takes the data and returns vectors
    '''
    if model == 'TfidfVectorizer':
        tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = {'english'})
    elif model == 'CountVectorizer':
        tf = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = {'english'})   
    matrix = tf.fit_transform(data['text'])
    return tf, matrix

@st.cache
def create_logistic_regression(X_train, y_train, X_test, y_test):
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)
    pred = classifier.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    return acc, pred, pred_labels

@st.cache
def create_decision_tree(X_train, y_train, X_test, y_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)
    pred = classifier.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    return acc, pred, pred_labels

@st.cache
def create_random_forest(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)
    pred = classifier.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    return acc, pred, pred_labels

@st.cache
def create_svm(X_train, y_train, X_test, y_test):
    classifier = SVC()
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)
    pred = classifier.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    return acc, pred, pred_labels

@st.cache(suppress_st_warning=True)
def create_neural_network(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    model = keras.Sequential([
    layers.Dense(10, input_shape=(input_dim,), activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(10, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(10, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(6, activation='softmax')
    ])
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = 'accuracy',
    )
    st.write('', model.summary())
    st.write('OK')
    history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=100,
    epochs=50,
    verbose=1
    )
    # Show the learning curves
    history_df = pd.DataFrame(history.history)
    st.write(history_df)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    st.pyplot()
    pred = pd.DataFrame(model.predict(X_test))
    pred.rename(columns={0: "anger", 1: "fear", 2: "happy", 3: "love", 4: "sadness", 5: "surprise"}, errors="raise", inplace=True)
    sentences_df = pd.DataFrame(sentences_test)
    sentences_pred = pd.concat([sentences_df, pred], axis=1)
    st.write(sentences_pred)


# App title

st.title('Emotion recognition')

# ------------------------------------------------------------------------------------------------

# User menu

chapter_input = st.sidebar.radio('Menu', ['Analysis and processing','Classification'])

st.sidebar.write('---')

# ------------------------------------------------------------------------------------------------

# Select the data source

data_input = st.sidebar.radio('Data', ['Kaggle','data.world'])
data, labels = load_data()
le = preprocessing.LabelEncoder()
y = le.fit_transform(data['emotion'])
labels_num = le.fit_transform(labels)

sentences = data['text'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.25, random_state=1000)

vectorizer_input = st.sidebar.radio('Vectorizer', ['CountVectorizer','TfidfVectorizer'])

tf, matrix = create_model(data, vectorizer_input)

tf.fit(sentences_train)



# ------------------------------------------------------------------------------------------------

# Analysis and processing

if chapter_input == 'Analysis and processing':
    if st.checkbox('Show dataframe'): 
        st.write(data)
    

    results = data['emotion'].value_counts()
    sns.histplot(data=data, y="emotion")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    st.write(matrix)

# ------------------------------------------------------------------------------------------------

# Classification

X_train = tf.transform(sentences_train)
X_test  = tf.transform(sentences_test)

if chapter_input == 'Classification':
    st.write('Classification models')
    alg=['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'Neural Network']
    #inverse label transformation using label encoder (le)
    y_test_labels = le.inverse_transform(y_test)

    acc_display_lr = 0
    acc_display_dt = 0
    acc_rf = 0
    acc_svm = 0
    acc_nn = 0

    classifier = st.selectbox('Which algorithm?', alg)
    if classifier=='Logistic Regression':
        acc_lr, pred_lr, pred_labels_lr = create_logistic_regression(X_train, y_train, X_test, y_test)
        st.write('Accuracy: ', acc_lr)
        st.write('Confusion matrix: ')
        #display confusion matrix with labels
        cm = confusion_matrix(y_test_labels, pred_labels_lr, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot()
    if classifier=='Decision Tree':
        acc_dt, pred_dt, pred_labels_dt = create_decision_tree(X_train, y_train, X_test, y_test)
        st.write('Accuracy: ', acc_dt)
        st.write('Confusion matrix: ')
        #display confusion matrix with labels
        cm = confusion_matrix(y_test_labels, pred_labels_dt, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot()
    elif classifier == 'Random Forest':
        acc_rf, pred_rf, pred_labels_rf = create_random_forest(X_train, y_train, X_test, y_test)
        st.write('Accuracy: ', acc_rf)
        st.write('Confusion matrix: ')
        #display confusion matrix with labels
        cm = confusion_matrix(y_test_labels, pred_labels_rf, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot()       
    elif classifier == 'Support Vector Machine':
        acc_svm, pred_svm, pred_labels_svm = create_svm(X_train, y_train, X_test, y_test)
        st.write('Accuracy: ', acc_svm)
        st.write('Confusion matrix: ')
        #display confusion matrix with labels
        cm = confusion_matrix(y_test_labels, pred_labels_svm, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot()
    elif classifier == 'Neural Network':
        # pred_nn, pred_labels_nn = 
        create_neural_network(X_train, y_train, X_test, y_test)

        # st.write('Confusion matrix: ')
        # #display confusion matrix with labels
        # cm = confusion_matrix(y_test_labels, pred_labels_nn, labels=labels)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
        # disp.plot(cmap=plt.cm.Blues)
        # st.pyplot()

    st.write('''Accuracy\n\n
    Logistic Regression: {}\n
    Decision Tree: {}\n
    Random Forest: {}\n
    Support Vector Machine: {}\n
    Neural Network: {}\n
        '''.format(0,0,0,0,0))