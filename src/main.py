# Import libraries

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fasttext
import io
import pickle
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# ------------------------------------------------------------------------------------------------

# Define functions 

@st.cache
def load_data():
    '''
    Load data based on the user's choice
    '''
    if data_input == 'Kaggle':
        DATA_URL = ('./data/raw/Emotion_final.csv')
        data = pd.read_csv(DATA_URL)
        labels = ['happy', 'sadness', 'love', 'anger', 'fear', 'surprise'] 
    elif data_input == 'data.world':
        DATA_URL = ('./data/raw/text_emotion.csv')
        data = pd.read_csv(DATA_URL)
        data['emotion'] = data['sentiment']
        data['text'] = data['content']
        data = data[['text', 'emotion']]
        labels = data['emotion'].unique()
    elif data_input == 'data.world binary':
        DATA_URL = ('./data/raw/text_emotion.csv')
        data = pd.read_csv(DATA_URL)
        data['emotion'] = data['sentiment']
        data['text'] = data['content']
        data = data[['text', 'emotion']]
        data = data.drop(data[(data.emotion == 'neutral') | (data.emotion == 'empty') | (data.emotion == 'surprise')].index)
        pos_emotions = ['enthusiasm', 'love', 'fun', 'happiness', 'relief']
        data.loc[data['emotion'].isin(pos_emotions), 'binary'] = 1
        data.loc[~data['emotion'].isin(pos_emotions), 'binary'] = 0
        data = data.rename(columns={'emotion':'emotion_label', 'binary': 'emotion'})
        labels = data['emotion'].unique()
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
    # classifier = LogisticRegression(max_iter=1000)
    # classifier.fit(X_train, y_train)
    if data_input == "Kaggle":
        if vectorizer_input == "CountVectorizer":
            filename = './data/models/lr_kaggle_count.sav'
        elif vectorizer_input == "TfidfVectorizer":
            filename = './data/models/lr_kaggle_tfidf.sav'
    elif data_input == "data.world":
        if vectorizer_input == "CountVectorizer":
            filename = './data/models/lr_world_count.sav'
        elif vectorizer_input == "TfidfVectorizer":
            filename = './data/models/lr_world_tfidf.sav'
    classifier = pickle.load(open(filename, 'rb'))
    acc = classifier.score(X_test, y_test)
    pred = classifier.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    return acc, pred, pred_labels

@st.cache
def create_decision_tree(X_train, y_train, X_test, y_test):
    # classifier = DecisionTreeClassifier()
    # classifier.fit(X_train, y_train)
    if data_input == "Kaggle":
        if vectorizer_input == "CountVectorizer":
            filename = './data/models/dt_kaggle_count.sav'
        elif vectorizer_input == "TfidfVectorizer":
            filename = './data/models/dt_kaggle_tfidf.sav'
    elif data_input == "data.world":
        if vectorizer_input == "CountVectorizer":
            filename = './data/models/dt_world_count.sav'
        elif vectorizer_input == "TfidfVectorizer":
            filename = './data/models/dt_world_tfidf.sav'
    classifier = pickle.load(open(filename, 'rb'))
    acc = classifier.score(X_test, y_test)
    pred = classifier.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    return acc, pred, pred_labels

@st.cache
def create_random_forest(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    # if data_input == "Kaggle":
    #     if vectorizer_input == "CountVectorizer":
    #         filename = './data/models/rf_kaggle_count.sav'
    #     elif vectorizer_input == "TfidfVectorizer":
    #         filename = './data/models/rf_kaggle_tfidf.sav'
    # elif data_input == "data.world":
    #     if vectorizer_input == "CountVectorizer":
    #         filename = './data/models/rf_world_count.sav'
    #     elif vectorizer_input == "TfidfVectorizer":
    #         filename = './data/models/rf_world_tfidf.sav'
    #classifier = pickle.load(open(filename, 'rb'))
    acc = classifier.score(X_test, y_test)
    pred = classifier.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    return acc, pred, pred_labels

@st.cache
def create_svm(X_train, y_train, X_test, y_test):
    # classifier = SVC()
    # classifier.fit(X_train, y_train)
    if data_input == "Kaggle":
        if vectorizer_input == "CountVectorizer":
            filename = './data/models/svm_kaggle_count.sav'
        elif vectorizer_input == "TfidfVectorizer":
            filename = './data/models/svm_kaggle_tfidf.sav'
    elif data_input == "data.world":
        if vectorizer_input == "CountVectorizer":
            filename = './data/models/svm_world_count.sav'
        elif vectorizer_input == "TfidfVectorizer":
            filename = './data/models/svm_world_tfidf.sav'
    classifier = pickle.load(open(filename, 'rb'))
    acc = classifier.score(X_test, y_test)
    pred = classifier.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    return acc, pred, pred_labels

def display_metrics(acc, pred, pred_labels):
        st.write('Accuracy: ', acc)
        st.write('Confusion matrix: ')
        #display confusion matrix with labels
        cm = confusion_matrix(y_test_labels, pred_labels, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.xticks(rotation=90)
        st.pyplot()
        report = classification_report(y_test_labels, pred_labels)
        st.code(report)

@st.cache(suppress_st_warning=True)
def create_neural_network(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    output_dim = len(labels)
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
    layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = 'accuracy',
    )
    early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
    )
    history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=100,
    epochs=50,
    verbose=1,
    callbacks=[early_stopping],
    )
    history_df = pd.DataFrame(history.history)
    #st.write(history_df)
    
    
    pred = pd.DataFrame(model.predict(X_test))

    temp_cols = pred.columns
    emotion_cols = le.inverse_transform(temp_cols)

    pred.columns = emotion_cols

    sentences_df = pd.DataFrame(sentences_test)
    sentences_pred = pd.concat([sentences_df, pred], axis=1)
    y_pred = pred.idxmax(axis=1)
    sentences_pred['y_pred'] = y_pred
    if data_input == 'Kaggle' or data_input == 'data.world':
        y_test = le.inverse_transform(y_test)
    acc = accuracy_score(y_test, y_pred)
    return sentences_pred, history_df, acc

def display_nn():
    st.write('Accuracy: ', acc_nn)
    st.write(sentences_pred)
    # Show the learning curves
    history_df.loc[:, ['loss', 'val_loss']].plot()
    st.pyplot()

@st.cache
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def make_ft():
    LOAD_MODEL = False
    model_file_name = "fasttext_{}.bin".format(data_input)
    df = data.copy()
    y = df.pop('emotion')
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=1, stratify=y)

    with open('train.txt', 'w') as f:
        for each_text, each_label in zip(X_train.text, y_train):
            f.writelines(f'__label__{each_label} {each_text}\n')
    with open('test.txt', 'w') as f:
        for each_text, each_label in zip(X_test.text, y_test):
            f.writelines(f'__label__{each_label} {each_text}\n')

    if LOAD_MODEL and pathlib.Path(model_file_name).exists():
            model = fasttext.load_model(model_file_name)
    else:
        model = fasttext.train_supervised("train.txt")
    
    model.save_model(model_file_name)
    st.write("vocabulary size: {}".format(len(model.words)))
    N, p, r = model.test('test.txt')
    st.write("Precision\t{:.3f}".format(p))
    st.write("Recall \t{:.3f}".format(r))

    predictions_df = pd.DataFrame(X_test.text)
    predictions_df['emotion'] = y_test
    def predict(row):
        return model.predict(row['text'])[0][0].split("__label__")[1]
    predictions_df['predictions'] = predictions_df.apply(predict,axis=1)
    st.write(predictions_df)

    user_sentence = st.text_input("Write a sentence", "I am happy")
    st.write("Your sentence is:", predict_emotion(model, user_sentence))

@st.cache(suppress_st_warning = True)
def predict_emotion(model: fasttext.FastText._FastText, sentence: str) -> str:
    predictions = model.predict(sentence)
    label = predictions[0][0].split("__label__")[1]
    label = label.title()
    confidence = predictions[1][0]
    return "{} ({:.2f}% confident)".format(label, confidence * 100)

# App title

st.title('Emotion recognition')

# ------------------------------------------------------------------------------------------------

# User menu

chapter_input = st.sidebar.radio('Menu', ['Analysis and processing','Classification'])

st.sidebar.write('---')

# ------------------------------------------------------------------------------------------------

# Select the data source

data_input = st.sidebar.radio('Data', ['Kaggle','data.world', 'data.world binary'])
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
    st.header('Analysis and processing')
    if st.checkbox('Show dataframe'): 
        st.write(data)
    

    results = data['emotion'].value_counts()
    sns.histplot(data=data, y="emotion")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Distribution of emotions in the dataset')

    st.pyplot()
    st.subheader('25 most frequent words')
    if data_input == 'Kaggle':
        st.image('./assets/frequency_kaggle.png')
    if data_input == 'data.world':
        st.image('./assets/frequency_world.png')




# ------------------------------------------------------------------------------------------------

# Classification


X_train = tf.transform(sentences_train)
X_test  = tf.transform(sentences_test)

if chapter_input == 'Classification':
    st.header('Classification')
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
        acc, pred, pred_labels = create_logistic_regression(X_train, y_train, X_test, y_test)
        display_metrics(acc, pred, pred_labels)

    if classifier=='Decision Tree':
        acc, pred, pred_labels = create_decision_tree(X_train, y_train, X_test, y_test)
        display_metrics(acc, pred, pred_labels)
    elif classifier == 'Random Forest':
        acc, pred, pred_labels = create_random_forest(X_train, y_train, X_test, y_test)
        display_metrics(acc, pred, pred_labels)   
    elif classifier == 'Support Vector Machine':
        acc, pred, pred_labels = create_svm(X_train, y_train, X_test, y_test)
        display_metrics(acc, pred, pred_labels)
    elif classifier == 'Neural Network':
        embedding_input = st.radio('Embedding', ['none','fastText'])
        if embedding_input == 'none':
            sentences_pred, history_df, acc_nn = create_neural_network(X_train, y_train, X_test, y_test)
            display_nn()
        elif embedding_input == 'fastText':
            make_ft()