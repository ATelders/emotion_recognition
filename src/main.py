import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

data_input = st.sidebar.radio('Data', ['Kaggle','data.world'])

@st.cache
def load_data():
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
    if model == 'TfidfVectorizer':
        tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = {'english'})
    elif model == 'CountVectorizer':
        tf = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = {'english'})   
    tfidf_matrix = tf.fit_transform(data)
    return tf, tfidf_matrix

st.title('Emotion recognition')


data, labels = load_data()

le = preprocessing.LabelEncoder()
y = le.fit_transform(data['emotion'])

sentences = data['text'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.25, random_state=1000)



vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

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

# @st.cache
# def create_xgboost(X_train, y_train, X_test, y_test):
#     classifier = XGBoost()
#     classifier.fit(X_train, y_train)
#     acc = classifier.score(X_test, y_test)
#     pred = classifier.predict(X_test)
#     pred_labels = le.inverse_transform(pred)
#     return acc, pred, pred_labels



chapter_input = st.sidebar.radio('Chapters', ['Analysis and processing','Classification'])

if chapter_input == 'Analysis and processing':
    if st.checkbox('Show dataframe'): 
        st.dataframe(data.head(10))

    results = data['emotion'].value_counts()
    sns.histplot(data=data, y="emotion")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

if chapter_input == 'Classification':
    st.write('Classification models')
    alg=['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine']
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

    st.write('''Accuracy\n\n
    Logistic Regression: {}\n
    Decision Tree: {}\n
    Random Forest: {}\n
    Support Vector Machine: {}\n
    Neural Network: {}\n
        '''.format(0,0,0,0,0))