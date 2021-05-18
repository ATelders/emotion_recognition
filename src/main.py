import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


data_input = st.sidebar.radio('Data', ['Kaggle','data.world'])

@st.cache
def load_data():
    if data_input == 'Kaggle':
        DATA_URL = ('../data/raw/Emotion_final.csv')
        data = pd.read_csv(DATA_URL)
    elif data_input == 'data.world':
        DATA_URL = ('../data/raw/text_emotion.csv')
        data = pd.read_csv(DATA_URL)
        data['emotion'] = data['sentiment']
        data['text'] = data['content']
        data = data[['text', 'emotion']]
    else:
        st.write('No data')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


@st.cache(allow_output_mutation=True)
def create_model(data, model):
    if model == 'TfidfVectorizer':
        tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = {'english'})
    elif model == 'CountVectorizer':
        tf = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = {'english'})   
    tfidf_matrix = tf.fit_transform(data)
    return tf, tfidf_matrix

st.title('Emotion recognition')


data = load_data()


le = preprocessing.LabelEncoder()
y = le.fit_transform(data['emotion'])

sentences = data['text'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.25, random_state=1000)



vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

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
    alg=['Decision Tree', 'Random Forest', 'Support Vector Machine']
    y_label = le.inverse_transform(y_test)

    classifier = st.selectbox('Which algorithm?', alg)
    if classifier=='Decision Tree':
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        acc = dtc.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = dtc.predict(X_test)
        cm_dtc=confusion_matrix(y_test,pred_dtc)
        st.write('Confusion matrix: ', cm_dtc)
    elif classifier == 'Random Forest':
        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm_clf=confusion_matrix(y_test,pred_clf)
        st.write('Confusion matrix: ', cm_clf)        
    elif classifier == 'Support Vector Machine':
        svm=SVC()
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_svm = svm.predict(X_test)
        cm=confusion_matrix(y_test,pred_svm)
        st.write('Confusion matrix: ', cm)