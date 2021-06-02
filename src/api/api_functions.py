# Import libraries

import pandas as pd
import seaborn as sns
import fasttext
import io
import pickle
import pathlib

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------------------------------------------

# Define functions 

@st.cache
def load_data():
    '''
    Load data
    '''
    data_input == 'Kaggle':
    DATA_URL = ('./data/raw/Emotion_final.csv')
    data = pd.read_csv(DATA_URL)
    labels = ['happy', 'sadness', 'love', 'anger', 'fear', 'surprise'] 
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data, labels

def create_model(data, model):
    '''
    Create a model that takes the data and returns vectors
    '''
    model == 'CountVectorizer':
    tf = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = {'english'})   
    matrix = tf.fit_transform(data['text'])
    return tf, matrix

def create_logistic_regression(X_train, y_train, X_test, y_test):
    '''
    Create a logistic regression model
    '''
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
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data



@st.cache(suppress_st_warning = True)
def predict_emotion(model: fasttext.FastText._FastText, sentence: str) -> str:
    predictions = model.predict(sentence)
    label = predictions[0][0].split("__label__")[1]
    label = label.title()
    confidence = predictions[1][0]
    return "{} ({:.2f}% confident)".format(label, confidence * 100)



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
    elif classifier=='Decision Tree':
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