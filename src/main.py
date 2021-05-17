import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = ('../data/raw/Emotion_final.csv')


@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

st.title('Emotion recognition')


data = load_data()


chapter_input = st.sidebar.radio('Chapters', ['Analysis and processing','Classification'])

if chapter_input == 'Analysis and processing':
    st.dataframe(data.head(10))

    results = data['emotion'].value_counts()
    sns.histplot(data=data, y="emotion")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.pyplot()

if chapter_input == 'Classification':
    st.write('Classification models')