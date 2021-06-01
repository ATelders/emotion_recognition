# Emotion Recognition App

Emotion Recognition is an app made with Streamlit in Python, which purpose is to test different machine learning and deep learning algorithms on two datasets, to find emotions in text.

https://share.streamlit.io/atelders/emotion_recognition/main/src/main.py

## Datasets

Two datasets were used:

From Kaggle : Emotion_final.csv
From data.world : text_emotion.csv

## Source Code

Source code of the app is in src/main.py

## File Structure

'''bash
├── assets
│   ├── frequency_kaggle.png
│   └── frequency_world.png
├── data
│   ├── models
│   │   ├── dt_kaggle_count.sav
│   │   ├── dt_kaggle_tfidf.sav
│   │   ├── dt_world_count.sav
│   │   ├── dt_world_tfidf.sav
│   │   ├── lr_kaggle_count.sav
│   │   ├── lr_kaggle_tfidf.sav
│   │   ├── lr_world_count.sav
│   │   ├── lr_world_tfidf.sav
│   │   ├── svm_kaggle_count.sav
│   │   ├── svm_kaggle_tfidf.sav
│   │   ├── svm_world_count.sav
│   │   └── svm_world_tfidf.sav
│   └── raw
│       ├── Emotion_final.csv
│       └── text_emotion.csv
├── environment.yml
├── fasttext_data.world.bin
├── fasttext_data.world binary.bin
├── fasttext_Kaggle.bin
├── LICENSE
├── notebooks
│   ├── 20210517-at-cs-jd-initial-data-exploration.ipynb
│   ├── neurons.ipynb
│   ├── Random_forest.ipynb
│   └── words_frequency.ipynb
├── README.md
├── src
│   ├── __init__.py
│   ├── main.py
│   └── words_frequency.py
├── test.txt
└── train.txt
'''