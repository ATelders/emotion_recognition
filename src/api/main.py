from fastapi import FastAPI
import pickle

app = FastAPI()

filename = '../data/models/lr_kaggle_tfidf.sav'
classifier = pickle.load(open(filename, 'rb'))
tf = pickle.load(open('../data/models/tfidf.sav', 'rb'))

@app.get("/")
async def root():
    return {"message": "happiness"}

@app.get("/{sentence}")
def predict_emotion(sentence):
    matrix = tf.transform([sentence])
    predictions = classifier.predict(matrix)
    labels = ["happy", "sadness", "love", "anger", "fear", "surprise"]
    labels.sort()
    return {'label': labels[int(predictions)]}