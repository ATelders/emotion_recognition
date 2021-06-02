from fastapi import FastAPI

app = FastAPI()


@app.get("/{sentence}")
async def root(sentence):

    return {"message": "happiness"}

