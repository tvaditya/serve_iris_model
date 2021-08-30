import joblib
from fastapi import FastAPI
from typing import List
import uvicorn

app = FastAPI()

joblib_file = "joblib_RL_Model.pkl"
model = joblib.load(joblib_file)


def classifier_iris(model, values):
    prediction = model.predict([values])
    return prediction


def run_server():
    uvicorn.run(app)


@app.get('/')
def get_root():
	return {'message': 'Iris classifier'}


@app.get('/classify/{values}')
async def iris_classify(values):
    return str(classifier_iris(model, eval(values)))