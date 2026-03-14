from fastapi import FastAPI, UploadFile
import shutil
from app.pipeline import train_model, predict

app = FastAPI()

DATASET_PATH = "datasets/data.csv"


@app.post("/upload-dataset")

async def upload_dataset(file: UploadFile):

    with open(DATASET_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Dataset uploaded successfully"}



@app.post("/train")

def train():

    message = train_model(DATASET_PATH)

    return {"message": message}



@app.post("/predict")

def make_prediction(features: list):

    result = predict(features)

    return {"prediction": result}
