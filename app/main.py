from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from app.pipeline import train_model, predict

app = FastAPI()

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "ML FastAPI Pipeline Running"}


# Upload dataset and train model
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = train_model(file_path)

    return {
        "message": "Dataset uploaded successfully",
        "training_status": result
    }


# Predict endpoint
@app.post("/predict")
async def make_prediction(features: str = Form(...)):
    try:
        # Convert input string to list of floats
        values = list(map(float, features.split(",")))

        result = predict(values)

        return {
            "prediction": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }
