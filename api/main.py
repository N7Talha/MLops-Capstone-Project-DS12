import os, wandb, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.model import load_model

# 1. download latest model at start
run = wandb.init(project="MLops_capstone", job_type="inference")
artifact = run.use_artifact('sk-model:latest', type='model')
model_path = artifact.download()
MODEL = load_model(f"{model_path}/model.joblib")
run.finish()

# 2. input schema
class Patient(BaseModel):
    age:int; trestbps:int; chol:int; thalach:int; oldpeak:float
    sex:int; cp:int; fbs:int; restecg:int; exang:int
    slope:int; ca:int; thal:int

app = FastAPI(title="Heart-Disease-API")

@app.post("/predict")
def predict(p: Patient):
    df = pd.DataFrame([p.dict()])
    pred = MODEL.predict(df)[0]
    prob = MODEL.predict_proba(df)[0,1]
    return {"prediction": int(pred), "probability": float(prob)}

@app.get("/health")
def health():
    return {"status":"ok"}