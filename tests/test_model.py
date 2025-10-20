import pandas as pd
from src.model import build_model, save_model, load_model
import joblib
import os

LABEL_COL = 'num'          # <<<< change here

def test_shape():
    model = build_model()
    df = pd.read_csv(r"D:/Work/Atomcamp/MLops/Project/heart_disease_uci.csv")
    X = df.drop(LABEL_COL, axis=1)   # <<<<
    assert X.shape[1] == 15          # update expected number if needed

def test_train_save_load():
    model = build_model()
    df = pd.read_csv(r"D:/Work/Atomcamp/MLops/Project/heart_disease_uci.csv")
    X = df.drop(LABEL_COL, axis=1)   # <<<<
    y = df[LABEL_COL]                # <<<<
    model.fit(X, y)
    save_model(model, "tmp.joblib")
    m2 = load_model("tmp.joblib")
    assert m2.predict(X.iloc[:5]).shape == (5,)
    os.remove("tmp.joblib")