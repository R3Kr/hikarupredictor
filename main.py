from enum import Enum
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pickle

class TimeControls(str, Enum):
    BLITZ = "blitz"
    BULLET = "bullet"
    RAPID = "rapid"

# Set up templates
templates = Jinja2Templates(directory="templates")

app = FastAPI()
# Load the model
with open('model1.pkl', 'rb') as f:
    pipeline = pickle.load(f)
    model: XGBClassifier = pipeline['model']
    le: LabelEncoder = pipeline['label_encoder']
    feature_names = pipeline['feature_names']

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "opponentRating": 2000
        }
    )



@app.post("/predict", response_class=HTMLResponse)
async def prediciton(request: Request, opponentRating: int = Form(...), gameTimeClass: TimeControls  = Form(...), isTournament: bool = Form(False)):
    input_df = pd.DataFrame([{gameTimeClass.value: 1, "opponentRating": opponentRating, "isTournament": isTournament}], columns=feature_names).fillna(0)
    probs = model.predict_proba(input_df)[0]
    output = {}
    for i in range(len(probs)):
        output[le.inverse_transform([i])[0]] = probs[i]
    # prediction = {"predict": le.inverse_transform([probs.argmax()])[0], "prob": probs}

    return templates.TemplateResponse(
        "prediction.html",
        {
            "request": request,
            "output": output,
            "opponentRating": opponentRating,
            "gameTimeClass": gameTimeClass,
            "isTournament": isTournament
        }
    )