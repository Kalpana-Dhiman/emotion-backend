from fastapi import FastAPI
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_emotion(data: TextInput):
    scores = analyzer.polarity_scores(data.text)

    if scores["compound"] >= 0.05:
        emotion = "joy"
    elif scores["compound"] <= -0.05:
        emotion = "sadness"
    else:
        emotion = "neutral"

    return {
        "emotion": emotion,
        "confidence": abs(scores["compound"])
    }
