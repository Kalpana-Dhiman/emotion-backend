from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load emotion detection model
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_emotion(data: TextInput):
    outputs = emotion_analyzer(data.text)

    # Handle different pipeline output formats safely
    if isinstance(outputs, list) and isinstance(outputs[0], list):
        emotions = outputs[0]
    else:
        emotions = outputs

    top_emotion = max(emotions, key=lambda x: x["score"])

    return {
        "emotion": top_emotion["label"],
        "score": round(float(top_emotion["score"]), 2)
    }
import os

port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)

