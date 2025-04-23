from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

app = FastAPI()

# Load FinBERT model
print("Loading FinBERT ESG model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
print("Model loaded.")

class ESGInput(BaseModel):
    text: str

@app.post("/esg")
async def analyze_esg(input: ESGInput):
    try:
        result = classifier(input.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
