from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import pipeline
import tempfile
import os
import langid  # 🔥 Add this for language detection

app = FastAPI()

# Load ClimateBERT models
print("Loading analysis model...")
model = pipeline("text-classification", model="ProsusAI/finbert")
print("Model loaded.")

class AnalysisInput(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(input: AnalysisInput):
    try:
        result = model(input.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Process file here
        result = {"filename": file.filename, "status": "uploaded"}
        
        os.remove(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)