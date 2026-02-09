from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import requests
import os
from dotenv import load_dotenv
from model.detector import predict_audio
from audio.preprocess import load_audio

# Checking API status
"""
app = FastAPI()
@app.post("/predict")
def predict():
    return {"status": "API working"}
"""

app = FastAPI()
load_dotenv()
API_KEY = os.getenv("API_KEY")

if API_KEY is None:
    raise RuntimeError("API_KEY not configured on server")

languages = {
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
}

#Request
class VoiceDetectionRequest(BaseModel):
    language: str | None = None
    audioFormat: str
    audioBase64: str | None = None
    audioUrl: str | None = None  # for MP3 URL
    message: str | None = None  
    
    class Config:
        extra = "allow"

# Response
class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

# endpoint
@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None, alias="x-api-key"),
    authorization: str = Header(None)
):
    # API key validation
    api_key = None

    if x_api_key:
        api_key = x_api_key

    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")

    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    # language validation
    if request.language and request.language not in languages:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language"
        )

    # audio format validation
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 format is supported"
        )

    # base64 validation
    audio_bytes = None

    if request.audioBase64:
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid Base64 audio data"
            )
    
    elif request.audioUrl:
        try:
            response = requests.get(request.audioUrl, timeout=10)
            response.raise_for_status()
            audio_bytes = response.content
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Unable to download audio from URL"
            )

    else:
        raise HTTPException(
            status_code=400,
            detail="audioBase64 or audioUrl must be provided"
        )

    # plugging ML model
    audio_tensor = load_audio(audio_bytes)
    result = predict_audio(audio_tensor)

    if "error" in result:
        raise HTTPException(
            status_code=400,
            detail=result["error"]
        )

    classification = result["classification"]
    confidence = float(result["ai_probability"])
    explanation = result["explanation"]

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }