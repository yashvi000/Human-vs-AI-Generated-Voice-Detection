from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import requests
import os
from dotenv import load_dotenv
from model.detector import predict_audio
from audio.preprocess import preprocess_audio
from pathlib import Path
import yaml
import time
from logger import get_logger

logger = get_logger(__name__, "api.log")

load_dotenv()
API_KEY = os.getenv("API_KEY")


# Loading config and paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

LANGUAGES = config["api"]["languages"]
TIMEOUT_SEC = config["api"]["timeout_sec"]


app = FastAPI()

if API_KEY is None:
    raise RuntimeError("API_KEY not configured on server")
logger.info("API key loaded successfully")

#Request
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str | None = None
    audioUrl: str | None = None  # for MP3 URL
 

# Response
class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str


# Checking API status
@app.get("/health")
def health_check():
    logger.info("Healthcheck called")
    return {"status" : "API working"}


# endpoint
@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: str | None = Header(None, alias="x-api-key"),
    authorization: str | None = Header(None)
):

    start_time = time.time()
    source_type = "base64" if request.audioBase64 else "MP3 url" if request.audioUrl else "none"
    logger.info(f"Request Received | Language : {request.language} | Source : {source_type}")

    # API key validation
    api_key = None

    if x_api_key:
        api_key = x_api_key

    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")

    if api_key != API_KEY:
        logger.warning(f"Authentication failed | Key provided : {'yes' if api_key else 'no'}")

        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    logger.info("Authentication Successful")

    # language validation
    if request.language not in LANGUAGES:
        logger.warning(f"Unsupported language : {request.language}")

        raise HTTPException(
            status_code=400,
            detail="Unsupported language"
        )

    # audio format validation
    if request.audioFormat.lower() != "mp3":
        logger.warning(f"Unsupported format : {request.audioFormat}")

        raise HTTPException(
            status_code=400,
            detail="Only MP3 format is supported"
        )

    # base64 validation
    audio_bytes = None

    if request.audioBase64 and request.audioUrl:
        logger.warning("Both audioBase64 and audioUrl provided | Using audioBase64")

    if request.audioBase64:
        try:
            logger.info(f"Decoding Base64 audio | Input length : {len(request.audioBase64)} chars")
            b64 = request.audioBase64
            b64 += "=" * ((4 - len(b64) % 4) % 4)   # padding
            audio_bytes = base64.b64decode(b64)
            logger.info(f"Base64 decoded | Size : {len(audio_bytes)} bytes")

        except Exception as e:
            logger.error(f"Base64 decode failed : {e}")

            raise HTTPException(
                status_code=400,
                detail="Invalid Base64 audio data"
            )
    
    elif request.audioUrl:
        try:
            logger.info(f"Downloading audio from URL: {request.audioUrl}")
            response = requests.get(request.audioUrl, timeout=TIMEOUT_SEC)
            response.raise_for_status()
            audio_bytes = response.content
            logger.info(f"URL download complete | Size : {len(audio_bytes)} bytes")

        except Exception as e:
            logger.error(f"URL download failed : {e}")

            raise HTTPException(
                status_code=400,
                detail="Unable to download audio from URL"
            )

    else:
        logger.warning("No audio source provided")
        raise HTTPException(
            status_code=400,
            detail="audioBase64 or audioUrl must be provided"
        )

    # Audio validation and preprocessing
    is_valid, audio_tensor, error_message = preprocess_audio(audio_bytes)

    if not is_valid:
        logger.error(f"Audio validation failed : {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    logger.info("Audio validation passed")

    logger.info("Running model inference")

    try:
        result = predict_audio(audio_tensor)

    except Exception as e:
        logger.exception(f"Model inference crashed : {e}")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed"
        )

    if "error" in result:
        logger.error(f"Model returned error : {result['error']}")
        raise HTTPException(
            status_code=400,
            detail=result["error"]
        )

    classification = result["classification"]
    confidence = float(result["confidence"])
    explanation = result["explanation"]

    elapsed = time.time() - start_time
    logger.info(
        f"Request Complete | "
        f"Classification : {classification} | "
        f"Confidence : {confidence:.3f} | "
        f"Language : {request.language} | "
        f"Time : {elapsed:.2f}s"
    )
    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }