import os
import sys
import torch
import torch.nn as nn
import numpy as np

from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from audio.preprocess import load_audio_from_bytes

# --------------------------------------------------
# FIX IMPORT PATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cpu")

# --------------------------------------------------
# LOAD WAVLM BASE+ (FROZEN)
# --------------------------------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base-plus"
)

wavlm = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus"
).to(device)

wavlm.eval()
for p in wavlm.parameters():
    p.requires_grad = False


# --------------------------------------------------
# CLASSIFIER (TRAINED)
# --------------------------------------------------
class AIHumanClassifier(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


classifier = AIHumanClassifier().to(device)

CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "model", "classifier.pth")

if not os.path.exists(CLASSIFIER_PATH):
    raise RuntimeError("classifier.pth not found. Train the classifier first.")

classifier.load_state_dict(
    torch.load(CLASSIFIER_PATH, map_location=device)
)
classifier.eval()


# --------------------------------------------------
# THRESHOLD (FN IS WORSE → LOWER THRESHOLD)
# --------------------------------------------------
THRESHOLD = 0.50   # tune later after validation


# --------------------------------------------------
# MAIN PREDICTION FUNCTION
# --------------------------------------------------
def predict_audio(audio_bytes: bytes):
    """
    audio_bytes: raw bytes from MP3/WAV
    returns dict compatible with app.py
    """

    waveform = load_audio_from_bytes(audio_bytes)

    if waveform is None or waveform.numel() == 0:
        return {"error": "Invalid or empty audio input"}

    # Limit length for latency (max 12 sec)
    max_len = 16000 * 12
    if waveform.numel() > max_len:
        waveform = waveform[:max_len]

    with torch.no_grad():
        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(device)

        outputs = wavlm(input_values)
        hidden_states = outputs.last_hidden_state  # [1, T, 768]

        pooled = hidden_states.mean(dim=1)         # [1, 768]

        logit = classifier(pooled)
        prob = torch.sigmoid(logit).item()

    classification = (
        "AI_GENERATED" if prob >= THRESHOLD else "HUMAN"
    )

    explanation = (
        "Synthetic speech artifacts and consistency patterns detected"
        if classification == "AI_GENERATED"
        else "Natural human speech variability detected"
    )

    return {
        "classification": classification,
        "ai_probability": round(prob, 3),
        "explanation": explanation
    }