import io
import librosa
import torch


# Load audio from file path
def load_audio(file_path, sample_rate=16000):
    try:
        audio, _ = librosa.load(
            file_path,
            sr=sample_rate,
            mono=True
        )
        return torch.tensor(audio, dtype=torch.float32)

    except Exception as e:
        print(f"[Audio Load Error] {file_path}: {e}")
        return None


# Load audio from raw bytes (API use)
def load_audio_from_bytes(audio_bytes, sample_rate=16000):
    try:
        audio, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=sample_rate,
            mono=True
        )
        return torch.tensor(audio, dtype=torch.float32)

    except Exception as e:
        print("[Audio Decode Error]:", e)
        return None
