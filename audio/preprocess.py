import io
from pathlib import Path

import librosa
import torch
import numpy as np
import yaml

from logger import get_logger

logger = get_logger(__name__, "preprocess.log")

# Load config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

preprocess_config = config["preprocess"]

MIN_DURATION_S = float(preprocess_config["min_duration_sec"])
RMS_THRESHOLD = float(preprocess_config["rms_threshold"])
CLIP_PCT_MAX = float(preprocess_config["clip_pct_max"])
CLIP_SAMPLE_THRESHOLD = float(preprocess_config["clip_sample_threshold"])
DEFAULT_SR = int(preprocess_config["target_sample_rate"])

logger.info(
    f"Preprocess config loaded | SR : {DEFAULT_SR} | "
    f"Min duration : {MIN_DURATION_S}s | "
    f"RMS : {RMS_THRESHOLD} | Clip % : {CLIP_PCT_MAX}"
)


def looks_like_usable_audio(audio):
    """Check audio is usable."""
    if audio is None:
        return False

    audio = np.asarray(audio)

    if audio.size == 0:
        return False

    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        return False

    if np.all(audio == 0):
        return False

    return True


def _pydub_decode(audio_bytes, sample_rate):
    """Fallback decode with pydub."""
    from pydub import AudioSegment

    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    segment = segment.set_frame_rate(sample_rate).set_channels(1)

    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    denom = float(1 << (8 * segment.sample_width - 1))

    if denom <= 0:
        raise RuntimeError("Invalid sample width from decoder")

    audio = samples / denom
    return audio.astype(np.float32), sample_rate


def decode_audio_bytes(audio_bytes, sample_rate=None):
    """Decode bytes to mono float32."""
    if sample_rate is None:
        sample_rate = DEFAULT_SR

    if not audio_bytes:
        logger.error("Empty audio payload received")
        raise RuntimeError("Empty audio payload")

    audio, sr = None, sample_rate
    librosa_ok = False

    # Try librosa first
    try:
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=sample_rate,
            mono=True,
        )

        if looks_like_usable_audio(audio):
            librosa_ok = True
            logger.info(
                f"Decoded with librosa | SR : {sr} | Samples : {len(audio)}")
        else:
            logger.warning(
                "librosa returned empty/silent audio | trying pydub")

    except Exception as e:
        logger.warning(f"librosa.load failed : {e} | trying pydub")

    # Fallback to pydub
    if not librosa_ok:
        try:
            audio, sr = _pydub_decode(audio_bytes, sample_rate)
            logger.info(
                f"Decoded with pydub fallback | SR : {sr} | Samples : {len(audio)}"
            )

        except Exception as e:
            logger.error(f"Both decoders failed : {e}")
            raise RuntimeError(
                f"Failed to decode audio (is ffmpeg installed?): {e}"
            ) from e

    return np.asarray(audio, dtype=np.float32), int(sr)


def validate_audio(audio, sr):
    """Validate audio."""
    if audio is None or len(np.atleast_1d(audio)) == 0:
        return False, "Audio file is empty: no data detected"

    audio = np.asarray(audio, dtype=np.float32)

    duration = librosa.get_duration(y=audio, sr=sr)
    if duration < MIN_DURATION_S:
        return False, (
            f"Audio too short: {duration:.2f}s "
            f"(minimum {MIN_DURATION_S}s required)"
        )

    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        return False, "Audio file corrupted: contains NaN or Inf values"

    if np.all(audio == 0):
        return False, "Audio file is empty: no data detected"

    rms = librosa.feature.rms(y=audio)[0]
    mean_rms = float(np.mean(rms))
    if mean_rms < RMS_THRESHOLD:
        return False, (
            f"No sound detected: RMS energy {mean_rms:.6f} "
            f"below threshold ({RMS_THRESHOLD})"
        )

    clipped_samples = int(np.sum(np.abs(audio) > CLIP_SAMPLE_THRESHOLD))
    clipping_pct = (clipped_samples / len(audio)) * 100.0
    if clipping_pct > CLIP_PCT_MAX:
        return False, (
            f"Audio severely clipped: {clipping_pct:.1f}% of samples "
            f"exceed {CLIP_SAMPLE_THRESHOLD} amplitude"
        )

    return True, "Valid audio"


def load_audio(file_path, sample_rate=None):
    """Entry point for file path."""
    if sample_rate is None:
        sample_rate = DEFAULT_SR

    logger.info(f"Loading audio from file : {file_path}")

    try:
        with open(file_path, "rb") as fh:
            audio_bytes = fh.read()

    except Exception as e:
        logger.error(f"[Audio Read Error] {file_path} : {e}")
        return None

    try:
        audio, sr = decode_audio_bytes(audio_bytes, sample_rate)
    except Exception as e:
        logger.error(f"[Audio Decode Error] {file_path} : {e}")
        return None

    is_valid, error_message = validate_audio(audio, sr)

    if not is_valid:
        logger.warning(f"[Validation Failed] {file_path} : {error_message}")
        return None

    logger.info(f"Audio loaded successfully : {file_path}")
    return torch.tensor(audio, dtype=torch.float32)


def load_audio_from_bytes(audio_bytes, sample_rate=None):
    """Entry point for raw bytes."""
    if sample_rate is None:
        sample_rate = DEFAULT_SR

    logger.info("Loading audio from raw bytes")

    try:
        audio, sr = decode_audio_bytes(audio_bytes, sample_rate)
    except Exception as e:
        logger.error(f"[Audio Decode Error] : {e}")
        return None

    is_valid, error_message = validate_audio(audio, sr)

    if not is_valid:
        logger.warning(f"[Validation Failed] : {error_message}")
        return None

    logger.info("Audio bytes loaded successfully")
    return torch.tensor(audio, dtype=torch.float32)


def preprocess_audio(audio_bytes=None, sample_rate=None):
    """Main function — calls decode and validation."""
    if sample_rate is None:
        sample_rate = DEFAULT_SR

    if audio_bytes is None:
        logger.warning("No audio bytes provided to main()")
        return False, None, "No audio data provided"

    logger.info("Starting audio decode and validation")

    try:
        audio, sr = decode_audio_bytes(audio_bytes, sample_rate)

    except Exception as e:
        logger.error(f"Decode failed : {e}")
        return False, None, str(e)

    is_valid, error_message = validate_audio(audio, sr)

    if not is_valid:
        logger.warning(f"Validation failed : {error_message}")
        return False, None, error_message

    duration = len(audio) / sr
    logger.info(f"Validation passed | Duration : {duration:.2f}s")

    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    return True, audio_tensor, "Valid audio"


if __name__ == "__main__":
    preprocess_audio()
