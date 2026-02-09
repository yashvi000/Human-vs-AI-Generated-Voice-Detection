import sys
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from audio.preprocess import load_audio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, "VALIDATION")
device = torch.device("cpu")

# Load WavLM Base+ (pretrained)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base-plus"
)

wavlm = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus"
).to(device)

wavlm.eval()

# Freeze WavLM
for p in wavlm.parameters():
    p.requires_grad = False


# Classifier (same architecture as training)
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


CLASSIFIER_PATH = os.path.join(
    PROJECT_ROOT, "model", "classifier.pth"
)

classifier = AIHumanClassifier().to(device)

if not os.path.exists(CLASSIFIER_PATH):
    raise RuntimeError("classifier.pth not found. Train classifier first.")

classifier.load_state_dict(
    torch.load(CLASSIFIER_PATH, map_location=device)
)

classifier.eval()
THRESHOLD = 0.50 

# Load validation dataset
def load_validation_data(root_dir):
    data = []

    for class_name in ["HUMAN", "AI_GENERATED"]:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        label = 0 if class_name == "HUMAN" else 1

        for language in os.listdir(class_dir):
            lang_dir = os.path.join(class_dir, language)
            if not os.path.isdir(lang_dir):
                continue

            for file in os.listdir(lang_dir):
                if file.lower().endswith((".wav", ".mp3")):
                    path = os.path.join(lang_dir, file)
                    audio_tensor = load_audio(path)
                    if audio_tensor is not None:
                        data.append((audio_tensor, label))

    return data

# Run validation
def run_validation():
    dataset = load_validation_data(DATASET_PATH)
    print(f"Loaded {len(dataset)} validation samples")

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for audio_tensor, label in dataset:
            inputs = feature_extractor(
                audio_tensor.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )

            input_values = inputs.input_values.to(device)
            outputs = wavlm(input_values)
            pooled = outputs.last_hidden_state.mean(dim=1)
            logit = classifier(pooled)
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= THRESHOLD else 0

            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    tn, fp, fn, tp = cm.ravel()

    print("\n\t\tVALIDATION RESULTS")
    print(f"Samples           : {len(y_true)}")
    print(f"Accuracy          : {acc * 100:.2f}%")
    print(f"ROC-AUC           : {auc:.3f}")
    print(f"False Negatives   : {fn}  (AI misclassified as HUMAN)")
    print(f"False Positives   : {fp}")
    print("Confusion Matrix:")
    print(cm)

    # Threshold check
    print("\n\t\tThreshold check")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        preds = [1 if p >= t else 0 for p in y_prob]
        acc_t = accuracy_score(y_true, preds)
        print(f"Threshold {t:.2f} → Accuracy {acc_t*100:.2f}%")

# Entry point
if __name__ == "__main__":
    run_validation()