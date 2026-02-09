import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from audio.preprocess import load_audio

# --------------------------------------------------
# FIX IMPORT PATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATASET_PATH = os.path.join(PROJECT_ROOT, "VALIDATION")
SAVE_PATH = os.path.join(PROJECT_ROOT, "model", "classifier.pth")

BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4
THRESHOLD = 0.5

device = torch.device("cpu")

# --------------------------------------------------
# LOAD WAVLM (FROZEN)
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
# CLASSIFIER (TRAINABLE)
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

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)


# --------------------------------------------------
# DATASET
# --------------------------------------------------
class VoiceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

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
                        self.samples.append(
                            (os.path.join(lang_dir, file), label)
                        )

        print(f"Loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform = load_audio(path)

        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = wavlm(inputs.input_values.to(device))
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embedding, torch.tensor(label, dtype=torch.float32)


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
def train():
    dataset = VoiceDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    classifier.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = classifier(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(classifier.state_dict(), SAVE_PATH)
    print(f"\n✅ Classifier saved to: {SAVE_PATH}")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    train()
