#!/usr/bin/env python3
"""
train_embedding.py — Universelles Sentence-Transformer Training
==============================================================
Ein Skript für zwei Aufgaben:
  task: "simcse"         → Embedding-Modell per SimCSE fine-tunen
  task: "classifier"     → Classifier auf Embedding trainieren

So wirds verwendet:
  1. CONFIG anpassen (Modell, Task, Datenpfad, Hyperparameter)
  2. python3 scripts/train_embedding.py

Modell und Dataset sind durch CONFIG vollständig austauschbar.
"""

# ============================================================
# KONFIGURATION — alles was du brauchst, hier anpassen
# ============================================================
CONFIG = {
    # --- Modell (V2 als Basis für Classifier) ---
    "model_name": "models/V2",
    "output_path": "models/V2BC_mlp",

    # --- Task ---
    "task": "classifier",

    # --- Daten ---
    "data_path": "data/blooms/translated/de_bloom_classifier.txt",
    "data_delimiter": "\t",

    # --- Hyperparameter ---
    "batch_size": 64,
    "epochs": 1,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_seq_length": 128,

    # --- Classifier (für task: classifier) ---
    "classifier_method": "mlp",

    # --- Contrastive Loss (nur für simcse relevant) ---
    "contrastive_scale": 10,

    # --- Hardware ---
    "device": "cpu",
}

import os
import sys
import logging
import random
import json
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
except ModuleNotFoundError:
    from sentence_transformers.losses import MultipleNegativesRankingLoss
from tqdm import tqdm


# ============================================================
# SETUP
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/train_{datetime.now():%Y%m%d_%H%M%S}.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logger = logging.getLogger()


def get_device(cfg: dict) -> torch.device:
    preference = cfg.get("device", "auto")
    if preference == "mps":
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# DATASETS
# ============================================================
class SimCSEDataset(Dataset):
    """Liest eine Textdatei (ein Satz pro Zeile). Jeder Satz wird
    mit sich selbst gepaart — das Dropout im Transformer erzeugt
    unterschiedliche Embeddings, die der Contrastive-Loss zusammenzieht."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f if line.strip()]
        logger.info(f"  SimCSE-Dataset: {len(self.sentences)} Sätze geladen")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]
        return s, s  # (anchor, positive) = gleicher Text


class ClassificationDataset(Dataset):
    """Liest eine TSV-Datei: satz<tab>label
    Gibt (satz, label) zurück."""

    def __init__(self, path: str, delimiter: str = "\t"):
        self.pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(delimiter, 1)
                if len(parts) == 2:
                    self.pairs.append((parts[0].strip(), int(parts[1].strip())))
        logger.info(f"  Classification-Dataset: {len(self.pairs)} Paare geladen")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ============================================================
# SIMCSE-TRAINING
# ============================================================
def train_simcse(model: SentenceTransformer, cfg: dict):
    logger.info("  >>> SIMCSE-TRAINING")
    logger.info(f"      Modell: {cfg['model_name']}")
    logger.info(f"      Daten:  {cfg['data_path']}")
    logger.info(f"      Batch:  {cfg['batch_size']} | Epochen: {cfg['epochs']}")

    dataset = SimCSEDataset(cfg["data_path"])
    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        collate_fn=lambda batch: list(zip(*batch)),
    )
    loss_fn = MultipleNegativesRankingLoss(model, scale=cfg.get("contrastive_scale", 20.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    total_steps = len(dataloader) * cfg["epochs"]
    warmup = int(total_steps * cfg["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, (s + 1) / warmup) if s < warmup else 1.0,
    )

    device = get_device(cfg)
    model.to(device)
    loss_fn.to(device)
    logger.info(f"      Device: {device} | {len(dataloader)} Batches/Epoche")
    logger.info(f"      Start: {datetime.now():%H:%M:%S}")

    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        n = 0
        epoch_start = time.time()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False)
        for anchors, positives in loop:
            feat_a = model.preprocess(list(anchors)).to(device)
            feat_b = model.preprocess(list(positives)).to(device)
            sentence_features = [feat_a, feat_b]
            labels = torch.tensor(0, device=device)

            loss = loss_fn(sentence_features, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n += 1

            if n == 1:
                first_batch = time.time()
            if n % 100 == 0:
                elapsed = time.time() - epoch_start
                batch_speed = n / elapsed
                remaining = (len(dataloader) - n) / batch_speed
                eta = datetime.fromtimestamp(time.time() + remaining).strftime("%H:%M")
                logger.info(
                    f"  Epoch {epoch+1}, Batch {n}/{len(dataloader)}: "
                    f"Loss={loss.item():.4f} | "
                    f"{batch_speed:.1f} Bat/s | "
                    f"ETA: {eta}"
                )
            loop.set_postfix(loss=f"{loss.item():.4f}")

            del feat_a, feat_b, sentence_features, labels, loss

        avg = total_loss / max(n, 1)
        logger.info(f"  Epoch {epoch+1}: Avg Loss = {avg:.4f}")

    logger.info("  SimCSE-Training abgeschlossen.")


# ============================================================
# CLASSIFIER-TRAINING
# ============================================================
def train_classifier(model: SentenceTransformer, cfg: dict):
    logger.info("  >>> CLASSIFIER-TRAINING")
    logger.info(f"      Modell: {cfg['model_name']}")
    logger.info(f"      Daten:  {cfg['data_path']}")

    dataset = ClassificationDataset(cfg["data_path"], cfg["data_delimiter"])
    texts = [p[0] for p in dataset.pairs]
    labels = [p[1] for p in dataset.pairs]
    n_classes = max(labels) + 1

    device = get_device(cfg)
    model.to(device)

    logger.info(f"      {len(texts)} Samples | {n_classes} Klassen")
    logger.info(f"      Extrahiere Embeddings...")

    model.eval()
    embeddings = []
    batch_size = min(64, cfg["batch_size"] * 4)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    logger.info(f"      Embeddings: {embeddings.shape}")

    # Classifier nach Konfiguration
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    method = cfg.get("classifier_method", "logistic")
    if method == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif method == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=2000,
            early_stopping=True, random_state=42
        )
    elif method == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        )
    else:
        raise ValueError(f"Unbekannte Classifier-Methode: {method}")

    logger.info(f"      Classifier: {type(clf).__name__}")
    scores = cross_val_score(clf, embeddings, labels, cv=min(5, len(set(labels))))
    logger.info(f"      Cross-Validation Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    clf.fit(embeddings, labels)

    # Classifier speichern
    import joblib
    os.makedirs(cfg["output_path"], exist_ok=True)
    joblib.dump(clf, os.path.join(cfg["output_path"], "classifier.joblib"))
    with open(os.path.join(cfg["output_path"], "classifier_meta.json"), "w") as f:
        json.dump({"n_classes": n_classes, "model": cfg["model_name"]}, f)
    logger.info(f"      Classifier gespeichert unter {cfg['output_path']}/")


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CONFIG
    logger.info("=" * 50)
    logger.info(f"Task: {cfg['task']}")
    logger.info(f"Modell: {cfg['model_name']}")
    logger.info("=" * 50)

    if cfg["task"] == "simcse":
        model = SentenceTransformer(cfg["model_name"], trust_remote_code=True)
        model.max_seq_length = cfg["max_seq_length"]
        train_simcse(model, cfg)
        os.makedirs(cfg["output_path"], exist_ok=True)
        model.save(cfg["output_path"])
        logger.info(f"Modell gespeichert: {cfg['output_path']}")

    elif cfg["task"] == "classifier":
        model = SentenceTransformer(cfg["model_name"], trust_remote_code=True)
        model.max_seq_length = cfg["max_seq_length"]
        train_classifier(model, cfg)

    else:
        logger.error(f"Unbekannter Task: {cfg['task']}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Fertig.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
