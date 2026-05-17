#!/usr/bin/env python3
"""
text_classifier.py — Universelle Text-Klassifikation
=====================================================
End-to-End Fine-Tuning mit HuggingFace. Lokal, einfach, genau.

Nutzung:
  python3 scripts/text_classifier.py --train              # Trainieren
  python3 scripts/text_classifier.py --predict "Text"     # Vorhersage

Datenformat (--train):
  data/mein_dataset.txt  →  satz<TAB>label  (label = Zahl 0, 1, 2, …)
  
Für Bloom siehe: data/blooms/translated/de_bloom_classifier.txt
"""

# ============================================================
# KONFIGURATION — hier anpassen für dein Projekt
# ============================================================
CONFIG = {
    # --- Daten ---
    "data_path": "data/blooms/translated/de_bloom_classifier.txt",
    "data_delimiter": "\t",       # Trennzeichen zwischen Text und Label

    # --- Modell ---
    "model_name": "bert-base-german-cased",  # Oder: "bert-base-multilingual-cased", "xlm-roberta-base"
    "output_path": "models/text_classifier",

    # --- Hyperparameter ---
    "num_labels": 6,              # Anzahl Klassen (z.B. 6 für Bloom, 2 für Sentiment)
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 3e-5,
    "max_seq_length": 128,
    "early_stopping_patience": 2, # Training stoppen wenn keine Verbesserung

    # --- Labels (optional, für --predict) ---
    "label_names": ["Erinnern", "Verstehen", "Anwenden", "Analysieren", "Bewerten", "Erschaffen"],

    # --- Hardware ---
    "device": "auto",             # "auto", "mps", "cpu"
}

BIN = CONFIG  # Kurzform

import os
import sys
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch


# ============================================================
# DATEN
# ============================================================
def load_data(path, delimiter="\t", num_labels=6):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter, 1)
            if len(parts) == 2:
                label = int(parts[1].strip())
                if 0 <= label < num_labels:
                    texts.append(parts[0].strip())
                    labels.append(label)
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ============================================================
# TRAINING
# ============================================================
def train():
    print("=" * 60)
    print(f"Klassifikation — Training")
    print(f"Modell: {BIN['model_name']}")
    print(f"Daten:  {BIN['data_path']}")
    print("=" * 60)

    # Daten laden
    texts, labels = load_data(BIN["data_path"], BIN["data_delimiter"], BIN["num_labels"])
    print(f"\nSamples: {len(texts)}")
    for i in range(BIN["num_labels"]):
        name = BIN["label_names"][i] if i < len(BIN["label_names"]) else str(i)
        count = labels.count(i)
        print(f"  {name:<12} {count:>5} ({count/len(labels)*100:.0f}%)")

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"\nTrain: {len(train_texts)} | Val: {len(val_texts)}")

    # Modell laden
    tokenizer = AutoTokenizer.from_pretrained(BIN["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        BIN["model_name"], num_labels=BIN["num_labels"]
    )

    # Device
    if BIN["device"] == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = BIN["device"]
    model.to(device)
    print(f"Device: {device}")

    # Tokenisierung
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length",
                         truncation=True, max_length=BIN["max_seq_length"])

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize, batched=True)

    # Training
    args = TrainingArguments(
        output_dir=BIN["output_path"],
        num_train_epochs=BIN["epochs"],
        per_device_train_batch_size=BIN["batch_size"],
        per_device_eval_batch_size=BIN["batch_size"],
        learning_rate=BIN["learning_rate"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(BIN["early_stopping_patience"])],
    )

    trainer.train()
    trainer.save_model(BIN["output_path"])
    tokenizer.save_pretrained(BIN["output_path"])

    # Finale Metriken
    metrics = trainer.evaluate()
    print(f"\n{'='*60}")
    print(f"Beste Accuracy: {metrics['eval_accuracy']:.1%}")
    print(f"Bester F1 (macro): {metrics['eval_f1_macro']:.3f}")
    print(f"{'='*60}")
    print(f"\nModell: {BIN['output_path']}/")


# ============================================================
# PREDICTION
# ============================================================
def predict(texts):
    if not os.path.exists(BIN["output_path"]):
        print(f"Modell nicht gefunden: {BIN['output_path']}")
        print("Führe zuerst --train aus.")
        return

    if isinstance(texts, str):
        texts = [texts]

    tokenizer = AutoTokenizer.from_pretrained(BIN["output_path"])
    model = AutoModelForSequenceClassification.from_pretrained(BIN["output_path"])
    device = "mps" if torch.backends.mps.is_available() and BIN["device"] != "cpu" else "cpu"
    model.to(device)
    model.eval()

    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=BIN["max_seq_length"]).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1).tolist()
    probs = torch.softmax(logits, dim=-1)

    print(f"\n{'='*60}")
    for i, (text, pred) in enumerate(zip(texts, preds)):
        name = BIN["label_names"][pred] if pred < len(BIN["label_names"]) else str(pred)
        conf = probs[i][pred].item()
        print(f"  → {name} ({conf:.0%})  |  {text[:70]}")
    print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Nutzung:")
        print("  python3 scripts/text_classifier.py --train")
        print("  python3 scripts/text_classifier.py --predict 'Dein Text hier'")
        return

    cmd = sys.argv[1]
    if cmd == "--train":
        train()
    elif cmd == "--predict":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        if text:
            predict(text)
        else:
            # Interactive mode
            print("Gib Texte ein (Enter = predict, Ctrl+C = Ende):")
            for line in sys.stdin:
                line = line.strip()
                if line:
                    predict(line)
    else:
        print(f"Unbekannt: {cmd}")


if __name__ == "__main__":
    main()
