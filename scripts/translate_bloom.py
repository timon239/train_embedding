#!/usr/bin/env python3
"""
translate_bloom.py — Übersetzt das englische Bloom-Dataset ins Deutsche.
===========================================================
Lädt einmal das Modell Helsinki-NLP/opus-mt-en-de und übersetzt
alle 8779 Fragen. Speichert nach jedem Batch, damit du bei Abbruch
nicht alles verlierst.

Nutzung:  python3 scripts/translate_bloom.py
Dauer:    ~30-60 min (einmalig)
"""

import os
import csv
import logging
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# ── Konfiguration ──────────────────────────────────────────
INPUT_PATH = "data/blooms/raw/en_bloom.csv"
OUTPUT_BILINGUAL = "data/blooms/translated/de_bloom_bilingual.csv"
OUTPUT_GERMAN = "data/blooms/processed/bloom_for_classifier.txt"
CHECKPOINT_PATH = "data/blooms/translated/_checkpoint.txt"
BATCH_SIZE = 32          # Sätze pro Batch (grösser = schneller, mehr RAM)
DEVICE = "mps"           # "mps" für Apple Silicon, "cpu" sonst
# ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger()


def load_data(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    log.info(f"Gelesen: {len(rows)} Zeilen aus {path}")
    return rows


def get_completed(checkpoint_path: str) -> set[int]:
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path) as f:
        return {int(line.strip()) for line in f if line.strip()}


def save_checkpoint(idx: int, checkpoint_path: str):
    with open(checkpoint_path, "a") as f:
        f.write(f"{idx}\n")


def main():
    os.makedirs(os.path.dirname(OUTPUT_BILINGUAL), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_GERMAN), exist_ok=True)

    rows = load_data(INPUT_PATH)
    completed = get_completed(CHECKPOINT_PATH)
    log.info(f"Bereits übersetzt: {len(completed)} Sätze (aus Checkpoint)")

    # Nur noch nicht übersetzte Sätze
    pending = [(i, r) for i, r in enumerate(rows) if i not in completed]
    if not pending:
        log.info("Alles bereits übersetzt. Starte neu mit komplettem Dataset.")
        pending = list(enumerate(rows))
        completed = set()

    log.info(f"Noch zu übersetzen: {len(pending)} Sätze")

    # Modell laden
    log.info("Lade Übersetzungsmodell Helsinki-NLP/opus-mt-en-de …")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    device = torch.device(DEVICE) if DEVICE == "mps" else torch.device("cpu")
    model.to(device)
    model.eval()
    log.info(f"Modell geladen auf {device}")

    # Übersetzen
    results = list(completed)  # indices
    translations = {}

    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i : i + BATCH_SIZE]
        texts = [r["Questions"] for _, r in batch]

        with torch.no_grad():
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=192).to(device)
            outputs = model.generate(**inputs, max_length=192,
                                     num_beams=2, early_stopping=True)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for (idx, row), de_text in zip(batch, decoded):
            translations[idx] = {
                "en": row["Questions"],
                "de": de_text,
                "label": row["Category"],
            }

        for idx, _, in batch:
            save_checkpoint(idx, CHECKPOINT_PATH)

        if (i // BATCH_SIZE) % 5 == 0:
            log.info(f"  Fortschritt: {min(i+BATCH_SIZE, len(pending))}/{len(pending)}")

    # Bilingual speichern
    with open(OUTPUT_BILINGUAL, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["en", "de", "category"])
        for idx in sorted(translations.keys()):
            t = translations[idx]
            writer.writerow([t["en"], t["de"], t["label"]])
    log.info(f"Bilingual gespeichert: {OUTPUT_BILINGUAL}")

    # German-only für Classifier-Training
    label_map = {"BT1": 0, "BT2": 1, "BT3": 2, "BT4": 3, "BT5": 4, "BT6": 5}
    with open(OUTPUT_GERMAN, "w", encoding="utf-8") as f:
        for idx in sorted(translations.keys()):
            t = translations[idx]
            label = label_map.get(t["label"], 0)
            f.write(f"{t['de']}\t{label}\n")
    log.info(f"Für Classifier: {OUTPUT_GERMAN}")

    # Checkpoint löschen (alles fertig)
    os.remove(CHECKPOINT_PATH)
    log.info("Übersetzung vollständig. Checkpoint gelöscht.")


if __name__ == "__main__":
    main()
