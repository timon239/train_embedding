#!/usr/bin/env python3
"""
translate.py — Übersetzt tabellarische Daten mit HuggingFace-Modell.
==============================================================
Nutzung:  python3 scripts/translate.py
Konfiguration oben anpassen.

Unterstützt jedes HuggingFace-Übersetzungsmodell, z.B.:
  - Helsinki-NLP/opus-mt-en-de   (Englisch → Deutsch)
  - Helsinki-NLP/opus-mt-de-en   (Deutsch → Englisch)
  - facebook/mbart-large-50-many-to-many-mmt  (multilingual)
"""

# ============================================================
# KONFIGURATION
# ============================================================
INPUT_PATH = "data/blooms/raw/en_bloom.csv"
OUTPUT_PATH = "data/blooms/translated/de_bloom.csv"
CHECKPOINT_PATH = "data/blooms/translated/_checkpoint.txt"

TEXT_COLUMN = "Questions"
TRANSLATED_COLUMN = "text_de"
KEEP_COLUMNS = True

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"         # Übersetzungsmodell
BATCH_SIZE = 32                                    # Sätze pro Batch
DEVICE = "mps"                                     # "mps" oder "cpu"

# ============================================================
import os
import csv
import logging
import time
from datetime import datetime
from transformers import MarianTokenizer, MarianMTModel
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger()


def load_data(path: str) -> tuple[list[dict], list[str]]:
    if not os.path.exists(path):
        log.error(f"Datei nicht gefunden: {path}")
        log.error("Lege Datei an oder passe INPUT_PATH in der Konfiguration an.")
        exit(1)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if TEXT_COLUMN not in rows[0]:
        log.error(f"Spalte '{TEXT_COLUMN}' nicht gefunden. Verfügbar: {list(rows[0].keys())}")
        exit(1)
    texts = [r[TEXT_COLUMN] for r in rows]
    log.info(f"Gelesen: {len(rows)} Zeilen aus {path}")
    return rows, texts


def get_completed(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {int(l.strip()) for l in f if l.strip()}


def save_checkpoint(idx: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{idx}\n")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rows, texts = load_data(INPUT_PATH)
    completed = get_completed(CHECKPOINT_PATH)
    log.info(f"Bereits übersetzt: {len(completed)}")

    pending = [(i, t) for i, t in enumerate(texts) if i not in completed]
    if not pending:
        log.info("Alles bereits übersetzt.")
        return

    log.info(f"Noch zu übersetzen: {len(pending)} Sätze")
    log.info(f"Lade Modell {MODEL_NAME} …")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    device = torch.device(DEVICE) if DEVICE == "mps" else torch.device("cpu")
    model.to(device)
    model.eval()
    log.info(f"Modell geladen auf {device}")

    translations = {}
    t0 = time.time()

    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i+BATCH_SIZE]
        batch_texts = [t for _, t in batch]

        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=192).to(device)
            outputs = model.generate(**inputs, max_length=192,
                                     num_beams=2, early_stopping=True)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for (idx, _), de in zip(batch, decoded):
            translations[idx] = de
            save_checkpoint(idx, CHECKPOINT_PATH)

        if (i // BATCH_SIZE) % 5 == 0:
            elapsed = time.time() - t0
            speed = (i + len(batch)) / elapsed
            remaining = (len(pending) - i - len(batch)) / speed
            eta = datetime.fromtimestamp(time.time() + remaining).strftime("%H:%M")
            log.info(f"  {min(i+BATCH_SIZE, len(pending))}/{len(pending)} | "
                     f"{speed:.1f} Sätze/s | ETA: {eta}")

    # Ausgabe
    fieldnames = list(rows[0].keys())
    if TRANSLATED_COLUMN not in fieldnames:
        fieldnames.append(TRANSLATED_COLUMN)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            row[TRANSLATED_COLUMN] = translations.get(idx, "")
            writer.writerow(row)

    log.info(f"Gespeichert: {OUTPUT_PATH}")

    # Classifier-Format schreiben: text_de\tlabel
    CLASSIFIER_PATH = OUTPUT_PATH.replace(".csv", "_classifier.txt")
    label_map = {"BT1": 0, "BT2": 1, "BT3": 2, "BT4": 3, "BT5": 4, "BT6": 5}
    with open(CLASSIFIER_PATH, "w", encoding="utf-8") as f:
        n_written = 0
        for idx, row in enumerate(rows):
            de_text = translations.get(idx, "").strip()
            if de_text:
                cat = row.get("Category", "BT1")
                label = label_map.get(cat, 0)
                f.write(f"{de_text}\t{label}\n")
                n_written += 1
    log.info(f"Classifier-Format: {CLASSIFIER_PATH} ({n_written} Zeilen)")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    log.info(f"Fertig. {len(translations)} Sätze übersetzt in "
             f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
