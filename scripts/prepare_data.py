#!/usr/bin/env python3
"""
prepare_data.py — Markdown → Satz-Extraktion
===========================================
Konfiguration am Anfang. Einfach anpassen und ausführen.

Unterstützt aktuell:
  - Extraktion aus .md-Dateien
  - SimCSE-Format: eine Zeile pro Satz (data/processed/sentences.txt)
  - Classification-Format: satz\\tlabel (noch nicht aktiv, Struktur vorbereitet)
"""

# ============================================================
# KONFIGURATION — hier anpassen
# ============================================================
INPUT_DIR = "data/knowledge/raw/"              # Wo liegen die .md-Dateien?
OUTPUT_SIMCSE = "data/knowledge/processed/sentences.txt"  # Ausgabe für SimCSE-Training
MIN_SENTENCE_LENGTH = 5            # Mindest-Wörter pro Satz
MAX_SENTENCE_LENGTH = 80           # Maximal-Wörter pro Satz (längere werden gesplittet)

import os
import re
import glob


def strip_markdown(text: str) -> str:
    text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\*Abb\.\s*.*?\*', '', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'[*_]{2,3}', '', text)
    text = re.sub(r'\|[^\n]+\|\s*\n', '\n', text)
    lines = []
    for l in text.split('\n'):
        l = l.strip()
        if not l:
            continue
        if re.match(r'^#+\s', l):
            l = re.sub(r'^#+\s*', '', l)
        if re.match(r'^[\s\-_=\|>#]+$', l):
            continue
        if re.match(r'^Seite\s+\d+|<!--.*-->$', l, re.I):
            continue
        lines.append(l)
    return '\n'.join(lines)


def split_sentences(text: str) -> list[str]:
    abbrs = r'\b(?:z\.B\.|bzw\.|etc\.|d\.h\.|u\.a\.|usw\.|ca\.|Mr\.|Dr\.|Prof\.|Nr\.|vgl\.|Abschn\.|Kap\.|S\.|Abb\.|Tab\.)'
    text = re.sub(abbrs, lambda m: m.group(0).replace('.', '<DOT>'), text)
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Za-z0-9ÄÖÜäöü])', text)
    sents = [s.replace('<DOT>', '.') for s in sents]
    return [s.strip() for s in sents if s.strip()]


def clean_sentence(sent: str) -> str | None:
    sent = sent.strip()
    if len(sent.split()) < MIN_SENTENCE_LENGTH:
        return None
    words = sent.split()
    if len(words) > MAX_SENTENCE_LENGTH:
        sent = ' '.join(words[:MAX_SENTENCE_LENGTH])
    if sent.endswith(','):
        sent = sent[:-1]
    if not sent.endswith(('.', '!', '?', ':')):
        sent += '.'
    if len(sent) < 15:
        return None
    # Nur Sätze mit deutschsprachigen Zeichen
    if not re.search(r'[A-Za-zÄÖÜäöüß]', sent):
        return None
    return sent


def main():
    os.makedirs(os.path.dirname(OUTPUT_SIMCSE), exist_ok=True)

    md_files = sorted(glob.glob(os.path.join(INPUT_DIR, '**', '*.md'), recursive=True))
    print(f"  -> {len(md_files)} .md-Dateien gefunden")

    all_sentences = []
    for fp in md_files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                raw = f.read()
        except Exception as e:
            print(f"  [WARN] {fp}: {e}")
            continue
        clean = strip_markdown(raw)
        sents = split_sentences(clean)
        for s in sents:
            cleaned = clean_sentence(s)
            if cleaned:
                all_sentences.append(cleaned)

    # Deduplizieren
    seen = set()
    unique = []
    for s in all_sentences:
        norm = s.lower().strip()
        if norm not in seen:
            seen.add(norm)
            unique.append(s)

    # Ausgabe SimCSE-Format
    with open(OUTPUT_SIMCSE, 'w', encoding='utf-8') as f:
        for s in unique:
            f.write(s + '\n')

    print(f"  -> {len(unique)} Sätze geschrieben nach {OUTPUT_SIMCSE}")


if __name__ == "__main__":
    main()
