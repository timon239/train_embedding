# Embedding Training — Das Lehrbuch

## 1. Konzepte

### Was ist ein Embedding-Modell?

Ein Embedding-Modell wandelt Text in Vektoren um (z. B. 384 Dimensionen),
die die *Bedeutung* eines Satzes kodieren. Ähnliche Sätze landen nah
beieinander im Vektorraum, unterschiedliche weit auseinander.

### SimCSE — Contrastive Learning

SimCSE trainiert das Modell ohne Label-Daten. Jeder Satz wird zweimal
encoded — unterschiedliche Dropout-Masken im Transformer erzeugen minimal
abweichende Vektoren. Der Contrastive Loss zieht diese beiden Vektoren
zusammen und stösst alle anderen Sätze im Batch auseinander.

Das Modell lernt dadurch die semantische Struktur deiner Texte:
Welche Wörter treten gemeinsam auf? Welche Konzepte sind verwandt?

### Klassifikation auf Embeddings

Für Text-Klassifikation (z. B. Bloom's Taxonomie, Sentiment, Themen) wird
ein einfacher Classifier (LogisticRegression) auf die Embeddings gesetzt:

```
Text → Embedding-Modell (frozen) → Vektor → Classifier → Klasse
```

Der Classifier ist extrem leichtgewichtig und braucht wenig Label-Daten
(~100-500 pro Klasse).

---

## 2. Workflow

### Schritt 0: Daten vorbereiten

```bash
python3 scripts/prepare_data.py
```

Liest `.md`-Dateien aus `INPUT_DIR`, entfernt Markdown-Syntax, YAML,
Bilder. Extrahiert Sätze nach `OUTPUT_SIMCSE` (ein Satz pro Zeile).

Parameter (oben im Script): `INPUT_DIR`, `OUTPUT_SIMCSE`,
`MIN_SENTENCE_LENGTH`, `MAX_SENTENCE_LENGTH`

### Schritt 1: Domain-Embedding trainieren

```bash
python3 scripts/train_embedding.py
```

```python
CONFIG = {
    "model_name": "intfloat/multilingual-e5-small",
    "output_path": "models/V1",
    "task": "simcse",
    "data_path": "data/knowledge/processed/sentences.txt",
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "max_seq_length": 128,
    "warmup_ratio": 0.1,
    "device": "auto",
}
```

**Parameter-Referenz:**

| Parameter | Werte | Default | Erklärung |
|-----------|-------|---------|-----------|
| `model_name` | HuggingFace-ID | `intfloat/multilingual-e5-small` | Basis-Modell |
| `output_path` | Pfad | `models/V1` | Speicherort |
| `task` | `"simcse"` / `"classifier"` | `simcse` | Trainingsart |
| `data_path` | Pfad | — | SimCSE: Satz/Zeile. Classifier: satz\\tlabel |
| `data_delimiter` | String | `\t` | Trennzeichen für Classifier |
| `batch_size` | 4–64 | `16` | Grösser = stabiler, mehr RAM |
| `epochs` | 1–10 | `3` | Mehr = stärkere Anpassung |
| `learning_rate` | 1e-6 – 5e-5 | `2e-5` | Schrittgrösse |
| `max_seq_length` | 64–512 | `128` | Token-Limit pro Satz |
| `warmup_ratio` | 0.0–0.5 | `0.1` | LR-Aufwärmphase |
| `device` | `"auto"`/`"mps"`/`"cpu"` | `auto` | MPS = Apple GPU |

### Schritt 2: Daten übersetzen (optional)

```bash
python3 scripts/translate.py --input data/raw/en.csv --output data/processed/de.csv
```

Übersetzt tabellarische Daten mit HuggingFace-Modell (Default:
`Helsinki-NLP/opus-mt-en-de`). Erwartet eine Spalte mit Text,
fügt eine übersetzte Spalte hinzu.

### Schritt 3: Classifier trainieren

```bash
python3 scripts/train_embedding.py
```

```python
CONFIG = {
    "model_name": "models/V1",           # ← Domain-modell als Basis
    "output_path": "models/V1BC",
    "task": "classifier",
    "data_path": "data/processed/classifier_data.txt",  # satz\tlabel
    "device": "cpu",
}
```

---

## 3. Entscheidungsbaum

```
Wenig Daten (< 500 Sätze)    → epochs: 5, batch_size: 8
Viel Daten (> 1000 Sätze)    → epochs: 3, batch_size: 16

Apple Silicon                → device: "auto" (MPS)
Wenig RAM / älterer Mac      → device: "cpu", batch_size: 8

Leichte Anpassung            → epochs: 1-2, learning_rate: 1e-5
Starke Domain-Adaptation     → epochs: 3-5, learning_rate: 2e-5
```

---

## 4. Modell-Versionierung (Beispiel)

```
Basis-Modell (z.B. e5-small)
    │
    ├── SimCSE(Domänen-Daten)  →  V1 (Domain-Embedding)
    │                               │
    │                               └── Classifier(Label-Daten) → V1BC
    │
    └── SimCSE(andere Daten)   →  V2
                                    │
                                    └── Classifier(andere Labels) → V2XY
```

---

## 5. Unterstützte Modelle

| HuggingFace ID | Parameter | Dim | RAM |
|----------------|-----------|-----|-----|
| `intfloat/multilingual-e5-small` | 118 M | 384 | ~450 MB |
| `intfloat/multilingual-e5-base` | 278 M | 768 | ~1.1 GB |
| `jinaai/jina-embeddings-v2-base-de` | 137 M | 768 | ~500 MB |
| `paraphrase-multilingual-MiniLM-L12-v2` | 117 M | 384 | ~420 MB |

---

## 6. Fehlerbehebung

| Problem | Lösung |
|---------|--------|
| MPS out of memory | `device: "cpu"`, `batch_size: 8` setzen |
| Modell nicht gefunden | Modell-ID auf HuggingFace prüfen |
| Keine Sätze extrahiert | `INPUT_DIR` in prepare_data.py prüfen |
| Loss = 0.0000 | batch_size erhöhen oder epochs reduzieren |
