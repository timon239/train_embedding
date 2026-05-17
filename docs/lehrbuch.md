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
| `output_path` | Pfad | `models/mein_modell` | Speicherort |
| `task` | `"simcse"` / `"classifier"` | `simcse` | Trainingsart |
| `data_path` | Pfad | — | SimCSE: Satz/Zeile. Classifier: satz\\tlabel |
| `data_delimiter` | String | `\t` | Trennzeichen für Classifier |
| `batch_size` | 4–128 | `32` | Grösser = mehr Negative im Contrastive Loss = stabiler |
| `epochs` | 1–3 | `1` | SimCSE: 1 reicht. Mehr = Collapse-Risiko |
| `learning_rate` | 1e-6 – 5e-5 | `1e-5` | Schrittgrösse |
| `weight_decay` | 0.0–0.1 | `0.01` | Bestrafung grosser Gewichte → verhindert Collapse |
| `warmup_ratio` | 0.0–0.5 | `0.1` | LR-Aufwärmphase |
| `max_seq_length` | 64–512 | `128` | Token-Limit pro Satz |
| `contrastive_scale` | 1.0–50.0 | `10` | Schärfe des Contrastive Loss. Nur für simcse. |
| `classifier_method` | `"mlp"` / `"logistic"` / `"rf"` | `"mlp"` | Classifier-Typ. MLP = beste Qualität, Logistic = schnell, RF = robust. |
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

## 5. Monitoring & Logs

Jeder Trainingslauf schreibt ein Log nach `logs/train_YYYYMMDD_HHMMSS.log`.

### Live verfolgen

```bash
# Terminal 1: Training laufen lassen
python3 scripts/train_embedding.py

# Terminal 2: Log live anzeigen
tail -f logs/train_*.log
```

### Log-Ablauf SimCSE

```
14:30:01 | >>> SIMCSE-TRAINING
14:30:01 |       Modell: intfloat/multilingual-e5-small
14:30:01 |       Device: mps | 2365 Batches/Epoche
14:30:01 |       Start: 14:30
14:31:05 | Epoch 1, Batch 100/2365: Loss=0.0421 | 19.2 Bat/s | ETA: 14:52
14:32:09 | Epoch 1, Batch 200/2365: Loss=0.0312 | 19.5 Bat/s | ETA: 14:52
14:35:12 | Epoch 1 abgeschlossen: Avg Loss = 0.0412
14:40:18 | Epoch 2 abgeschlossen: Avg Loss = 0.0083
14:45:22 | Epoch 3 abgeschlossen: Avg Loss = 0.0047
14:45:22 | Modell gespeichert: models/mein_modell
```

**Spalten im Log:**
- `Loss`: Aktueller Batch-Verlust (sollte Epoche für Epoche fallen)
- `Bat/s`: Verarbeitete Batches pro Sekunde
- `ETA`: Geschätzte Fertigstellungszeit dieser Epoche

### Log-Ablauf Classifier

```
14:45:01 | >>> CLASSIFIER-TRAINING
14:45:01 |       100 Samples | 6 Klassen
14:45:05 |       Cross-Validation Accuracy: 0.723 (+/- 0.031)
```

### Terminal-Progress-Bar (tqdm)

Während des Trainings siehst du im Terminal live:

```
Epoch 1/3:  12%|██▌       | 284/2365 [00:15<01:48, 19.2batch/s]
```

Bedeutung:
- `284/2365` = Batches in dieser Epoche
- `00:15` = vergangene Zeit
- `01:48` = geschätzte Restzeit
- `19.2batch/s` = Geschwindigkeit

### Interpretation

```
Loss sinkt von Epoche zu Epoche        → ✅ Modell lernt
Loss = 0.0000                          → ❌ Overfitting (nur bei SimCSE)
Accuracy > 0.7 (Classifier)            → ✅ Brauchbar
Accuracy > 0.85 (Classifier)           → ✅ Sehr gut
```

---

## 6. Fehlerbehebung

| Problem | Lösung |
|---------|--------|
| MPS out of memory | `device: "cpu"`, `batch_size: 8` setzen |
| Modell nicht gefunden | Modell-ID auf HuggingFace prüfen |
| Keine Sätze extrahiert | `INPUT_DIR` in prepare_data.py prüfen |
| Loss = 0.0000 | batch_size erhöhen oder epochs reduzieren |
