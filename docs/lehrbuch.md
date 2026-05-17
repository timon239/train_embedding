# Embedding Training — Das Lehrbuch

Alles was du brauchst, um Embedding-Modelle zu trainieren, zu verstehen und
für eigene Projekte anzupassen.

---

## 1. Konzepte

### Was ist ein Embedding-Modell?

Ein Embedding-Modell wandelt Text in Vektoren um — Zahlenreihen, die die
*Bedeutung* eines Satzes kodieren. Ähnliche Sätze landen nah beieinander
im Vektorraum, unterschiedliche weit auseinander.

Beispiel (vereinfacht, 2D statt 384D):
```
"Die Kalotte ist die tragende Hülle."         → [0.92, 0.12]
"Die Schirmkappe wölbt sich unter Druck."     → [0.89, 0.15]  ← nah (gleiches Thema)
"Die Gruppe entwickelt ein Wir-Gefühl."       → [0.23, 0.88]  ← fern (anderes Thema)
```

### SimCSE — Contrastive Learning für Embeddings

SimCSE (Simple Contrastive Learning) trainiert das Modell ohne Label-Daten.
Jeder Satz wird zweimal encoded — unterschiedliche Dropout-Masken im
Transformer erzeugen minimal abweichende Vektoren. Der Loss zieht diese
beiden Vektoren zusammen und stösst alle anderen Sätze im Batch auseinander.

```
Satz A ──→ Dropout-Maske 1 ──→ Vektor A₁ ──┐
                                             ├─ nah (gleicher Satz)
Satz A ──→ Dropout-Maske 2 ──→ Vektor A₂ ──┘

Satz B ──→ Dropout-Maske 1 ──→ Vektor B₁ ──→ fern von A₁ (anderer Satz)
```

Das Modell lernt dadurch die semantische Struktur deiner Texte:
Welche Wörter treten gemeinsam auf? Welche Konzepte sind verwandt?

### Klassifikation auf Embeddings

Für Aufgaben wie Bloom's Taxonomie wird ein einfacher Classifier
(LogisticRegression) auf die Embeddings gesetzt:

```
Text → Embedding-Modell (frozen) → Vektor → Classifier → Bloom-Stufe (0-5)
```

Der Classifier ist extrem leichtgewichtig und braucht wenig Label-Daten
(~100-500 Sätze pro Stufe), weil die Embeddings bereits die Domänen-Semantik
enthalten.

---

## 2. Workflow

### Schritt 0: Daten vorbereiten

```bash
python3 scripts/prepare_data.py
```

Liest alle `.md`-Dateien aus `data/knowledge/raw/`, entfernt Markdown-Syntax
und extrahiert Sätze nach `data/knowledge/processed/sentences.txt`.

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `INPUT_DIR` | `data/knowledge/raw` | Ordner mit `.md`-Dateien |
| `OUTPUT_SIMCSE` | `data/knowledge/processed/sentences.txt` | Extrahierte Sätze |
| `MIN_SENTENCE_LENGTH` | 5 | Min. Wörter pro Satz |
| `MAX_SENTENCE_LENGTH` | 80 | Max. Wörter pro Satz |

### Schritt 1: V1 — Domain-Embedding trainieren

```bash
python3 scripts/train_embedding.py
```

Konfiguration in `CONFIG` oben in der Datei:

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

**Parameter im Detail:**

| Parameter | Werte | Default | Wirkung |
|-----------|-------|---------|---------|
| **model_name** | HuggingFace-ID | `intfloat/multilingual-e5-small` | Basis-Modell. Alle SentenceTransformer-kompatiblen Modelle. |
| **output_path** | Pfad | `models/V1` | Speicherort für das trainierte Modell. |
| **task** | `"simcse"` / `"classifier"` | `simcse` | Trainingsart. SimCSE = Embedding, Classifier = Bloom. |
| **data_path** | Pfad | `data/knowledge/processed/sentences.txt` | SimCSE: ein Satz pro Zeile. Classifier: satz\\tlabel. |
| **batch_size** | 4–64 | `16` | Sätze pro Batch. Grösser = stabiler, mehr RAM. |
| **epochs** | 1–10 | `3` | Trainingsdauer. Mehr = stärkere Anpassung, Overfitting-Risiko. |
| **learning_rate** | 1e-6 – 5e-5 | `2e-5` | Schrittgrösse. Zu hoch = Modell vergisst, zu niedrig = nichts passiert. |
| **max_seq_length** | 64–512 | `128` | Max. Token pro Satz. Längere werden gekappt. |
| **warmup_ratio** | 0.0–0.5 | `0.1` | Anteil mit steigender LR. Stabilisiert den Start. |
| **device** | `"auto"`/`"mps"`/`"cpu"` | `auto` | Auto = MPS wenn verfügbar, sonst CPU. |

**Entscheidungsbaum für Parameter:**

```
Daten < 500 Sätze  → epochs: 5, batch_size: 8
Daten > 1000 Sätze → epochs: 3, batch_size: 16

Wenig RAM (16GB MacBook) → device: "auto" (nutzt MPS), batch_size: 16
Wenig Speicher           → device: "cpu", batch_size: 8

Leichte Anpassung        → epochs: 1-2, learning_rate: 1e-5
Starke Domain-Adaptation → epochs: 3-5, learning_rate: 2e-5
```

### Schritt 1b: Bloom-Daten übersetzen (einmalig)

```bash
python3 scripts/translate_bloom.py
```

Übersetzt das englische Bloom-Dataset (`data/blooms/raw/en_bloom.csv`)
mit Helsinki-NLP/opus-mt-en-de ins Deutsche.

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `BATCH_SIZE` | 32 | Sätze pro Übersetzungs-Batch |
| `DEVICE` | `"mps"` | "mps" für Apple Silicon, "cpu" sonst |

Output:
- `data/blooms/translated/de_bloom_bilingual.csv` — EN + DE + Label
- `data/blooms/processed/bloom_for_classifier.txt` — DE + Label (für Training)

### Schritt 2: V1BC — Bloom-Klassifikation trainieren

```bash
python3 scripts/train_embedding.py
```

```python
CONFIG = {
    "model_name": "models/V1",          # ← V1 als Basis
    "output_path": "models/V1BC",
    "task": "classifier",               # ← Task wechseln
    "data_path": "data/blooms/processed/bloom_for_classifier.txt",
    "data_delimiter": "\t",
    "device": "cpu",
}
```

Extrahiert Embeddings aus V1, trainiert LogisticRegression und zeigt
die Cross-Validation-Accuracy an. Speichert den Classifier als
`models/V1BC/classifier.joblib`.

**Datenformat für Classification:**
```
satz<TAB>label
Die Schüler definieren den Begriff der Gruppe.	0
Die Schüler vergleichen zwei Modelle.	        3
Die Schüler entwerfen ein Experiment.	        5
```

**Bloom-Stufen (Labels):**
| Label | Stufe | Beispiel |
|-------|-------|----------|
| 0 | Erinnern | "Die Schüler nennen die fünf Sinne." |
| 1 | Verstehen | "Die Schüler erklären den Treibhauseffekt." |
| 2 | Anwenden | "Die Schüler berechnen den Luftwiderstand." |
| 3 | Analysieren | "Die Schüler vergleichen Demokratie und Diktatur." |
| 4 | Bewerten | "Die Schüler beurteilen die Ethik der Gentechnik." |
| 5 | Erschaffen | "Die Schüler entwerfen ein eigenes Experiment." |

---

## 3. Modell-Versionierung

```
V1:   e5-small + SimCSE(Vault)           → Domain-Embedding
V1BC: V1 + Classifier(Bloom)             → Bloom-Klassifikation
```

Wenn du später ein neues Projekt startest (z.B. andere Texte, andere
Klassifikation), erstellst du einfach:

```
V2:   [anderes Modell] + SimCSE(neue Daten)
V2XY: V2 + Classifier(andere Labels)
```

### Unterstützte Modelle

| Modell | Parameter | Dim | RAM | Geeignet für |
|--------|-----------|-----|-----|--------------|
| `intfloat/multilingual-e5-small` | 118 M | 384 | ~450 MB | Standard — leicht & schnell |
| `intfloat/multilingual-e5-base` | 278 M | 768 | ~1.1 GB | Höhere Qualität, mehr RAM |
| `jinaai/jina-embeddings-v2-base-de` | 137 M | 768 | ~500 MB | Deutscher Fokus, 8192 Token |
| `paraphrase-multilingual-MiniLM-L12-v2` | 117 M | 384 | ~420 MB | Sehr stabil, breit getestet |

## 4. Monitoring

```bash
tail -f logs/train_*.log
```

### SimCSE-Training
```
14:30:01 | >>> SIMCSE-TRAINING
14:30:01 |       Modell: intfloat/multilingual-e5-small
14:30:05 | Epoch 1: Avg Loss = 0.0421   ← startet höher
14:35:12 | Epoch 2: Avg Loss = 0.0083   ← fällt
14:40:18 | Epoch 3: Avg Loss = 0.0047   ← stabilisiert sich
```

Ein sinkender Loss ist normal. Solange er nicht auf 0.0000 fällt, lernt das
Modell. Loss = 0.0000 wäre ein Zeichen für Overfitting (Modell rät "immer
denselben Vektor" — nutzlos für Embeddings).

### Classifier-Training
```
14:45:01 | >>> CLASSIFIER-TRAINING
14:45:02 |       Cross-Validation Accuracy: 0.723 (+/- 0.031)
```

Accuracy > 0.7 = brauchbar. > 0.85 = sehr gut.

## 5. Modell verwenden

```python
from sentence_transformers import SentenceTransformer

# V1 laden (Embedding)
model = SentenceTransformer("models/V1")
embedding = model.encode("Die Kalotte ist die tragende Hülle.")
# → np.array mit 384 Werten

# V1BC laden (Bloom-Klassifikation)
import joblib
clf = joblib.load("models/V1BC/classifier.joblib")
text = "Die Schüler erklären die Funktion der Mitochondrien."
emb = model.encode(text)
bloom_level = clf.predict([emb])[0]  # → 1 (Verstehen)
```

## 6. Fehlerbehebung

| Problem | Lösung |
|---------|--------|
| MPS out of memory | `device: "cpu"`, `batch_size: 8` setzen |
| Modell nicht gefunden | Modell-ID auf HuggingFace prüfen |
| Keine Sätze extrahiert | `INPUT_DIR` in prepare_data.py prüfen |
| Übersetzung zu langsam | `BATCH_SIZE` in translate_bloom.py erhöhen |
| Bloom-Accuracy < 0.5 | Mehr Label-Daten oder zuerst SimCSE auf Vault laufen lassen |
| Loss = 0.0000 im SimCSE | batch_size erhöhen oder epochs reduzieren |
