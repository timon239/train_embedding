# Modell-Vergleich — German Embedding & Classification

Stand: Mai 2026
Zielsetzung: Domain-Adaptation + Bloom's Taxonomie-Klassifikation

---

## Infrastruktur (dein MacBook)

| Ressource | Verfügbar |
|-----------|-----------|
| RAM | 16 GB (unified) |
| GPU | Apple MPS (20 GB Limit) |
| CPU | Multi-Core |
| Speicher | ~50 GB frei |

→ Modelle bis ~500 MB sind komfortabel, bis ~1 GB machbar.

---

## Embedding-Modelle (Sentence-Transformer-kompatibel)

### Vergleichstabelle

| Modell | Parameter | Dim | RAM | Sprachen | Stärke | Schwäche |
|--------|-----------|-----|-----|----------|--------|----------|
| **jinaai/jina-embeddings-v2-base-de** | 137 M | 768 | ~500 MB | Deutsch | Max 8192 Token, deutsch-optimiert | Custom BERT → Kompatibilitätsprobleme |
| **intfloat/multilingual-e5-small** | 118 M | 384 | ~450 MB | 100+ | SOTA Multilingual, leicht, SimCSE-trainiert | Kleine Dim (384) |
| **intfloat/multilingual-e5-base** | 278 M | 768 | ~1.1 GB | 100+ | SOTA Multilingual, beste Balance | Grösser |
| **intfloat/multilingual-e5-large** | 560 M | 1024 | ~2.2 GB | 100+ | SOTA, höchste Qualität | Zu schwer für dein Setup |
| **T-Systems/cross-en-de-roberta-sentence-transformer** | 278 M | 768 | ~1.1 GB | DE/EN | Spezifisch DE/EN trainiert | Nischenmodell |
| **paraphrase-multilingual-MiniLM-L12-v2** | 117 M | 384 | ~420 MB | 50+ | Sehr stabil, breit getestet | Multilingual = weniger deutsch-spezifisch |
| **BAAI/bge-m3** | 567 M | 1024 | ~2.3 GB | 100+ | SOTA + sparse retrieval | Zu schwer |

### Empfohlen: Top 3

| Rang | Modell | Grund |
|------|--------|-------|
| 🥇 | **multilingual-e5-base** | Beste Qualität, 768-dim, SimCSE-vortrainiert (ideal für Fine-Tuning), 278 M passen in RAM |
| 🥈 | **jina-embeddings-v2-base-de** | Deutscher Fokus, lange Kontexte (8192), klein (500 MB) — aber Custom-Code-Probleme |
| 🥉 | **multilingual-e5-small** | 118 M, superschnell, 384-dim reichen für viele Tasks — bester Kompromiss Speed/Qualität |

---

## Bloom's Taxonomie — Modell-Empfehlung

Bloom's Taxonomie hat 6 Ebenen:
1. Erinnern (Remember)
2. Verstehen (Understand)
3. Anwenden (Apply)
4. Analysieren (Analyze)
5. Bewerten (Evaluate)
6. Erschaffen (Create)

**Zwei Strategien:**

### A) Embedding + Classifier (empfohlen)

```
Text → Embedding-Modell → Vektor (768-dim) → Linear-Classifier → Bloom-Ebene
                                                      ↕
                                           trainierst du mit ~100-500 labelierten Sätzen
```

**Vorteil:** Ein Modell für beides — Domain-Adaptation + Bloom.
**Ablauf:** 
1. Embedding-Modell per SimCSE auf deine Domänen fine-tunen
2. Frozen Embeddings + trainierbaren Classifier oben drauf setzen
3. Bloom-Klassifikation mit wenig Label-Daten

**Empfohlenes Modell dafür:** `multilingual-e5-base` oder `multilingual-e5-small`

### B) Separater Classifier

```
Text → bert-base-german-cased → [CLS] → Classifier → Bloom-Ebene
```

**Vorteil:** End-to-End, oft bessere Accuracy
**Nachteil:** Separates Modell, kein gemeinsames Embedding

---

## Meine Empfehlung

| Task | Modell | Begründung |
|------|--------|------------|
| **Domain-Adaptation** | `multilingual-e5-base` | SimCSE-vortrainiert, 768-dim, passt in RAM |
| **Bloom-Klassifikation** | Gleiches Modell + Classifier-Head | Wiederverwendung der Embeddings |
| **Budget-Alternative** | `multilingual-e5-small` | 118 M, schnell, 384-dim reichen oft |

### Vergleich e5-small vs e5-base

```
                     e5-small              e5-base
                     (118M, 384dim)        (278M, 768dim)
Training:             ~0.3s/Batch (CPU)     ~0.6s/Batch (CPU)
                      4 min/Epoche          8 min/Epoche
RAM:                   ~450 MB               ~1.1 GB
Qualität:             gut                   sehr gut
Bloom-Klassifikation:  ausreichend           besser                       
```

---

## Nächster Schritt

Modell auswählen → ich baue `prepare_data.py` und `train_embedding.py` mit deinem Wunschmodell.
