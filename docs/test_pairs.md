# Validierungs-Paare — Vorlage für eigene Evaluation

Dieses Dokument zeigt exemplarisch, wie du Test-Paare für dein Embedding-Modell
definieren kannst, um die Qualität der Domain-Adaptation zu prüfen.

## Prinzip

- **Synonym-Paare:** Zwei Sätze mit ähnlicher Bedeutung aus DEMSELBEN Fachbereich.
  → Ähnlichkeit (Cosinus) soll nach dem Training **steigen**.
- **Homonym-Paare:** Zwei Sätze mit gleichem Begriff aus UNTERSCHIEDLICHEN
  Fachbereichen. → Ähnlichkeit soll nach dem Training **sinken**.

## Beispiel-Struktur

```
(SYNONYM) "Definition von Fachbegriff A" ↔ "Erklärung von Fachbegriff A"
(HOMONYM) "Begriff X in Domäne 1"       ↔ "Begriff X in Domäne 2"
```

## Eigene Paare erstellen

Lege eine CSV-Datei an mit folgendem Format:

```csv
typ,paar_beschreibung,satz_a,satz_b
syn,Kalotte ↔ Schirmkappe,Die Kalotte bildet die tragende Fläche.,Die Schirmkappe überträgt die aerodynamischen Kräfte.
hom,Profil — Aero vs. Sozio,Schmale symmetrische Flügelprofile.,Das Rollenprofil umfasst die Verhaltenserwartungen.
```

Dann evaluieren mit:

```python
from sentence_transformers import SentenceTransformer
import csv

model = SentenceTransformer("models/V1")
with open("test_pairs.csv") as f:
    for row in csv.DictReader(f):
        emb_a = model.encode(row["satz_a"])
        emb_b = model.encode(row["satz_b"])
        sim = float(emb_a @ emb_b)  # Cosinus-Ähnlichkeit
        print(f"{row['typ']} | {row['paar_beschreibung']}: {sim:.4f}")
```

## Interpretation

```
Alle Synonyme steigen (+Δ) + Alle Homonyme sinken (−Δ)  → ✅ Training erfolgreich
Alle Werte > 0.97                                         → ❌ Representation Collapse
Kaum Veränderung                                          → ⚠️ Training wirkungslos
```
