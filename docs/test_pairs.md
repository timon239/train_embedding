# Validierungs-Paare — Erweiterte Matrix (40 Paare)

Diese Paare dokumentieren die semantischen Beziehungen zwischen den Domänen
und können verwendet werden, um ein trainiertes Modell zu evaluieren.

**Idee:** Vorher/Nachher-Cosinus-Ähnlichkeit messen.
- Synonyme (Teil 1): Ähnlichkeit **sollte steigen** nach Domain-Adaptation
- Homonyme (Teil 2): Ähnlichkeit **sollte sinken** nach Domain-Adaptation

Basiert auf Extraktion aus vier Hauptdomänen:
**Gleitschirm (G), Soziologie (S), Forstwart (F), Biologie (B)**

---

## TEIL 1: SYNONYM-TESTS (20 Paare) — Ähnlichkeit SOLL steigen

### Gleitschirm-Domäne (1–8)

| # | Konzept A | Konzept B | Quellenbeleg |
|---|-----------|-----------|-------------|
| 1 | *Kalotte* | *Schirmkappe* | G: "Die aus Ober- und Untersegel bestehende **Kalotte**…" / "Rückmeldungen von der **Schirmkappe**…" |
| 2 | *Gleitzahl* | *Gleitverhältnis* | G: "Der Winkel zwischen der Flugbahn und der Horizontalen heisst **Gleitwinkel**." / "**Gleitzahl** als Leistungskriterium" |
| 3 | *Anstellwinkel* | *Profilsehne zum Fahrtwind* | G: "Die Profilsehne bildet zusammen mit der Strömungsrichtung den **Anstellwinkel**." / "**Trimmung** = Einstellung der **Profilsehne zum anströmenden Fahrtwind**." |
| 4 | *Flächenbelastung* | *Belastung pro Flächeneinheit* | G: "Die **Flächenbelastung** beeinflusst die Flugeigenschaften wesentlich." / "Zuladung in Relation zu dessen Fläche" |
| 5 | *Fussbeschleuniger* | *Beschleunigungssystem* | G: "**Beschleunigungs-** und Trimmsysteme" / "Beim Einsatz des **Fussbeschleunigers**…" |
| 6 | *Streckung* | *aerodynamische Güte* | G: "Die **Streckung** ist ein Mass für die **aerodynamische Güte** eines Flügels." |
| 7 | *Randwirbel* | *induzierter Widerstand* | G: "Je höher die Streckung, desto geringer der negative Einfluss der **Randwirbel**." / "**Induzierter Widerstand** 40" |
| 8 | *Strömungsabriss* | *Stall* | G: "**Strömungsabriss** bzw. **Stall** (engl. 'überzogener Flug')." |

### Soziologie-Domäne (9–15)

| # | Konzept A | Konzept B | Quellenbeleg |
|---|-----------|-----------|-------------|
| 9 | *Soziale Normen* | *Verhaltenserwartungen* | S: "**Soziale Normen** sind verbindliche Verhaltensvorschriften…" / "…Wert- und Normvorstellungen sowie mit den **Verhaltenserwartungen** einer Gesellschaft." |
| 10 | *Gruppenkohäsion* | *Gruppenzusammenhalt* | S: "Die Stärke des Zusammenhalts… wird als **Gruppenkohäsion** bezeichnet." / "Der **Gruppenzusammenhalt** wächst, es entsteht ein Wir-Gefühl." |
| 11 | *Rollendifferenzierung* | *Rollenverhalten* | S: "Diese Vielgestaltigkeit im **Rollenverhalten** bezeichnet man als **Rollendifferenzierung**." |
| 12 | *Sanktionen* | *Massnahmen zur Normdurchsetzung* | S: "**Sanktionen** sind Massnahmen, die für die Einhaltung von sozialen Normen sorgen sollen." |
| 13 | *Soziale Kontrolle* | *Überwachung der Normeinhaltung* | S: "**Soziale Kontrolle** bedeutet die Beaufsichtigung und **Überwachung der Einhaltung von sozialen Normen**." |
| 14 | *Rang* | *soziale Stellung* | S: "**Rang** bedeutet, dass ein Gruppenmitglied mit höherer **Stellung** mehr Macht besitzt." |
| 15 | *Gruppenidentifikation* | *Wir-Gefühl* | S: "**Gruppenidentifikation** bedeutet ein emotionales Sich-Gleichsetzen…" / "…**Wir-Gefühl**… gefühlsmässige Teilhabe des Einzelnen an der Gruppe." |

### Forstwart-Domäne (16–20)

| # | Konzept A | Konzept B | Quellenbeleg |
|---|-----------|-----------|-------------|
| 16 | *Naturverjüngung* | *natürliche Wiederbewaldung* | F: "Vorrang der **Naturverjüngung** vor der Pflanzung" / "Der Wald verjüngt sich auf **natürliche Weise**." |
| 17 | *Bestandesdichte* | *Schlussgrad der Kronen* | F: "Die **Bestandesdichte** wird mit dem **Schlussgrad der Baumkronen** gemessen." |
| 18 | *Pionierwald* | *Pioniervegetation* | F: "Die erste Generation – auch **Pionierwald** genannt – besteht aus Lichtbaumarten." / "…einer lichten **Pioniervegetation**." |
| 19 | *Kronenschluss* | *Kronendach* | F: "…stellt sich später oder früher **Kronenschluss** ein." / "Das **Kronendach** schützt das Innere des Waldes vor Wind." |
| 20 | *Durchforstung* | *Pflegeeingriff* | F: "…durch regelmässige Eingriffe (**Pflegeeingriffe, Durchforstungen**)." |

---

## TEIL 2: HOMONYM-/KONTEXT-TESTS (20 Paare) — Ähnlichkeit SOLL sinken

Die Paare nutzen identische oder sehr ähnliche Wörter aus **unterschiedlichen Domänen**.

### Profil — Aerodynamik vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 21 | G: Flügelprofil | S: Rollenprofil | G: "Schnell fliegende Flugzeuge haben schmale, nahezu symmetrische **Flügelprofile**." ↔ S: "Die Summe der Verhaltenserwartungen an eine soziale Position = **Rollenprofil**." |

### Rolle — Funktion vs. Verhaltenserwartung

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 22 | G: (Bedeutung) | S: soziale Rolle | G: "Eine wesentliche **Rolle** spielt dagegen die Streckung." ↔ S: "Mit jeder Position gibt die Gesellschaft ihm eine **Rolle** in die Hand, die er zu spielen hat." |

### Stellung — Forst vs. Soziologie **(Homonym-König!)**

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 23 | F: soziale Stellung der Bäume | S: soziale Stellung in der Gruppe | F: "Es werden fünf **soziale Stellungen** unterschieden: vorherrschend, herrschend, mitherrschend, beherrscht, unterdrückt." ↔ S: "**Rang** bedeutet, dass ein Gruppenmitglied mit höherer **Stellung** mehr Macht besitzt." |

### Druck — Physik vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 24 | G: Luftdruck | S: Konformitätsdruck | G: "Mit zunehmender Höhe nimmt der **Luftdruck** ab." ↔ S: "Der Einzelne ist einem **Konformitätsdruck** ausgesetzt." |

### Gruppe — Forst vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 25 | F: Baumgruppe | S: soziale Gruppe | F: "**Gruppe**: Bäume derselben Baumart auf einer Fläche von 5 bis 10 Aren." ↔ S: "Eine **Gruppe** sind mehrere Personen, die miteinander in Wechselbeziehung stehen und ein Wir-Gefühl entwickeln." |

### Leistung — Aerodynamik vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 26 | G: Flugleistung | S: Gruppenleistung | G: "**Leistungsdaten** sind von untergeordneter Bedeutung." ↔ S: "Gruppenprozesse beeinflussen die **Gruppenleistung**." |

### Widerstand — Aerodynamik vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 27 | G: Luftwiderstand | S: Widerstand gegen Normen | G: "Der **Widerstand** nimmt enorm zu und der Auftrieb sinkt." ↔ S: "Abweichung = **Widerstand** gegen soziale Normen." |

### Auftrieb — Aerodynamik vs. emotional

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 28 | G: aerodynamischer Auftrieb | S: emotionaler Auftrieb | G: "Damit ein Flugzeug fliegt, muss an der Tragfläche der dynamische **Auftrieb** angreifen." ↔ S: "Der Einzelne erlebt ein Wohlgefühl, was ihm **Kraft** gibt, sich zu entfalten." |

### Stabilität — Aerodynamik vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 29 | G: Flugstabilität | S: Gruppenstabilität | G: "Ein sicheres Fluggerät zeichnet sich durch hohe Eigen**stabilität** aus." ↔ S: "Normen und Rollen verleihen der Gruppe **Stabilität**." |

### Kontrolle — Steuerung vs. soziale Überwachung

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 30 | G: Schirmkontrolle | S: soziale Kontrolle | G: "Die **Kontrolle** über das Schirmverhalten, insbesondere die Flugrichtung." ↔ S: "**Soziale Kontrolle** bedeutet die Beaufsichtigung und Überwachung der Normen." |

### Norm — Technik vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 31 | G: EN-Norm (technisch) | S: soziale Norm | G: "Die Europäischen **Normen** (EN) und die deutschen Lufttüchtigkeitsforderungen (LTF)." ↔ S: "**Soziale Normen** sind verbindliche Verhaltensvorschriften." |

### Bestand — Forst vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 32 | F: Waldbestand | S: sozialer Bestand | F: "Ein **Bestand** ist ein Waldteil, der sich von der Umgebung unterscheidet." ↔ S: "Der soziale **Bestand** einer Gesellschaft…" |

### Entwicklung — Forst vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 33 | F: Waldentwicklung | S: Persönlichkeitsentwicklung | F: "Die natürliche **Waldentwicklung** dauert 300 bis 600 Jahre." ↔ S: "Der Aufbau der eigenen **Persönlichkeit** kann nur durch Kontakte zu anderen gelingen." |

### Mischung — Forst vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 34 | F: Baumartenmischung | S: soziale Mischung | F: "Jeder Wald hat eine spezifische **Baumartenmischung** und -verteilung." ↔ S: "Von **sozialer Mischung** wird in der Stadtsoziologie gesprochen." |

### Wert — Forst (materiell) vs. Soziologie (normativ)

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 35 | F: Holzwert | S: sozialer Wert | F: "Der **Holzwert** steigt mit zunehmendem Durchmesser des Stammes." ↔ S: "**Soziale Werte** sind Vorstellungen über das Wünschenswerte." |

### Schicht — Forst (vertikal) vs. Soziologie (horizontal)

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 36 | F: Wald-Schichtung | S: soziale Schicht | F: "Die Pflanzen bilden deutlich unterscheidbare **Schichten**: Kronenschicht, Strauchschicht, Krautschicht." ↔ S: "**Soziale Schicht** = Gruppe mit gemeinsamem sozioökonomischen Status." |

### Struktur — Forst vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 37 | F: Bestandesstruktur | S: Gruppenstruktur | F: "Die **Bestandesstruktur** beschreibt den Aufbau eines Waldes." ↔ S: "Jede Gruppe weist eine soziale **Struktur** – eine Gruppenstruktur – auf." |

### Position — Gurtzeug vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 38 | G: Sitzposition | S: Alpha-Position | G: "Das Gurtzeug soll dem Piloten eine entspannte **Position** ermöglichen." ↔ S: "Die **Alpha-Position** ist die ranghöchste und ihr Inhaber hat den grössten Einfluss." |

### Gefüge — Forst vs. Soziologie

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 39 | F: Bestandesgefüge | S: Sozialgefüge | F: "Nadelbäume: Aufbau eines **Bestandesgefüges**." ↔ S: "**Sozialgefüge** = Geflecht sozialer Beziehungen." |

### Potenzial — Forst vs. allgemein

| # | Domäne A | Domäne B | Sätze |
|---|---------|---------|-------|
| 40 | F: WuchsPotenzial | S: Entwicklungspotenzial | F: "Das **WuchsPotenzial** des Standortes bestimmt die Baumartenwahl." ↔ S: "Das menschliche **Entwicklungspotenzial** entfaltet sich in sozialer Interaktion." |

---

## Validierungslogik

- **Synonyme (1–20):** Erwartung: Cos-Sim **steigt** (Delta > 0)
- **Homonyme (21–40):** Erwartung: Cos-Sim **sinkt** (Delta < 0)
- **Kollaps-Warnung:** Wenn alle 40 Paare nach dem Training > 0.97 Ähnlichkeit aufweisen

Stand: 16. Mai 2026 — Extrahiert aus 4 Domänen, 12 Quelldateien, ~15.000 Zeilen Text.
