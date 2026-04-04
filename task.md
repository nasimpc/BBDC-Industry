# Bremen Big Data Challenge 2026

## Daten-Bereitstellung

Die Daten sind in Trainings- und Testdaten unterteilt. Alle Daten stehen in folgenden Ordnern zur Verfügung:
- **Training Set**: 100 Dokumente mit Labels (`train/` Ordner + `train_labels.csv`)
- **Test Set**: 1400 Dokumente ohne Labels (`test/` Ordner + `test_skeleton.csv`)

## Aufgabenstellung

Die Bremen Big Data Challenge 2026 behandelt das Aufgabengebiet der **Multi-Label Text-Klassifikation** mit Fokus auf die **Sustainable Development Goals ([SDGs](https://sdgs.un.org/goals))** der Vereinten Nationen.

### Hintergrund zu den SDGs

Die 17 Sustainable Development Goals (Ziele für nachhaltige Entwicklung) wurden 2015 von den Vereinten Nationen verabschiedet und bilden einen globalen Plan zur Förderung nachhaltigen Friedens und Wohlstands und zum Schutz unseres Planeten. Die 17 Ziele umfassen:

1. Keine Armut - Beendigung der Armut in all ihren Formen überall auf der Welt durch Zugang zu grundlegenden Ressourcen und sozialen Sicherungssystemen.
2. Kein Hunger - Sicherstellung der Ernährungssicherheit, Verbesserung der Ernährung und Förderung einer nachhaltigen Landwirtschaft für alle Menschen.
3. Gesundheit und Wohlergehen - Gewährleistung eines gesunden Lebens und Förderung des Wohlbefindens für alle Menschen in jedem Alter.
4. Hochwertige Bildung - Sicherstellung einer inklusiven, chancengerechten und hochwertigen Bildung sowie Förderung lebenslangen Lernens für alle.
5. Geschlechtergleichheit - Erreichen der Gleichstellung der Geschlechter und Stärkung aller Frauen und Mädchen zur Selbstbestimmung.
6. Sauberes Wasser und Sanitäreinrichtungen - Gewährleistung der Verfügbarkeit und nachhaltigen Bewirtschaftung von Wasser und Sanitärversorgung für alle.
7. Bezahlbare und saubere Energie - Sicherstellung des Zugangs zu bezahlbarer, verlässlicher, nachhaltiger und moderner Energie für alle Menschen.
8. Menschenwürdige Arbeit und Wirtschaftswachstum - Förderung von nachhaltigem, inklusivem Wirtschaftswachstum, produktiver Vollbeschäftigung und menschenwürdiger Arbeit für alle.
9. Industrie, Innovation und Infrastruktur - Aufbau widerstandsfähiger Infrastrukturen, Förderung inklusiver und nachhaltiger Industrialisierung sowie Unterstützung von Innovationen.
10. Weniger Ungleichheiten - Verringerung der Ungleichheit innerhalb von und zwischen Staaten durch gezielte politische Maßnahmen.
11. Nachhaltige Städte und Gemeinden - Gestaltung von Städten und Siedlungen inklusiv, sicher, widerstandsfähig und nachhaltig für alle Bewohner.
12. Nachhaltige/r Konsum und Produktion - Sicherstellung nachhaltiger Konsum- und Produktionsmuster durch effiziente Nutzung von Ressourcen und Vermeidung von Abfall.
13. Maßnahmen zum Klimaschutz - Ergreifung dringender Maßnahmen zur Bekämpfung des Klimawandels und seiner Auswirkungen auf globaler Ebene.
14. Leben unter Wasser - Erhaltung und nachhaltige Nutzung der Ozeane, Meere und Meeresressourcen für eine nachhaltige Entwicklung.
15. Leben an Land - Schutz, Wiederherstellung und Förderung der nachhaltigen Nutzung von terrestrischen Ökosystemen und Wäldern sowie Bekämpfung der Wüstenbildung.
16. Frieden, Gerechtigkeit und starke Institutionen - Förderung friedlicher und inklusiver Gesellschaften, Zugang zur Justiz für alle und Aufbau effektiver, rechenschaftspflichtiger Institutionen.
17. Partnerschaften zur Erreichung der Ziele - Stärkung der Umsetzungsmittel und Wiederbelebung der globalen Partnerschaft für nachhaltige Entwicklung.

### Die Aufgabe

Ziel ist es, für politische Dokumente (z.B. parlamentarische Anfragen, Drucksachen) automatisch zu erkennen, welche der 17 SDGs in den jeweiligen Texten thematisiert werden. Ein Dokument kann **mehrere SDGs gleichzeitig** behandeln (Multi-Label Klassifikation) oder auch kein SDG behandeln.

Die Challenge beinhaltet:
- **100 gelabelte Trainingsdokumente** mit Informationen darüber, welche SDGs in den jeweiligen Dokumenten vorkommen
- **1400 Test-Dokumente**, für die die SDG-Zuordnungen vorhergesagt werden sollen

Für jedes Dokument soll für jedes der 17 SDGs vorhergesagt werden, ob es im Dokument thematisiert wird (1) oder nicht (0).

### 1. Training Set: `train/` Ordner
Enthält 100 Textdokumente (`doc_0.txt` bis `doc_99.txt`) mit politischen Texten.

### 2. Training Labels: `train_labels.csv`
Enthält die Labels für die Trainingsdaten im folgenden Format:

| Spalte | Beschreibung |
|---|---|
| `doc_id` | Eindeutige Dokument-ID (z.B. `doc_0`, `doc_1`, ...) |
| `SDG1` bis `SDG17` | Binäre Labels (0 oder 1) für jedes der 17 SDGs |

**Werte:**
- `1` = SDG kommt in diesem Dokument vor
- `0` = SDG kommt nicht vor

**Beispiel:**
```
doc_id,SDG1,SDG2,SDG3,SDG4,SDG5,SDG6,SDG7,SDG8,SDG9,SDG10,SDG11,SDG12,SDG13,SDG14,SDG15,SDG16,SDG17
doc_0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0
doc_1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
doc_2,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0
...
```

In diesem Beispiel:
- `doc_0` behandelt SDG 11
- `doc_1` behandelt kein SDG
- `doc_2` behandelt SDG 10, 11 und 16 (Multi-Label)

### 3. Test Set: `test/` Ordner
Enthält 1400 Textdokumente (`doc_100.txt` bis `doc_1499.txt`), für die die SDG-Zuordnungen vorhergesagt werden sollen.

### 4. Test Skeleton: `test_skeleton.csv`
Diese Datei gibt die Struktur für die Vorhersagen vor und soll mit den vorhergesagten SDG-Labels "gefüllt" werden. Die Datei hat das gleiche Format wie `train_labels.csv`:

```
doc_id,SDG1,SDG2,SDG3,SDG4,SDG5,SDG6,SDG7,SDG8,SDG9,SDG10,SDG11,SDG12,SDG13,SDG14,SDG15,SDG16,SDG17
doc_100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
doc_101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
...
```

**Wichtig:** 
- Die Reihenfolge der Dokument-IDs und die Anzahl der Zeilen sollten **nicht verändert** werden
- Alle Zellen müssen mit `0` oder `1` ausgefüllt werden
- Die erste Zeile (Header) muss unverändert bleiben

### Beispiel-Dokumente
Im Ordner `examples` befinden sich annotierte Beispiel-Dokumente, in denen das Vorkommen der SDGs markiert und eine kurze Erklärung/Begründung angegeben ist, warum die gekennzeichnete Textstelle dem Ziel zugeordnet werden kann.

## Abgabe

Die ausgefüllte Datei `test_skeleton.csv` kann im BBDC 2026 Submission Portal hochgeladen werden.

**Beispiel einer ausgefüllten Submission:**
```
doc_id,SDG1,SDG2,SDG3,SDG4,SDG5,SDG6,SDG7,SDG8,SDG9,SDG10,SDG11,SDG12,SDG13,SDG14,SDG15,SDG16,SDG17
doc_100,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0
doc_101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
doc_102,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0
...
```

## Scoring: Macro-Averaged F1-Score

Der finale Score wird über den **Macro-Averaged F1-Score** über alle 17 SDGs berechnet. Je höher der Score, desto besser. Der minimale Score beträgt 0.0 (0%) und der maximale Score beträgt 1.0 (100%).

Für jedes Dokument $d \in D$ und jedes SDG $s \in {1, 2, ..., 17}$ wird vorhergesagt:
$$y_{d,s} \in {0, 1} \quad \text{wobei } y_{d,s} = \begin{cases} 1 & \text{SDG } s \text{ kommt in Dokument } d \text{ vor} \\ 0 & \text{SDG } s \text{ kommt nicht vor} \end{cases}$$
Die Vorhersage wird als $\hat{y}_{d,s} \in {0, 1}$ notiert.

**Berechnung**

**Schritt 1: F1-Score pro SDG**

Für jedes SDG $s$ werden über alle Dokumente berechnet:
$$TP_s = \sum_{d \in D} \mathbb{1}[y_{d,s} = 1 \land \hat{y}_{d,s} = 1]$$
$$FP_s = \sum_{d \in D} \mathbb{1}[y_{d,s} = 0 \land \hat{y}_{d,s} = 1]$$
$$FN_s = \sum_{d \in D} \mathbb{1}[y_{d,s} = 1 \land \hat{y}_{d,s} = 0]$$
$$\text{Precision}_s = \frac{TP_s}{TP_s + FP_s}, \quad \text{Recall}_s = \frac{TP_s}{TP_s + FN_s}$$
$$F1_s = \frac{2 \cdot \text{Precision}_s \cdot \text{Recall}_s}{\text{Precision}_s + \text{Recall}_s}$$
**Sonderfälle:**

Falls $TP_s + FP_s = 0$, dann $\text{Precision}_s = 0$
Falls $TP_s + FN_s = 0$, dann $\text{Recall}_s = 0$
Falls $\text{Precision}_s + \text{Recall}_s = 0$, dann $F1_s = 0$

**Schritt 2: Macro-Averaging**
$$\text{Score}{\text{final}} = \frac{1}{17} \sum{s=1}^{17} F1_s$$

---

# Bremen Big Data Challenge 2026

## Data Provision

The data is divided into training and test data. All data is available in the following folders:
- **Training Set**: 100 documents with labels (`train/` folder + `train_labels.csv`)
- **Test Set**: 1400 documents without labels (`test/` folder + `test_skeleton.csv`)

## Task Description

The Bremen Big Data Challenge 2026 addresses the task area of **Multi-Label Text Classification** with a focus on the United Nations' **Sustainable Development Goals ([SDGs](https://sdgs.un.org/goals))**.

### Background on the SDGs

The 17 Sustainable Development Goals were adopted by the United Nations in 2015 and form a global plan to promote sustainable peace and prosperity and to protect our planet. The 17 goals include:

1. No Poverty - Ending poverty in all its forms everywhere through access to basic resources and social security systems.
2. Zero Hunger - Ensuring food security, improving nutrition, and promoting sustainable agriculture for all people.
3. Good Health and Well-being - Ensuring healthy lives and promoting well-being for all people at all ages.
4. Quality Education - Ensuring inclusive, equitable, and quality education and promoting lifelong learning opportunities for all.
5. Gender Equality - Achieving gender equality and empowering all women and girls for self-determination.
6. Clean Water and Sanitation - Ensuring availability and sustainable management of water and sanitation for all.
7. Affordable and Clean Energy - Ensuring access to affordable, reliable, sustainable, and modern energy for all people.
8. Decent Work and Economic Growth - Promoting sustained, inclusive economic growth, productive full employment, and decent work for all.
9. Industry, Innovation and Infrastructure - Building resilient infrastructure, promoting inclusive and sustainable industrialization, and fostering innovation.
10. Reduced Inequalities - Reducing inequality within and among countries through targeted policy measures.
11. Sustainable Cities and Communities - Making cities and human settlements inclusive, safe, resilient, and sustainable for all residents.
12. Responsible Consumption and Production - Ensuring sustainable consumption and production patterns through efficient use of resources and waste reduction.
13. Climate Action - Taking urgent action to combat climate change and its impacts at a global level.
14. Life Below Water - Conserving and sustainably using the oceans, seas, and marine resources for sustainable development.
15. Life on Land - Protecting, restoring, and promoting sustainable use of terrestrial ecosystems and forests, and combating desertification.
16. Peace, Justice and Strong Institutions - Promoting peaceful and inclusive societies, access to justice for all, and building effective, accountable institutions.
17. Partnerships for the Goals - Strengthening the means of implementation and revitalizing the global partnership for sustainable development.

### The Task

The goal is to automatically identify which of the 17 SDGs are addressed in political documents (e.g., parliamentary inquiries, printed matters). A document can address **multiple SDGs simultaneously** (multi-label classification) or no SDG at all.

The challenge includes:
- **100 labeled training documents** with information about which SDGs occur in the respective documents
- **1400 test documents** for which the SDG assignments should be predicted

For each document, it should be predicted for each of the 17 SDGs whether it is addressed in the document (1) or not (0).

### 1. Training Set: `train/` folder
Contains 100 text documents (`doc_0.txt` to `doc_99.txt`) with political texts.

### 2. Training Labels: `train_labels.csv`
Contains the labels for the training data in the following format:

| Column | Description |
|---|---|
| `doc_id` | Unique document ID (e.g., `doc_0`, `doc_1`, ...) |
| `SDG1` to `SDG17` | Binary labels (0 or 1) for each of the 17 SDGs |

**Values:**
- `1` = SDG occurs in this document
- `0` = SDG does not occur

**Example:**
```
doc_id,SDG1,SDG2,SDG3,SDG4,SDG5,SDG6,SDG7,SDG8,SDG9,SDG10,SDG11,SDG12,SDG13,SDG14,SDG15,SDG16,SDG17
doc_0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0
doc_1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
doc_2,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0
...
```

In this example:
- `doc_0` addresses SDG 11
- `doc_1` addresses no SDG
- `doc_2` addresses SDG 10, 11, and 16 (multi-label)

### 3. Test Set: `test/` folder
Contains 1400 text documents (`doc_100.txt` to `doc_1499.txt`) for which the SDG assignments should be predicted.

### 4. Test Skeleton: `test_skeleton.csv`
This file provides the structure for the predictions and should be "filled" with the predicted SDG labels. The file has the same format as `train_labels.csv`:

```
doc_id,SDG1,SDG2,SDG3,SDG4,SDG5,SDG6,SDG7,SDG8,SDG9,SDG10,SDG11,SDG12,SDG13,SDG14,SDG15,SDG16,SDG17
doc_100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
doc_101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
...
```

**Important:** 
- The order of document IDs and the number of rows should **not be changed**
- All cells must be filled with `0` or `1`
- The first row (header) must remain unchanged

### Example documents
The folder `examples` contains annotated text documents, in which the sdg occurence is marked. Additionally, each relevant section has an explanation why the sdgs were chosen.

## Submission

The completed file `test_skeleton.csv` can be uploaded to the BBDC 2026 Submission Portal.

**Example of a completed submission:**
```
doc_id,SDG1,SDG2,SDG3,SDG4,SDG5,SDG6,SDG7,SDG8,SDG9,SDG10,SDG11,SDG12,SDG13,SDG14,SDG15,SDG16,SDG17
doc_100,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0
doc_101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
doc_102,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0
...
```

## Scoring: Macro-Averaged F1-Score

The final score is calculated using the **Macro-Averaged F1-Score** across all 17 SDGs. The higher the score, the better. The minimum score is 0.0 (0%) and the maximum score is 1.0 (100%).

For each document $d \in D$ and each SDG $s \in {1, 2, ..., 17}$, the prediction is:
$$y_{d,s} \in {0, 1} \quad \text{where } y_{d,s} = \begin{cases} 1 & \text{SDG } s \text{ occurs in document } d \\ 0 & \text{SDG } s \text{ does not occur} \end{cases}$$
The prediction is denoted as $\hat{y}_{d,s} \in {0, 1}$.

**Calculation**

**Step 1: F1-Score per SDG**

For each SDG $s$, calculated across all documents:
$$TP_s = \sum_{d \in D} \mathbb{1}[y_{d,s} = 1 \land \hat{y}_{d,s} = 1]$$
$$FP_s = \sum_{d \in D} \mathbb{1}[y_{d,s} = 0 \land \hat{y}_{d,s} = 1]$$
$$FN_s = \sum_{d \in D} \mathbb{1}[y_{d,s} = 1 \land \hat{y}_{d,s} = 0]$$
$$\text{Precision}_s = \frac{TP_s}{TP_s + FP_s}, \quad \text{Recall}_s = \frac{TP_s}{TP_s + FN_s}$$
$$F1_s = \frac{2 \cdot \text{Precision}_s \cdot \text{Recall}_s}{\text{Precision}_s + \text{Recall}_s}$$
**Special cases:**

If $TP_s + FP_s = 0$, then $\text{Precision}_s = 0$
If $TP_s + FN_s = 0$, then $\text{Recall}_s = 0$
If $\text{Precision}_s + \text{Recall}_s = 0$, then $F1_s = 0$

**Step 2: Macro-Averaging**
$$\text{Score}_{\text{final}} = \frac{1}{17} \sum_{s=1}^{17} F1_s$$