# 🦅 Falken-Klassifikation – Machine Learning Projekt
## SVM • Random Forest • Logistic Regression • Naive Bayes • Feature Engineering

In diesem Projekt wird der **Hawks-Datensatz** verwendet, um Falkenarten anhand biologischer Merkmale zu klassifizieren.  
Das Ziel ist, verschiedene klassische Machine-Learning-Modelle miteinander zu vergleichen und ein reproduzierbares, gut strukturiertes Data-Science-Projekt für Bewerbungen zu zeigen.

---

## Projektbeschreibung

Dieses Projekt klassifiziert Falkenarten anhand biologischer Merkmale des Hawks-Datensatzes.
Es zeigt eine vollständige Machine-Learning-Pipeline mit:
- Datenbereinigung
- Feature Engineering
- Skalierung der Merkmale
- Training mehrerer Klassifikationsmodelle
- Genauigkeitsvergleich
- Confusion-Matrizen
- Modularer Code über src/

Das Projekt dient als Demonstrator für saubere Datenanalyse, Modellierung und strukturierte Python-Projektorganisation.
---
##  Projektstruktur

FALKEN_KLASSIFIKATION/
│
├── data/
│   └── hawks.csv                     # Datensatz (lokale Kopie des Hawks-Datensatzes)
│
├── notebooks/
│   └── falken_classifikation.ipynb   # Haupt-Notebook mit der Analyse
│
├── src/                              # Wiederverwendbare Python-Module
│   ├── __init__.py
│   ├── data_preprocessing.py         # Laden, Bereinigung, Train/Test-Split
│   ├── feature_engineering.py        # Skalierung, weitere Transformationen
│   ├── model_training.py             # Training verschiedener ML-Modelle
│   └── evaluation.py                 # Auswertung & Visualisierung
│
├── requirements.txt                  # Python-Abhängigkeiten
└── README.md


---

## Vorgehensweise
### 1. Datenvorbereitung
- CSV einlesen
- Relevante Features auswählen
- Zielvariable encodieren
- Grundlegende Statistiken anzeigen
### 2. Datenaufteilung & Skalierung
***X_train, X_test, y_train, y_test = split_data(X, y)***
***X_train_s, X_test_s, scaler = scale_features(X_train, X_test)***
### 3. Training verschiedener Modelle

Verwendete Modelle:
- SVC (linear, poly, rbf, sigmoid)
- Logistische Regression
- Gaussian Naive Bayes
- Random Forest
### 4. Evaluation & Visualisierung
- Accuracy-Vergleich
- Confusion-Matrix
- Classification Report
### 5.  Ergebnisse
- Modelle erreichen 95–99 % Genauigkeit
- Beste Modelle:
      - SVC (RBF)
      - Random Forest
- Confusion-Matrizen zeigen eine sehr präzise Klassifikation
- Skalierung verbessert besonders SVM-Modelle deutlich


| Bereich          | Technologie                |
|------------------|---------------------------|
| Programmiersprache | Python                  |
| Datenanalyse     | Pandas, NumPy             |
| Machine Learning | Scikit-Learn             |
| Visualisierung   | Matplotlib, Seaborn       |
| Projektstruktur  | modularer Code (`src/`)   |


### Nutzung
**Repository klonen**

git clone https://github.com/NataliaArchipenko/FALKEN_KLASSIFIKATION.git

**Requirements installieren**

pip install -r requirements.txt

**Notebook starten**

jupyter notebook


Autorin

Natalia Archipenko
Fachinformatikerin für Daten- und Prozessanalyse

Schwerpunkte:
Datenanalyse • Machine Learning • Klassifikation • Feature Engineering

LinkedIn: www.linkedin.com/in/natalia-archipenko-335357271
---








