# Falken-Klassifikation 🦅

In diesem Projekt wird der **Hawks-Datensatz** verwendet, um Falkenarten anhand biologischer Merkmale zu klassifizieren.  
Das Ziel ist, verschiedene klassische Machine-Learning-Modelle miteinander zu vergleichen und ein reproduzierbares, gut strukturiertes Data-Science-Projekt für Bewerbungen zu zeigen.

---

##  Projektüberblick

**Fragestellung:**  
Kann man Falkenarten anhand weniger biologischer Merkmale (z. B. Flügelspannweite, Gewicht) zuverlässig klassifizieren?

**Schritte im Projekt:**

1. Daten laden und verstehen  
2. Daten bereinigen und vorbereiten  
3. Relevante Features auswählen  
4. Daten skalieren  
5. Mehrere Klassifikationsmodelle trainieren  
6. Modellgenauigkeit vergleichen  
7. Confusion-Matrizen und Reports auswerten  

Verwendete Modelle:

- Support Vector Machine (verschiedene Kernel)
- Logistische Regression
- Gaussian Naive Bayes
- Random Forest

Die eigentliche Analyse befindet sich im Notebook  
`notebooks/falken_classifikation.ipynb`.

---

##  Projektstruktur

```text
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
│   └── evaluation.py                 # Auswertung & Visualisierung (Accuracy, Confusion Matrix)
│
├── requirements.txt                  # Python-Abhängigkeiten
└── README.md
