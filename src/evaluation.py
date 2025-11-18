from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title: Optional[str] = None,
    normalize: Optional[str] = None,
):
    """
    Zeichnet eine Confusion Matrix für ein Modell.

    Parameters
    ----------
    normalize : {'true', 'pred', 'all', None}
        Optional: Normalisierungsmethode der Confusion-Matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", colorbar=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Gibt den Classification Report leserlich aus.
    """
    print(classification_report(y_true, y_pred, target_names=target_names))


def build_accuracy_table(results: Dict[str, float]) -> pd.DataFrame:
    """
    Baut aus einem Dict {modellname: accuracy} ein sortiertes DataFrame.
    """
    df = (
        pd.DataFrame.from_dict(results, orient="index", columns=["Genauigkeit"])
        .sort_values("Genauigkeit", ascending=False)
    )
    return df


def plot_accuracy_bar(
    df: pd.DataFrame,
    ymin: float = 0.9,
    ymax: float = 1.0,
    title: str = "Genauigkeitsvergleich der Modelle",
):
    """
    Balkendiagramm der Modellgenauigkeiten mit Beschriftung über den Balken.
    """
    plt.figure(figsize=(10, 5))
    bars = plt.bar(df.index, df["Genauigkeit"])

    plt.ylim(ymin, ymax)
    plt.ylabel("Genauigkeit")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    # Werte über die Balken schreiben
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.002,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()
