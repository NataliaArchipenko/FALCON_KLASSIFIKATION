import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple

# Standard-Einstellungen für dieses Projekt
DEFAULT_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Hawks.csv"
FEATURE_COLS = ["Wing", "Weight", "Culmen", "StandardTail", "KeelFat"]
TARGET_COL = "Species"


def load_data(url: str = DEFAULT_URL) -> pd.DataFrame:
    """
    Lädt den Hawks-Datensatz von einer URL (oder einem lokalen Pfad).

    Parameters
    ----------
    url : str
        URL oder Dateipfad zur CSV-Datei.

    Returns
    -------
    pd.DataFrame
        Geladener DataFrame.
    """
    return pd.read_csv(url)


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[pd.Series, LabelEncoder]:
    """
    Encodiert die Zielvariable mit LabelEncoder.

    Returns
    -------
    y : pd.Series (int)
        Enkodierte Zielvariable.
    le : LabelEncoder
        Trainierter Encoder (für spätere Inverse-Transformation / Labels).
    """
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    return pd.Series(y, name=target_col), le


def select_and_clean_features(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Wählt relevante Features aus, behandelt fehlende Werte und encodiert die Zielvariable.

    - numerische Features: fehlende Werte -> Spaltenmittelwert
    - Zielvariable: LabelEncoding

    Returns
    -------
    X : pd.DataFrame
        Bereinigte Eingangsvariablen.
    y : pd.Series
        Enkodierte Zielvariable.
    le : LabelEncoder
        Encoder für die Zielvariable.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = df.copy()

    # Features auswählen
    X = df[feature_cols]

    # Fehlende Werte durch Spaltenmittelwert ersetzen
    X = X.fillna(X.mean())

    # Zielvariable encodieren
    y, le = encode_target(df, target_col=target_col)

    return X, y, le


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    """
    Teilt die Daten in Trainings- und Testset.

    Parameters
    ----------
    stratify : bool
        Ob nach y geschichtet werden soll (empfohlen bei Klassifikation).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

