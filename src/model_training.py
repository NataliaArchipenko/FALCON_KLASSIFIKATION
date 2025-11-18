from typing import Tuple, Dict

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_svc(
    kernel: str,
    X_train,
    y_train,
    random_state: int = 42,
    **kwargs,
) -> SVC:
    """
    Trainiert ein SVC-Modell mit gegebenem Kernel.
    """
    model = SVC(kernel=kernel, random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train,
    y_train,
    max_iter: int = 200,
    **kwargs,
) -> LogisticRegression:
    """
    Trainiert eine logistische Regression.
    """
    model = LogisticRegression(max_iter=max_iter, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(
    X_train,
    y_train,
) -> GaussianNB:
    """
    Trainiert ein Gaussian Naive Bayes Modell.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train,
    y_train,
    random_state: int = 42,
    **kwargs,
) -> RandomForestClassifier:
    """
    Trainiert einen RandomForest-Klassifikator.
    """
    model = RandomForestClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> Tuple[float, np.ndarray]:
    """
    Berechnet die Genauigkeit und gibt gleichzeitig die Vorhersagen zurück.

    Returns
    -------
    accuracy : float
        Accuracy-Score.
    y_pred : np.ndarray
        Modellvorhersagen.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred


def train_svc_variants(
    kernels,
    X_train,
    y_train,
    X_test,
    y_test,
) -> Dict[str, Dict[str, object]]:
    """
    Trainiert mehrere SVC-Varianten und liefert eine Zusammenfassung.

    Returns
    -------
    results : dict
        z.B.:
        {
          "SVC (rbf)": {"model": model, "accuracy": 0.98, "y_pred": array(...)},
          ...
        }
    """
    results = {}

    for kernel in kernels:
        name = f"SVC ({kernel})"
        model = train_svc(kernel, X_train, y_train)
        acc, y_pred = evaluate_model(model, X_test, y_test)
        results[name] = {"model": model, "accuracy": acc, "y_pred": y_pred}

    return results
