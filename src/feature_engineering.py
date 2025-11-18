from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_features(
    X_train,
    X_test,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Skaliert Features mit StandardScaler.

    Der Scaler wird auf X_train gefittet und auf X_train & X_test angewendet.

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray
    scaler : StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

