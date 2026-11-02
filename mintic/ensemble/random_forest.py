"""Random Forest basado en árboles ID3 para datos categóricos.

La implementación evita librerías que ya contienen algoritmos de aprendizaje
automático. NumPy se utiliza únicamente para arreglos y muestreo.
"""

from __future__ import annotations

import numpy as np


def bootstrap_sample(
    X: np.ndarray, y: np.ndarray, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene una muestra de ``N`` filas con reemplazo."""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional.")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y debe ser un vector con una etiqueta por fila de X.")
    if len(X) == 0:
        raise ValueError("X y y no pueden estar vacíos.")

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(X), size=len(X))
    return X[indices], y[indices]


def _class_sort_key(value: object) -> str:
    """Clave estable para desempatar clases alfabéticamente."""
    return str(value)


def _majority_class(y: np.ndarray) -> object:
    """Devuelve la clase más frecuente; en empate, la primera alfabéticamente."""
    values, counts = np.unique(y, return_counts=True)
    highest_count = counts.max()
    tied = values[counts == highest_count]
    return sorted(tied.tolist(), key=_class_sort_key)[0]


def _entropy(y: np.ndarray) -> float:
    """Calcula la entropía de Shannon de un vector de clases."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _information_gain(feature: np.ndarray, y: np.ndarray) -> float:
    """Calcula la ganancia de información de una característica."""
    parent_entropy = _entropy(y)
    weighted_entropy = 0.0

    for value in np.unique(feature):
        subset = y[feature == value]
        weighted_entropy += (len(subset) / len(y)) * _entropy(subset)

    return parent_entropy - weighted_entropy


def build_id3_tree(X: np.ndarray, y: np.ndarray) -> dict:
    """Construye recursivamente un árbol ID3 para características categóricas."""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional.")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y debe ser un vector con una etiqueta por fila de X.")
    if len(y) == 0:
        raise ValueError("No se puede construir un árbol sin datos.")

    def grow(features: np.ndarray, labels: np.ndarray, feature_indices: list[int]):
        majority = _majority_class(labels)

        if len(np.unique(labels)) == 1:
            return {"type": "leaf", "class": labels[0]}
        if not feature_indices:
            return {"type": "leaf", "class": majority}

        gains = np.array(
            [_information_gain(features[:, i], labels) for i in feature_indices]
        )
        best_position = int(np.argmax(gains))
        best_feature = feature_indices[best_position]

        node = {
            "type": "node",
            "feature": best_feature,
            "majority": majority,
            "children": {},
        }
        remaining_features = [
            index for index in feature_indices if index != best_feature
        ]

        for value in np.unique(features[:, best_feature]):
            mask = features[:, best_feature] == value
            node["children"][value] = grow(
                features[mask], labels[mask], remaining_features
            )

        return node

    return grow(X, y, list(range(X.shape[1])))


def build_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_trees: int = 10,
    random_state: int | None = None,
) -> list[dict]:
    """Entrena ``n_trees`` árboles ID3 usando muestras bootstrap."""
    if not isinstance(n_trees, (int, np.integer)) or n_trees <= 0:
        raise ValueError("n_trees debe ser un entero positivo.")

    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    trees = []

    for _ in range(int(n_trees)):
        sample_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        X_sample, y_sample = bootstrap_sample(X, y, random_state=sample_seed)
        trees.append(build_id3_tree(X_sample, y_sample))

    return trees


def _predict_tree(tree: dict, row: np.ndarray) -> object:
    """Obtiene la predicción de un árbol para una sola fila."""
    current = tree
    while current["type"] != "leaf":
        feature = current["feature"]
        value = row[feature]
        child = current["children"].get(value)
        if child is None:
            return current["majority"]
        current = child
    return current["class"]


def predict_ensemble(trees: list[dict], X: np.ndarray) -> np.ndarray:
    """Predice por voto mayoritario y desempata alfabéticamente."""
    if not trees:
        raise ValueError("trees debe contener al menos un árbol.")

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional.")

    predictions = []
    for row in X:
        votes = [_predict_tree(tree, row) for tree in trees]
        predictions.append(_majority_class(np.asarray(votes, dtype=object)))

    return np.asarray(predictions)
