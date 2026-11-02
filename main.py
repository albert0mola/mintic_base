"""Script ejecutor básico del proyecto mintic-base."""
import pandas as pd

DATASET_PATH = "data/sample_dataset.csv"


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Carga el dataset de prueba desde data/ usando pandas."""
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())

    from mintic.ensemble.random_forest import (
        build_random_forest,
        predict_ensemble,
    )

    # El reto trabaja con datos categóricos; separamos la última columna como
    # etiqueta y usamos las demás columnas como características.
    X = df.iloc[:, :-1].astype(str).to_numpy()
    y = df.iloc[:, -1].astype(str).to_numpy()
    forest = build_random_forest(X, y, n_trees=10, random_state=42)
    predictions = predict_ensemble(forest, X[:5])
    print("Predicciones de ejemplo:", predictions)
