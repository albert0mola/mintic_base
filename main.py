"""Script ejecutor básico del proyecto mintic-base."""
import pandas as pd

DATASET_PATH = "data/sample_dataset.csv"


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Carga el dataset de prueba desde data/ usando pandas."""
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())

    # TODO(alumno): importar el subpaquete de mintic que desee probar, p. ej.:
    # from mintic.kmeans import KMeansClustering
