"""
Descarga y muestreo reproducible del dataset "All Lending Club loan data"
(Kaggle: wordsforthewise/lending-club).

Por qué muestreamos en vez de usar el dataset completo: accepted tiene
~2.26M filas y rejected ~27.6M filas. Para un proyecto de portafolio no
se necesita esa escala -- una muestra de ~150k filas por archivo (~300k
en total) ya es representativa para entrenar modelos de demanda, PD y
LGD, y reduce drásticamente los tiempos de EDA, entrenamiento e
iteración en notebooks.

Método de muestreo: en vez de un muestreo por chunks con probabilidad
aproximada, contamos primero el número exacto de filas de datos de cada
archivo (excluyendo el header) y elegimos, con numpy.random.RandomState
sembrado en random_state=42, exactamente n_sample índices de fila sin
reemplazo. El resto de las filas se pasan a `pandas.read_csv` como
`skiprows`, de forma que pandas nunca materializa en memoria las filas
descartadas. Esto da una muestra aleatoria simple, exacta y
100% reproducible (mismo random_state -> mismas filas), sin sesgo de
"primeras N filas" ni de orden cronológico del archivo original.
"""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DOWNLOAD_DIR = RAW_DIR / "_download"
KAGGLE_DATASET = "wordsforthewise/lending-club"

FILES = {
    "accepted": "accepted_2007_to_2018Q4.csv.gz",
    "rejected": "rejected_2007_to_2018Q4.csv.gz",
}

N_SAMPLE = 150_000
RANDOM_STATE = 42


def download_if_missing() -> None:
    """Descarga ambos archivos crudos vía Kaggle CLI si no están presentes."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        dest = DOWNLOAD_DIR / fname
        if dest.exists():
            continue
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-f",
                fname,
                "-p",
                str(DOWNLOAD_DIR),
            ],
            check=True,
        )


def count_data_rows(path: Path) -> int:
    """Cuenta filas de datos (excluye el header) de un csv.gz grande sin cargarlo en memoria."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        n = sum(1 for _ in fh)
    return n - 1


def sample_csv_gz(path: Path, n_total_rows: int, n_sample: int, random_state: int) -> pd.DataFrame:
    """Toma una muestra aleatoria simple y reproducible de n_sample filas de un csv.gz."""
    rng = np.random.RandomState(random_state)
    chosen = rng.choice(n_total_rows, size=n_sample, replace=False)

    keep_mask = np.zeros(n_total_rows, dtype=bool)
    keep_mask[chosen] = True
    # +1 porque la fila 0 del archivo es el header (no se debe saltar)
    skip_rows = np.nonzero(~keep_mask)[0] + 1

    return pd.read_csv(path, skiprows=skip_rows, low_memory=False)


def main() -> None:
    download_if_missing()

    for key, fname in FILES.items():
        src_path = DOWNLOAD_DIR / fname
        out_path = RAW_DIR / f"{key}_sample.csv.gz"

        n_total = count_data_rows(src_path)
        print(f"[{key}] filas totales en el archivo original: {n_total:,}")

        sample_df = sample_csv_gz(
            src_path, n_total_rows=n_total, n_sample=N_SAMPLE, random_state=RANDOM_STATE
        )
        sample_df.to_csv(out_path, index=False, compression="gzip")
        print(f"[{key}] muestra guardada: {out_path} ({len(sample_df):,} filas)")


if __name__ == "__main__":
    main()
