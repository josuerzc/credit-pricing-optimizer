"""
Modelo de demanda: probabilidad de que un solicitante acepte un préstamo
(`acepto_prestamo`) en función de la tasa y su perfil de riesgo.

Se entrena a nivel de CLIENTE INDIVIDUAL (ver README, sección "Decisión
metodológica: granularidad individual vs. sub_grade") usando solo
features presentes en ambos datasets (accepted y rejected), porque el
dataset de entrenamiento es la unión de los dos.

--------------------------------------------------------------------------
Dos decisiones de feature engineering que resuelven asimetrías entre
accepted y rejected (ninguna se resuelve en la limpieza de la Fase 2/3
porque son decisiones de MODELADO, no de limpieza de datos):
--------------------------------------------------------------------------

1. `risk_percentile` -- normaliza `fico` (accepted) y `risk_score`
   (rejected) a una misma escala comparable. No son el mismo score
   (distinta fuente, distinto rango), así que en vez de forzarlos a una
   escala numérica común arbitraria, se convierte cada uno al percentil
   que ocupa DENTRO DE SU PROPIO DATASET (0-100, donde 100 = el perfil
   de menor riesgo relativo de ese dataset). Es una normalización
   ordinal, no una equivalencia de score real.

2. `tasa` -- `int_rate` solo existe para `accepted` (un rechazo nunca
   llega a que se le cotice una tasa). Si se usara `int_rate` crudo con
   NaN para todo `rejected`, el modelo aprendería trivialmente
   "tasa faltante -> rechazado" (missing e `acepto_prestamo=0` están
   perfectamente correlacionados por construcción), lo cual es
   inservible: el objetivo de este modelo es poder simular la
   probabilidad de aceptación de CUALQUIER cliente (incluidos los
   perfiles que hoy son rechazados) a distintas tasas candidatas
   (Fase 6). Para eso, se entrena un modelo auxiliar de pricing
   (`estimate_offered_rate`) SOLO sobre `accepted` -- que aprende qué
   tasa le pone LendingClub a un perfil de riesgo/monto/dti dado -- y se
   usa para imputar una "tasa contrafactual" a los rechazados: la tasa
   que un cliente con ese perfil habría recibido de haber sido evaluado
   para precio. Se documenta con un flag `tasa_estimada` para que quede
   trazable cuáles tasas son observadas y cuáles son estimadas.

--------------------------------------------------------------------------
Restricción de monotonicidad en `tasa` para el modelo XGBoost (hallazgo
de la Fase 6, ver notebooks/04_aggregation_subgrade.ipynb):
--------------------------------------------------------------------------

Sin restringir, XGBoost aprende una relación `tasa -> p(aceptar)` NO
monótona (incluso creciente en el tramo alto): en los datos de
entrenamiento, las tasas más altas quedan dominadas por préstamos de
`accepted` con grados de riesgo altos (G, F...) que SIEMPRE tienen
`acepto_prestamo=1` por construcción (todo lo que está en `accepted` fue
otorgado), mientras que la tasa contrafactual estimada para `rejected`
rara vez llega a esos extremos. El modelo termina aprendiendo un patrón
espurio de "a mayor tasa, más aceptación" en la cola alta, causado por
confusión entre "fue un préstamo realmente originado" y "el cliente
prefiere tasas altas" -- exactamente lo opuesto a la relación de demanda
que todo el proyecto necesita.

Esto NO es una elección de modelado discrecional: una curva de demanda
no decreciente en el precio no es economicamente admisible (es la
premisa base de Phillips 2013 y de cualquier problema de pricing). Por
eso `demand_model_pipeline` fuerza una restricción de monotonicidad
(`monotone_constraints`, soportado nativamente por XGBoost) que impone
`p(aceptar)` no creciente en `tasa`, sin restringir el resto de
features. Esto no es "hacer trampa" para forzar el resultado que
queremos -- es imponer un prior económico válido y bien establecido
sobre un modelo que, sin restricción, aprendió un artefacto espurio de
los datos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import PURPOSE_CATEGORIES

SHARED_NUMERIC_FEATURES = ["monto", "dti", "risk_percentile", "emp_length_years"]
SHARED_CATEGORICAL_FEATURES = ["purpose", "addr_state"]
DEMAND_FEATURES = ["tasa", *SHARED_NUMERIC_FEATURES, *SHARED_CATEGORICAL_FEATURES]

# Los 51 estados/territorios presentes en accepted+rejected (ver
# notebooks/01_eda.ipynb). Se fija la lista de categorías del encoder (en
# vez de dejar que se infieran del set de entrenamiento) para que el ancho
# de la matriz de features sea determinístico -- necesario para poder
# construir el vector de `monotone_constraints` de XGBoost sin tener que
# inspeccionar el encoder ya ajustado.
ADDR_STATE_CATEGORIES = [
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]


def add_risk_percentile(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega `risk_percentile`: percentil (0-100) de riesgo relativo dentro
    de cada dataset de origen (100 = perfil de menor riesgo relativo).
    """
    out = demand_df.copy()
    out["risk_percentile"] = np.nan

    is_accepted = out["source_dataset"] == "accepted"
    out.loc[is_accepted, "risk_percentile"] = out.loc[is_accepted, "fico"].rank(pct=True) * 100

    is_rejected = out["source_dataset"] == "rejected"
    out.loc[is_rejected, "risk_percentile"] = (
        out.loc[is_rejected, "risk_score"].rank(pct=True) * 100
    )

    return out


def _make_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Imputa (mediana) + escala las numéricas, one-hot-encodea las
    categóricas. Se imputa/escala incluso para modelos que no lo necesitan
    (XGBoost, HistGradientBoosting) para poder reusar el mismo
    preprocesamiento con LogisticRegression, que sí lo requiere.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_encoder = OneHotEncoder(
        categories=[PURPOSE_CATEGORIES, ADDR_STATE_CATEGORIES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_encoder, SHARED_CATEGORICAL_FEATURES),
        ]
    )


def _rate_model_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _make_preprocessor(SHARED_NUMERIC_FEATURES)),
            (
                "model",
                HistGradientBoostingRegressor(random_state=42, max_iter=200),
            ),
        ]
    )


def estimate_offered_rate(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Entrena un modelo de pricing sobre `accepted` (donde `int_rate` es
    observado) y lo usa para imputar una tasa contrafactual en `rejected`.

    Agrega dos columnas:
    - `tasa`: `int_rate` real para accepted; tasa contrafactual estimada
      para rejected.
    - `tasa_estimada`: bool, True solo para las filas de rejected.
    """
    out = demand_df.copy()

    train_mask = out["source_dataset"] == "accepted"
    feature_cols = SHARED_NUMERIC_FEATURES + SHARED_CATEGORICAL_FEATURES

    pipeline = _rate_model_pipeline()
    pipeline.fit(out.loc[train_mask, feature_cols], out.loc[train_mask, "int_rate"])

    out["tasa"] = out["int_rate"]
    out["tasa_estimada"] = False

    reject_mask = ~train_mask
    out.loc[reject_mask, "tasa"] = pipeline.predict(out.loc[reject_mask, feature_cols])
    out.loc[reject_mask, "tasa_estimada"] = True

    return out


def build_demand_features(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Aplica `add_risk_percentile` + `estimate_offered_rate` y devuelve el
    dataset listo para entrenar el modelo de demanda (columnas de
    `DEMAND_FEATURES` + `acepto_prestamo`).
    """
    out = add_risk_percentile(demand_df)
    out = estimate_offered_rate(out)
    return out


def demand_monotone_constraints() -> tuple[int, ...]:
    """Vector de `monotone_constraints` para XGBoost: -1 (no creciente) en
    `tasa`, sin restricción (0) en el resto. El orden debe coincidir
    exactamente con la salida de `_make_preprocessor`: numéricas primero
    (`tasa` es la primera), luego el one-hot de `purpose` + `addr_state`.
    """
    n_numeric = 1 + len(SHARED_NUMERIC_FEATURES)  # tasa + el resto de numéricas
    n_categorical = len(PURPOSE_CATEGORIES) + len(ADDR_STATE_CATEGORIES)
    return (-1,) + (0,) * (n_numeric - 1) + (0,) * n_categorical


def demand_model_pipeline(model) -> Pipeline:
    """Pipeline sklearn genérico (preprocesamiento + `model`) para el modelo
    de demanda. `model` es cualquier clasificador sklearn-compatible
    (LogisticRegression, XGBClassifier, ...). Si `model` es un XGBClassifier,
    debe construirse con `monotone_constraints=demand_monotone_constraints()`
    para que `tasa` tenga efecto no creciente sobre `p(aceptar)` (ver
    docstring del módulo).
    """
    preprocess = _make_preprocessor(["tasa", *SHARED_NUMERIC_FEATURES])
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])
