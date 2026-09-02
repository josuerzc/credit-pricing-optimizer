"""
Agregación de las predicciones individuales (demanda, PD, LGD) al nivel
de sub_grade que usa el optimizador (Fases 8-9).

Principio central (ver README, "Decisión metodológica: granularidad
individual vs. sub_grade"): NO se reentrena un modelo agregado. Se toman
los clientes reales de cada sub_grade, se corre el modelo individual
sobre cada uno, y se PROMEDIA. Solo `accepted` tiene `sub_grade`
asignado (es una etiqueta que LendingClub pone a los préstamos que sí
otorga, tras underwriting) -- por eso la agregación usa `accepted`, no
el dataset de demanda combinado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_model import LGD_INDUSTRY_BENCHMARK

# ~15 tasas candidatas cubriendo el rango real de int_rate observado en
# accepted (~5%-31%), usadas tanto para las curvas de elasticidad (Fase 6)
# como para las variables de decisión del optimizador (Fase 8-9).
RATE_GRID = np.round(np.linspace(5, 30, 15), 2)

# Mínimo de préstamos Charged Off dentro de un sub_grade para confiar en
# su LGD modelado. Por debajo de este umbral, la muestra es demasiado
# chica para que el promedio sea estable, y se usa el benchmark de
# industria (ver src/risk_model.LGD_INDUSTRY_BENCHMARK) como respaldo.
MIN_CHARGEOFF_SAMPLE = 30


def add_risk_percentile_accepted(accepted_with_risk: pd.DataFrame) -> pd.DataFrame:
    """Agrega `risk_percentile` calculado sobre `accepted` únicamente.

    Es el mismo cálculo que `demand_model.add_risk_percentile` aplica a las
    filas de `accepted` del dataset de demanda combinado (el percentil se
    calcula siempre dentro del propio dataset de origen, así que el
    resultado es idéntico lo aplique sobre accepted solo o sobre la unión).
    """
    out = accepted_with_risk.copy()
    out["risk_percentile"] = out["fico"].rank(pct=True) * 100
    return out


def build_demand_feature_frame(accepted_with_risk: pd.DataFrame, tasa) -> pd.DataFrame:
    """Arma la matriz de features del modelo de demanda para `accepted`,
    fijando `tasa` a un valor (o array) de tasa candidata en vez de usar
    la tasa realmente ofrecida -- así se puede evaluar F̄(r) para
    cualquier `r` hipotético, no solo el histórico.
    """
    return pd.DataFrame(
        {
            "tasa": tasa,
            "monto": accepted_with_risk["funded_amnt"].to_numpy(),
            "dti": accepted_with_risk["dti_clean"].to_numpy(),
            "risk_percentile": accepted_with_risk["risk_percentile"].to_numpy(),
            "emp_length_years": accepted_with_risk["emp_length_years"].to_numpy(),
            "purpose": accepted_with_risk["purpose_clean"].to_numpy(),
            "addr_state": accepted_with_risk["addr_state"].to_numpy(),
        }
    )


def compute_elasticity_curve(
    accepted_with_risk: pd.DataFrame, demand_pipeline, rate_grid=RATE_GRID
) -> pd.DataFrame:
    """Calcula p_{i,k}: probabilidad de aceptación promedio por sub_grade
    (i) y tasa candidata (k), evaluando el modelo de demanda individual
    sobre cada cliente real del sub_grade a cada tasa de `rate_grid` y
    promediando.

    Devuelve un DataFrame largo con columnas: sub_grade, tasa, p_aceptacion.
    """
    chunks = []
    for k in rate_grid:
        X_k = build_demand_feature_frame(accepted_with_risk, tasa=k)
        p_k = demand_pipeline.predict_proba(X_k)[:, 1]
        chunk = pd.DataFrame(
            {"sub_grade": accepted_with_risk["sub_grade"].to_numpy(), "tasa": k, "p": p_k}
        )
        chunks.append(chunk.groupby(["sub_grade", "tasa"], as_index=False)["p"].mean())

    curve = pd.concat(chunks, ignore_index=True)
    return curve.rename(columns={"p": "p_aceptacion"}).sort_values(["sub_grade", "tasa"])


def aggregate_pd_lgd(accepted_with_risk: pd.DataFrame) -> pd.DataFrame:
    """Agrega PD y LGD individuales a nivel de sub_grade, promediando las
    predicciones de los clientes reales de cada sub_grade.

    Para LGD, si un sub_grade tiene menos de `MIN_CHARGEOFF_SAMPLE`
    préstamos Charged Off, el promedio modelado es poco confiable y se
    reemplaza por `LGD_INDUSTRY_BENCHMARK` (ver justificación en
    `src/risk_model.py` y `notebooks/03_risk_estimation.ipynb`).
    """
    pd_agg = accepted_with_risk.groupby("sub_grade")["pd_individual"].mean().rename("PD")

    n_chargeoff = (
        accepted_with_risk[accepted_with_risk["loan_status"] == "Charged Off"]
        .groupby("sub_grade")
        .size()
        .rename("n_chargeoff")
    )
    lgd_modeled = accepted_with_risk.groupby("sub_grade")["lgd_individual"].mean().rename("LGD")
    n_clients = accepted_with_risk.groupby("sub_grade").size().rename("n_clients")

    out = pd.concat([pd_agg, lgd_modeled, n_chargeoff, n_clients], axis=1)
    out["n_chargeoff"] = out["n_chargeoff"].fillna(0).astype(int)

    out["LGD_source"] = np.where(
        out["n_chargeoff"] >= MIN_CHARGEOFF_SAMPLE, "modelado", "benchmark_industria"
    )
    out["LGD"] = np.where(
        out["n_chargeoff"] >= MIN_CHARGEOFF_SAMPLE, out["LGD"], LGD_INDUSTRY_BENCHMARK
    )

    return out.reset_index()
