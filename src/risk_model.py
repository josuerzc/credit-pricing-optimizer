"""
Modelos individuales de riesgo de crédito: probabilidad de default (PD) y
pérdida dado el default (LGD). Ambos se entrenan solo sobre `accepted`
(no tiene sentido estimar riesgo de un préstamo que nunca se otorgó) y a
nivel de CLIENTE INDIVIDUAL, igual que el modelo de demanda -- ver README.

También se documenta aquí el supuesto de costo de fondeo (`r_c` en la
notación de Phillips), que no existe en los datos de LendingClub (es una
plataforma P2P, no un banco con costo de fondeo propio reportado).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

RISK_NUMERIC_FEATURES = [
    "fico",
    "dti_clean",
    "funded_amnt",
    "term",
    "revol_util",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec_bankruptcies",
    "open_acc",
    "mort_acc",
    "annual_inc",
]
RISK_CATEGORICAL_FEATURES = ["home_ownership", "verification_status"]
RISK_FEATURES = RISK_NUMERIC_FEATURES + RISK_CATEGORICAL_FEATURES

# Desenlaces finales (préstamo "resuelto") vs. censurados (aún en curso o
# atrasado sin desenlace definitivo). Solo los resueltos sirven para
# entrenar PD: un préstamo "Current" puede terminar bien o mal, y
# etiquetarlo como 0 (no default) introduciría un sesgo optimista
# sistemático -- son datos censurados, no observaciones de "no default".
DEFAULT_STATUSES = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
FULLY_PAID_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
RESOLVED_STATUSES = DEFAULT_STATUSES | FULLY_PAID_STATUSES

# Benchmark de industria para LGD de crédito personal sin garantía, usado
# como respaldo en la Fase 4/6 cuando un sub_grade tiene muestra
# insuficiente de charged-off para confiar en el LGD modelado. Basel usa
# 45% como referencia supervisoria estándar (LGD "foundation IRB" para
# exposiciones retail sin garantía); el rango 40-60% es el típico
# reportado en la industria de crédito personal no garantizado.
LGD_INDUSTRY_BENCHMARK = 0.45

# Costo de fondeo (r_c de Phillips). LendingClub es P2P y no reporta esta
# variable -- es un supuesto documentado, no un dato observado. 3.5% se
# ubica en el rango de costo de fondeo retail bancario de referencia
# (ver Federal Reserve Bank of Minneapolis, "How do lenders set interest
# rates on loans", y series de costo de depósitos de la Fed). Se define
# también un rango para el análisis de sensibilidad de la Fase 8.
FUNDING_COST_BASE = 0.035
FUNDING_COST_SENSITIVITY = [0.02, 0.04, 0.06]


def filter_resolved_loans(accepted_clean: pd.DataFrame) -> pd.DataFrame:
    """Filtra a préstamos con desenlace final y agrega la columna `default`
    (1 = Charged Off/Default, 0 = Fully Paid).
    """
    out = accepted_clean[accepted_clean["loan_status"].isin(RESOLVED_STATUSES)].copy()
    out["default"] = out["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    return out


def _risk_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), RISK_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                RISK_CATEGORICAL_FEATURES,
            ),
        ]
    )


def pd_model_pipeline(**xgb_params) -> Pipeline:
    """Pipeline de PD: preprocesamiento + XGBClassifier."""
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    params.update(xgb_params)
    model = XGBClassifier(eval_metric="logloss", **params)
    return Pipeline(steps=[("preprocess", _risk_preprocessor()), ("model", model)])


def compute_lgd_target(accepted_clean: pd.DataFrame) -> pd.DataFrame:
    """Calcula el LGD realizado sobre los préstamos Charged Off.

    LGD = 1 - (recoveries - collection_recovery_fee) / EAD

    `out_prncp` (saldo pendiente) resultó **inservible** como EAD en este
    dataset: LendingClub lo pone en 0 para todo préstamo ya dado de baja
    (verificado empíricamente sobre la muestra: 100% de los Charged Off
    tienen `out_prncp == 0`). En su lugar se aproxima el EAD como
    `funded_amnt - total_rec_prncp` (monto original menos lo que el
    cliente alcanzó a pagar de principal antes del default) -- es la
    alternativa que sugiere el enunciado del proyecto y, a diferencia de
    `out_prncp`, sí tiene variación real en los datos.

    El LGD se recorta a [0, 1]: un puñado de préstamos (recoveries menos
    la comisión de cobranza superando el EAD estimado) produce LGD
    negativo, que no es interpretable como fracción de pérdida y se trata
    como ruido de datos, no como una señal real de "ganancia" en el
    default.
    """
    charged_off = accepted_clean[accepted_clean["loan_status"] == "Charged Off"].copy()
    charged_off["ead_est"] = charged_off["funded_amnt"] - charged_off["total_rec_prncp"]
    charged_off = charged_off[charged_off["ead_est"] > 0].copy()

    charged_off["lgd"] = (
        1
        - (charged_off["recoveries"] - charged_off["collection_recovery_fee"])
        / charged_off["ead_est"]
    )
    charged_off["lgd"] = charged_off["lgd"].clip(lower=0, upper=1)

    return charged_off


def lgd_model_pipeline(**xgb_params) -> Pipeline:
    """Pipeline de LGD: preprocesamiento + XGBRegressor."""
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    params.update(xgb_params)
    model = XGBRegressor(**params)
    return Pipeline(steps=[("preprocess", _risk_preprocessor()), ("model", model)])
