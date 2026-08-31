"""
Limpieza y estandarización de los datasets `accepted` y `rejected` de
Lending Club.

Cada dataset se limpia con sus propias reglas (`clean_accepted`,
`clean_rejected`) porque tienen columnas y formatos distintos, pero se
homogeneizan hacia un esquema común de features compartidas (tasa,
monto, dti, riesgo, antigüedad laboral, propósito, ubicación) para que
en la Fase 3 puedan unirse en un único dataset de "demanda"
(`acepto_prestamo` = 1/0). Ver README para el detalle de por qué esa
unión es un proxy y no una medición perfecta de aceptación.
"""

from __future__ import annotations

import re

import pandas as pd

# --------------------------------------------------------------------------
# Mapeo de antigüedad laboral (formato idéntico en accepted y rejected, solo
# cambia el nombre de columna: emp_length vs. Employment Length)
# --------------------------------------------------------------------------
EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}

# --------------------------------------------------------------------------
# Mapeo de propósito del préstamo. `accepted.purpose` ya viene como una
# categoría limpia (14 valores fijos). `rejected["Loan Title"]` es texto
# libre que mezcla las mismas categorías (a veces literalmente, ej.
# "debt_consolidation") con frases humanas (ej. "Debt consolidation",
# "Credit card refinancing"). Normalizamos a las mismas 14 categorías de
# `purpose` para poder usar un solo feature categórico en el modelo de
# demanda (Fase 5).
# --------------------------------------------------------------------------
PURPOSE_CATEGORIES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "vacation",
    "moving",
    "house",
    "wedding",
    "renewable_energy",
    "educational",
    "other",
]

_PURPOSE_KEYWORDS = [
    ("debt_consolidation", r"debt.?consolidat|consolidation"),
    ("credit_card", r"credit.?card"),
    ("home_improvement", r"home.?improvement"),
    ("house", r"home buying|house|mortgage"),
    ("major_purchase", r"major purchase"),
    ("small_business", r"business"),
    ("car", r"\bcar\b|auto|vehicle"),
    ("medical", r"medical"),
    ("moving", r"moving|relocation"),
    ("vacation", r"vacation"),
    ("wedding", r"wedding"),
    ("renewable_energy", r"renewable|green loan|solar"),
    ("educational", r"educat|student|school|tuition"),
]


def map_loan_title_to_purpose(title: str) -> str:
    """Normaliza un `Loan Title` de rejected a una de las categorías de `purpose`."""
    if pd.isna(title):
        return "other"
    text = str(title).strip().lower()
    if text in PURPOSE_CATEGORIES:
        return text
    for category, pattern in _PURPOSE_KEYWORDS:
        if re.search(pattern, text):
            return category
    return "other"


def _parse_emp_length(series: pd.Series) -> pd.Series:
    return series.map(EMP_LENGTH_MAP)


def clean_accepted(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset de préstamos otorgados (accepted).

    Decisiones de limpieza:
    - Se descartan las filas de resumen que Lending Club agrega al final
      del CSV (ej. "Total amount funded in policy code 1: ...") -- no son
      préstamos reales, se identifican porque `funded_amnt` es NaN.
    - `term` se convierte de texto (" 36 months") a entero (36).
    - `emp_length` se convierte a años numéricos; los NaN (~6.5% de las
      filas) se dejan como NaN -- LendingClub no distingue "0 años" de
      "no reportado", y forzar un valor introduciría sesgo. Se imputa
      más adelante, a nivel de pipeline de modelado, no aquí.
    - Se crea `fico` como el punto medio de `fico_range_low`/`fico_range_high`
      (LendingClub solo publica el rango, nunca el score exacto).
    - `dti` tiene un grupo pequeño (176 de 150,000, ~0.12%) de valores
      extremos hasta 999, físicamente implausibles para un préstamo ya
      otorgado (dti > 100% no es un ratio deuda/ingreso creíble, es un
      error de captura o un caso residual mal calculado por LC). Se
      convierten a NaN en vez de eliminarse la fila completa, para no
      perder el resto de las variables de esos clientes.
    """
    out = df.copy()
    out = out[out["funded_amnt"].notna()].copy()

    out["term"] = out["term"].str.extract(r"(\d+)").astype(float)
    out["emp_length_years"] = _parse_emp_length(out["emp_length"])
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2

    out["dti_clean"] = out["dti"].where(out["dti"] <= 100)

    out["purpose_clean"] = out["purpose"].where(out["purpose"].isin(PURPOSE_CATEGORIES), "other")

    return out


def clean_rejected(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset de solicitudes rechazadas (rejected).

    Decisiones de limpieza:
    - `Debt-To-Income Ratio` viene como texto con "%" (ej. "11.06%"); se
      convierte a float en la misma escala que `dti` de accepted.
    - Se descartan las ~7 filas con `Amount Requested == 0` (monto de
      solicitud nulo no es una solicitud real, es un error de captura).
    - `Risk_Score` tiene ~66.6% de missing en la muestra completa, pero
      la tasa de missing NO es uniforme en el tiempo: es <15% hasta 2014
      y sube a 45-93% entre 2015 y 2018 (LendingClub dejó de poblar
      sistemáticamente este campo para la mayoría de rechazos en años
      recientes). Esto es una limitación de datos real, no un artefacto
      del muestreo -- se documenta explícitamente en el README. Se
      conserva la columna con NaN (no se imputa aquí) y se agrega
      `risk_score_missing` como flag, para que el modelo de demanda
      (Fase 5) pueda usar ambas señales.
    - `Employment Length` se mapea con la misma función que `emp_length`
      de accepted.
    - `Loan Title` se normaliza a las categorías de `purpose` vía
      `map_loan_title_to_purpose`.
    """
    out = df.copy()
    out = out[out["Amount Requested"] > 0].copy()

    out["dti_clean"] = out["Debt-To-Income Ratio"].astype(str).str.rstrip("%").astype(float)
    out["risk_score_missing"] = out["Risk_Score"].isna()
    out["emp_length_years"] = _parse_emp_length(out["Employment Length"])
    out["purpose_clean"] = out["Loan Title"].map(map_loan_title_to_purpose)

    return out
