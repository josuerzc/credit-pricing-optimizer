"""
Optimizador de tasas por sub_grade (Fase 8) con PuLP.

Implementa las Ecuaciones 4 y 5 de Phillips (2013), replicadas al nivel de
sub_grade (35 segmentos) usando los `p_{i,k}` (Fase 6), `PD_i`/`LGD_i`
(Fase 4) y `P_i`/`n_i` (monto y plazo promedio, Fase 4) ya agregados.

Dos versiones de función objetivo (ver README y notebooks/05_optimization.ipynb
para la comparación completa):

- REVENUE (ingenua): max Σ r_k·n_i·P_i·p_{i,k}·x[i,k] -- ingreso bruto
  esperado, ignora costo de fondeo y pérdida esperada.
- PROFITABILITY (Phillips, Ecuación 4): max Σ P_i·p_{i,k}·(n_i·(r_k-c) -
  PD_i·LGD_i)·x[i,k] -- resta la pérdida esperada (PD·LGD) directamente
  del margen bruto, SIN multiplicar por (1-PD) (ver Concepto clave 2 del
  README: es la aproximación lineal que Phillips demuestra formalmente).

`n_i` se agregó en meses (Fase 4); se convierte a años aquí porque
`tasa`/`r_c` son tasas anuales (APR) -- `n_i_years·(r-c)` representa el
margen de interés neto acumulado a lo largo de la vida del préstamo,
consistente con la aproximación lineal de la Ecuación 4.
"""

from __future__ import annotations

import pandas as pd
import pulp

MONTHS_PER_YEAR = 12


def build_optimization_table(
    subgrade_params: pd.DataFrame, elasticity: pd.DataFrame
) -> pd.DataFrame:
    """Une PD/LGD/monto/plazo por sub_grade con p_{i,k} por (sub_grade, tasa),
    y agrega los términos de la función objetivo para cada combinación
    (sub_grade, tasa candidata).
    """
    table = elasticity.merge(subgrade_params, on="sub_grade", how="inner")
    table["n_years"] = table["n_i"] / MONTHS_PER_YEAR
    table = table.sort_values(["sub_grade", "tasa"]).reset_index(drop=True)
    return table


def add_objective_terms(table: pd.DataFrame, funding_cost: float) -> pd.DataFrame:
    """Agrega `revenue_term` y `profit_term` -- el aporte a la función
    objetivo de asignarle la tasa `tasa` al sub_grade, si `x[i,k]=1`.
    """
    out = table.copy()
    out["revenue_term"] = out["tasa"] / 100 * out["n_years"] * out["P_i"] * out["p_aceptacion"]

    pvnii_per_dollar = out["n_years"] * (out["tasa"] / 100 - funding_cost) - out["PD"] * out["LGD"]
    out["profit_term"] = out["P_i"] * out["p_aceptacion"] * pvnii_per_dollar

    out["volume_term"] = out["P_i"] * out["p_aceptacion"]
    return out


def _risk_order(table: pd.DataFrame) -> list[str]:
    """Sub_grades ordenados de menor a mayor riesgo (PD ascendente)."""
    return table.drop_duplicates("sub_grade").sort_values("PD")["sub_grade"].tolist()


def solve_pricing(
    table: pd.DataFrame,
    objective: str = "profitability",
    min_volume: float | None = None,
) -> dict:
    """Resuelve el problema de asignación de tasas óptimas por sub_grade.

    `table` debe venir de `add_objective_terms(build_optimization_table(...))`.
    `objective`: "profitability" (Ecuación 4-5 de Phillips) o "revenue"
    (ingreso bruto ingenuo, para comparar).
    `min_volume`: restricción de volumen mínimo (Σ P_i·p_{i,k}·x[i,k] >= B),
    usada en la Fase 9 para construir la frontera eficiente. `None` = sin
    restricción de volumen.

    Devuelve un dict con `status`, `assignment` (DataFrame con la tasa
    asignada y las métricas de cada sub_grade) y los totales de
    portafolio (`total_revenue`, `total_profit`, `total_volume`).
    """
    term_col = "profit_term" if objective == "profitability" else "revenue_term"

    prob = pulp.LpProblem("credit_pricing", pulp.LpMaximize)

    x = {
        (row.sub_grade, row.tasa): pulp.LpVariable(f"x_{row.sub_grade}_{row.tasa}", cat="Binary")
        for row in table.itertuples()
    }

    prob += pulp.lpSum(
        getattr(row, term_col) * x[(row.sub_grade, row.tasa)] for row in table.itertuples()
    )

    for sub_grade, group in table.groupby("sub_grade"):
        prob += pulp.lpSum(x[(sub_grade, k)] for k in group["tasa"]) == 1, f"una_tasa_{sub_grade}"

    if min_volume is not None:
        prob += (
            pulp.lpSum(row.volume_term * x[(row.sub_grade, row.tasa)] for row in table.itertuples())
            >= min_volume,
            "volumen_minimo",
        )

    risk_order = _risk_order(table)

    def assigned_rate_expr(sub_grade: str):
        rates = table.loc[table["sub_grade"] == sub_grade, "tasa"]
        return pulp.lpSum(k * x[(sub_grade, k)] for k in rates)

    for less_risky, more_risky in zip(risk_order[:-1], risk_order[1:]):
        prob += (
            assigned_rate_expr(less_risky) <= assigned_rate_expr(more_risky),
            f"monotonicidad_{less_risky}_{more_risky}",
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    rows = []
    for row in table.itertuples():
        if x[(row.sub_grade, row.tasa)].value() == 1:
            rows.append(
                {
                    "sub_grade": row.sub_grade,
                    "tasa_asignada": row.tasa,
                    "p_aceptacion": row.p_aceptacion,
                    "PD": row.PD,
                    "LGD": row.LGD,
                    "P_i": row.P_i,
                    "revenue_term": row.revenue_term,
                    "profit_term": row.profit_term,
                    "volume_term": row.volume_term,
                }
            )
    assignment = pd.DataFrame(rows).sort_values("sub_grade").reset_index(drop=True)

    return {
        "status": pulp.LpStatus[prob.status],
        "assignment": assignment,
        "total_revenue": assignment["revenue_term"].sum(),
        "total_profit": assignment["profit_term"].sum(),
        "total_volume": assignment["volume_term"].sum(),
    }
