import numpy as np
import pandas as pd

from src.optimization import add_objective_terms, build_optimization_table, solve_pricing


def _subgrade_params_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sub_grade": ["A1", "B1"],
            "PD": [0.05, 0.20],
            "LGD": [0.9, 0.9],
            "P_i": [10000.0, 10000.0],
            "n_i": [36.0, 36.0],
        }
    )


def _elasticity_fixture() -> pd.DataFrame:
    rates = np.array([5.0, 15.0, 25.0])
    # demanda decreciente en tasa para ambos sub_grades, B1 algo mas inelastico
    rows = []
    for sub_grade, drop in [("A1", 0.03), ("B1", 0.01)]:
        for r in rates:
            p = max(0.05, 0.9 - drop * r)
            rows.append({"sub_grade": sub_grade, "tasa": r, "p_aceptacion": p})
    return pd.DataFrame(rows)


def test_build_optimization_table_converts_term_to_years():
    table = build_optimization_table(_subgrade_params_fixture(), _elasticity_fixture())
    assert (table["n_years"] == 3.0).all()
    assert len(table) == 2 * 3


def test_solve_pricing_respects_one_rate_per_segment():
    table = add_objective_terms(
        build_optimization_table(_subgrade_params_fixture(), _elasticity_fixture()),
        funding_cost=0.035,
    )
    result = solve_pricing(table, objective="profitability")
    assert result["status"] == "Optimal"
    assert len(result["assignment"]) == 2
    assert set(result["assignment"]["sub_grade"]) == {"A1", "B1"}


def test_solve_pricing_enforces_risk_monotonicity():
    table = add_objective_terms(
        build_optimization_table(_subgrade_params_fixture(), _elasticity_fixture()),
        funding_cost=0.035,
    )
    result = solve_pricing(table, objective="profitability")
    assignment = result["assignment"].set_index("sub_grade")
    # B1 tiene mayor PD que A1 -> su tasa asignada debe ser >= la de A1
    assert assignment.loc["B1", "tasa_asignada"] >= assignment.loc["A1", "tasa_asignada"]


def test_min_volume_constraint_forces_lower_rates():
    table = add_objective_terms(
        build_optimization_table(_subgrade_params_fixture(), _elasticity_fixture()),
        funding_cost=0.035,
    )
    unconstrained = solve_pricing(table, objective="profitability")
    constrained = solve_pricing(
        table, objective="profitability", min_volume=unconstrained["total_volume"] * 1.3
    )
    assert constrained["status"] == "Optimal"
    assert constrained["total_volume"] >= unconstrained["total_volume"]
    assert constrained["total_profit"] <= unconstrained["total_profit"]
