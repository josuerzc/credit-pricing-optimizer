import numpy as np
import pandas as pd

from src.demand_model import DEMAND_FEATURES, add_risk_percentile, build_demand_features


def _demand_fixture() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 40
    accepted = pd.DataFrame(
        {
            "acepto_prestamo": 1,
            "monto": rng.uniform(1000, 30000, n),
            "dti": rng.uniform(1, 35, n),
            "fico": rng.uniform(640, 820, n),
            "risk_score": np.nan,
            "emp_length_years": rng.randint(0, 10, n),
            "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
            "zip_code": "100xx",
            "addr_state": rng.choice(["CA", "NY", "TX"], n),
            "int_rate": rng.uniform(5, 30, n),
            "fecha_solicitud": "Jan-2015",
            "source_dataset": "accepted",
        }
    )
    rejected = pd.DataFrame(
        {
            "acepto_prestamo": 0,
            "monto": rng.uniform(1000, 30000, n),
            "dti": rng.uniform(1, 35, n),
            "fico": np.nan,
            "risk_score": rng.uniform(600, 990, n),
            "emp_length_years": rng.randint(0, 10, n),
            "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
            "zip_code": "200xx",
            "addr_state": rng.choice(["CA", "NY", "TX"], n),
            "int_rate": np.nan,
            "fecha_solicitud": "2016-01-05",
            "source_dataset": "rejected",
        }
    )
    return pd.concat([accepted, rejected], ignore_index=True)


def test_add_risk_percentile_uses_within_dataset_ranking():
    demand = _demand_fixture()
    out = add_risk_percentile(demand)

    accepted_rows = out[out["source_dataset"] == "accepted"]
    rejected_rows = out[out["source_dataset"] == "rejected"]

    assert accepted_rows["risk_percentile"].between(0, 100).all()
    assert rejected_rows["risk_percentile"].between(0, 100).all()
    # el perfil de mayor fico debe tener el mayor percentil dentro de accepted
    top_fico_idx = accepted_rows["fico"].idxmax()
    assert (
        accepted_rows.loc[top_fico_idx, "risk_percentile"] == accepted_rows["risk_percentile"].max()
    )


def test_build_demand_features_fills_tasa_for_rejected_without_using_raw_nan():
    demand = _demand_fixture()
    out = build_demand_features(demand)

    assert out["tasa"].notna().all()
    assert (~out.loc[out["source_dataset"] == "accepted", "tasa_estimada"]).all()
    assert out.loc[out["source_dataset"] == "rejected", "tasa_estimada"].all()
    for col in DEMAND_FEATURES:
        assert col in out.columns
