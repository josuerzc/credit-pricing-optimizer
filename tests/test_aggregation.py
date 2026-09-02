import numpy as np
import pandas as pd

from src.aggregation import (
    MIN_CHARGEOFF_SAMPLE,
    add_risk_percentile_accepted,
    aggregate_pd_lgd,
    build_demand_feature_frame,
)
from src.risk_model import LGD_INDUSTRY_BENCHMARK


def _accepted_with_risk_fixture() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n_a1, n_g5 = 50, 5  # A1: muestra grande; G5: muestra chica (< MIN_CHARGEOFF_SAMPLE)

    a1 = pd.DataFrame(
        {
            "sub_grade": "A1",
            "fico": rng.uniform(700, 820, n_a1),
            "dti_clean": rng.uniform(1, 30, n_a1),
            "funded_amnt": rng.uniform(5000, 20000, n_a1),
            "emp_length_years": rng.randint(0, 10, n_a1),
            "purpose_clean": "debt_consolidation",
            "addr_state": "CA",
            "pd_individual": rng.uniform(0.02, 0.1, n_a1),
            "lgd_individual": rng.uniform(0.8, 0.95, n_a1),
            "loan_status": ["Charged Off"] * (MIN_CHARGEOFF_SAMPLE + 5)
            + ["Fully Paid"] * (n_a1 - MIN_CHARGEOFF_SAMPLE - 5),
        }
    )
    g5 = pd.DataFrame(
        {
            "sub_grade": "G5",
            "fico": rng.uniform(640, 680, n_g5),
            "dti_clean": rng.uniform(20, 35, n_g5),
            "funded_amnt": rng.uniform(5000, 20000, n_g5),
            "emp_length_years": rng.randint(0, 10, n_g5),
            "purpose_clean": "debt_consolidation",
            "addr_state": "TX",
            "pd_individual": rng.uniform(0.3, 0.5, n_g5),
            "lgd_individual": rng.uniform(0.85, 0.95, n_g5),
            "loan_status": ["Charged Off"] * 2 + ["Fully Paid"] * (n_g5 - 2),
        }
    )
    return pd.concat([a1, g5], ignore_index=True)


def test_add_risk_percentile_accepted_is_bounded():
    fixture = _accepted_with_risk_fixture()
    out = add_risk_percentile_accepted(fixture)
    assert out["risk_percentile"].between(0, 100).all()


def test_build_demand_feature_frame_broadcasts_scalar_tasa():
    fixture = add_risk_percentile_accepted(_accepted_with_risk_fixture())
    X = build_demand_feature_frame(fixture, tasa=12.5)
    assert (X["tasa"] == 12.5).all()
    assert len(X) == len(fixture)


def test_aggregate_pd_lgd_falls_back_to_benchmark_for_small_sample():
    fixture = _accepted_with_risk_fixture()
    agg = aggregate_pd_lgd(fixture)

    a1_row = agg[agg["sub_grade"] == "A1"].iloc[0]
    g5_row = agg[agg["sub_grade"] == "G5"].iloc[0]

    assert a1_row["LGD_source"] == "modelado"
    assert g5_row["LGD_source"] == "benchmark_industria"
    assert g5_row["LGD"] == LGD_INDUSTRY_BENCHMARK
    assert a1_row["PD"] < g5_row["PD"]
