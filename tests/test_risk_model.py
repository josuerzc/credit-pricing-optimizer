import numpy as np
import pandas as pd

from src.risk_model import compute_lgd_target, filter_resolved_loans


def _accepted_clean_fixture() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 20
    statuses = (
        ["Fully Paid"] * 8
        + ["Charged Off"] * 6
        + ["Current"] * 3
        + ["Late (31-120 days)"] * 2
        + ["Does not meet the credit policy. Status:Charged Off"] * 1
    )
    return pd.DataFrame(
        {
            "loan_status": statuses,
            "fico": rng.uniform(640, 820, n),
            "dti_clean": rng.uniform(1, 35, n),
            "funded_amnt": rng.uniform(2000, 30000, n),
            "term": rng.choice([36, 60], n),
            "revol_util": rng.uniform(0, 100, n),
            "delinq_2yrs": rng.randint(0, 3, n),
            "inq_last_6mths": rng.randint(0, 5, n),
            "pub_rec_bankruptcies": rng.randint(0, 2, n),
            "open_acc": rng.randint(1, 20, n),
            "mort_acc": rng.randint(0, 5, n),
            "annual_inc": rng.uniform(20000, 150000, n),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
            "verification_status": rng.choice(["Verified", "Not Verified"], n),
            "recoveries": rng.uniform(0, 2000, n),
            "collection_recovery_fee": rng.uniform(0, 200, n),
            "total_rec_prncp": rng.uniform(500, 10000, n),
        }
    )


def test_filter_resolved_loans_excludes_censored_statuses():
    fixture = _accepted_clean_fixture()
    out = filter_resolved_loans(fixture)

    assert "Current" not in out["loan_status"].values
    assert "Late (31-120 days)" not in out["loan_status"].values
    assert len(out) == 8 + 6 + 1


def test_filter_resolved_loans_labels_default_correctly():
    fixture = _accepted_clean_fixture()
    out = filter_resolved_loans(fixture)

    assert (out.loc[out["loan_status"] == "Fully Paid", "default"] == 0).all()
    assert (out.loc[out["loan_status"] == "Charged Off", "default"] == 1).all()
    assert (
        out.loc[
            out["loan_status"] == "Does not meet the credit policy. Status:Charged Off",
            "default",
        ]
        == 1
    ).all()


def test_compute_lgd_target_only_uses_charged_off_and_clips_to_unit_interval():
    fixture = _accepted_clean_fixture()
    # fuerza un caso de EAD pequeño y recoveries grandes -> lgd fuera de rango
    fixture.loc[fixture["loan_status"] == "Charged Off", "total_rec_prncp"] = (
        fixture.loc[fixture["loan_status"] == "Charged Off", "funded_amnt"] - 1
    )
    fixture.loc[fixture["loan_status"] == "Charged Off", "recoveries"] = 999999

    out = compute_lgd_target(fixture)

    assert (out["loan_status"] == "Charged Off").all()
    assert out["lgd"].between(0, 1).all()
