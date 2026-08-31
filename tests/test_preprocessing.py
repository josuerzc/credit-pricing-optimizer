import numpy as np
import pandas as pd

from src.preprocessing import (
    build_demand_dataset,
    clean_accepted,
    clean_rejected,
    map_loan_title_to_purpose,
)


def _accepted_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1", "2", "Total amount funded in policy code 1: 123"],
            "funded_amnt": [10000.0, 5000.0, np.nan],
            "int_rate": [12.5, 9.0, np.nan],
            "term": [" 36 months", " 60 months", np.nan],
            "emp_length": ["10+ years", "< 1 year", np.nan],
            "fico_range_low": [700, 680, np.nan],
            "fico_range_high": [704, 684, np.nan],
            "dti": [15.0, 999.0, np.nan],
            "purpose": ["credit_card", "unknown_bucket", np.nan],
            "zip_code": ["100xx", "200xx", np.nan],
            "addr_state": ["NY", "CA", np.nan],
            "issue_d": ["Jan-2015", "Feb-2015", np.nan],
        }
    )


def _rejected_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Amount Requested": [1000.0, 0.0, 5000.0],
            "Debt-To-Income Ratio": ["11.06%", "5%", "20.5%"],
            "Risk_Score": [650.0, np.nan, np.nan],
            "Employment Length": ["3 years", "10+ years", "< 1 year"],
            "Loan Title": ["Debt consolidation", "debt_consolidation", "Some weird title"],
            "Zip Code": ["300xx", "400xx", "500xx"],
            "State": ["TX", "FL", "WA"],
            "Application Date": ["2016-01-05", "2016-02-10", "2016-03-15"],
        }
    )


def test_clean_accepted_drops_summary_rows():
    out = clean_accepted(_accepted_fixture())
    assert len(out) == 2
    assert out["term"].tolist() == [36.0, 60.0]
    assert out["emp_length_years"].tolist() == [10, 0]
    assert out["fico"].tolist() == [702.0, 682.0]


def test_clean_accepted_treats_extreme_dti_as_missing():
    out = clean_accepted(_accepted_fixture())
    assert out["dti_clean"].iloc[0] == 15.0
    assert np.isnan(out["dti_clean"].iloc[1])


def test_clean_accepted_falls_back_unknown_purpose_to_other():
    out = clean_accepted(_accepted_fixture())
    assert out["purpose_clean"].tolist() == ["credit_card", "other"]


def test_clean_rejected_drops_zero_amount_requests():
    out = clean_rejected(_rejected_fixture())
    assert len(out) == 2
    assert (out["Amount Requested"] > 0).all()


def test_clean_rejected_parses_percentage_dti():
    out = clean_rejected(_rejected_fixture())
    assert out["dti_clean"].tolist() == [11.06, 20.5]


def test_clean_rejected_flags_missing_risk_score():
    out = clean_rejected(_rejected_fixture())
    assert out["risk_score_missing"].tolist() == [False, True]


def test_map_loan_title_to_purpose_handles_raw_category_and_free_text():
    assert map_loan_title_to_purpose("debt_consolidation") == "debt_consolidation"
    assert map_loan_title_to_purpose("Debt consolidation") == "debt_consolidation"
    assert map_loan_title_to_purpose("Credit card refinancing") == "credit_card"
    assert map_loan_title_to_purpose(np.nan) == "other"
    assert map_loan_title_to_purpose("something totally unrelated") == "other"


def test_build_demand_dataset_labels_and_stacks_rows():
    accepted_out = clean_accepted(_accepted_fixture())
    rejected_out = clean_rejected(_rejected_fixture())

    demand = build_demand_dataset(accepted_out, rejected_out)

    assert len(demand) == len(accepted_out) + len(rejected_out)
    assert set(demand["acepto_prestamo"].unique()) == {0, 1}
    assert (demand.loc[demand["source_dataset"] == "accepted", "acepto_prestamo"] == 1).all()
    assert (demand.loc[demand["source_dataset"] == "rejected", "acepto_prestamo"] == 0).all()


def test_build_demand_dataset_keeps_fico_and_risk_score_separate():
    accepted_out = clean_accepted(_accepted_fixture())
    rejected_out = clean_rejected(_rejected_fixture())

    demand = build_demand_dataset(accepted_out, rejected_out)

    accepted_rows = demand[demand["source_dataset"] == "accepted"]
    rejected_rows = demand[demand["source_dataset"] == "rejected"]

    assert accepted_rows["risk_score"].isna().all()
    assert rejected_rows["fico"].isna().all()
    assert rejected_rows["int_rate"].isna().all()
