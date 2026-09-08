import pandas as pd
from src.data_preprocessing import build_preprocessor, split_xy


def test_customer_id_and_target_are_not_features():
    df = pd.DataFrame({
        "customerID": ["A", "B"],
        "tenure": [1, 2],
        "MonthlyCharges": [30.0, 50.0],
        "TotalCharges": [30.0, 100.0],
        "Contract": ["Month-to-month", "One year"],
        "Churn": [1, 0],
    })
    X, y = split_xy(df)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert list(y) == [1, 0]


def test_preprocessor_fit_transform():
    X = pd.DataFrame({
        "tenure": [1, 2, 3],
        "MonthlyCharges": [30.0, 50.0, 70.0],
        "TotalCharges": [30.0, None, 210.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
    })
    Xt = build_preprocessor(X).fit_transform(X)
    assert Xt.shape[0] == 3
