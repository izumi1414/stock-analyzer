from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA20",
    "MA50",
    "MA200",
    "PreviousDayChange",
]
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class PredictionResult:
    model: LinearRegression
    metrics: dict[str, float]
    actual_values: pd.Series
    predicted_values: pd.Series
    next_close_prediction: float


def prepare_prediction_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Create prediction features and the next-day Close target."""
    data = stock_data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"必要な株価カラムがありません: {', '.join(missing_columns)}")

    data = data[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["PreviousDayChange"] = data["Close"].pct_change()
    data["Target"] = data["Close"].shift(-1)
    return data.dropna(subset=FEATURE_COLUMNS + ["Target"])


def train_and_evaluate(stock_data: pd.DataFrame) -> PredictionResult:
    prediction_data = prepare_prediction_data(stock_data)
    if len(prediction_data) < 10:
        raise ValueError("予測に必要なデータが不足しています。")

    split_index = int(len(prediction_data) * 0.8)
    if split_index == 0 or split_index == len(prediction_data):
        raise ValueError("学習データとテストデータを分割できません。")

    features = prediction_data[FEATURE_COLUMNS]
    target = prediction_data["Target"]
    train_features = features.iloc[:split_index]
    test_features = features.iloc[split_index:]
    train_target = target.iloc[:split_index]
    test_target = target.iloc[split_index:]

    model = LinearRegression()
    model.fit(train_features, train_target)
    predicted_values = pd.Series(model.predict(test_features), index=test_target.index)
    metrics = {
        "MAE": mean_absolute_error(test_target, predicted_values),
        "RMSE": mean_squared_error(test_target, predicted_values) ** 0.5,
        "R2": r2_score(test_target, predicted_values),
    }
    next_close_prediction = float(model.predict(features.iloc[[-1]])[0])

    return PredictionResult(
        model=model,
        metrics=metrics,
        actual_values=test_target,
        predicted_values=predicted_values,
        next_close_prediction=next_close_prediction,
    )
