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


@dataclass
class BacktestResult:
    results: pd.DataFrame
    metrics: dict[str, float]


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


def run_backtest(
    stock_data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> BacktestResult:
    """Run expanding-window next-day predictions without using future data."""
    prediction_data = prepare_prediction_data(stock_data)
    prediction_data = prediction_data.sort_index()
    prediction_data = prediction_data.loc[
        (prediction_data.index >= start_date) & (prediction_data.index <= end_date)
    ]

    full_prediction_data = prepare_prediction_data(stock_data).sort_index()
    predictions = []
    for prediction_date, row in prediction_data.iterrows():
        training_data = full_prediction_data.loc[full_prediction_data.index < prediction_date]
        if len(training_data) < 10:
            continue

        model = LinearRegression()
        model.fit(training_data[FEATURE_COLUMNS], training_data["Target"])
        predicted_value = float(model.predict(row[FEATURE_COLUMNS].to_frame().T)[0])
        actual_value = float(row["Target"])
        previous_close = float(row["Close"])
        predictions.append(
            {
                "date": prediction_date,
                "predicted_value": predicted_value,
                "actual_value": actual_value,
                "prediction_error": predicted_value - actual_value,
                "predicted_direction": "Up" if predicted_value >= previous_close else "Down",
                "actual_direction": "Up" if actual_value >= previous_close else "Down",
            }
        )

    results = pd.DataFrame(
        predictions,
        columns=[
            "date",
            "predicted_value",
            "actual_value",
            "prediction_error",
            "predicted_direction",
            "actual_direction",
        ],
    )
    if results.empty:
        raise ValueError("バックテストに必要なデータが不足しています。")

    metrics = {
        "MAE": mean_absolute_error(results["actual_value"], results["predicted_value"]),
        "RMSE": mean_squared_error(results["actual_value"], results["predicted_value"]) ** 0.5,
        "Directional Accuracy": (
            results["predicted_direction"] == results["actual_direction"]
        ).mean(),
    }
    return BacktestResult(results=results, metrics=metrics)
