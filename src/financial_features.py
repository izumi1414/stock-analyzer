from collections.abc import Iterable

import numpy as np
import pandas as pd


FINANCIAL_FEATURE_COLUMNS = [
	"revenue",
	"revenue_growth",
	"operating_income",
	"operating_income_growth",
	"operating_margin",
	"net_income",
	"net_income_growth",
	"net_profit_margin",
	"eps",
	"eps_growth",
	"days_since_earnings",
]


_COLUMN_ALIASES = {
	"reporting_date": ("reporting_date", "決算期", "date"),
	"available_date": (
		"available_date",
		"announcement_date",
		"決算発表日",
		"発表日",
	),
	"revenue": ("revenue", "売上高"),
	"operating_income": ("operating_income", "営業利益"),
	"net_income": ("net_income", "純利益"),
	"eps": ("eps", "EPS"),
}


def _first_existing_column(
	frame: pd.DataFrame, candidates: Iterable[str]
) -> str | None:
	return next((column for column in candidates if column in frame.columns), None)


def _normalise_date(values: pd.Series) -> pd.Series:
	return pd.to_datetime(values, errors="coerce").dt.tz_localize(None)


def _normalise_financial_data(financial_data: pd.DataFrame) -> pd.DataFrame:
	result = pd.DataFrame(index=financial_data.index)
	for name, aliases in _COLUMN_ALIASES.items():
		column = _first_existing_column(financial_data, aliases)
		result[name] = (
			financial_data[column] if column is not None else np.nan
		)

	result["reporting_date"] = _normalise_date(result["reporting_date"])
	available_date = result["available_date"]
	result["available_date"] = _normalise_date(available_date)
	result["available_date"] = result["available_date"].fillna(
		result["reporting_date"]
	)
	for column in ("revenue", "operating_income", "net_income", "eps"):
		result[column] = pd.to_numeric(result[column], errors="coerce")

	result = result.dropna(subset=["available_date"])
	result = result.sort_values("reporting_date").reset_index(drop=True)
	for column in ("revenue", "operating_income", "net_income", "eps"):
		previous = result[column].shift(1)
		result[f"{column}_growth"] = (
			(result[column] - previous) / previous.replace(0, np.nan)
		)
	result["operating_margin"] = (
		result["operating_income"] / result["revenue"].replace(0, np.nan)
	)
	result["net_profit_margin"] = (
		result["net_income"] / result["revenue"].replace(0, np.nan)
	)
	return result


def create_earnings_features(
	stock_data: pd.DataFrame,
	financial_data: pd.DataFrame,
) -> pd.DataFrame:
	"""Join the latest publicly available earnings data to each stock date."""
	stock = stock_data.copy()
	if "date" in stock.columns:
		stock_dates = stock["date"]
	else:
		stock_dates = pd.Series(stock.index, index=stock.index)
	stock_dates = _normalise_date(stock_dates)
	result = pd.DataFrame({"date": stock_dates}).sort_values("date")
	result = result.dropna(subset=["date"]).reset_index(drop=True)

	if result.empty:
		return pd.DataFrame(columns=["date", *FINANCIAL_FEATURE_COLUMNS])
	if financial_data.empty:
		return result.assign(
			**{column: np.nan for column in FINANCIAL_FEATURE_COLUMNS}
		)[["date", *FINANCIAL_FEATURE_COLUMNS]]

	financial = _normalise_financial_data(financial_data)
	if financial.empty:
		return result.assign(
			**{column: np.nan for column in FINANCIAL_FEATURE_COLUMNS}
		)[["date", *FINANCIAL_FEATURE_COLUMNS]]

	financial = financial.rename(columns={"available_date": "earnings_date"})
	merged = pd.merge_asof(
		result,
		financial.sort_values("earnings_date"),
		left_on="date",
		right_on="earnings_date",
		direction="backward",
	)
	merged["days_since_earnings"] = (
		merged["date"] - merged["earnings_date"]
	).dt.days
	return merged[["date", *FINANCIAL_FEATURE_COLUMNS]]
