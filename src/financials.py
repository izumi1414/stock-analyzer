from collections.abc import Iterable

import pandas as pd


FINANCIAL_COLUMNS = [
	"売上高",
	"営業利益",
	"純利益",
	"EPS",
	"粗利益",
	"EBITDA",
	"総資産",
	"総負債",
	"フリーキャッシュフロー",
	"営業利益率",
	"純利益率",
]


def _get_financial_frame(ticker: object, attribute: str) -> pd.DataFrame:
	try:
		frame = getattr(ticker, attribute)
	except Exception:
		return pd.DataFrame()
	return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _find_row(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
	for candidate in candidates:
		if candidate in frame.index:
			return pd.to_numeric(frame.loc[candidate], errors="coerce")
	return pd.Series(dtype="float64")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
	if frame.empty:
		return frame
	result = frame.copy()
	result.columns = pd.to_datetime(result.columns, errors="coerce")
	result = result.loc[:, ~result.columns.isna()]
	return result


def get_financial_data(ticker: object) -> pd.DataFrame:
	"""Return annual financial metrics in one row per reporting period."""
	income = _normalise_columns(_get_financial_frame(ticker, "income_stmt"))
	balance_sheet = _normalise_columns(_get_financial_frame(ticker, "balance_sheet"))
	cashflow = _normalise_columns(_get_financial_frame(ticker, "cashflow"))

	frames = [income, balance_sheet, cashflow]
	periods = sorted({period for frame in frames for period in frame.columns})
	if not periods:
		return pd.DataFrame(columns=["決算期", *FINANCIAL_COLUMNS])

	def values(candidates: Iterable[str], frame: pd.DataFrame) -> pd.Series:
		series = _find_row(frame, candidates)
		return series.reindex(periods)

	revenue = values(("Total Revenue", "Operating Revenue"), income)
	operating_income = values(("Operating Income",), income)
	net_income = values(
		("Net Income", "Net Income Common Stockholders"), income
	)
	result = pd.DataFrame(
		{
			"決算期": periods,
			"売上高": revenue.to_numpy(),
			"営業利益": operating_income.to_numpy(),
			"純利益": net_income.to_numpy(),
			"EPS": values(("Basic EPS", "Diluted EPS"), income).to_numpy(),
			"粗利益": values(("Gross Profit",), income).to_numpy(),
			"EBITDA": values(("EBITDA", "Normalized EBITDA"), income).to_numpy(),
			"総資産": values(("Total Assets",), balance_sheet).to_numpy(),
			"総負債": values(
				("Total Liabilities Net Minority Interest", "Total Liabilities"),
				balance_sheet,
			).to_numpy(),
			"フリーキャッシュフロー": values(("Free Cash Flow",), cashflow).to_numpy(),
		}
	)
	result["営業利益率"] = (result["営業利益"] / result["売上高"] * 100).where(
		result["売上高"].ne(0)
	)
	result["純利益率"] = (result["純利益"] / result["売上高"] * 100).where(
		result["売上高"].ne(0)
	)
	return (
		result[["決算期", *FINANCIAL_COLUMNS]]
		.dropna(how="all", subset=FINANCIAL_COLUMNS)
		.sort_values("決算期")
		.reset_index(drop=True)
	)
