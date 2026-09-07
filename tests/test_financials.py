import unittest

import pandas as pd

from src.financials import get_financial_data


class FakeTicker:
	def __init__(self, income_stmt, balance_sheet, cashflow):
		self.income_stmt = income_stmt
		self.balance_sheet = balance_sheet
		self.cashflow = cashflow


class FinancialDataTests(unittest.TestCase):
	def test_financial_data_combines_metrics_and_calculates_margins(self):
		period = pd.Timestamp("2025-12-31")
		ticker = FakeTicker(
			pd.DataFrame(
				{
					period: {
						"Total Revenue": 1000,
						"Operating Income": 200,
						"Net Income": 100,
						"Basic EPS": 1.5,
					}
				}
			),
			pd.DataFrame({period: {"Total Assets": 5000}}),
			pd.DataFrame({period: {"Free Cash Flow": 150}}),
		)

		result = get_financial_data(ticker)

		self.assertEqual(len(result), 1)
		self.assertEqual(result.loc[0, "売上高"], 1000)
		self.assertEqual(result.loc[0, "EPS"], 1.5)
		self.assertEqual(result.loc[0, "営業利益率"], 20)
		self.assertEqual(result.loc[0, "純利益率"], 10)

	def test_empty_or_missing_financial_data_does_not_raise(self):
		result = get_financial_data(
			FakeTicker(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
		)
		self.assertTrue(result.empty)
		self.assertIn("売上高", result.columns)

	def test_missing_metric_is_nan(self):
		period = pd.Timestamp("2025-12-31")
		result = get_financial_data(
			FakeTicker(
				pd.DataFrame({period: {"Total Revenue": 1000}}),
				pd.DataFrame(),
				pd.DataFrame(),
			)
		)
		self.assertTrue(pd.isna(result.loc[0, "純利益"]))


if __name__ == "__main__":
	unittest.main()
