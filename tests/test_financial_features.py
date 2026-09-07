import unittest

import numpy as np
import pandas as pd

from src.financial_features import create_earnings_features


def financial_data():
	return pd.DataFrame(
		{
			"決算期": ["2024-12-31", "2025-12-31"],
			"発表日": ["2025-02-10", "2026-02-10"],
			"売上高": [100.0, 120.0],
			"営業利益": [20.0, 30.0],
			"純利益": [10.0, 18.0],
			"EPS": [1.0, 1.5],
		}
	)


class EarningsFeatureTests(unittest.TestCase):
	def test_normal_data_creates_growth_margins_and_days(self):
		stock = pd.DataFrame(index=pd.to_datetime(["2025-02-11", "2026-02-11"]))

		result = create_earnings_features(stock, financial_data())

		self.assertAlmostEqual(result.loc[1, "revenue_growth"], 0.2)
		self.assertAlmostEqual(result.loc[1, "operating_income_growth"], 0.5)
		self.assertAlmostEqual(result.loc[1, "operating_margin"], 0.25)
		self.assertAlmostEqual(result.loc[1, "net_profit_margin"], 0.15)
		self.assertAlmostEqual(result.loc[1, "eps_growth"], 0.5)
		self.assertEqual(result.loc[1, "days_since_earnings"], 1)

	def test_before_first_announcement_has_no_earnings_features(self):
		stock = pd.DataFrame(index=pd.to_datetime(["2025-02-09"]))

		result = create_earnings_features(stock, financial_data())

		self.assertTrue(pd.isna(result.loc[0, "revenue"]))
		self.assertTrue(pd.isna(result.loc[0, "days_since_earnings"]))

	def test_first_year_growth_is_nan(self):
		stock = pd.DataFrame(index=pd.to_datetime(["2025-02-11"]))

		result = create_earnings_features(stock, financial_data())

		self.assertTrue(pd.isna(result.loc[0, "revenue_growth"]))

	def test_missing_values_remain_nan(self):
		data = financial_data()
		data.loc[1, "営業利益"] = np.nan
		data.loc[1, "EPS"] = np.nan
		stock = pd.DataFrame(index=pd.to_datetime(["2026-02-11"]))

		result = create_earnings_features(stock, data)

		self.assertTrue(pd.isna(result.loc[0, "operating_income"]))
		self.assertTrue(pd.isna(result.loc[0, "operating_income_growth"]))
		self.assertTrue(pd.isna(result.loc[0, "eps"]))

	def test_zero_previous_values_do_not_divide(self):
		data = financial_data()
		data.loc[0, ["売上高", "営業利益", "純利益", "EPS"]] = 0
		stock = pd.DataFrame(index=pd.to_datetime(["2026-02-11"]))

		result = create_earnings_features(stock, data)

		self.assertTrue(pd.isna(result.loc[0, "revenue_growth"]))
		self.assertTrue(pd.isna(result.loc[0, "eps_growth"]))

	def test_zero_revenue_does_not_divide(self):
		data = financial_data()
		data.loc[1, "売上高"] = 0
		stock = pd.DataFrame(index=pd.to_datetime(["2026-02-11"]))

		result = create_earnings_features(stock, data)

		self.assertTrue(pd.isna(result.loc[0, "operating_margin"]))
		self.assertTrue(pd.isna(result.loc[0, "net_profit_margin"]))

	def test_empty_financial_data_returns_feature_schema(self):
		stock = pd.DataFrame(index=pd.to_datetime(["2025-02-11"]))

		result = create_earnings_features(stock, pd.DataFrame())

		self.assertEqual(len(result), 1)
		self.assertTrue(pd.isna(result.loc[0, "revenue"]))
		self.assertEqual(list(result.columns), ["date", "revenue", "revenue_growth", "operating_income", "operating_income_growth", "operating_margin", "net_income", "net_income_growth", "net_profit_margin", "eps", "eps_growth", "days_since_earnings"])


if __name__ == "__main__":
	unittest.main()
