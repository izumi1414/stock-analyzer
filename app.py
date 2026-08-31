import plotly.express as px
import pandas as pd
import streamlit as st
from pytrends.request import TrendReq
import yfinance as yf
from src.prediction import train_and_evaluate


def get_google_trends(keyword: str, timeframe: str) -> pd.DataFrame:
	trends = TrendReq(hl="ja-JP", tz=540)
	trends.build_payload([keyword], timeframe=timeframe, geo="JP")
	trend_data = trends.interest_over_time().reset_index()

	if trend_data.empty:
		return pd.DataFrame(columns=["date", "search_index"])

	trend_data = trend_data.rename(columns={"date": "date", keyword: "search_index"})
	trend_data["date"] = pd.to_datetime(trend_data["date"])
	trend_data["search_index"] = pd.to_numeric(trend_data["search_index"], errors="coerce")
	return trend_data[["date", "search_index"]].dropna()


def get_correlation_interpretation(correlation: float) -> str:
	if correlation >= 0.7:
		return "強い正の相関があります。"
	if correlation >= 0.3:
		return "中程度の正の相関があります。"
	if correlation > -0.3:
		return "相関が弱いです。"
	if correlation > -0.7:
		return "中程度の負の相関があります。"
	return "強い負の相関があります。"


stocks = {
	"Apple": "AAPL",
	"Microsoft": "MSFT",
	"NVIDIA": "NVDA",
	"Amazon": "AMZN",
	"Tesla": "TSLA",
	"Toyota": "7203.T",
	"Sony": "6758.T",
}

selected_name = st.selectbox("銘柄を選択", stocks)
selected_ticker = stocks[selected_name]
st.title(f"{selected_name} 株価分析")

data = yf.download(selected_ticker, period="2y", auto_adjust=False)

if data.empty:
	st.error("株価データを取得できませんでした。")
else:
	close = data["Close"]
	if hasattr(close, "columns"):
		close = close[selected_ticker]
	close = close.dropna()

	if len(close) < 2:
		st.error("株価データが不足しています。")
	else:
		current_price = close.iloc[-1]
		previous_price = close.iloc[-2]
		daily_change = current_price - previous_price
		daily_return = daily_change / previous_price * 100

		def period_return(periods):
			if len(close) <= periods:
				return None
			return (current_price / close.iloc[-periods - 1] - 1) * 100

		ma20 = close.rolling(20).mean().iloc[-1]
		ma50 = close.rolling(50).mean().iloc[-1]
		ma200 = close.rolling(200).mean().iloc[-1]

		metric_columns = st.columns(5)
		metric_columns[0].metric("現在価格", f"{current_price:,.2f}")
		metric_columns[1].metric("前日比", f"{daily_change:+,.2f}", f"{daily_return:+.2f}%")
		for column, label, periods in zip(
			metric_columns[2:],
			["1週間騰落率", "1ヶ月騰落率", "1年間騰落率"],
			[5, 21, 252],
		):
			return_rate = period_return(periods)
			column.metric(label, "データ不足" if return_rate is None else f"{return_rate:+.2f}%")

		ma_columns = st.columns(3)
		ma_columns[0].metric("20日移動平均", f"{ma20:,.2f}")
		ma_columns[1].metric("50日移動平均", f"{ma50:,.2f}")
		ma_columns[2].metric("200日移動平均", f"{ma200:,.2f}")

		chart_data = close.to_frame(name="株価")
		chart_data["20MA"] = close.rolling(20).mean()
		chart_data["50MA"] = close.rolling(50).mean()
		chart_data = chart_data.tail(252)

		figure = px.line(
			chart_data,
			x=chart_data.index,
			y=["株価", "20MA", "50MA"],
			title=f"{selected_name} 過去1年間の株価と移動平均",
		)
		figure.update_layout(xaxis_title="日付", yaxis_title="株価")
		st.plotly_chart(figure, use_container_width=True)

		st.subheader("株価予測")
		try:
			prediction_result = train_and_evaluate(data)
			predicted_price = prediction_result.next_close_prediction
			prediction_change = (predicted_price - current_price) / current_price * 100
			prediction_summary = st.columns(3)
			prediction_summary[0].metric("最新の実際の株価", f"{close.iloc[-1]:,.2f}")
			prediction_summary[1].metric(
				"翌日の予測株価", f"{predicted_price:,.2f}"
			)
			prediction_summary[2].metric("予測騰落率", f"{prediction_change:+.2f}%")
			if predicted_price > current_price:
				st.success("上昇予測")
			elif predicted_price < current_price:
				st.warning("下落予測")
			else:
				st.info("横ばい予測")
			st.caption("この予測は過去の株価データを機械学習した結果であり、将来の株価を保証するものではありません。")

			prediction_columns = st.columns(3)
			prediction_columns[0].metric("MAE", f"{prediction_result.metrics['MAE']:,.2f}")
			prediction_columns[1].metric("RMSE", f"{prediction_result.metrics['RMSE']:,.2f}")
			prediction_columns[2].metric("R²", f"{prediction_result.metrics['R2']:.2f}")

			comparison_data = pd.concat(
				[
					prediction_result.actual_values.rename("Actual Close"),
					prediction_result.predicted_values.rename("Predicted Close"),
				],
				axis=1,
			).dropna()
			if comparison_data.empty:
				st.error("実測値と予測値を比較するためのデータがありません。")
			else:
				prediction_figure = px.line(
					comparison_data,
					x=comparison_data.index,
					y=["Actual Close", "Predicted Close"],
					title="実測値と予測値の比較",
				)
				prediction_figure.update_layout(xaxis_title="日付", yaxis_title="株価")
				st.plotly_chart(prediction_figure, use_container_width=True)
		except ValueError as error:
			st.error(str(error))

st.header("Google Trends検索トレンド")
keyword = st.text_input("検索キーワード", value=selected_name).strip()
trend_periods = {
		"過去12ヶ月": "today 12-m",
		"過去5年": "today 5-y",
	}
selected_period = st.selectbox("検索期間", trend_periods)

if not keyword:
	st.warning("検索キーワードを入力してください。")
else:
	try:
		trend_data = get_google_trends(keyword, trend_periods[selected_period])
		if trend_data.empty:
			st.warning("Google Trendsのデータを取得できませんでした。")
		else:
			trend_figure = px.line(
				trend_data,
				x="date",
				y="search_index",
				title=f"Google Trends - {keyword}",
			)
			trend_figure.update_layout(xaxis_title="日付", yaxis_title="検索指数")
			st.plotly_chart(trend_figure, use_container_width=True)

			if not data.empty and len(close) >= 2:
				stock_weekly = close.resample("W").last().rename("stock_price").reset_index()
				stock_weekly["date_key"] = pd.to_datetime(stock_weekly.iloc[:, 0]).dt.date
				trend_data = trend_data.copy()
				trend_data["date_key"] = trend_data["date"].dt.date
				correlation_data = pd.merge(
					stock_weekly[["date_key", "stock_price"]],
					trend_data[["date_key", "search_index"]],
					on="date_key",
					how="inner",
				).dropna()

				st.subheader("Google Trends × 株価 相関分析")
				if len(correlation_data) < 2:
					st.error("相関係数を計算するためのデータが不足しています。")
				else:
					correlation = correlation_data["stock_price"].corr(
						correlation_data["search_index"], method="pearson"
					)
					if pd.isna(correlation):
						st.error("相関係数を計算できませんでした。")
					else:
						st.metric("相関係数", f"{correlation:.2f}")
						st.write(get_correlation_interpretation(correlation))
	except Exception as error:
		st.error(f"Google Trendsデータの取得中に通信エラーが発生しました: {error}")
