import plotly.express as px
import streamlit as st
import yfinance as yf


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
