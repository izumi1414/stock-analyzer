# 株価分析アプリ

## プロジェクト概要

yfinanceから取得した株価データとGoogle Trendsの検索トレンドを使って、銘柄分析や株価予測を行うStreamlitアプリです。

## 使用技術

- Python
- Streamlit
- yfinance
- pandas
- Plotly
- pytrends
- scikit-learn

## 主な機能

- 複数銘柄の選択と株価表示
- 現在価格、前日比、期間別騰落率の表示
- 20日、50日、200日移動平均線の表示
- Google Trends検索トレンドの可視化
- 株価とGoogle Trendsの相関分析
- LinearRegressionによる翌日の終値予測
- MAE、RMSE、R²による予測評価
- 時系列バックテストとDirectional Accuracyの表示

## 使い方

1. 銘柄を選択します。
2. 株価チャートと各種指標を確認します。
3. Google Trendsの検索キーワードと期間を指定します。
4. 株価予測結果とバックテスト期間を確認します。

株価およびGoogle Trendsデータの取得にはインターネット接続が必要です。

## セットアップ手順

### 1. 仮想環境を作成・有効化

```bash
python3 -m venv .venv
source .venv/bin/activate
```

既存の `.env` 環境を利用する場合は、次のコマンドを実行します。

```bash
source .env/bin/activate
```

### 2. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

### 3. アプリを起動

```bash
streamlit run app.py
```

起動後、ブラウザで http://localhost:8501 を開いてください。

終了する場合は、ターミナルで `Ctrl+C` を押します。

## ディレクトリ構成

```text
market-scope/
├── app.py                 # Streamlitアプリのエントリーポイント
├── requirements.txt       # Python依存パッケージ
├── README.md              # プロジェクト説明とセットアップ手順
├── .gitignore             # Git管理対象外ファイルの設定
└── src/
	└── prediction.py      # 株価予測・バックテストロジック
```
