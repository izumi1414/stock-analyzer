# 株価分析アプリ

Streamlitとyfinanceを使った株価分析アプリです。

## 起動方法

### 1. 仮想環境を作成

```bash
python3 -m venv .venv
source .venv/bin/activate
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

株価データの取得にはインターネット接続が必要です。
