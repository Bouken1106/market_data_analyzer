# Lead-Lag Subspace PCA

このモジュールは、論文「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」に沿った処理を、既存の `MarketDataHub` と独立して追加するための実装です。

## 構成

- `data_adapter.py`: 既存 `hub.historical_payload()` をそのまま使う薄い adapter
- `preprocessing.py`: OHLC から `rcc` / `roc` と rolling 標準化行列を生成
- `subspace_pca.py`: `V0`, `C0`, `Creg_t`, 固有分解
- `signals.py`: `f_t`, `zhat_J`, `B_t = VJ VU^T`
- `evaluation.py`: 上位/下位分位の等ウェイト long-short 評価
- `service.py`: API 向けオーケストレーション

## 既定値

- `L = 60`
- `lambda = 0.9`
- `K = 3`
- `q = 0.3`
- `Cfull = 2010-01-01 .. 2014-12-31`

すべて API request から上書きできます。固定値のベタ書きにはしていません。

## データ前提

- U.S. 側の情報集合: 当日 `Close-to-Close`
- Japan 側の評価対象: 翌営業日 `Open-to-Close`
- 既存取得系の OHLC 形式: `{"t","o","h","l","c","v"}`

## 安全側の実装上の扱い

- 取得失敗や履歴不足の銘柄は、自動で除外し、理由を `data_summary.excluded_symbols` に返します。
- `Ct` と `Cfull` は pairwise correlation を作り、不足要素は prior target `C0` で補ってから相関行列へ射影します。
- `B_t` は `include_transfer_matrix=true` のときだけ API レスポンスへ載せます。

## 使い方

1. `GET /api/leadlag/config` で既定設定を取得
2. `POST /api/leadlag/analyze` へ設定を送信
3. `latest_signal` と `strategy.summary` を参照
