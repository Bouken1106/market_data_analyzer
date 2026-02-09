# Market Data Analyzer (Twelve Data Basic)

Twelve Data の **Basic プラン**を使って、米国株の現在価格をリアルタイム監視するローカルシステムです。

- 通常時: WebSocket で価格更新を受信
- 接続不安定時: REST `/price` へ自動フォールバック
- Basic 制限を超えないよう、フォールバック時のポーリング間隔を自動調整

## 前提

- Python 3.11 or 3.12（`torch==2.5.1` のため 3.13 は未対応）
- Twelve Data API キー
- Basic プラン想定（WebSocket trial 枠と API クレジット制限に準拠）
- 対応OS: Ubuntu（WSL 含む）/ macOS（Intel, Apple Silicon）

## OSごとの仮想環境

- macOS: `.venv.macos`
- Ubuntu/WSL: `.venv.wsl`
- Windows: `.venv.windows`

`requirements.txt` は OS に応じて PyTorch を自動選択します。

- Ubuntu/WSL: `torch==2.5.1+cpu`
- macOS: `torch==2.5.1`

## セットアップ（macOS）

```bash
bash scripts/setup_macos.sh
source .venv.macos/bin/activate
```

## セットアップ（Ubuntu/WSL）

```bash
bash scripts/setup_wsl.sh
source .venv.wsl/bin/activate
```

## セットアップ（Windows PowerShell）

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv.windows\Scripts\Activate.ps1
```

`.env` を編集して API キーを設定:

```env
TWELVE_DATA_API_KEY=your_api_key
DEFAULT_SYMBOLS=AAPL,MSFT,NVDA,AMZN,GOOGL
API_LIMIT_PER_MIN=8
API_LIMIT_PER_DAY=800
DAILY_BUDGET_UTILIZATION=0.75
PER_MIN_LIMIT_UTILIZATION=0.9
REST_MIN_POLL_INTERVAL_SEC=30
MARKET_CLOSED_SLEEP_SEC=60
SYMBOL_CATALOG_COUNTRY=United States
SYMBOL_COUNTRY_MAP=
SYMBOL_CATALOG_TTL_SEC=86400
SYMBOL_CATALOG_MAX_ITEMS=25000
HISTORICAL_DEFAULT_YEARS=5
HISTORICAL_MAX_YEARS=10
HISTORICAL_CACHE_TTL_SEC=43200
HISTORICAL_INTERVAL=1day
HISTORICAL_MAX_POINTS=2000
```

## 起動

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

ブラウザで `http://localhost:8000` を開くと、ダッシュボードが表示されます。

## ページ構成

右上の `Pages` ボタンから各ページに遷移できます。

- `/`: US Stock Live Monitor（リアルタイム監視）
- `/ml-lab`: ML Forecast Lab（モデル一覧から選択して翌営業日分布を推定）
- `/strategy-lab`: Strategy Lab（戦略検証の準備ページ）
- `/compare-lab`: Model Compare Lab（複数銘柄 × 複数モデルの一括比較）

## 使い方

1. 検索欄をクリックすると、Twelve Data から取得した米国株シンボル候補が表示される（入力時はシンボルの頭文字一致で絞り込み）
2. 候補をクリックすると監視銘柄に追加され、即時反映される（上限8）
3. 画面上部に日本時間（JST）・米国時間（ET）・米国市場の通常取引時間（09:30-16:00 ET）と開場状態を表示
4. 下のテーブルの `Symbol` 欄の銘柄名をクリックすると、`/historical/{symbol}` に遷移してヒストリカルチャートとリスク指標を表示
5. 下のテーブルの `Symbol` 欄にある `x` を押すと監視対象から除外
6. テーブルに価格・更新時刻が表示され、取得ソース（`websocket` / `rest` / `stored`）は更新時刻の下に小さく表示
   - `change(%)` は営業中は「現在値 vs 前営業日終値」、休場中は「直近営業日終値 vs その1つ前の営業日終値」で表示
7. `Refresh Credits` で日次の残APIクレジットを手動更新（`/api_usage` を呼ぶため 1 クレジット消費）
8. `/ml-lab` では Model Catalog からモデルを選択し、実行可能なモデルで以下を表示:
   - 0.1%〜99.9%分位点の分位点関数プロット（代表日を複数比較）
   - ファンチャート（q50, q25-q75, q05-q95, 実測値）
   - 最新時点から翌営業日分布（PDF/CDF、Returns/Prices切替）
   - 予測に毎日従った場合の「過去60営業日」実績バックテスト（戦略 vs 固定株数ホールド）
   - 戦略は「1日1%超損失の確率が3%を超えない上限」の範囲で期待リターン最大の株式比率を採用
   - test の平均ピンボール損失 / 被覆率（q05-q95, q25-q75）
   - 時系列分割は「直近2カ月を test、残りを train/val=4:1」で分割
   - 現在 `Ready` は Quantile LSTM / PatchTST Quantile、他モデルは `Coming Soon` として UI 上で選択可能
9. `/compare-lab` では、複数銘柄（例: AAPL, MSFT, GOOG, JPM, XOM, UNH, WMT, META, LLY, BRK.B, NVDA, HD）に同一ハイパーパラメータを適用してモデルを一括学習・比較
   - 比較期間（test）は「最新データから直近2カ月」
   - 学習/検証は残り期間を `4:1` で分割（train=80%, val=20%）
   - モデル別サマリーと、銘柄×モデル詳細（Pinball/MAE/RMSE/MAPE/SMAPE/Coverage）を表示

補足:
- 起動時に `/api_usage` を1回呼び、日次残量を初期化します（1クレジット消費）。
- その後は REST フォールバック時の実行回数に応じて、日次残量を推定で減算表示します（`(est)` 表示）。
- 正確な日次残量に再同期したいときは `Refresh Credits` を押してください。
- シンボル一覧は `/stocks?country=United States` を取得してキャッシュし、UI検索に使います（`SYMBOL_CATALOG_TTL_SEC` で再取得間隔を調整可能）。
- 最終取得価格は `app/cache/last_prices.json` に保存され、銘柄追加時は保存済み価格（`stored`）を即表示します。
- ヒストリカルデータは `GET /api/historical/{symbol}` で取得し、サーバー側でTTLキャッシュします。
- 価格の自動更新（WebSocket購読/RESTフォールバック）は「各国マーケットの営業日・営業時間内」のみ実行します（営業時間外は `market-closed`）。
- 国判定は `SYMBOL_COUNTRY_MAP`（例: `AAPL:United States,7203.T:Japan,9988.HK:Hong Kong`）を優先し、未設定時はシンボル接尾辞ヒント（`.T`, `.HK`, `.L` など）→ `SYMBOL_CATALOG_COUNTRY` の順で判定します。
- この時間判定は通常取引時間ベースです（祝日・臨時休場・昼休みは未考慮）。

## Basic プラン向け実装ポイント

- 監視銘柄は最大 8（WebSocket trial 制約に合わせる）
- REST フォールバック時は「分あたり上限」と「日次上限」の両方で自動抑制
- WebSocket が切断・失速した場合のみ REST を有効化

レート制限が厳しい場合は、`.env` でさらに遅くできます。

- `API_LIMIT_PER_MIN`: 分あたり上限（Basic は 8）
- `API_LIMIT_PER_DAY`: 日次上限（Basic は 800）
- `DAILY_BUDGET_UTILIZATION`: 日次上限の使用率（`0.75` なら 75%まで）
- `PER_MIN_LIMIT_UTILIZATION`: 分上限の使用率（`0.9` なら 90%まで）
- `REST_MIN_POLL_INTERVAL_SEC`: 最短リクエスト間隔（秒）

デフォルト設定（上記）では、REST は最大でも約 `600 requests/day` を目安に動きます。  
監視銘柄が 5 つの場合、1銘柄あたり約 144 秒間隔、全銘柄1巡は約 12 分です。

## API エンドポイント

- `GET /api/snapshot`: 現在の状態と価格スナップショット
- `POST /api/symbols`: 監視銘柄更新 (`{"symbols":"AAPL,MSFT"}`)
- `GET /api/credits`: 現在把握している日次クレジット情報
- `GET /api/credits?refresh=true`: Twelve Data `/api_usage` から日次残クレジットを再取得（1クレジット消費）
- `GET /api/symbol-catalog`: 検索候補用シンボル一覧（キャッシュ）
- `GET /api/symbol-catalog?refresh=true`: シンボル一覧を強制再取得
- `GET /api/historical/{symbol}?years=5`: 過去N年ヒストリカルデータ（デフォルト5年）
- `GET /api/ml/models`: ML Forecast Lab のモデル一覧（Ready / Coming Soon）
- `GET /api/ml/quantile-lstm?...`: Quantile LSTM を学習・推論し、分位点/評価/描画データを返却（`months=3..60`, デフォルト `60`）
- `GET /api/ml/patchtst?...`: PatchTST Quantile を学習・推論し、分位点/評価/描画データを返却（`months=3..60`, デフォルト `60`）
- `POST /api/ml/quantile-lstm/jobs`: Quantile LSTM 非同期ジョブを開始
- `POST /api/ml/patchtst/jobs`: PatchTST 非同期ジョブを開始
- `POST /api/ml/compare/jobs`: 複数銘柄 × 複数モデル比較ジョブを開始（直近2カ月評価、残り4:1分割）
- `GET /api/ml/jobs/{job_id}`: 非同期ジョブ状態を取得（`queued` / `running` / `cancelling` / `completed` / `failed` / `cancelled`）
- `POST /api/ml/jobs/{job_id}/cancel`: 実行中ジョブの停止を要求
- `GET /api/stream`: SSE でリアルタイム配信
- `GET /ml-lab`: モデル選択式の分位点予測ページ
- `GET /strategy-lab`: 戦略検証ページ（準備用）
- `GET /compare-lab`: モデル一括比較ページ

## 注意

- この実装は「監視用途」の最小構成です。注文機能や資産管理は含みません。
- API キーはサーバー側環境変数で管理し、クライアント側に露出させない設計です。
