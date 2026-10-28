# 端到端 range-count latency benchmark

此目錄只新增 benchmark。不修改 ARM、histogram 核心、既有模型或部署流程。

## 目前 checkout 有什麼

起點為 `f5a21ce`。沒有 AGENTS.md。現有流程為：

1. `sigspatial25/CountHistExp` 與 `DataDrivenRangeQueryMergingRateExp` 載入來源、執行 ARM、評估 count-histogram。
2. `SimpleSpatialHistogramOpt.RangeQuery` 用重疊面積比例估算。
3. `RangeQueryWithRatio` 提供逐筆 Skewness score。尚無 calibration gate。
4. `CountHistExp` 讀取預先給定的答案來算誤差，沒有線上 exact fallback。
5. `BuildRangeAns` 展示原始 point 的閉區間判斷及可選權重，但它是離線 GT 產生器。
6. `.github/workflows/build-upload-and-run.yml` 在 DBR 14.3 LTS / Scala 2.12 / single-user 上執行幾何摘要建構，不是 ML latency job。舊 `job-create.json` 使用另一組 DBFS 路徑。

**此 checkout 沒有報告中的 LightGBM/v0/v1、feature pipeline、模型、既有 split，或研究用 10.5M points。** 不會用新 heuristic 冒充既有 ML。正式 ML run 必須提供下述接入資料。

## 快速本機驗證

Java 11、Maven、Python 3.10+。從 repository 根目錄執行：

```bash
cd SynopsisLake
mvn -B package org.apache.maven.plugins:maven-dependency-plugin:3.8.1:build-classpath -Dmdep.outputFile=target/classpath.txt
cd ..
python -m venv /tmp/synopsislake-bench-venv
/tmp/synopsislake-bench-venv/bin/pip install -r experiments/latency_benchmark/requirements-local.txt
python -m unittest discover -s experiments/latency_benchmark/tests -v
/tmp/synopsislake-bench-venv/bin/python experiments/latency_benchmark/make_smoke_config.py \
  --classpath SynopsisLake/target/classpath.txt \
  --jar SynopsisLake/target/SynopsisLake-0.0.1-SNAPSHOT.jar \
  --config /tmp/synopsislake-smoke.json \
  --output /tmp/synopsislake-smoke-results
/tmp/synopsislake-bench-venv/bin/python -m experiments.latency_benchmark \
  --config /tmp/synopsislake-smoke.json
```

如環境需要 Maven proxy/truststore，沿用環境的 Maven settings；不要關閉 TLS 驗證。
output 必須是新目錄，避免混入舊結果。再次執行請換 output 路徑。

fixture 只有六個原始 point，是 correctness smoke，不是效能或 ML 研究結果。
Java smoke 執行真實核心 estimator、Skewness、calibration 與原始 CSV fallback。
Python unit tests 的模型替身只測試 runner 分支、無 label 洩漏和 timing；不是 v0/v1。

選用的實際 Spark/Booster integration checks（已驗證 CPython 3.12、Spark 3.5.0、LightGBM 4.6.0）：

```bash
/tmp/synopsislake-bench-venv/bin/pip install pyspark==3.5.0 lightgbm==4.6.0
SYNOPSISLAKE_TEST_JAR="$PWD/SynopsisLake/target/SynopsisLake-0.0.1-SNAPSHOT.jar" \
  /tmp/synopsislake-bench-venv/bin/python -m unittest discover \
  -s experiments/latency_benchmark/tests -v
```

這些只驗證接入契約。測試用 Booster 在暫存目錄建立，不是團隊模型；不在正式環境更換訓練依賴。
沒有 optional dependencies / JAR 時，對應 integration tests 會明確 skipped。

## 正式輸入契約

每次 run 固定一份 merged histogram / merge rate / 模型。用同一份既有 query split 分別執行各設定。
本工具不訓練、不建立新 split、不用 test 決定 cutoff。

Query CSV 必須有以下 header：

```csv
query_id,xmin,ymin,xmax,ymax,split
```

`split` 僅接受 `train`、`calibration`、`test`。query ID 必須唯一。
請從團隊的**既有** split 匯出這份 manifest，不要重新亂數切分。
同一 query 跨 merge rates 必須維持相同 split。本 runner 不會把其他欄位傳給 extractor。
train rows 不會拿來校準或計時 test。

Histogram/source JSON 格式：

```json
{"x_edges":[0,1,2],"y_edges":[0,1,2],"counts":[[2,0],[2,2]]}
```

`counts[x][y]`。請從原 pipeline 的實際 source/merged arrays 匯出，保留原始浮點邊界。
不在 benchmark 重建 ARM。不要把 geometric-histogram Parquet 當成 count-histogram。
只有 features 真的需要來源時才設定 `sources`，並把需要的來源全部列出。

原始資料在 `exact.files` 逐一宣告。座標欄為從 0 起算的 `x_col` / `y_col`，預設 0/1。
`delimiter` 預設逗號；`header` 預設 false；有 `weight_col` 才做加權 count，否則每筆為 1。
閉區間 `xmin <= x <= xmax && ymin <= y <= ymax` 沿用 `BuildRangeAns`。
要確認這個語義與團隊原 query truth 一致，尤其是邊界點與權重。

`scan` 每次 exact 都重新開啟、解析、掃描原始 CSV，沒有答案快取。
對帶有尾端 synopsis 的自訂檔案，必須明確填入 `data_rows` 排除尾端；不足列數會失敗。
`spark` 支援 point-only CSV，不支援帶尾端格式；需要先另外匯出純原始 point。
Spark 每次 exact 都執行 filter/aggregate/collect action，不讀取預算好的 GT。
兩者是參考掃描器；研究結論只能針對選用的 executor，不能泛化成已索引資料庫的效能。

## 接入既有 LightGBM

`ml.factory` 是 `module:function`。factory 在初始化階段載入既有模型。
返回物件須提供：

```python
def features(query, estimate, histogram, sources):
    # query 只有 ID 與 bounds；沒有 exact count 或 split。
    # histogram.getHistogram() 是既有 Java SimpleSpatialHistogramOpt。
    # sources 同樣是 bridge 物件。
    ...

def predict(features):
    # 單一分數，越小越接受；不可在此讀取離線 query 分數或 GT。
    ...
```

可以直接接團隊自己的 factory。或使用內建 `lightgbm_adapter:load`：

```json
{
  "factory":"experiments.latency_benchmark.lightgbm_adapter:load",
  "training_seconds":null,
  "options":{
    "model_path":"/Volumes/CATALOG/SCHEMA/VOLUME/model.txt",
    "feature_callable":"team_features:extract",
    "feature_names":["EXACT_EXISTING_MODEL_FEATURE_NAMES_IN_ORDER"],
    "num_threads":1
  }
}
```

`team_features.extract(query, estimate, histogram, sources)` 必須重用既有線上特徵程式，回傳 feature-name 到 value 的 dict。
model schema 與名稱順序不符會失敗。`features_ns` 包括 extractor 與 array 建構；`inference_ns` 包括 Booster.predict。
模型 score 若是 log error，可直接用原分數做排序校準，不需要改模型。
非有限 score 一律拒絕。Histogram/source 讀取在 runner 計時；extractor 另做的 query-time I/O 也必須在這些計時方法內。
禁止在 factory 預先讀取 test features、分數或 GT。

訓練由既有 pipeline 完成。`training_seconds` 沒有可信原紀錄時保留 null，不填 0。
提供數值時它是外部紀錄，不代表此 benchmark 重跑過訓練。
保留原 pipeline 的 training、early stopping、calibration 分工，使用 training 環境相同的 LightGBM 版本。
自訂 factory 可提供 JSON-compatible `metadata`，記錄 model/schema/code hashes 與版本。

## Databricks 執行

1. 在上述獨立分支 build JAR。將產物放到團隊授權的 UC Volume。不要執行原 CI workflow 來代替本步。
2. 使用 dedicated/single-user classic compute，建議與原研究相同的 DBR 14.3 LTS。此 adapter 使用 `_jvm`，不支援 Spark Connect/serverless。
3. 在 compute Libraries 安裝本分支的 JAR，再啟動 Python session。它只包含 project classes；Spark 由 DBR 提供。
4. ML 使用團隊原套件版本與既有 feature module。所有方法在同一 compute、同一原始資料、同一 split 下執行。
5. 複製 `databricks.example.json` 到自己的 Volume。替換所有 `REPLACE` 路徑及 schema，記錄不可變 dataset/model/split 版本、DBR、node type 和 worker 數。
6. 在 notebook 中執行，路徑以你的 Git folder 為準：

```python
import sys
sys.path.insert(0, "/Workspace/Shared/SynopsisLake")
from experiments.latency_benchmark.runner import run
run("/Volumes/CATALOG/SCHEMA/VOLUME/benchmark/config.json")
```

讀取 raw/hist/source/model、寫入 output 都需要對應 Volume 權限。
這些 Volume 不會由 Git clone 建立。本 PR 不建立 job、修改權限或部署。
ML 檔案尚未取得時，可以明確選 `methods: ["always_exact", "skewness"]`；它不是 ML 結果。

**Skewness 限制：** 本 checkout 的 `ComputeBetaDist` 在長方形網格使用錯誤 stride。
bridge 對長方形的 Skewness 初始化明確失敗，避免回報失真的比較；長方形的 histogram estimate 仍支援。
正式 histogram 若不是方形，移除 `skewness` 方法並記錄原因，待團隊另行修正核心。
Skewness score 直接重用既有方法。它內部重算一次 estimate，此成本完整列入 `features_ns`。

## 計時與結果解讀

預設 epsilon=25%、delta=5%。cutoff 僅由 calibration 決定：依 score 排序，整組 ties 一起接受，選 empirical violation <= delta 的最大 prefix。
沒有 feasible prefix 就全部拒絕。這不保證 test violation <=5%。**不能根據 test 結果改 cutoff。**
零真值時：答案為 0 則 error=0，其他答案則 error=+inf。校準與 test 都納入，不平滑分母，也不在執行時用真值篩選。
`truth_group` 另標記 zero、small(<10)、at_least_10，便於分析與原報告排除小真值的政策差異。
既有 checkout 原始誤差除法沒有完整零值政策；這是本 benchmark 明確新增的定義。

每次查詢計入：hist/source lookup → estimate → feature extraction → inference → gate → 必要的 raw exact。
query manifest 解析、輸出 CSV、事後品質評估不在 response latency 內。Query ID/座標在所有方法間相同。
各 query 的方法順序與各 repeat 的 query 順序使用固定 seed 打散。warmup 只使用 calibration queries。
exact baseline 每個 repeat、每個 query 都真正執行。其結果只在計時結束後用來驗證答案，絕不餵給 gate 或 fallback。
訓練、模型載入、JVM 初始化、resident histogram/source/Beta 初始化、exact executor 初始化、calibration、warmup 分開記錄。

`histogram_cache=resident`：小摘要啟動時載入，每次查詢計 lookup 成本，載入成本放 initialization。
`histogram_cache=per_query`：每筆重新讀取並建構摘要；Skewness 也重新初始化 Beta，成本進 lookup。
`exact.cache=none`：不設定應用層資料 cache。Spark 可選 `persist`，materialization 成本計入初始化，只清除此 runner 的 persistence。
**這些都不代表 OS、UC/FUSE 或 object-store cache 是 cold。** interleaved methods 共用 storage cache；不要宣稱 cold-cache latency。
請用不同 output、相同環境分別跑兩種 cache 條件及至少 5 repeats。

輸出：

| 檔案 | 內容 |
|---|---|
| `queries.csv` | 每筆、每 repeat、每 method 的答案來源、接受決策、score、truth、error、各階段及 E2E ns |
| `calibration.csv` | calibration 的原始 exact label 與 score 計算耗時，非線上預算答案 |
| `calibration_summary.csv` | 每個 gate 凍結的 cutoff、calibration 接受/違規樣本數 |
| `summary.csv` | 每 repeat 及整體的 mean/p95、接受比例、接受與違規樣本數、違規率、品質目標是否觀察到達標、speedup |
| `run.json` | 完整設定、cutoff、初始化/校準/訓練時間、split sizes、artifact hashes、raw file metadata、版本、git SHA/dirty flag |

重複執行不是新增獨立品質樣本；整體 accepted sample count 依 query ID 去重。
無接受答案時 violation rate/quality_target_met 留白，不宣稱達標。
p95 使用 nearest rank `ceil(0.95*n)`。`speedup_vs_exact` 是 mean exact / mean method 的 **steady-state** 比值。
不含另外列出的初始化與 calibration。一次性建構成本是否值得，要用實際服務 query 數另外攤提，不可隱藏。
速度與 quality_target_met 必須一起呈現；即使 observed=true，也只是該 test 集的觀察值，不是統計保證。
只有 `run.json` 的 `result_status=complete` 才代表完成；失敗產生的目錄不可拿來當結果。

## 正式 ML 實驗尚缺

團隊既有 feature/model 程式與 schema、原 train/calibration/test manifest、同版本 count histograms、原始 point 檔案與權重/邊界語義、模型與訓練耗時紀錄、Databricks compute/Volume 權限。
目前只能驗證 runner 與小資料流程，不能宣稱團隊 ML 加速、5% 達標，或已在 Databricks 驗證。
