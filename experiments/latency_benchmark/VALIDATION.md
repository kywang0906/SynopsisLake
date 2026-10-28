# 本次驗證紀錄

此紀錄只涵蓋本機正確性。尚未使用團隊模型或在 Databricks 執行研究效能測試。

- Base revision：`f5a21ce366ab8a13d9a219a37282d4ad37073185`。
- Branch：`feat/e2e-query-latency`。
- 環境：Linux、Java 11.0.28、Maven 3.9.9、CPython 3.12.14。
- Optional integration packages：JPype1 1.6.0、PySpark 3.5.0、Py4J 0.10.9.7、LightGBM 4.6.0、NumPy 2.5.4、SciPy 1.18.1。

## 執行過的檢查

從 repository 根目錄：

```bash
cd SynopsisLake
mvn -B package org.apache.maven.plugins:maven-dependency-plugin:3.8.1:build-classpath -Dmdep.outputFile=target/classpath.txt
cd ..
SYNOPSISLAKE_TEST_JAR="$PWD/SynopsisLake/target/SynopsisLake-0.0.1-SNAPSHOT.jar" \
  python -m unittest discover -s experiments/latency_benchmark/tests -v
```

實際使用環境已有的 Maven proxy/settings、Java truststore，及工作區的 Python venv。
README 提供不含機器特定路徑的重現指令。

| 檢查 | 結果 |
|---|---|
| Maven package | 成功；12 個測試通過，0 失敗、0 errors、0 skipped。其中 11 個為新增 bridge tests |
| Python tests，包含 optional integrations | 14 個通過，0 失敗、0 skipped |
| Java 核心透過 JPype 的完整 fixture CLI | 成功；實際產生逐筆及彙總 CSV |
| Spark JVM bridge + Spark raw exact | 測試通過；包含接受答案與 raw fallback |
| 真正 LightGBM Booster 的儲存、重載、schema 與推論契約 | 測試通過；僅使用暫存 fixture model |
| 修改 raw file 後再次 fallback | 答案從 1 變 2，確認沒有使用先前真值查表 |
| calibration/test 隔離 | 測試通過；test 超標保留原 cutoff 並回報 quality_target_met=false |
| repeated samples / cache modes / zero truth / ties / malformed raw data | 測試通過 |

小資料的可解釋結果：

| Test query | Gate 回答來源 | 回傳答案 | 原始掃描答案 |
|---|---|---:|---:|
| `test_full_cell` | histogram | 2 | 2 |
| `test_partial` | raw fallback | 1 | 1 |
| `test_empty` | histogram | 0 | 0 |

不報告六點 fixture 的研究加速比。樣本太小，不能推論效能或 5% 風險控制能力。
正式實驗仍需要團隊模型/特徵、原始 split、source/merged count arrays、原始 point workload 與 Databricks 權限。
