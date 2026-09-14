# 最佳化進展與獨立評估

主機為 Apple M3；這裡只描述本機合成模型，不推論 x86-64 或生產服務 SLA。

## 第一階段：依模型自動選擇產碼配置

每個模型用獨立 tuning 資料從 12 個原型候選選一個，原版與修改版 TL2cgen 各自選模；選定並寫入 selection.json 後才讀取新的 evaluation 資料。修改版只把累加常數轉成 float32，包含最後的 base score；它不是原版 TL2cgen。

評估完成後固定所有配置，以另一個原生行程和亂數種子重跑確認。下面每格為 **首次 / 重跑**，單位 µs／筆，代表 native loop 平均時間的跨輪中位數。

| 模型 | 舊配置 | 選中的原型 | 原版 TL2cgen | 修改 float32 常數的 TL2cgen |
|---|---:|---:|---:|---:|
| 100x4 | 0.243 / 0.249 | 0.239 / 0.222 | 0.371 / 0.360 | 0.318 / 0.316 |
| 300x6 | 2.697 / 2.740 | 2.067 / 2.133 | 3.324 / 3.362 | 2.764 / 2.765 |

300×6 選到 `clang_d3_adaptive`：較少見分支比例至少 15% 才使用 eager select，最大子樹深度 3。相對同輪舊的 `clang_d1`，首次與重跑均減少約 22–23% 的時間。小模型選到 `clang_d1_block32`，收益在兩次測量中差異較大，暫不視為穩定的顯著改善。

拆分函式不是大型模型的必然最佳選擇：此輪大型模型的 block16/32/64 候選都輸給未拆分的深度 3 配置。所有候選留在 tuning.json 中，包括退步配置。

## 可使用的產物

- [100×4 完整結果](results/optimized-100x4/report.json) / [選定的模型 metadata](results/optimized-100x4/selected_model/metadata.json)

- [300×6 完整結果](results/optimized-300x6/report.json) / [選定的模型 metadata](results/optimized-300x6/selected_model/metadata.json)

函式庫、model.o、model.h 都位於各結果目錄的 selected_model/。使用原本的 `Predictor` 介面或 C ABI。

## 測量限制

每次量測均為 11 輪、每輪 8192 次單筆呼叫。不同引擎使用相同的隨機 row sequence，每輪隨機調整引擎順序。沒有 CPU 綁核或頻率隔離；所有絕對時間與較小差距仍可能受桌面負載影響。

TL2cgen 候選本輪固定使用 LTO，覆蓋 annotation / quantization 的四種組合與可用的 prepared / dense 介面。這並非窮舉所有 TL2cgen 編譯配置。

原型保留 NaN 語意與 float32 逐樹累加順序。所有候選在 tuning 資料先驗證，最後選中的模型也在 evaluation 資料核對 XGBoost；詳細數值誤差保留在報告中。

## 第二階段：更深的自適應 select 與比較重用

這一階段改用新的 tuning / evaluation 矩陣（亂數種子 9331 / 9332），評估檔案仍在 selection.json 寫入之後才被讀取。訓練模型和 calibration 與第一階段相同。

入選的是 `clang_d4_adaptive`，未啟用比較重用。對照固定為上一階段的 `clang_d3_adaptive`。下表單位 µs；block 欄是每筆攤提時間，p99 是原生呼叫的計時觀測值，含 timer 成本。

| 配置 | Block 首次 | Block 重跑 | 呼叫 p99 重跑 |
|---|---:|---:|---:|
| 上一階段原型（深度 3 自適應） | 2.604 | 2.687 | 4.208 |
| 本階段入選原型（深度 4 自適應） | 2.456 | 2.604 | 3.708 |
| 原版 TL2cgen 入選配置 | 4.035 | 3.982 | 6.250 |
| 修改 float32 常數的 TL2cgen 入選配置 | 3.458 | 3.305 | 5.375 |

相對同輪上一版，首次 / 重跑分別減少 **5.7% / 3.1%** 的時間。提升比第一階段小，仍應在部署硬體多次確認。
不同階段的絕對時間有漂移，不能直接拿第二階段的時間與第一階段相比；上面的百分比只使用同輪參考配置。

比較重用採用校準資料的預期執行次數排序，限制 4/8/16/32 個條件；功能與正確性測試已保留，但本輪沒有選中該配置，不能宣稱它對本模型提供額外增益。
直接對所有深度 4 子樹使用 select，在 tuning 資料上大幅退步，說明條件篩選比一味增加 select 範圍更重要。

### 第二階段 tuning 候選（全部原型）

以下數值只供了解 tuning 行為，不是 final-test 結果，單位 ns／筆。

| 配置 | Block median |
|---|---:|
| clang_d4_adaptive | 1686.7 |
| clang_d3_preload | 1707.4 |
| clang_d3_hoist16 | 1740.8 |
| clang_d3_hoist8 | 1753.0 |
| clang_d3_hoist4 | 1766.5 |
| clang_d3_hoist32 | 1768.7 |
| clang_d3_adaptive | 1795.7 |
| clang_d4_hoist8 | 1817.5 |
| clang_d3_block32 | 2067.1 |
| clang_d5_adaptive | 2161.7 |
| llvm_d3_adaptive | 2278.5 |
| clang_d4 | 4695.0 |

[第二階段完整結果](results/optimized-phase2-300x6/report.json) / [入選模型 metadata](results/optimized-phase2-300x6/selected_model/metadata.json)

### 使用本輪選好的模型

```python
from xgb_latency import Predictor

predictor = Predictor("results/optimized-phase2-300x6/selected_model/model.dylib")
raw_margin = predictor.predict(rows)
```

這是本機 macOS 的產物；Linux 需重新編譯，副檔名為 `.so`。C/C++ 可連結同目錄的 model.o，使用 model.h 的 predict_row 介面。

若要在自己的模型重做第二階段選模：

```sh
.venv/bin/python -m benchmarks.optimize --preset expanded \
  --model model.json --calibration calibration.npy \
  --tuning tuning.npy --evaluation evaluation.npy \
  --output results/tuned-expanded
```

確認重跑固定已選中的配置，沒有改選或重新編譯；其亂數種子為 8130，詳見報告的 confirmation 欄及 evaluation-repeat.json。

目前共 56 項測試通過；包括分塊後逐位元一致、重複比較與 NaN 路徑、TL2cgen 修改版、以及讀取 evaluation 前先固定配置的流程檢查。
