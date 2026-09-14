# 利用已知模型結構產生 LLVM IR

本專案的流程為：XGBoost save_model JSON → 經驗證的 Forest → 模型專用 LLVM IR (`model.ll`) → O3 最佳化 IR (`model.opt.ll`) → 原生物件檔與共享函式庫。
llvmlite backend 使用 LLVM pass builder；Clang backend 將相同原始 IR 交給與 TL2cgen 對照相同的 Clang。兩條路徑都保存最佳化前後 IR 與組合語言。

這個編譯器架構參考 [lleaves 作者的說明](https://siboehm.com/articles/21/lleaves)；[lleaves 專案](https://github.com/siboehm/lleaves)也提供直接連結 native binary 的方式。本文新增的位元索引查表是本專案的實驗，並非宣稱 lleaves 採用同一演算法。
TL2cgen 的 C 同樣能經過 Clang/LLVM，因此直接產生 IR 本身不構成更快的證據。

## 本輪新增：編譯時子樹查表

模型的拓樸、特徵索引、float32 門檻、NaN 預設方向與葉值，在編譯前已知。對原本會 eager evaluate 的小子樹，先找出不同的完整比較條件，再列舉所有布林組合，沿原始樹走訪並填入葉值。

例如三個比較 c0、c1、c2，索引為 `c0 | (c1 << 1) | (c2 << 2)`，可在 8 格唯讀表中取得葉值。原本一連串 float select 的依賴，改為比較、整數組合與一次索引讀取。這仍需 LLVM 與目標 CPU 驗證；不能假定查表永遠更快。

- 相同特徵、門檻與 NaN 方向的比較共用一個位元；NaN 方向不同不能合併。
- 即使有些布林組合因門檻關係不可能出現，仍完整填表，避免對未見過的輸入做不安全假設。
- 只改寫葉值的選擇；base margin 與逐樹 float32 加總順序保持不變。
- 只在 select 策略選中的區域內嘗試轉表。較大的區域可遞迴使用較小的表；不強迫整棵樹查表。
- 每表最多 `2 ** leaf_table_bits` 個 float32。預設 0 關閉，上限 8；表數與最佳化前表格位元組數記錄在 metadata 中。LLVM 可合併或移除常數表，因此該數字不等同最終 binary 的資料段大小。
- 額外的比較、索引計算、資料快取與 load dependency 都可能抵銷收益。保留既有配置，使用獨立 tuning 資料選擇。

## 使用

```sh
.venv/bin/python -m xgb_latency.cli model.json build/table-model \
  --backend clang --select-depth 4 --select-policy profile \
  --calibration calibration.npy --leaf-table-bits 3

.venv/bin/python -m benchmarks.optimize --preset tables \
  --model model.json --calibration calibration.npy \
  --tuning tuning.npy --evaluation evaluation.npy \
  --output results/tuned-tables
```

`tables` preset 含兩個既有配置，以及 Clang／llvmlite 的深度 2、3、4 與每表 3 或 5 個比較候選。TL2cgen 原版及 float32 常數修改版各比較 annotation／quantization 組合，全部啟用 LTO；可用時包括 prepared ABI。這並未窮舉所有編譯配置。

## 正確性與評估

新增測試對完整輸入組合檢查 NaN、正負無限大、signed zero、門檻及其相鄰 float32 值，也驗證重複條件與不同 missing 方向。兩個 backend、分塊與比較共用的組合都與獨立走樹或 XGBoost 的結果比對。
選模流程另檢查 `selection.json` 已寫入後才讀取 evaluation 資料。

效能評估維持原模型與 calibration，另外產生 4096×32 的 tuning／evaluation 矩陣，各取標準常態分布並放入約 3% NaN，亂數種子分別為 10431／10432。量測使用 11 輪、每輪 8192 次，原生 harness 種子 10430；最終評估使用 10431。
這是 Apple M3 上未綁核的 warm-cache 桌面實驗，不能直接代表服務 p99 或其他 CPU 的效能。

## 本輪結果

完整測試為 **75 passed**。兩個模型的全部原型候選在 tuning 上、入選原型在 evaluation 上，對 XGBoost 的最大絕對差異均為 0；TL2cgen 仍依既有容許誤差驗證，不能將原型的零誤差套用到所有對照或其他模型。

每個模型固定 tuning 選中的配置與 binary，再以三個新行程、種子 10433／10434／10435 重跑相同 evaluation rows。未重新編譯或改選，報告保存各 binary 的 SHA-256。下表是這三次 `block_median_ns_per_row` 的中位數，單位 µs／筆，並非 request latency percentile。

| 模型 | 入選原型 | 原版 TL2cgen 入選配置 | float32 常數修改版 TL2cgen |
|---|---:|---:|---:|
| 100×4 | 0.168 | 0.272 | 0.208 |
| 300×6 | 1.441 | 2.438 | 2.014 |

100×4 選中既有 `clang_d1_block32`，**沒有選中查表**。300×6 選中 `clang_d4_table5_profile`，但與同輪固定參考 `clang_d4_adaptive` 的差距很小：

| 300×6 量測 | 查表原型 ns／筆 | 既有原型 ns／筆 | 查表減少時間 |
|---|---:|---:|---:|
| 首次 evaluation | 1438.1 | 1449.8 | 0.80% |
| 重跑 10433 | 1479.1 | 1479.1 | 約 0% |
| 重跑 10434 | 1441.1 | 1450.1 | 0.62% |
| 重跑 10435 | 1438.6 | 1435.7 | -0.20% |

**這輪確認了引擎在這兩個合成模型上仍快於已測 TL2cgen 配置，沒有證明新查表技巧相對既有引擎有穩定提升。** 保持 `leaf_table_bits=0` 為預設；不能把引擎原有優勢歸因於本輪新增技巧。

300×6 入選查表配置在原始 IR 產生 3240 個表、共 115840 bytes，函式庫為 658504 bytes；既有配置為 462792 bytes。查表增加約 42% 的函式庫大小，目前微小的時間差不足以支持預設啟用。沒有硬體計數器資料，因此不把快取壓力當作已證明的效能原因。

- [100×4 報告與固定配置重跑](results/optimized-tables-100x4/report.json)
- [100×4 所有 tuning 候選](results/optimized-tables-100x4/tuning.json)
- [300×6 報告與固定配置重跑](results/optimized-tables-300x6/report.json)
- [300×6 所有 tuning 候選](results/optimized-tables-300x6/tuning.json)

## 重現本輪輸入與選模

先依 [初始實驗](BENCHMARKS.md) 的命令產生兩個合成模型及 calibration，並將下方 `results/controlled-{shape}` 改為對應輸出目錄。接著建立新的輸入：

```python
from pathlib import Path
import numpy as np

root = Path("results/optimization-inputs")
root.mkdir(parents=True, exist_ok=True)
for shape in ("100x4", "300x6"):
    for kind, seed in (("tuning", 10431), ("evaluation", 10432)):
        rng = np.random.default_rng(seed)
        rows = rng.normal(size=(4096, 32)).astype(np.float32)
        rows[rng.random(rows.shape) < .03] = np.nan
        np.save(root / f"{shape}-tables-{kind}.npy", rows)
```

```sh
for shape in 100x4 300x6; do
  .venv/bin/python -m benchmarks.optimize --preset tables \
    --model "results/controlled-$shape/model.json" \
    --calibration "results/controlled-$shape/calibration.npy" \
    --tuning "results/optimization-inputs/$shape-tables-tuning.npy" \
    --evaluation "results/optimization-inputs/$shape-tables-evaluation.npy" \
    --output "results/reproduce-tables-$shape" \
    --seed 10430 --rounds 11 --samples 8192
done
```

輸出目錄必須是新目錄。資料及模型可重建，但不同執行環境的計時與 tuning 選擇不保證相同。
