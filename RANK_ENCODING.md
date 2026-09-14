# 高頻特徵的精確門檻編碼

本輪新增 `rank_feature_limit`，以模型已知的所有門檻，將部分高頻特徵在每次單筆呼叫開始時編碼為整數。其他特徵仍直接使用原本浮點比較，預設 0 關閉。

## 語意與實作

對選中的特徵，將模型內的 float32 門檻去重、排序為 `t[0] … t[n-1]`。非 NaN 輸入的編碼為：

```text
rank(x) = 模型門檻中 <= x 的數量
x < t[i]  ⇔  rank(x) <= i
```

這裡沒有降低輸入精度或改動門檻，`x == t[i]` 會產生大於 i 的編碼，維持 XGBoost 嚴格 `<` 的語意。這個簡化利用了目前支援的數值樹都使用同一比較運算；不能未經修改就套用到含 `<=` 或 categorical split 的模型。

NaN 編碼為 i32 的 -1。節點預設往左時，使用 signed `rank <= i`，-1 會成立；預設往右時，使用 unsigned 比較，同一位元表示會大於所有合法門檻索引，因此不成立。每個節點保留自己的 missing 方向。

編碼採固定輪數的二分搜尋：每輪載入一個已知排序表中的門檻，比較後以 select 更新位置。表格填補到 `2^k - 1` 個元素，使所有索引在範圍內；以正無限大填補，最後將位置限制在真實門檻數，正無限大輸入也維持正確編碼。`k = threshold_count.bit_length()`。

若啟用樹分塊，編碼只做一次並存於固定大小的本次呼叫堆疊空間，各組共用該結果；沒有 heap 配置、全域可變快取或跨筆重用。模型函式本身完成編碼，因此整個成本包含在 native benchmark 裡。

特徵選擇以 calibration 的原始走樹平均使用次數減去二分搜尋輪數排序，保留正值並受 `rank_feature_limit` 限制。這只是啟發式，未包含所有載入、暫存器與 LLVM 產碼成本。**門檻集合始終取自完整模型**，不刪除校準資料未見過的門檻。

[TL2cgen 官方文件](https://tl2cgen.readthedocs.io/en/latest/tutorials/optimize.html)亦描述將門檻轉成整數的最佳化，並指出每筆輸入的轉換成本。本實驗的差異是選擇部分高頻特徵、使用適合嚴格 `<` 的 rank 編碼、在 LLVM IR 展開固定輪數搜尋，以及把 NaN 方向整合到 signed／unsigned 比較。效能仍需與 TL2cgen 的 quantized 配置實測。

## 使用

```sh
.venv/bin/python -m xgb_latency.cli model.json build/rank-model \
  --backend clang --select-depth 4 --select-policy profile \
  --calibration calibration.npy --rank-feature-limit 4

.venv/bin/python -m benchmarks.optimize --preset ranks \
  --model model.json --calibration calibration.npy \
  --tuning tuning.npy --evaluation evaluation.npy \
  --output results/tuned-ranks
```

metadata 的 `rank_features` 記錄實際選中的特徵、門檻數與搜尋輪數。設定為 32 不代表一定選滿 32 個特徵。

候選包含 Clang 深度 4 profile 搭配最多 2／4／8／32 個特徵、llvmlite 的最多 4 個特徵，以及成本策略分塊 32 搭配最多 4 個特徵。既有大模型配置 `clang_d4_adaptive` 與小模型配置 `clang_cost4_block32` 都保留為候選及最終固定對照。
TL2cgen 原版與 float32 常數修改版各包含 annotation／quantization 組合、LTO，以及可用的 prepared／dense 介面。

## 評估設定

沿用原本 100×4／300×6 合成模型與 calibration，另外以 RNG 種子 12651／12652 生成 tuning／evaluation，各 4096×32 標準常態 float32、約 3% NaN。選定並保存 `selection.json` 後才讀取 evaluation。
native harness 採 11 輪、每輪 8192 次呼叫，tuning／evaluation 計時種子 12650／12651。Apple M3 上的 warm-cache 桌面測量，未綁核或隔離頻率，不能視為服務 p99。

## 結果

完整測試 **118 passed**。直接檢查 rank 編碼的 12 組門檻數包含二次方長度與其相鄰長度，輸入涵蓋門檻、相鄰 float32、signed zero、正負無限大及多種 NaN 位元表示。兩個 backend 的分塊／未分塊模型也與 XGBoost 比對，並測試與成本策略、比較共用及葉值查表的組合。
本輪所有原型候選在 tuning 上、入選及固定原型對照在 evaluation 上，對 XGBoost 的最大絕對誤差均為 0；這不是跨模型 bitwise 保證。

選模結果：

- **100×4**：`clang_cost4_block32_rank4`，對特徵 3／0／1／2 每筆各做 7 輪搜尋，共用於所有分塊。首次 evaluation 為 150.6 ns／筆，上一版 `clang_cost4_block32` 為 159.4 ns／筆。
- **300×6**：`clang_d4_rank32`，本輪實際選取全部 32 個特徵，每個 8 輪。首次 evaluation 為 1102.5 ns／筆，上一版 `clang_d4_adaptive` 為 1571.9 ns／筆。沒有預設「局部一定勝過全部編碼」。

固定 tuning 選中的 binary，不重新編譯或改選，以新原生行程、種子 12653／12654／12655 重跑相同 evaluation 資料。下表是 `block_median_ns_per_row`，不是 request p50：

| 模型 | 重跑種子 | 新配置 ns／筆 | 同輪上一版 ns／筆 | 減少時間 |
|---|---:|---:|---:|---:|
| 100×4 | 12653 | 151.7 | 160.7 | 5.59% |
| 100×4 | 12654 | 151.5 | 160.3 | 5.50% |
| 100×4 | 12655 | 150.8 | 158.9 | 5.13% |
| 300×6 | 12653 | 1022.4 | 1479.4 | 30.89% |
| 300×6 | 12654 | 1014.1 | 1447.0 | 29.92% |
| 300×6 | 12655 | 1019.9 | 1475.7 | 30.88% |

對照包含同輪固定的 TL2cgen 入選配置。以下為三次 block median 的中位數，單位 µs／筆：

| 模型 | 新原型 | 原版 TL2cgen | float32 常數修改版 TL2cgen |
|---|---:|---:|---:|
| 100×4 | 0.151 | 0.273 | 0.206 |
| 300×6 | 1.020 | 2.521 | 2.049 |

小模型兩個 TL2cgen family 都選中 profiled prepared，大模型都選中 quantized profiled dense，均包含 LTO。修改版維持獨立標示；本輪未加入完整 PGO。

300×6 原生呼叫的觀測 p99，三次新配置為 1417／1375／1417 ns，上一版為 2167／2125／2167 ns。這些含 timer 成本與解析度影響，不代表服務端端到端 p99。

## 產碼變化

| 模型 | 上一版函式庫 bytes | 新配置函式庫 bytes |
|---|---:|---:|
| 100×4 | 66456 | 50088 |
| 300×6 | 462792 | 365016 |

在 300×6 的 Clang 最佳化 IR 中，靜態 `fcmp` 指令出現次數從 14635 降至 288；新增 14669 次 `icmp`，主要樹比較改成整數比較。這是靜態 IR 計數，**不是每筆動態執行的指令數**。編碼本身每筆仍有搜尋與載入成本。
函式庫縮小與實際計時改善都有觀察到；未取得硬體計數器，因此不能量化其中多少來自快取、比較成本或其他編譯器變化。

這一輪有比上一版更明確的收益，但資料仍是兩個合成模型。`rank_feature_limit` 預設保持 0，可透過 `--preset ranks` 在自己的模型和部署硬體上選擇。

- [100×4 完整報告與固定配置重跑](results/optimized-ranks-100x4/report.json)
- [100×4 所有 tuning 候選](results/optimized-ranks-100x4/tuning.json)
- [300×6 完整報告與固定配置重跑](results/optimized-ranks-300x6/report.json)
- [300×6 所有 tuning 候選](results/optimized-ranks-300x6/tuning.json)

## 重現本輪實驗

模型與 calibration 使用 [初始實驗](BENCHMARKS.md) 的設定。對 tuning／evaluation 各用 `np.random.default_rng(12651)`／`np.random.default_rng(12652)`，產生 `normal(size=(4096,32)).astype(np.float32)`，再將同一 RNG 的 `random(rows.shape) < .03` 位置設為 NaN。
分別保存為 `results/optimization-inputs/{shape}-ranks-{tuning,evaluation}.npy`，再執行：

```sh
for shape in 100x4 300x6; do
  .venv/bin/python -m benchmarks.optimize --preset ranks \
    --model "results/controlled-$shape/model.json" \
    --calibration "results/controlled-$shape/calibration.npy" \
    --tuning "results/optimization-inputs/$shape-ranks-tuning.npy" \
    --evaluation "results/optimization-inputs/$shape-ranks-evaluation.npy" \
    --output "results/reproduce-ranks-$shape" \
    --seed 12650 --rounds 11 --samples 8192
done
```

請使用新輸出目錄，模型路徑需指向實際產生的初始模型目錄。報告保存資料與模型雜湊、每輪時間、套件版本，以及固定配置重跑的 binary SHA-256；重建同一資料不保證重現相同延遲或選模結果。
