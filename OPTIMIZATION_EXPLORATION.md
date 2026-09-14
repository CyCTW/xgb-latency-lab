# 其他優化方向：子樹成本策略與後續候選

日期：2026-09-13。目標仍為單筆、單執行緒、dense float32、保留 NaN 語意與逐樹加總順序。

## 本輪選擇：以整棵子樹成本決定 eager select

上一版的 profile 策略只看高度上限與該節點較少見分支是否至少占 15%。深度相同的子樹，節點數與深層分支比例仍可能很不一樣；對整棵子樹 eager evaluation 可能多做許多比較。

在原有 calibration rows 上，模型每筆預期經過約 400／1799 個分裂節點（100×4／300×6）；深度 4 profile 策略在 LLVM 最佳化前，估計會執行約 1072／3276 個比較。這是依模型結構與校準路徑計算的操作量，不是量到的 CPU 指令數或 cycles。

新增 `select_policy="cost"`，從葉往根計算兩種代價：

```text
EagerCost(node)  = 子樹全部分裂節點數
BranchCost(node) = 1 + p_left * BestCost(left) + p_right * BestCost(right)
                    + min(p_left, p_right) * branch_penalty
```

只有該節點有校準樣本、未超過高度上限且 EagerCost 嚴格較低時才採用 eager select；否則保留分支與各子節點各自的策略。`branch_penalty` 是相對於比較的可調權重，**不是該 CPU 的真實 branch-miss cycles**；`min(p_left,p_right)` 也不等同硬體實際誤判率。

未觀察到的路徑不會刪掉。策略只改寫運算方式，不改門檻、葉值、NaN 判斷或加總順序。完整 PGO 尚未實作；這個成本模型也沒有包含所有載入、常數、暫存器與快取成本。

| 300×6 策略 | 預期比較／筆 | 預期保留分支節點／筆 |
|---|---:|---:|
| 原深度 4 profile | 3276 | 1034 |
| cost，penalty 2 | 1799 | 1500 |
| cost，penalty 4 | 1891 | 1407 |
| cost，penalty 8 | 2049 | 1310 |
| cost，penalty 16 | 2394 | 1159 |

降低比較數可能增加分支，因此仍需測量；不能以這張表預測誰最快。[結構分析數據](results/cost-policy-structure.json)

## 如何使用

```sh
.venv/bin/python -m xgb_latency.cli model.json build/cost-model \
  --backend clang --calibration calibration.npy \
  --select-depth 6 --select-policy cost --select-branch-penalty 8

.venv/bin/python -m benchmarks.optimize --preset cost \
  --model model.json --calibration calibration.npy \
  --tuning tuning.npy --evaluation evaluation.npy \
  --output results/tuned-cost
```

`cost` preset 測試 Clang penalty 2／4／8／16、llvmlite penalty 4、Clang penalty 4 分塊 32，以及既有的小模型與大模型配置。所有成本候選高度上限為 6。原版與 float32 常數修改版 TL2cgen 各測試 annotation／quantization 與 LTO 的組合及可用的 prepared ABI。

仍然先寫入 `selection.json` 再讀取 evaluation，並把既有 `clang_d4_adaptive` 與 `clang_d1_block32` 都列入固定最終對照。預設編譯策略沒有改為 cost。

## 其他方向與優先順序

| 方向 | 預期機會 | 需要確認的成本／條件 | 狀態 |
|---|---|---|---|
| 子樹成本策略 | 減少不划算的 eager evaluation | 多算比較與分支失誤之間的取捨 | 本輪實作、測量 |
| 完整 PGO 與程式碼配置 | 根據執行 profile 改善區塊排列與編譯決策 | 原型與 TL2cgen 都必須加入同等 PGO；profile 不可讀取 evaluation | 尚未實作 |
| 局部門檻量化 | 只預先編碼重複使用多次的特徵，減少浮點比較與常數載入 | rank 查找成本、NaN、精確等於門檻的語意；需把編碼納入每筆計時 | 後續已實作，見 [門檻編碼實驗](RANK_ENCODING.md) |
| 依使用頻率拆分程式碼 | 將罕見路徑移出常用函式，嘗試縮小熱路徑 | 額外呼叫成本與分布漂移；不能刪除罕見路徑 | 尚未實作 |
| 跨樹並行計算葉值 | 提高單筆推論的指令層級並行或 SIMD 利用率 | 不規則路徑、暫存器與額外讀取；最終仍要按原順序加總 | 尚未實作 |

[LLVM branch weights](https://llvm.org/docs/BranchWeightMetadata.html) 是編譯器提示，不是硬體命中保證。[Clang PGO 文件](https://clang.llvm.org/docs/UsersManual.html#profile-guided-optimization)描述以 profile 輔助最佳化的流程；它與目前自行統計樹分支、附上 weights 的做法不同。
[TL2cgen 官方最佳化說明](https://tl2cgen.readthedocs.io/en/latest/tutorials/optimize.html)包含 branch annotation 與 integer threshold quantization；局部量化是擬探索的變體，不能宣稱已超越完整量化對照。

本輪沒有加入 `noalias` 或 fast-math：前者會對輸入／輸出重疊增加 API 契約要求，後者可能改變既有數值語意。也不以多個部分和重新排列 float32 加總。這些不能當作免費的編譯器提示。

## 評估設定

沿用兩個已訓練的合成模型及 calibration；以標準常態分布另產生 4096×32 的 tuning／evaluation 矩陣，各約 3% NaN，種子 11541／11542。native harness 為 11 輪、每輪 8192 次，tuning／evaluation 計時種子 11540／11541。
測試在 Apple M3、未綁核的 macOS 桌面環境進行，使用 warm-cache 原生迴圈，不能視為服務端 p99。

## 本輪測量結果

**96 項測試通過。** 包含 cost 策略在兩個 backend、不同分塊、查表／比較共用組合上的數值一致性，門檻相鄰 float32 與 NaN 測試，以及校準資料全是 NaN 時仍保留其他路徑的驗證。所有原型候選在 tuning 上與 XGBoost 的最大絕對差異為 0，入選原型及固定參考在 evaluation 上也為 0；這不是所有模型的 bitwise 保證。

100×4 選中 `clang_cost4_block32`：高度上限 6、成本權重 4、每組 32 棵樹。首次 evaluation 為 159.7 ns／筆，同輪原本的 `clang_d1_block32` 為 168.4 ns／筆，減少約 5.2%。兩者函式庫檔案大小均為 66456 bytes。

配置固定後，不重新編譯或改選，以三個新原生行程及亂數種子重跑相同 evaluation rows：

| 100×4 重跑種子 | 新成本策略 ns／筆 | 原本分塊配置 ns／筆 | 減少時間 |
|---|---:|---:|---:|
| 11543 | 157.9 | 167.8 | 5.88% |
| 11544 | 159.5 | 169.3 | 5.76% |
| 11545 | 158.1 | 167.1 | 5.42% |

這些是 `block_median_ns_per_row`，不是單次 request percentile。三次同輪重跑均改善，但仍需更多模型與部署硬體驗證，不能宣稱普遍有效。

300×6 沒有選中新成本策略，仍選中 `clang_d4_adaptive`。新策略中 tuning 最快的 `clang_cost16` 為 1474.9 ns／筆，既有配置為 1435.5 ns／筆；不把這些 tuning 數字當作最終評估結果。也沒有因為新策略減少估算比較數就強制採用。

下表為三次固定配置重跑的 block median 再取中位數，單位 µs／筆：

| 模型 | 入選原型 | 原版 TL2cgen 入選配置 | float32 常數修改版 TL2cgen |
|---|---:|---:|---:|
| 100×4 | 0.158 | 0.271 | 0.204 |
| 300×6 | 1.435 | 2.416 | 1.999 |

TL2cgen 亦先在 tuning 選配置：小模型選到 profiled prepared，大模型選到 quantized profiled dense；兩者均含 LTO。預設編譯策略仍保持原設定，想試新策略時使用 CLI 或 `--preset cost`。

- [100×4 完整報告及固定配置重跑](results/optimized-cost-100x4/report.json)
- [100×4 所有 tuning 候選](results/optimized-cost-100x4/tuning.json)
- [300×6 完整報告及固定配置重跑](results/optimized-cost-300x6/report.json)
- [300×6 所有 tuning 候選](results/optimized-cost-300x6/tuning.json)

## 重現本輪實驗

模型與 calibration 使用 [初始實驗](BENCHMARKS.md) 的設定。建立新資料時，對 tuning／evaluation 各用 `np.random.default_rng(11541)`／`np.random.default_rng(11542)`，產生 `normal(size=(4096,32)).astype(np.float32)`，再將同一 RNG 的 `random(rows.shape) < .03` 位置設為 NaN。
輸入保存為 `results/optimization-inputs/{shape}-cost-{tuning,evaluation}.npy`。下列命令的模型路徑需指向你實際產生的模型目錄：

```sh
for shape in 100x4 300x6; do
  .venv/bin/python -m benchmarks.optimize --preset cost \
    --model "results/controlled-$shape/model.json" \
    --calibration "results/controlled-$shape/calibration.npy" \
    --tuning "results/optimization-inputs/$shape-cost-tuning.npy" \
    --evaluation "results/optimization-inputs/$shape-cost-evaluation.npy" \
    --output "results/reproduce-cost-$shape" \
    --seed 11540 --rounds 11 --samples 8192
done
```

每次需使用新的輸出目錄。報告保存模型及資料雜湊、固定配置重跑的 binary SHA-256、套件版本與每輪計時；重建輸入不保證重現相同延遲或 tuning 勝出配置。
