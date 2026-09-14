# 精確單筆推論：布局、葉值表、排程與完整 PGO

本輪沿用 model → LLVM IR → optimized LLVM IR → native code，新增七類實作／編譯實驗，並進一步測試樹分塊及 select 成本參數的組合。模型仍是兩個固定合成模型：100 棵深度 4、300 棵深度 6，各 32 個特徵。沒有使用 fast-math、樹加總重排或近似 threshold quantization。

## 各方向的做法與發現

| 方向 | 實作 | 第一輪觀察 |
|---|---|---|
| Eytzinger 搜尋布局 | 將精確 rank 的門檻表排成 breadth-first 完整二元樹，固定回合搜尋 | 正確性通過；兩個原配置都沒有明顯收益 |
| SIMD 門檻計數 | 4-lane float32 ordered comparison，整數累計所有 `threshold <= x` 的數量 | 比 serial search 多做比較；兩模型皆較慢 |
| 緊湊葉值表 | eager 子樹先用整數 select 決定葉索引，再讀取每片葉子一格的 float32 常數表 | 深度 2 候選在兩模型的 tuning 勝出，進入組合實驗 |
| 延後累加 | 先求 4／16 棵樹的葉值，再保持原順序加入 float32 accumulator | Clang 將兩個原配置的指令內容還原成相同內容，未增加有效排程自由度 |
| O2／Os／Oz | 對相同 IR 改用不同編譯速度／大小策略 | 沒有穩定領先；Oz 在大模型原配置上明顯退步。llvmlite 0.49 無 size-level 介面，Os／Oz 明確限用 Clang |
| 完整 instrumentation PGO | Clang 插入計數器、以 calibration 跑原生函式、合併 profile、重新編譯 | 原型未勝出；TL2cgen 大模型有收益，因此保留更強的 PGO 對照 |
| IEEE 位元前綴分桶 | 以保序的 float32 位元前綴查出起始 rank，再用完整門檻做精確搜尋；索引採可容納的最小 8／16／32-bit 整數 | 12-bit 將主要特徵搜尋從 7–8 輪減為 3–4 輪；另測 8／10／14／16-bit 前綴及更多特徵 |
| 組合與重新分塊 | 從第一輪 tuning 選定的原型出發，交叉調整 rank 布局、累加排程、compact 深度、cost penalty 與 block size | 使用全新 tuning／evaluation，結果見下表 |

緊湊葉值表不同於先前 truth table：後者需要最多 `2^比較數` 格，現在的表只需要 `葉子數` 格。不過表格載入仍有依賴與快取成本，且總函式庫可能更大；不能因為名稱是「緊湊」就推論 binary 較小。底層 LLVM 仍可把 select 改為跳躍。

## 實驗流程

- 校準資料沿用原模型的獨立 calibration。新資料每份 4,096 × 32 float32，標準常態並有約 3% NaN。
- 第一輪 tuning／evaluation 資料 seed 為 13761／13762，native seed 13760／13761；組合輪資料 seed 14871／14872，native tuning seed 14870。
- 每次 tuning 都先驗證數值，再計時並寫入 `selection.json`，之後才開啟該輪 evaluation。組合輪使用前一輪 tuning 的選擇，不使用前一輪 evaluation 選配置。
- 每次計時 11 輪、每輪 8,192 筆；引擎順序隨機化、同輪使用相同 row sequence。表中的單位是原生迴圈 `block_median_ns_per_row`，包含索引、checksum 與呼叫成本。
- 組合輪額外以 seed 14872／14873／14874 執行三個全新 native process，沿用固定 binary 與 evaluation，沒有重新選模。
- TL2cgen 的 stock／float32 常數修改版分列，兩者皆包含 LTO、annotation／quantization 組合，以及適用的 prepared／dense 介面。PGO 也分列。
- 第一輪 TL PGO 從 dense 介面訓練。組合輪進一步對未量化版本同時訓練 dense／prepared 介面並合併 profile；保留先前 PGO 作 tuning 候選。量化版只使用會重建輸入的 dense 介面。

PGO profile 必須包含非零函式執行計數，並將 profile 不相符警告視為錯誤。instrumented 函式的計時只供訓練紀錄，從未當作效能分數。[Clang PGO 說明](https://clang.llvm.org/docs/UsersManual.html#profile-guided-optimization)

分桶輪與前綴粒度輪也使用新的 tuning／evaluation：資料 seed 分別為 15981／15982、17091／17092，native tuning seed 為 15980、17090；各自再執行三個固定配置確認程序。每輪都沿用 tuning 選出的前一輪配置，baseline 各 family 也只由該輪 tuning 選取。

分桶以 canonical zero 合併 +0／−0，再建立保持數值順序的 IEEE 整數 key。directory 儲存該前綴之前的門檻數；該桶最多包含幾個門檻在編譯時已知，因此可縮短固定回合搜尋。跨到下一桶的門檻必定大於輸入，不會被誤計；尾端 padding 的 +∞ 結果會 clamp，NaN 最後回傳 −1 sentinel。沒有丟棄低位元、近似門檻或假設測試資料分布。directory 對極端指數與空桶也有定義。

## 正確性與環境

265 項測試通過，涵蓋四種 rank encoder 的門檻／相鄰 float32／±∞／NaN，以及隨機位元模式、正負零、subnormal、最大有限值與 8／12／16-bit 分桶、兩個 backend 的組合、原生 PGO、TL2cgen 兩種 ABI 與 PGO object 重新連結。完整記錄：`results/exploration-20260913/tests-final-265.log`。測試覆蓋不是所有 XGBoost 模型的形式化證明。

本機為 Apple M3、macOS 26.6.2、Apple Clang 17，llvmlite 0.49／LLVM 22.1；TL2cgen 1.0.0、Treelite frontend 4.7.2，內建 Treelite 4.1.2 的警告仍存在。未綁核、隔離 CPU 頻率或移除其他桌面工作。絕對 ns 在各時段有波動，因此改善幅度只對同一 native process 的固定參考配置計算，不跨不同時段直接相減。此處不是服務端 p99，也沒有跨 CPU／模型必勝保證。

Ubuntu 已嘗試既有 machine 與獨立 Ubuntu 24.04 容器，皆未成功執行。服務能回應清單，但 machine 命令逾時、新容器卡在 Starting container。Linux 與硬體計數器尚無結果，詳見 [Ubuntu 重現紀錄](UBUNTU_BENCHMARK.md)。

## 可重現指令

```sh
.venv/bin/python -m benchmarks.optimize \
  --model results/controlled-100x4/model.json \
  --calibration results/controlled-100x4/calibration.npy \
  --tuning results/optimization-inputs/100x4-explore-tuning.npy \
  --evaluation results/optimization-inputs/100x4-explore-evaluation.npy \
  --output results/new-explore-100x4 --preset explore --seed 13760 --pgo-training dense

.venv/bin/python -m benchmarks.combine \
  --previous results/new-explore-100x4/report.json \
  --model results/controlled-100x4/model.json \
  --calibration results/controlled-100x4/calibration.npy \
  --tuning results/optimization-inputs/100x4-combine-tuning.npy \
  --evaluation results/optimization-inputs/100x4-combine-evaluation.npy \
  --output results/new-combine-100x4
```

大模型將路徑中的 `100x4` 換成 `300x6`。前一輪函式庫與 C／IR 不得刪除或覆蓋，組合實驗會重用它們；output 必須是新目錄。核心 `.o`、`.h`、動態函式庫及 metadata 複製到各輪的 `selected_model/`。


接續第三、四輪時，用 `benchmarks.combine --mode bucket --seed 15980` 和 `--mode prefix --seed 17090`，`--previous` 分別指向組合輪及分桶輪的 report，並提供相應全新資料與 output。資料仍用 `np.random.default_rng(seed).normal(size=(4096,32)).astype(np.float32)`，再以同一 RNG 的下一批 uniform 值將約 3% 元素設成 NaN。合成模型的來源 seed 為 2026（`benchmarks.run`）。

## 最終固定配置與確認結果

下表為最後一輪 **三個獨立 native process** 的範圍，非跨不同時段挑選最低值。原配置指本輪開始前已驗證的 threshold-rank 配置。所有數值皆為 ns／筆。

| 模型 | 最終原型 | 原配置 | TL stock＋PGO | TL 修改 float32＋PGO | 相對原配置延遲下降 |
|---|---:|---:|---:|---:|---:|
| 100x4 | 116.1–116.8 | 150.6–152.4 | 269.5–272.4 | 204.3–204.4 | 22.9–23.6% |
| 300x6 | 755.0–758.2 | 1014.3–1020.5 | 2343.3–2345.4 | 1914.7–1938.3 | 25.6–25.7% |

### 100x4

選定 `prefix_bits16_rank16`：select policy `cost`、depth 6、block 16、compact leaf depth 2，以 16-bit 前綴編碼 14 個特徵。
函式庫從原配置 **50,088 bytes** 增至 **486,728 bytes**。上一輪 12-bit 分桶的 `bucket_rank8` 為 **73,496 bytes**，可作空間取捨參考；跨輪絕對延遲不可直接比較。
TL PGO 選定 `tl_stock_pgo_prepared`；float32 修改版選定 `tl_f32_modified_pgo_prepared`，兩者都有 LTO。
| 確認 seed | 原型 | 原配置 | stock＋PGO | 修改 float32＋PGO |
|---|---:|---:|---:|---:|
| 17092 | 116.394 | 152.445 | 272.415 | 204.366 |
| 17093 | 116.806 | 151.738 | 272.146 | 204.269 |
| 17094 | 116.130 | 150.579 | 269.516 | 204.285 |

可直接載入的本機產物：`results/prefix-macos-100x4/selected_model/model.dylib`。
完整候選、tuning 決定與逐輪數字：[100x4 report](results/prefix-macos-100x4/report.json)。

### 300x6

選定 `prefix_bits16`：select policy `profile`、depth 4、block 16、compact leaf depth 2，以 16-bit 前綴編碼 32 個特徵。
函式庫從原配置 **365,016 bytes** 增至 **1,515,848 bytes**。上一輪 12-bit 分桶的 `bucket_rank` 為 **525,128 bytes**，可作空間取捨參考；跨輪絕對延遲不可直接比較。
TL PGO 選定 `tl_stock_q_profiled_dual_pgo_dense`；float32 修改版選定 `tl_f32_modified_q_profiled_dual_pgo_dense`，兩者都有 LTO。
| 確認 seed | 原型 | 原配置 | stock＋PGO | 修改 float32＋PGO |
|---|---:|---:|---:|---:|
| 17092 | 758.250 | 1020.477 | 2345.418 | 1935.526 |
| 17093 | 755.020 | 1014.282 | 2343.608 | 1914.724 |
| 17094 | 756.653 | 1017.176 | 2343.277 | 1938.349 |

可直接載入的本機產物：`results/prefix-macos-300x6/selected_model/model.dylib`。
完整候選、tuning 決定與逐輪數字：[300x6 report](results/prefix-macos-300x6/report.json)。

可攜的結果摘要（含所有 tuning、確認測量、配置與來源雜湊）：[exploration-20260913.json](reports/exploration-20260913.json)。四輪原型在各自 tuning／evaluation 上與 XGBoost 的觀測最大絕對誤差皆為 0；未將未量測模型納入此結論。
