# 初始效能實驗

日期：2026-09-12（Asia/Taipei）。主機：Apple M3 / macOS arm64。
這是本專案的合成資料研究結果，尚未代表真實部署模型或 x86-64 的效能。

## 結果摘要

下表單位為 **ns／筆**，數值是各輪 native loop 平均每筆時間的中位數；並非單次 request 的 p50。
原型的兩個 profiled 配置都使用 `select_depth=1` 與獨立校準資料。
TL2cgen 一欄取該輪所有已測配置中最低的 block median，包含 LTO、annotation 與 quantization 對照。
這是列舉實驗結果，沒有宣稱已用獨立 tuning set 選出可部署的最佳策略。

| 模型 | TL2cgen 本輪最低 | 原型預設（無校準） | 原型 LLVM 22 + 校準 | 原型同 Clang 17 + 校準 |
|---|---:|---:|---:|---:|
| 100x4 | 438.4 | 341.3 | 309.3 | 291.0 |
| 300x6 | 3751.8 | 4777.0 | 3400.1 | 2972.3 |

較深模型的未校準原型會輸給調過參數的 TL2cgen；分支權重是目前有效的改善。
同一支 Clang 的 IR 對照仍有優勢，表示差距不只來自 LLVM 版本。
不過原型採直接 float32 累加，TL2cgen 生成的部分 leaf 加法在本機包含 float32/float64 轉換；這是另一項影響因素。
本輪尚未修改 TL2cgen C 常數為 float32，因此不能把提升全部歸因於分支配置，也未證明優於所有可能的 TL2cgen 產碼調整。

## 設定與正確性

- 每個模型使用 8192 筆 training rows、4096 筆 calibration rows、4096 筆獨立 evaluation rows，32 個輸入特徵，約 3% NaN。目標由前四個特徵的線性與非線性函式加噪音產生，不是通用模型代表集。
- 每輪抽樣 8192 次單筆呼叫，共 11 輪；每引擎有 90,112 筆個別呼叫時間。每輪打亂引擎順序，相同輪次使用相同 row sequence。
- 定時在 C++ 內完成；原型 native function 沒有 Python 或記憶體配置。每輪、每引擎都有暖機。
- 所有生成引擎與 XGBoost raw-margin prediction 比對後才進行 benchmark。本輪原型所有配置的最大絕對差異都是 0；這不是跨模型 bitwise 相等保證。
- `prepared` 直接呼叫 TL2cgen 的生成函式，定時不含 Entry 轉換；`dense` 包含必要轉換。quantized 版本每次重建 Entry，避免重用被修改的輸入。
- llvmlite 0.49.0 / LLVM 22.1.0、Apple Clang 17.0.0、XGBoost 3.4.1、TL2cgen 1.0.0、Treelite frontend 4.7.2。TL2cgen 內建 Treelite 4.1.2 的版本警告仍存在；每個輸出的數值驗證已通過。
- 本機沒有 CPU 綁核、頻率隔離或專用執行環境。個別呼叫百分位包含計時器成本；不同輪執行的絕對數值可有明顯漂移。沒有 cold-cache、服務排隊或端到端 SLA 測量。

## 完整配置

所有數值為 ns；p50 / p99 是個別 native call 的計時觀測值，含計時器成本。

### 100x4

[原始 JSON](results/controlled-100x4/report.json)，包含 compiler flags、library bytes、各輪時間及最大數值誤差。

| 配置 | Block median / row | Call p50 | Call p99 |
|---|---:|---:|---:|
| llvm_select0 | 340.5 | 375 | 625 |
| llvm_select1 | 341.3 | 375 | 667 |
| llvm_select2 | 341.0 | 334 | 500 |
| llvm_select1_preload | 387.1 | 375 | 708 |
| llvm_select1_profiled | 309.3 | 333 | 583 |
| llvm_select1_clang | 332.1 | 375 | 667 |
| llvm_select1_profiled_clang | 291.0 | 333 | 625 |
| tl2cgen_dense | 483.0 | 458 | 1000 |
| tl2cgen_prepared | 491.2 | 417 | 834 |
| tl2cgen_profiled_dense | 444.5 | 417 | 916 |
| tl2cgen_profiled_prepared | 444.9 | 417 | 1000 |
| tl2cgen_quantized_dense | 843.3 | 833 | 1167 |
| tl2cgen_profiled_quantized_dense | 843.6 | 833 | 1125 |
| tl2cgen_lto_dense | 460.3 | 458 | 917 |
| tl2cgen_lto_prepared | 455.5 | 417 | 833 |
| tl2cgen_profiled_lto_dense | 464.2 | 417 | 959 |
| tl2cgen_profiled_lto_prepared | 438.4 | 417 | 916 |
| tl2cgen_quantized_lto_dense | 815.7 | 792 | 1167 |
| tl2cgen_profiled_quantized_lto_dense | 795.0 | 792 | 1166 |

### 300x6

[原始 JSON](results/controlled-300x6/report.json)，包含 compiler flags、library bytes、各輪時間及最大數值誤差。

| 配置 | Block median / row | Call p50 | Call p99 |
|---|---:|---:|---:|
| llvm_select0 | 4797.3 | 4709 | 6375 |
| llvm_select1 | 4777.0 | 4667 | 6333 |
| llvm_select2 | 4368.0 | 4291 | 5833 |
| llvm_select1_preload | 4708.2 | 4584 | 6291 |
| llvm_select1_profiled | 3400.1 | 3292 | 5083 |
| llvm_select1_clang | 4846.7 | 4667 | 6375 |
| llvm_select1_profiled_clang | 2972.3 | 2959 | 4584 |
| tl2cgen_dense | 7116.6 | 6958 | 9208 |
| tl2cgen_prepared | 7122.8 | 6917 | 9417 |
| tl2cgen_profiled_dense | 4500.7 | 4417 | 6875 |
| tl2cgen_profiled_prepared | 4456.8 | 4375 | 6791 |
| tl2cgen_quantized_dense | 5678.3 | 5542 | 7542 |
| tl2cgen_profiled_quantized_dense | 3826.5 | 3708 | 5708 |
| tl2cgen_lto_dense | 6973.2 | 6958 | 9083 |
| tl2cgen_lto_prepared | 7041.8 | 6875 | 9041 |
| tl2cgen_profiled_lto_dense | 4645.4 | 4541 | 6917 |
| tl2cgen_profiled_lto_prepared | 4672.9 | 4500 | 6958 |
| tl2cgen_quantized_lto_dense | 5789.0 | 5666 | 7667 |
| tl2cgen_profiled_quantized_lto_dense | 3751.8 | 3667 | 5625 |

## 重現

```sh
.venv/bin/python -m benchmarks.run --output results/reproduce-100x4 \
  --trees 100 --depth 4 --rounds 11 --samples 8192 --tl-lto --clang-control
.venv/bin/python -m benchmarks.run --output results/reproduce-300x6 \
  --trees 300 --depth 6 --rounds 11 --samples 8192 --tl-lto --clang-control
```

在部署 CPU 上改用真實模型及代表性的校準/評估資料，是下一個最有價值的驗證。
之後優先加入自動配置選擇，以及修正 TL2cgen leaf 常數精度後的對照；再依 branch misses 與 instruction-cache misses 決定是否需要新的樹表示或 SIMD。
