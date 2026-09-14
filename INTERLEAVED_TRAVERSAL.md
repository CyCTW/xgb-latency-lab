# 同一筆資料的跨樹交錯遍歷（2026-09-14）

這個實驗同時推進數棵樹，讓 CPU 有機會重疊獨立的 node／feature 讀取。它處理的是同一筆輸入的不同樹，**不是多筆 batch 推論，也沒有建立額外 thread**。

## 實作

- `traversal_lanes=0` 預設關閉；可選 1、2、4、8、16。
- 全部樹保存為 preorder 的 8-byte node，加上 root index 表。每一組沿原始樹順序分組，由共享的 `noinline` LLVM helper 推進。
- 同一組每層先讀取各 lane 的 node，再讀取對應 feature。迴圈次數是此組樹的最大高度；LLVM loop metadata 禁止完整展開迴圈，以保留小型指令工作集。
- 已到葉子的 lane 保持在原 leaf，餘下層級讀取有效的 feature 0；其 dummy 比較不會改變 node 或輸出。不同高度的樹、stump、constant tree、空森林與最後一組不足寬度都保留正確語意。
- 遍歷完成後才依原始樹順序執行 float32 加法，累加值傳入下一組。沒有局部和重排、fast-math 或 threshold／leaf 近似。

`traversal_mode=scalar` 使用獨立 scalar 比較；`vector` 將四棵樹的比較放進 LLVM `<4 x float>`，支援總寬度 4／8／16。Apple Clang 17 的組合語言已確認出現 `fcmgt.4s` 與 `fcmeq.4s`。Node／feature 的地址不連續，讀取仍需要 scalar load／lane insertion；有 SIMD 比較不代表整段遍歷都被向量化，更不保證更快。

第一輪使用 `traversal_leaf_layout=sentinel`：葉子 control 為 -1，每層清除葉子的 feature index 並保留其 node。第二輪新增 `self_loop`：split 的 control 高 bit 是左 child 位移 1，低位保留 feature／NaN default 與右 child 相對位移；leaf control 是 0，左右位移都為 0。Leaf 自然停在原 node，省掉額外的 leaf mask／node 保留判斷。兩種布局都是 8 bytes，右 offset／feature 超過可表示範圍時拒絕編譯。

這是獨立的 lowering，開啟時不能同時使用 rank、hybrid subtree、leaf table、predicate hoist、tree block、preload 或 accumulation batching。既有的 select／branch-profile 選項不控制這個固定層數迴圈。Calibration 不決定其模型結構；模型全部節點都保留。

## 使用

```sh
.venv/bin/python -m xgb_latency.cli model.json build/interleaved \
  --backend clang --traversal-lanes 16 --traversal-mode scalar \
  --traversal-leaf-layout self_loop
```

`compile_model` 接受同名 Python keyword arguments，Clang／llvmlite 都支援；C ABI 仍為 `predict_row(const float*, float*)`。Metadata 記錄 mode、lanes、leaf layout、node／root table 大小、helper 數及 group 數。Native ns 不包括 Python ctypes 開銷。編譯器預設未變更。

## 實驗方法

沿用 [COLD_CACHE.md](COLD_CACHE.md) 的六個 same-thread profile。每一筆都執行前置工作，再推論；分別量測核心、p99 與完整 pipeline。軟體壓力不是已驗證的硬體 cache flush。全部在 Apple M3 macOS，沒有 core pinning／頻率隔離，也沒有 Linux／perf 結果。

每輪比較 scalar 寬度 1／2／4／8／16，vector 寬度 4／8／16，以及 scalar8／vector4 的 O2 對照；其他使用 O3。保留全部既有候選（含 TL2cgen 的 36 個 stock／f32 修改／PGO／ABI 對照）。完全相同 object 去重。

各 family 在 tuning 初選後再用三個獨立程序確認 shortlist；六個 profile 的 selection 全部寫定才開 evaluation，每個 profile 再測三個獨立程序，每引擎每程序 512×7 次核心呼叫及另跑的完整 pipeline。舊 cold 配置、wide／compact hybrid、單 lane／scalar4／vector4 皆為預先固定對照。第二輪另外保留第一輪 tuning 選定的各 profile winner。

兩個模型仍是 100 棵／最大深度 4、300 棵／最大深度 6，各 32 features。每輪新 tuning／evaluation 各 4,096 rows，standard-normal 加 3% NaN：第一輪 seeds 24821／24822（benchmark seed 24820），第二輪 seeds 26121／26122（benchmark seed 26120）。不同輪的絕對耗時不能作因果比較；百分比使用同一程序配對。

```sh
.venv/bin/python -m benchmarks.interleaved_candidates \
  --previous results/hybrid8-cold-macos-300x6/report.json \
  --model results/controlled-300x6/model.json \
  --output results/interleaved-build-repeat-300x6
```

第二輪以第一輪 report 為 `--previous`，加 `--leaf-layout self_loop`。候選 builder 依賴既有研究 artifact，不是可以獨立下載即跑的泛用 benchmark package。新的探索要用新的 tuning／evaluation，不能利用同一 evaluation 反覆改選。

原始結果保存在 `results/interleaved-cold-macos-SHAPE/report.json` 及 `results/interleaved-self-cold-macos-SHAPE/report.json`。

## 結果

已完成兩輪；第一輪共 67／70 個候選介面，第二輪 75／78 個，均包括保留的 36 個 TL2cgen 對照。完整數值、配置、各程序結果及 binary／object／IR 雜湊見 [可攜 JSON 報告](reports/interleaved-traversal-20260914.json)。

**大模型的 code-pressure 與 mixed-pressure 選到 `interleaved_self_scalar16_O3`，兩種情境的核心與 p99 都在三個 held-out 程序中勝過原 cold 配置。** 小模型沒有選交錯遍歷；大模型的 hot／features／data-only profile 也仍選既有 `directory_seed`。沒有更改編譯器預設為交錯遍歷。

第一輪 sentinel 表示：大模型 code-pressure 的 scalar16 為 1,614.3–1,673.6 ns，相較同程序原 cold 配置降低 2.6–7.4%；mixed-pressure 為 1,658.0–1,908.5 ns，與原 cold 配置互有勝負。SIMD 版本沒有勝出。第二輪才加入 self-loop 表示，並使用新資料評估。

第二輪大模型的結果如下。核心是 `model_mean_median_ns`，包含個別計時與 dispatch 成本；範圍為三次程序結果的最小–最大值，不是 request p50 或信賴區間。

| Profile | 新 self-loop scalar16（ns） | 原 cold `cold_depth3`（ns） | 原 warm `directory_seed`（ns） | TL stock＋PGO（ns） | TL f32 修改＋PGO（ns） |
|---|---:|---:|---:|---:|---:|
| code_pressure | **1,402.8–1,443.1** | 1,674.0–1,791.2 | 1,751.4–1,849.3 | 3,340.3–3,569.4 | 2,948.0–3,076.4 |
| mixed_pressure | **1,432.9–1,513.4** | 1,748.6–1,811.6 | 1,801.7–1,889.3 | 3,486.2–3,583.1 | 2,958.8–3,117.3 |

相同程序配對計算：

| 比較 | code-pressure 核心延遲降低 | mixed-pressure 核心延遲降低 |
|---|---:|---:|
| 新版 vs 原 cold 配置 | **16.2–19.4%** | **13.5–19.8%** |
| 新版 vs 原 warm 配置 | 19.9–22.0% | 16.0–23.7% |
| self-loop scalar16 vs sentinel scalar16 | 13.9–14.6% | 12.8–17.9% |

相對同輪 TL stock＋PGO 的核心速度比是 code **2.38–2.47×**、mixed **2.30–2.50×**；相對另外標示的 f32 修改＋PGO，是 **2.10–2.13×／1.97–2.17×**。各 family 的選擇仍只用 tuning，不在 evaluation 挑最快 baseline。

尾端與完整流程也分開保留：

| Profile | 新版 model p99（ns） | 原 cold model p99（ns） | 新版 pipeline（µs） | 原 cold pipeline（µs） |
|---|---:|---:|---:|---:|
| code_pressure | **1,584–1,666** | 2,416–2,458 | 12.574–12.626 | 12.766–12.893 |
| mixed_pressure | **1,750–1,875** | 2,584–2,750 | 30.513–30.581 | 30.570–30.752 |

相對原 cold 配置，code 的 p99 降低 **31.0–34.5%**，mixed 降低 **27.4–33.3%**。完整 pipeline 延遲則只降低 **1.50–2.07%／0.12–0.78%**。這些是配對程序的百分比，不能由表格不相配的區間端點相除。相對 TL stock＋PGO 的完整 pipeline 延遲降低約 **11.8–13.0%／5.0–5.6%**，仍遠小於核心速度倍數。

## 沒有勝出的情況

- 100×4 六個 workload 都沒有選交錯遍歷：hot／feature／data-only／mixed 選 `cold_compact4_cost8`，code 選既有 `hybrid8_h2_p1`。這不是新的小模型加速宣稱。
- 小模型 code-pressure 中，固定 scalar4 從 sentinel 的 498.7–500.5 ns 改善到 self-loop 的 418.1–419.2 ns，但仍慢於既有選定 hybrid 的 291.3–310.6 ns。
- 300×6 hot／features／data-only 仍選原 warm 配置。新 cold winner 沒有因此被用作全域預設。
- SIMD 確實產生四路比較指令，但 scalar load、lane insertion／extraction 等成本仍在。以大模型 code-pressure 的固定四樹對照為例：sentinel scalar4 為 2,715.7–2,756.0 ns、vector4 為 4,011.5–4,110.8 ns；self-loop scalar4 為 2,161.4–2,210.0 ns、vector4 為 3,443.5–3,624.5 ns。所有測試寬度的 vector 候選都沒有獲選，不能只看到 SIMD 指令就宣稱比較快。
- 16 棵樹是這次有限候選中的選擇，不代表已找到最佳寬度；尚未測更寬、不同分組順序或 target-specific gather。

資料顯示交錯與 self-loop 表示有用，但尚無 PMU 計數器能把收益精確分解為 memory-level parallelism、cache miss、分支或指令數的貢獻。保留原始 held-out selection，不因為某些固定對照在 evaluation 意外勝出而改寫選模結果。

## 大小、使用與驗證

300×6 原 cold 配置的 `__text` 為 247,560 bytes；sentinel scalar16 為 **3,456 bytes**，self-loop scalar16 為 **3,052 bytes**。交錯版 node＋root table 為 **246,480 bytes**。模型表示把指令空間換成資料讀取，並非把模型整體壓成 3 KB。

新大模型 code／mixed 的原生 artifact 已保存於：

- `results/interleaved-self-cold-macos-300x6/selected_models/code_pressure/`
- `results/interleaved-self-cold-macos-300x6/selected_models/mixed_pressure/`

各目錄包含 `model.dylib`、`model.o`、`model.h`、`metadata.json`。C++ 可直接連結 object；Python 可依 report 的 `selected_libraries` 載入：

```python
import json
from xgb_latency import Predictor

with open("results/interleaved-self-cold-macos-300x6/report.json") as f:
    report = json.load(f)
predictor = Predictor(report["selected_libraries"]["mixed_pressure"])
raw_margin = predictor.predict(rows)
```

重跑既有第二輪候選（output 使用新目錄）：

```sh
.venv/bin/python -m benchmarks.interference \
  --shape 300x6 --seed 26120 --shortlist-runs 3 \
  --candidate-manifest results/interleaved-self-builds-300x6/candidates.json \
  --model results/controlled-300x6/model.json \
  --tuning results/optimization-inputs/300x6-interleaved-self-tuning.npy \
  --evaluation results/optimization-inputs/300x6-interleaved-self-evaluation.npy \
  --output results/interleaved-self-repeat-300x6
```

完整套件 **479 passed in 163.05s**，日誌 `results/interleave-smoke/self-full-tests.log`。新增測試涵蓋兩個 backend、兩種 leaf layout、全部 lane／mode 組合、不同樹高／剩餘 lane／constant／empty、門檻及相鄰 float32、NaN、正負無限、正負零，並以獨立 Python 遍歷及 XGBoost 交叉比對。所有 benchmark 原型在實際產生的 features 上 `max_abs_error = 0`。

效能期間沒有同時執行編譯或測試。這仍是兩個合成模型與軟體干擾 workload 的研究結果，不能保證真實服務 p99 或其他 CPU。下一步可針對真實 feature producer、資料布局與 caller 融合繼續探索；跨樹交錯與 SIMD 比較已從未嘗試方向移入實驗完成項目。
