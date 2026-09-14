# 同 thread 工作後的單筆推論（2026-09-13）

兩輪實驗已完成。在 Apple M3 上，模型每次被呼叫前都先執行指定工作，再量測推論。第二輪在 tuning 選定的配置，在混合資料／程式碼干擾下，100×4 模型為 **296.9–313.3 ns**、300×6 模型為 **1,653.6–1,756.3 ns**。相同程序中，TL2cgen stock＋PGO 分別為 **729.5–761.6 ns**、**3,352.9–3,502.1 ns**。

這是軟體工作負載造成的 cache／branch predictor 干擾，沒有透過硬體計數器驗證所有模型 cache line 都被清除。它用來逼近「同 thread 算完 feature 才進 inference」，不是 process cold start、page fault 或已驗證的硬體 cache flush。

完整數值、三次程序各自的結果、模型配置、雜湊及候選大小保存在 [可攜 JSON 報告](reports/cold-cache-20260913.json)。原始報告為 `results/cold-macos-{100x4,300x6}/report.json` 與 `results/cold-refined-macos-{100x4,300x6}/report.json`。

## 工作負載與計時

`benchmarks/interference.cc` 在同一 thread 依序執行：資料／程式碼干擾 → feature 計算 → inference。**每一筆**都執行前置工作，而非只在整個 benchmark 開始前清一次 cache。

| Profile | Feature history 容量 | 每筆額外資料掃描 | 每筆額外程式碼工作 |
|---|---:|---:|---|
| hot_control | 0 | 0 | 無 |
| features_64k | 64 KiB | 0 | 無 |
| features_4m | 4 MiB | 0 | 無 |
| data_pressure_2m | 64 KiB | 2 MiB | 無 |
| code_pressure | 64 KiB | 0 | 256 個函式 |
| mixed_pressure | 64 KiB | 2 MiB | 256 個函式 |

Feature producer 對每個 feature 讀取 16 個散布在 history 中的 float32，依序加總後對原始 feature 做小幅變換，NaN 保留。32 features 每筆是 512 次 scalar history read；**4 MiB 是可存取工作集容量，不是每筆讀完 4 MiB**。XGBoost oracle 使用這些實際產生的 features，並在定時區間外驗證。

資料干擾讀取 2 MiB buffer，每 64 bytes 一次 volatile read，以排列後的順序拜訪全部位置。程式碼干擾是 256 個不內聯、內容各異的整數運算函式；獨立 object 的 `__TEXT` 為 246,840 bytes。它也會擾動 branch predictor，不能將差異全部歸因於 instruction cache。

報告區分兩個指標：

- `model_mean_median_ns`：每輪 512 次個別 model call 的平均值，再取 7 輪中位數；包含 clock／dispatch 成本。以下核心表格使用此數值，**不是 request p50，也不能直接和先前約 115／750 ns 的 hot block 指標相比**。
- `pipeline_block_median_ns`：另跑完整的工作→推論迴圈，僅在整段外計時，避免每筆讀取 clock；包含前置工作、轉換、推論、索引與 checksum。

另保留個別 model／pipeline 的 p50、p95（model）、p99。計時器有離散解析度，未扣掉空 clock pair；桌面 OS 沒有 core pinning 或頻率隔離。這些 p99 不是服務端 SLA。

## 選模與公平性

使用兩個既有合成 regression 模型（100 棵樹／最大深度 4、300 棵樹／最大深度 6），各 32 features。新 tuning／evaluation 各 4,096 rows，standard-normal 加 3% NaN；第二輪 seeds 為 20521／20522。模型校準資料沿用既有獨立 calibration，沒有用 evaluation 訓練 PGO 或選分支。

第一輪探索所有既有 TL2cgen family／介面與多種原型表示。縮小 bucket directory 的候選在一次 tuning 中獲勝，但 holdout 平均延遲沒有穩定超越固定 warm winner。因此第二輪增加：

1. 從前輪 **mixed tuning 選定** 配置產生 15 個編譯候選：compact leaf depth、select depth、樹分塊、cost policy、Os／Oz、LLVM machine outliner。
2. 對完全相同的 engine object 去重，保留原 warm winner 作固定對照，避免相同機器碼因多個名稱取得額外選模機會。最終 61／66 個介面候選，包含每個模型 36 個 TL2cgen 對照介面。
3. 初輪 tuning 各 family 取前兩名，加上 warm winner，在 tuning 上以三個獨立程序確認，依這三次的中位數選擇。
4. 六種 workload 的 `selection.json` **全部寫定後才開啟 evaluation**。每種 workload 再用三個獨立 native process 測試，未依 evaluation 改選。

每輪隨機排列引擎，使用相同 row sequence；每個程序每個引擎收集 3,584 次 model call，另有完整 pipeline block 計時。下表是三個程序結果的最小–最大值，並非信賴區間。

TL2cgen 包含 annotation、quantization、PGO、dense／prepared 介面，Clang O3、native CPU、LTO，沒有 fast-math。prepared 的 Entry packing 由 feature producer 執行，完整 pipeline 包含此成本；dense adapter 的轉換在 model call 內。量化版本每次重建 Entry，沒有錯誤重用被修改的 buffer。

`tl_f32_modified` 是另外標示的 C 常數修改對照，非 stock TL2cgen：使用 float32 additive literals，仍保留其原本的累加順序。原型維持 XGBoost float32 樹累加順序及 NaN 路由；本次所有原型在實際生成的 features 上 `max_abs_error = 0`。TL2cgen 的浮點差異以既有 tolerance 驗證。不能把全部效能差異歸功於 LLVM IR 或分支策略。

## 第二輪結果

單位皆為 ns，數值為 model call 指標；各列的配置都在 evaluation 前選定。

| 模型 | Profile | 原型 | TL stock＋PGO | TL f32 修改＋PGO |
|---|---|---:|---:|---:|
| 100×4 | hot_control | 160.8–161.6 | 419.2–429.8 | 354.3–369.7 |
| 100×4 | features_64k | 159.8–167.6 | 407.8–443.4 | 347.4–370.1 |
| 100×4 | features_4m | 158.9–160.3 | 374.8–412.6 | 314.3–354.8 |
| 100×4 | data_pressure_2m | 170.5–172.3 | 415.6–434.9 | 343.1–365.7 |
| 100×4 | code_pressure | 301.2–315.5 | 732.0–749.2 | 620.6–641.6 |
| 100×4 | mixed_pressure | 296.9–313.3 | 729.5–761.6 | 612.1–641.6 |
| 300×6 | hot_control | 886.5–906.7 | 2,536.9–2,636.1 | 2,191.5–2,246.3 |
| 300×6 | features_64k | 894.0–912.9 | 2,564.3–2,609.4 | 2,152.5–2,197.2 |
| 300×6 | features_4m | 905.6–911.1 | 2,558.0–2,576.3 | 2,180.3–2,204.4 |
| 300×6 | data_pressure_2m | 916.3–939.8 | 2,561.2–2,725.2 | 2,152.7–2,253.3 |
| 300×6 | code_pressure | 1,667.2–1,801.3 | 3,385.5–3,555.1 | 2,911.8–3,052.2 |
| 300×6 | mixed_pressure | 1,653.6–1,756.3 | 3,352.9–3,502.1 | 2,980.3–3,074.0 |

原型也快於本次選定的兩個非 PGO TL family；完整數值在 JSON。PGO 並非每種 workload 都勝過非 PGO，表格不代表已窮舉 TL2cgen 的所有編譯技巧。

混合干擾下，相對 stock＋PGO 的同程序核心速度比是 **2.43–2.46×／1.98–2.05×**；相對 f32 修改＋PGO 是 **2.05–2.09×／1.70–1.80×**。完整流程沒有同樣倍數的加速：

| 模型 | 原型 pipeline（µs） | TL stock＋PGO pipeline（µs） | 原型 model p99（ns） | TL stock＋PGO model p99（ns） |
|---|---:|---:|---:|---:|
| 100×4 | 29.113–29.220 | 29.559–29.618 | 459–500 | 1,125–1,125 |
| 300×6 | 30.548–30.845 | 32.097–32.319 | 2,209–2,625 | 4,583–5,000 |

相同程序配對計算，整段 pipeline 相對 stock＋PGO 的延遲降低約 **1.3–1.5%／4.5–5.0%**。前置工作主導時，核心加快不會等比例轉換為端到端收益。

## 哪些配置有幫助

100×4 的 code／mixed 選到 `cold_compact4_block8`：8 個 rank features、12-bit bucket、cost select depth 6／penalty 4、compact leaf depth 4、每函式 8 棵樹。300×6 選到 `cold_depth3`：32 個 rank features、正負分開的 14-bit bucket directory、profile select depth 3、compact leaf depth 2、每函式 16 棵樹。

| 模型 | 相對 warm winner：code model 延遲降低 | 相對 warm winner：mixed model 延遲降低 |
|---|---:|---:|
| 100×4 | 7.6–9.3% | 5.6–9.3% |
| 300×6 | 1.8–3.4% | 2.3–5.5% |

這是三個程序分別配對的結果，不是用兩個區間的最佳端點相除。小模型 mixed p99 從 warm winner 的 583–666 ns 降到 459–500 ns；大模型 code p99 並未改善，不能宣稱所有尾端指標都更好。

選定配置的大小取捨：

| 模型／配置 | `__text` bytes | `__const` bytes | dylib bytes |
|---|---:|---:|---:|
| 100×4 warm winner | 33,100 | 431,272 | 486,728 |
| 100×4 cold mixed | 33,704 | 20,816 | 73,624 |
| 300×6 warm winner | 229,248 | 1,126,676 | 1,515,848 |
| 300×6 cold mixed | 247,560 | 84,456 | 475,192 |

兩個 cold winner 都大幅降低常數資料量，機器指令本身反而略增。降低查表工作集、改變 eager evaluation 範圍及分塊是共同變化，這一輪沒有把每項的獨立效益分離，不能把全部收益歸因於單一選項。

更深的 compact leaf 表、Os／Oz 和 machine outliner 有縮小部分候選的 code，但沒有贏得最終 code／mixed 選模。小模型的 compact depth 4 和部分 depth 2 候選也產生相同 object，已去重；名稱不同不等於真的換了指令。Machine outliner 提供的是 compiler 的重複機器碼共用，**不是**模型 profile 感知的 cold subtree interpreter。

300×6 在 hot／feature／data-only profile 仍選原 warm winner；沒有修改編譯器預設值以強推 cold 配置。小模型的其他 profile 選 `cold_compact4_cost8`，但相對 warm winner 的收益不一致。大模型 mixed 的完整 pipeline 相對 warm winner 變化約 -0.7% 到 +0.1%（正值表示延遲降低），沒有穩定端到端改善。

## 使用與重跑

各 profile 選定的 library、object、header、metadata 已保存於 `results/cold-refined-macos-SHAPE/selected_models/PROFILE/`。這些 artifact 使用 Apple M3 CPU 特性，部署到其他 CPU 要重新編譯。

```python
import json
from xgb_latency import Predictor

with open("results/cold-refined-macos-100x4/report.json") as f:
    report = json.load(f)
predictor = Predictor(report["selected_libraries"]["mixed_pressure"])
raw_margin = predictor.predict(rows)
```

C++ 可連結同目錄的 `model.o`，include `model.h`，直接呼叫 `predict_row(features, &margin)`；上面的 ns 數字是 native harness 的量測，沒有包含 Python ctypes 呼叫成本。

重跑本次已編譯的候選（output 必須是新目錄）：

```sh
.venv/bin/python -m benchmarks.interference \
  --shape 100x4 --seed 20520 --shortlist-runs 3 \
  --candidate-manifest results/cold-refine-builds-100x4/candidates.json \
  --model results/controlled-100x4/model.json \
  --tuning results/optimization-inputs/100x4-cold-refine-tuning.npy \
  --evaluation results/optimization-inputs/100x4-cold-refine-evaluation.npy \
  --output results/cold-repeat-100x4
```

將 `100x4` 換成 `300x6` 可重跑大模型。新的探索應使用新的 tuning／evaluation；同份 evaluation 重跑僅作重現確認，不能繼續用它改選配置。

要編譯本輪候選，可執行：

```sh
.venv/bin/python -m benchmarks.cold_candidates \
  --previous results/cold-macos-100x4/report.json \
  --model results/controlled-100x4/model.json \
  --calibration results/controlled-100x4/calibration.npy \
  --output results/cold-build-repeat-100x4
```

這兩個研究 runner 預設沿用既有候選 artifact；不是拿任意模型就能直接跑的獨立 benchmark package。自己的模型需先編譯相符候選，提供 manifest `{ "entries": [...], "warm_winner": {...} }`，每個 entry 包含 `name`、`family`、`library`、`symbol`、`prepared`。原型候選會檢查模型 SHA256，全部候選都會驗證實際生成 features 的預測。

`--metric model_p99_ns` 可依尾端選模，`--metric pipeline_block_median_ns` 可依整段流程選模；本次仍以 `model_mean_median_ns` 為目標。若真實需求是端到端延遲，應改用對應 metric 並重跑獨立 tuning／evaluation。

新增 `--machine-outliner`（Clang backend），預設關閉；沒有因為編譯器支援此選項就宣稱會加速。

## 驗證與下一步

完整套件 **354 passed in 107.92s**，日誌 `results/interference-smoke/full-final-tests.log`。涵蓋 feature producer 的獨立 float32 比對、六種干擾模式、TL dense／prepared／quantized ABI、evaluation 前凍結選擇、shortlist 確認、部署 artifact 與 outliner 正確性。效能量測期間沒有同時執行編譯或測試。

下一個有價值的工作是接入真實 feature producer／模型／輸入分布，並在 Linux 目標機取得 cycles、branch misses、instruction-cache misses，區分瓶頸。Ubuntu 容器服務可回應，但啟動／exec 逾時，**本輪沒有 Linux 結果**，詳見 [Ubuntu 記錄](UBUNTU_BENCHMARK.md)。模型感知的混合 code/data 遍歷、跨樹 SIMD、caller LTO 融合、prefetch／post-link layout 仍未完成，不能宣稱單筆優化已探索完或任何模型一定快於 TL2cgen。

後續更新：已完成 16-byte／8-byte 混合樹遍歷的兩輪實驗，小模型 code-pressure 有改善，但不是通用替代；詳見 [HYBRID_TRAVERSAL.md](HYBRID_TRAVERSAL.md)。最新完整測試數見 README，本報告的 354 項是 cold 初期階段紀錄。
