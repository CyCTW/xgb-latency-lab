# XGBoost Latency Lab

這是一個以 **單筆、單執行緒 CPU 推論** 為目標的 XGBoost → LLVM AOT 編譯原型。
研究目標是超越 TL2cgen 產生的原生程式碼；目前的實作與測量只是一個起點，沒有跨模型、跨 CPU 的效能保證。

初始測量見 [BENCHMARKS.md](BENCHMARKS.md)，後續自動選模與獨立評估見 [OPTIMIZATION.md](OPTIMIZATION.md)。目前完整套件 **1049 項測試通過**，包含正確性、門檻編碼、PGO、baseline 介面、同 thread feature／cache 干擾與選模流程。[Cold 干擾](COLD_CACHE.md)、[混合樹遍歷](HYBRID_TRAVERSAL.md)、[跨樹交錯探索](INTERLEAVED_TRAVERSAL.md) 與 [資料布局探索](LAYOUT_TRAVERSAL.md) 均完成獨立 tuning／evaluation。

Cold 情境的 [多目標選模實驗](MULTIOBJECTIVE_COLD.md) 已完成六情境、四個目標與七次獨立程序驗證：核心、pipeline 與 p99 可能需要不同配置，且部分 tuning 選擇在 holdout 退步。

[模型感知 prefetch](PREFETCH_TRAVERSAL.md) 已完成多目標效能與新資料配對實驗，未確認穩定整體收益；預設保持關閉。

[Leaf 分離表示](SEPARATED_LEAVES.md) 已通過正確性測試，模型表格 payload 約縮小 24%，但六情境／四目標量測沒有新 winner，同配置對照變慢，未改用新表示。

[QuickScorer 特徵導向 lowering](QUICKSCORER.md) 是首個 x86-64 Linux 資料點，也與原型逐位元相同。100×4 的 dense 表變體在 hot／feature／code 情境的核心延遲比前輪最強配置低 5.5–23%，但在 data_pressure 慢 20%，六情境中有 3 個被選中；300×6 不適用。另外兩項發現：pairwise 累加診斷顯示 float32 依序累加鏈不是主要瓶頸；路徑隱含冗餘 split 數為 0。

[Batch=1 方法總比較](BATCH1_STUDY.md) 加入 VPred、機率導向節點布局（tree framing／Forest Packing）、RapidScorer、直接 if/else、區塊混用與自動調參（`benchmarks.autotune`，TSC ticks 指標）。所有新 lowering 都與原型逐位元相同。在 x86 上，VPred 於 300×6／1000×4／300×8 比既有交錯遍歷快 12–17%；lossguide 不平衡樹則以既有交錯遍歷最佳。RapidScorer 與跨樹 SIMD 在所有模型都落後。

## 已實作

- 讀取 `Booster.save_model("model.json")`，建立經過驗證的樹結構。
- 直接產生 LLVM IR，使用 host CPU 的指令集與 O3 編譯，輸出原生物件檔、共享函式庫、header、IR、組合語言與 metadata。
- 三個實驗控制：小子樹 `select`、特徵預先載入、由獨立校準資料提供 branch weights。
- 可改由 Clang 最佳化並編譯相同 IR，讓產碼方式能在相同 compiler backend 下比較。
- 可依分支比例限制 `select`，並以固定樹數拆成較小的函式；跨函式傳遞累加值，保持 float32 加總順序。
- 可將校準資料顯示經常重複執行的比較，提前計算一次並供不同樹重用。
- 可利用已知的子樹結構，在編譯時建立比較位元到葉值的查表，作為 eager select 的實驗替代。
- 可將高頻特徵依完整模型門檻編碼為精確整數 rank，供不同樹共用，並保留 NaN 預設方向。
- 可用獨立 tuning 資料自動選模，再以尚未讀取的 evaluation 資料做最終評估。
- C/C++ 可直接呼叫：`void predict_row(const float *features, float *raw_margin)`。函式內沒有 heap 配置、Python 或執行緒池；預設展開樹，可選混合表示的共享 subtree traversal。
- 原生 benchmark，同時比較 TL2cgen 預設、branch annotation、threshold quantization 與兩者組合。

`select` 是 IR 層的選擇，LLVM 仍可能把它改成條件跳躍；不能宣稱產出的組合語言一定沒有分支。

## 使用

需要 Python 3.11+ 與 Clang；本機已建立 `.venv`。若從新環境安裝：

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

以下指令皆從專案根目錄執行：

```sh
# 正確性測試
.venv/bin/python -m pytest -q

# 編譯自己的模型
.venv/bin/python -m xgb_latency.cli model.json build/my-model \
  --select-depth 1 --calibration calibration.npy

# 合成資料基準
.venv/bin/python -m benchmarks.run --output results/my-run \
  --trees 100 --depth 4 --features 32 --rounds 11 --samples 8192 \
  --tl-lto --clang-control

# 真實模型；校準資料與評估資料應分開
.venv/bin/python -m benchmarks.run --model model.json \
  --data heldout.npy --calibration calibration.npy --output results/real-model
```

Python 便利介面：

```python
from xgb_latency import compile_model, Predictor

library = compile_model("model.json", "build/model", select_depth=1)
raw_margin = Predictor(library).predict(rows)  # shape: (n_rows,)
```

此 Python wrapper 逐筆呼叫 native function，用來整合與驗證。**不要用它的執行時間判斷核心函式延遲。**
效能測試使用 `benchmarks/native.cc`，定時區間完全在原生程式中。

## 自動選擇編譯配置

```sh
.venv/bin/python -m benchmarks.optimize \
  --model model.json --calibration calibration.npy \
  --tuning tuning.npy --evaluation evaluation.npy \
  --output results/tuned-model
```

三份資料必須分開，output 必須使用新目錄。程式會先編譯並驗證候選，在 tuning 資料上計時後寫入 `selection.json`，然後才讀取 evaluation 資料。最後不會根據 evaluation 表現改選。
預設以 `block_median_ns_per_row` 選模；也可用 `--metric p50_ns` 或 `--metric p99_ns`。
桌面計時雜訊可能影響選擇，尤其是很小的模型；此 runner 尚未加入多次獨立程序的選模確認；下方的 cold runner 可使用 `--shortlist-runs 3`，在 evaluation 前重測 tuning shortlist。

候選包含 12 組原型配置與兩個 TL2cgen family。TL2cgen family 都固定開啟 LTO，分別測試 annotation / quantization 的四個組合與可用的 prepared / dense 介面；沒有窮舉所有 compiler flags。
`--preset expanded` 改為測試較深的 `select`、比較重用與預載入候選，並以 `clang_d3_adaptive` 作固定參考配置；預設 `initial` 仍保留第一階段的候選與 `clang_d1` 參考配置，便於重現。
`tl_f32_modified` 是明確修改過的對照：把生成 C 中所有 `result[...] += 常數`（含最後的 base score）改成精確的 float32 hexadecimal literal；分裂門檻和量化邏輯不變。它不是未修改的 TL2cgen。

選好的原型函式庫、物件檔、header 與 metadata 會複製到 `selected_model/`，可直接使用：

```python
import json
from xgb_latency import Predictor

report = json.load(open("results/tuned-model/report.json"))
predictor = Predictor(report["selected_library"])
```

若要手動控制新選項：

```sh
.venv/bin/python -m xgb_latency.cli model.json build/blocked-model \
  --backend clang --calibration calibration.npy --select-depth 2 \
  --select-policy profile --tree-block-size 32
```

`tree_block_size=0` 保持單一函式；正整數表示每組最多幾棵樹，並保留函式邊界。
`select_policy=profile` 只在該節點有校準樣本且較少見分支比例至少 15% 時，把深度範圍內的子樹交給 eager select；這是實驗性規則，LLVM 仍可能再次改寫控制流程。

`--predicate-hoist-limit 8` 表示每個生成函式最多提前計算 8 個比較，必須提供 calibration。
只挑選至少出現在兩個節點、且校準資料顯示每列預期執行次數超過 1.5 的比較；特徵、float32 門檻及 NaN 預設方向都相同才可重用。metadata 記錄 IR 中實際提前計算的數量，LLVM 最後仍可能移動這些指令。預設 0 關閉，不會根據校準資料刪掉未觀察到的分支。

`--leaf-table-bits 3` 會把符合 select 條件、且含 2–3 個不同比較的子樹，轉為最多 8 格 float32 常數表。超過上限時，繼續檢查更小的子樹；未轉表的部分保留原本 select。每個比較位元包括 NaN 路由，無須假定輸入不含 missing。上限可設 0–8，預設 0 關閉；表格大小以比較數呈指數成長，因此較大上限未必更快。
`--preset tables` 會在 tuning 資料上比較查表候選、上一版配置與兩個 TL2cgen family。設計與實驗見 [LLVM_SPECIALIZATION.md](LLVM_SPECIALIZATION.md)。

`--select-policy cost --select-branch-penalty 8` 會依整個子樹的節點數與校準分支比例，自底向上比較 eager evaluation 與保留分支的估計成本；必須提供 calibration。權重不是 CPU 實測 cycles，需在 tuning 資料上選擇。`--preset cost` 提供相應候選，詳見 [OPTIMIZATION_EXPLORATION.md](OPTIMIZATION_EXPLORATION.md)。

`--rank-feature-limit 4` 會選取最多 4 個高頻特徵，以完整模型門檻預先計算每筆的精確整數 rank，再供各棵樹比較。需要 calibration，預設 0 關閉；保留嚴格 `<`、NaN 路由與輸入資料。分塊時只編碼一次，各組共用本次呼叫的固定大小堆疊緩衝。編碼成本包含在單筆計時內；`--preset ranks` 提供公平對照，詳見 [RANK_ENCODING.md](RANK_ENCODING.md)。

新一輪探索可用 `--preset explore`，比較 Eytzinger rank 搜尋、SIMD 門檻計數、緊湊葉值表、延後累加、O2／Os／Oz 與完整 instrumentation PGO。PGO 只使用 calibration；未量化 TL2cgen 同時訓練 dense／prepared 介面。需要與 Clang 相容的 `llvm-profdata`，macOS 可由 `xcrun` 尋找。

新增的手動控制：

- `--rank-strategy binary|eytzinger|simd|bucket|bucket_split`：精確 rank 的實作方式；須搭配非零 rank limit。
- `--rank-bucket-bits 12`：bucket 策略使用的 IEEE 位元前綴長度（8–16）；只定位搜尋起點，仍保留全部原始門檻與精確比較。較長前綴可減少搜尋，但會增大索引表。
- `--compact-leaf-depth 2`：在 eager 子樹內，以整數 select 選出葉索引，再讀取每片葉子一格的 float32 表；預設 0 關閉。
- `--accumulation-batch 4`：延後一組樹的加法，仍依原順序累加；LLVM 可能改回原有指令排程。
- `--optimization O3|O2|Os|Oz`：編譯模式；Os／Oz 目前僅提供 Clang backend。
- `--hybrid-depth 3 --hybrid-max-probability 0.1 --hybrid-layout compact`：將符合高度與到達比例的子樹交給共享資料遍歷；`wide` 每 node 16 bytes、`compact` 8 bytes，保持原 threshold／leaf 值。預設 depth 0 關閉，機率門檻小於 1 時需要 calibration。
- `--traversal-lanes 16 --traversal-mode scalar --traversal-leaf-layout self_loop`：同一筆資料跨樹交錯遍歷；也可測試 `vector` 的四路比較。這是獨立的完整樹資料表示，不能和 rank／hybrid／table／block 等 lowering 同時啟用；預設 lanes 0 關閉，詳見 [INTERLEAVED_TRAVERSAL.md](INTERLEAVED_TRAVERSAL.md)。
- `--traversal-leaf-layout separate`：使用 8-byte split 與 4-byte leaf 分表；需指定非零 lanes，這一版不與 prefetch 合用。
- `--traversal-leaf-state packed`：搭配 `separate`，將 internal／leaf index 打包成單一 64-bit traversal state；正確性已驗證，效能診斷尚未完成。
- `--traversal-prefetch none|roots|next|both`：實驗性的模型資料預取；root lookahead 用 `--traversal-prefetch-distance 1|2|4`，locality hint 用 `--traversal-prefetch-locality 0|1|2|3`。僅適用交錯遍歷，預設 none。
- `--traversal-data-layout aos|soa|soa8`、`--traversal-alignment 16|64|128|4096`：控制交錯遍歷的 node 資料表示與對齊；lanes 支援至 32。`--traversal-load-schedule staged|direct|lane` 比較 LLVM 載入產生順序，預設 staged 保留原本 AoS 機器碼；布局與排程已完成獨立量測，但 mixed 排名不穩定，詳見 [LAYOUT_TRAVERSAL.md](LAYOUT_TRAVERSAL.md) 與 [SCHEDULE_TRAVERSAL.md](SCHEDULE_TRAVERSAL.md)。
- `--machine-outliner`：Clang backend 嘗試共用重複的機器碼；預設關閉，本次 cold 選模沒有選中。
- `--lowering quickscorer`：改用 QuickScorer bitvector lowering，由 Clang 編譯 C，並忽略樹 lowering 選項。`--qs-stride 0|1|2^k` 分別為 classic／dense／checkpoint，省略時依 `--qs-budget-bytes`（預設 256 KiB）選擇；rank 由 `--qs-rank-linear-max` 與 `--qs-rank-search two_level|binary` 控制。詳見 [QUICKSCORER.md](QUICKSCORER.md)。
- `accumulation_order="pairwise_inexact"`（Python API）：**不精確**的診斷選項，只用於估計依序累加鏈成本；metadata 標記 `exact_accumulation_order: false`，不可部署。
- `bucket_split` 會分開儲存正負數的 bucket directory，以額外整數操作換取較小常數表；仍保留精確門檻比較。

`benchmarks.combine` 可從前一輪 **tuning 選定** 的配置探索組合，要求新的 tuning／evaluation 檔，並在定案後執行三個獨立 native process 確認結果。接著可用 `--mode bucket` 探索精確位元前綴分桶，及 `--mode prefix` 比較分桶粒度，兩者都需要新的 tuning／evaluation。詳見 [本輪探索報告](EXPLORATION_20260913.md)；Ubuntu 狀態與重現方式見 [UBUNTU_BENCHMARK.md](UBUNTU_BENCHMARK.md)。

## 同 thread feature 計算與 cold 干擾

`benchmarks.interference` 每筆先做前置工作，再進模型；包含 feature history、資料掃描、程式碼與混合干擾共六個 profile。分別記錄 model call、完整 pipeline 及 p99，所有 profile 都在開啟 evaluation 前選定配置。

```sh
.venv/bin/python -m benchmarks.interference \
  --shape 100x4 --seed 20520 --shortlist-runs 3 \
  --candidate-manifest results/cold-refine-builds-100x4/candidates.json \
  --model results/controlled-100x4/model.json \
  --tuning results/optimization-inputs/100x4-cold-refine-tuning.npy \
  --evaluation results/optimization-inputs/100x4-cold-refine-evaluation.npy \
  --output results/cold-repeat-100x4
```

以上使用本機既有 artifact；自己的模型要先建立相符的候選 manifest。預設選模 metric 是 `model_mean_median_ns`，可改為 `model_p99_ns` 或 `pipeline_block_median_ns`。各 profile 的 library／object／header／metadata 保存於 `selected_models/PROFILE/`，Python 可用 `Predictor(report["selected_libraries"]["mixed_pressure"])` 載入。

混合干擾下，本次 100×4／300×6 的核心延遲分別為 297–313 ns／1.65–1.76 µs，快於選定的 TL stock＋PGO；完整流程只降低約 1.3–1.5%／4.5–5.0%。這是軟體 cache／branch predictor 壓力，沒有驗證硬體 cache flush。方法、完整數值、限制與重現方式見 [COLD_CACHE.md](COLD_CACHE.md)。

混合遍歷的第二輪實驗中，100×4 的 code-pressure 核心比前輪 cold 配置快 5.1–11.9%，但 mixed-pressure 慢 4.9–9.3%，大模型也未獲得通用收益。預設配置保持不變，詳見 [HYBRID_TRAVERSAL.md](HYBRID_TRAVERSAL.md)。

最新跨樹實驗在 300×6 的 code／mixed 干擾選到 scalar16＋self-loop，混合干擾核心為 1.43–1.51 µs，比同程序原 cold 配置降低 13.5–19.8%，p99 降低 27.4–33.3%；完整 pipeline 只降低 0.12–0.78%。小模型與 hot／feature 情境仍保留既有配置。詳見 [跨樹交錯報告](INTERLEAVED_TRAVERSAL.md)。

## 目前支援範圍

| 項目 | 支援 |
|---|---|
| 模型格式 | `save_model` JSON；非 dump JSON、UBJSON 或 pickle |
| booster | `gbtree` |
| objective | `reg:squarederror`、`reg:absoluteerror`、`binary:logistic`、`reg:logistic` |
| 樹 | 數值分裂、scalar leaf、單輸出、深度不超過 64 |
| 輸入 | 連續、對齊的 dense float32；NaN 代表 missing |
| 輸出 | raw margin；分類 probability 的 sigmoid 由呼叫端另行處理 |
| 累加 | float32，base margin 起算、依序加入各棵樹，無 fast-math |

不支援的 objective、categorical split、DART、gblinear、多分類、多目標、向量 leaf、deleted-node 模型會明確拒絕。
不提供 per-row base margin、稀疏輸入、非 NaN missing sentinel、iteration range 或自動套用 `best_iteration`；會使用檔案內的所有樹。
需要 early-stopping 範圍時，請先在 XGBoost 中切出要部署的 booster 再存檔。

原生 API 由 caller 保證輸入長度為 `num_feature`、輸出至少有一個 float，記憶體指標有效且對齊。
函式庫使用編譯機的 CPU 特性，部署時應在目標機或相容環境重新編譯。這版沒有可攜 binary cache 或跨平台編譯功能。

## 比較方式

所有引擎都對同一模型、同一組 held-out rows、相同 raw margin 進行驗證。校準資料另存於 `calibration.npy`。
每一輪隨機調整引擎順序，並對每個引擎使用相同的隨機 row sequence；暖機後才開始定時。

- `llvm_*`：一次 indirect call，直接消費 dense float32。
- `tl2cgen_*_dense`：同樣的 dense 輸入，定時包含 `Entry` 轉換、輸出初始化、生成函式本身；量化也在定時範圍內。
- `tl2cgen_*_prepared`：提前準備 `Entry`，**直接呼叫生成的 `predict`**，沒有額外 adapter call；定時包含必要的輸出歸零。提供較強的核心函式對照。
- 量化版本會就地修改 `Entry`，因此每次從 dense 資料重建，不提供會錯誤重用 quantized buffer 的 prepared 測量。

`p50_ns`、`p95_ns`、`p99_ns` 是個別呼叫的觀測值，包含計時器與 indirect call 成本。
`block_median_ns_per_row` 是多輪迴圈平均每筆時間的中位數，包含索引與 checksum 的成本，**不是 request p50/p99**。
計時器解析度可能讓個別呼叫數字呈階梯狀，報告附有空的 clock pair p50，沒有直接扣除它。

TL2cgen 使用 Clang O3、native CPU、禁止 fast-math；LLVM backend 同樣 O3、native CPU、沒有浮點重排。
加上 `--tl-lto` 會另外測試四組 TL2cgen 的 link-time optimization 版本，需 linker 支援 Clang LTO。
加上 `--clang-control` 會把原型的 default / profiled IR 交給同一支 Clang 編譯；也可單獨使用編譯 CLI 的 `--backend clang`。
兩者 LLVM 版本可能不同，report 保留版本與旗標。TL2cgen 先加樹再加 base score，XGBoost 先加 base score。
此外，本機 TL2cgen 生成的 C leaf 常數沒有 `f` 後綴；在其組合語言中觀察到 float32/float64 轉換與 double 加法。原型使用 float32 加法，因此精度路徑也不同，檢查容許小幅浮點差異，效能提升不能全部歸因於分支配置。
本機 TL2cgen 1.0.0 內建 Treelite 4.1.2，但 Python frontend 是 4.7.2，載入時有版本警告；每個編譯結果都與 XGBoost 比對後才計時。目前的模型與測試已通過，仍不能據此保證其他模型的相容性。

目前已有 warm-cache 與同 thread 工作／cache 干擾的單執行緒桌面測量；沒有綁定 CPU、隔離頻率，也未驗證硬體 cache flush 或服務端端到端尾端延遲。
不可僅用一次合成模型結果，推論生產環境一定更快。

## 後續優先順序

1. 用真正的目標 CPU、模型與線上特徵分布重跑，先決定要最佳化核心運算還是呼叫端端到端延遲。
2. Cold runner 已加入 tuning shortlist 的三次獨立程序確認；繼續以真實前置工作、更多模型與資料分布驗證選模穩定性。
3. 在目標 Linux 機器分析 cycles、instructions、branch misses、L1 instruction-cache misses；根據瓶頸再加入 code-size 限制、改進已實作的混合樹表示／跨樹交錯遍歷，以及資料布局與呼叫端融合。
4. 完整 PGO 與兩種 TL2cgen 介面訓練對照已加入；接著需在真實模型與 Linux 上驗證結果能否重現。
5. 對目標模型需要的 categorical / multiclass / base-margin 等語意逐項增加支援與 differential tests。

不能預設「改用 LLVM 就更快」：TL2cgen 的 C 也能經過 Clang/LLVM。可驗證的機會在於樹表示、NaN 條件化簡、load placement 與分支/程式碼大小的取捨。

## 原始參考

- [lleaves 作者的編譯架構說明](https://siboehm.com/articles/21/lleaves)：LightGBM → LLVM IR → native code。
- [lleaves API](https://lleaves.readthedocs.io/en/latest/)：單筆推論與 cache blocking 的不同需求。
- [TL2cgen optimization guide](https://tl2cgen.readthedocs.io/en/latest/tutorials/optimize.html)：branch annotation 與 threshold quantization。
- [XGBoost Model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)：`save_model` 與 `dump_model` 的差異。
- [llvmlite optimization passes](https://llvmlite.pydata.org/en/stable/user-guide/binding/optimization-passes.html)：LLVM pass builder。
