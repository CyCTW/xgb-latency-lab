# 混合程式碼／資料遍歷實驗

本輪嘗試把模型的一部分子樹從 LLVM 展開指令改為共享迴圈遍歷。目的在於降低指令工作集，同時保留精確 raw margin；是否更快由每筆執行前置工作的獨立評估決定。

## 表示與語意

`hybrid_depth=0` 預設關閉。開啟時，從樹根往下選取高度 2 到指定上限的子樹；高度 1 的 stump 保留原生指令。`hybrid_max_probability` 可限制校準資料中的到達比例，分母是該樹根的樣本數，並非父節點的樣本數。小於 1 時必須提供 calibration。

選中的子樹第一輪完整保存在唯讀資料表，每節點 16 bytes：feature／NaN default flag、原 float32 threshold 或 leaf、左右 child index。所有子樹共用一個 `noinline` LLVM 遍歷函式，回傳 leaf value，由原生成函式依照原本樹順序進行 float32 加總。沒有 fast-math、浮點重排、模型近似、heap allocation 或刪掉低頻分支。

校準中完全沒被走到的子樹也保留所有節點。生成上層不會用 eager select 吃掉已指定資料遍歷的下層。資料遍歷直接讀取 float32 feature，不消費 rank；仍展開的區域可使用既有 rank 編碼。整棵樹都轉表時，最佳化器可以刪除不再使用的 rank 計算。

第一輪用同一種 16-byte node 表示 split 和 leaf。第二輪新增 `hybrid_layout="compact"`：DFS preorder 讓左 child 固定為本節點的下一筆；feature、NaN default flag 與右 child 相對位移合併為 32-bit control，搭配原 float32 threshold／leaf，合計 **8 bytes**。葉節點由 control 的最高 bit 識別，位移／feature 無法容納時明確拒絕，不截斷索引。

Compact 遍歷多了 mask／shift／add，但少讀兩個 child index，並把資料表縮小一半。這與近似量化無關，原 float32 數值未改動。`hybrid_layout="wide"` 保留第一輪表示，預設 hybrid 仍然關閉。尚未做 leaf 分表、跨樹 SIMD 或分層 prefetch，因此不能用單次成敗排除整個混合表示方向。

## 使用

```sh
.venv/bin/python -m xgb_latency.cli model.json build/hybrid \
  --backend clang --calibration calibration.npy \
  --hybrid-depth 3 --hybrid-max-probability 0.1 \
  --select-depth 3 --tree-block-size 16
```

Python `compile_model` 同樣接受 `hybrid_depth`／`hybrid_max_probability`／`hybrid_layout`；Clang 與 llvmlite backend 都支援。Metadata 記錄選中的 subtree 數、node 數與 table bytes。保持預設關閉，應在部署 workload 上選模。

研究候選由 `benchmarks.hybrid_candidates` 從前輪 **mixed tuning 選定** 的配置建立，包含不同高度／機率門檻，以及停用 rank 的對照。完全相同的 engine object 去重。既有 TL2cgen 36 個對照介面全部保留，前輪各 profile 的 engine winner 和 warm winner 也保留。

```sh
.venv/bin/python -m benchmarks.hybrid_candidates \
  --previous results/cold-refined-macos-300x6/report.json \
  --model results/controlled-300x6/model.json \
  --calibration results/controlled-300x6/calibration.npy \
  --output results/hybrid-build-repeat-300x6

.venv/bin/python -m benchmarks.interference \
  --shape 300x6 --seed 21920 --shortlist-runs 3 \
  --candidate-manifest results/hybrid-builds-300x6/candidates.json \
  --model results/controlled-300x6/model.json \
  --tuning results/optimization-inputs/300x6-hybrid-tuning.npy \
  --evaluation results/optimization-inputs/300x6-hybrid-evaluation.npy \
  --output results/hybrid-repeat-300x6
```

第二段使用本輪既有候選；若使用第一段新建的候選，請把 manifest 路徑改為 `hybrid-build-repeat-300x6/candidates.json`。Output 必須是新目錄。新的探索需要新的 tuning／evaluation，不能根據同一份 evaluation 持續改選。

本輪 tuning／evaluation 分別使用 seeds 21921／21922，各 4,096 rows、32 features、standard-normal 加 3% NaN。先對各 family shortlist 在 tuning 上做三個獨立程序確認，六個 workload 決定全部凍結後才讀取 evaluation。上一輪 cold winner 透過 manifest 的 `references` 保證出現在 shortlist 與最終評估中，即使它不再是本輪 winner。

工作負載、TL prepared／dense 成本歸屬、計時器限制與 cold 定義沿用 [COLD_CACHE.md](COLD_CACHE.md)。原始結果保存於 `results/hybrid-cold-macos-{100x4,300x6}/report.json`；第二輪對照保存於 `results/hybrid8-cold-macos-{100x4,300x6}/report.json`。

第二輪使用相同前輪 cold seed，加 `--layout compact --include-manifest results/hybrid-builds-SHAPE/candidates.json`，將 wide 與 compact 候選放在同一輪選模。新的 tuning／evaluation seeds 是 23121／23122，benchmark seed 是 23120。預先固定的 wide／compact height-2 對照即使沒選中也保留在最終 evaluation；避免只呈現獲勝的 hybrid。

## 結果（2026-09-13 完成）

已完成兩輪編譯／選模／獨立 evaluation。第一輪候選數為 49／50，第二輪為 59／62，均包括 TL2cgen 的 36 個對照介面。完整數值、配置、binary／object／IR 雜湊與每次程序結果見 [可攜 JSON 報告](reports/hybrid-traversal-20260913.json)。

**小模型的 code-pressure 情境有改善；mixed-pressure 與大模型沒有找到可取代既有配置的 hybrid。** 因此保留此功能供 workload 選模，沒有修改預設引擎。

第一輪 16-byte 表示：100×4 的 `hybrid_h2_p1` 在 code-pressure tuning 中獲選，但 held-out 三次 model call 為 357.3–384.7 ns，前輪 cold 配置為 353.5–367.0 ns；三次配對均未勝過前輪 cold 配置。大模型沒有 hybrid 成為 winner。

第二輪 8-byte 表示的固定 height-2 對照如下，單位為 ns，範圍是三個獨立程序的 `model_mean_median_ns` 最小–最大值，**不是 p50 或信賴區間**：

| 模型 | Profile | 8-byte hybrid | 16-byte hybrid | 前輪 cold 配置 |
|---|---|---:|---:|---:|
| 100×4 | hot_control | 289.4–301.3 | 302.8–321.9 | 188.2–200.4 |
| 100×4 | features_64k | 303.8–314.5 | 326.8–334.5 | 202.3–209.3 |
| 100×4 | data_pressure_2m | 386.8–419.8 | 373.5–413.6 | 240.2–246.8 |
| 100×4 | code_pressure | **311.9–328.2** | 328.9–349.3 | 333.1–372.6 |
| 100×4 | mixed_pressure | 368.4–430.2 | 387.3–400.4 | **350.4–393.8** |
| 300×6 | hot_control | 2,100.2–2,511.0 | 2,304.9–2,324.5 | 1,151.9–1,194.0 |
| 300×6 | features_64k | 2,135.2–2,297.9 | 2,305.9–2,447.6 | 1,168.4–1,203.9 |
| 300×6 | data_pressure_2m | 2,283.9–2,394.6 | 2,547.0–2,669.2 | 1,276.8–1,311.9 |
| 300×6 | code_pressure | 2,215.9–2,385.2 | 2,389.2–2,651.3 | **1,963.4–2,068.1** |
| 300×6 | mixed_pressure | 2,574.0–2,824.7 | 2,583.4–2,887.6 | **2,264.0–2,427.3** |

固定 height-2 hybrid 並非每個 profile 最好的 hybrid；較低 probability 門檻的配置在一些 tuning profile 更好，但仍未取代原生配置。完整 tuning 表保存在 JSON。第二輪 300×6 六個 profile 都選既有 `directory_seed`；上表刻意固定前輪 `cold_depth3` 作同程序對照，未在 evaluation 挑有利的比較對象。

100×4 code-pressure 的同程序配對結果：

- 相對前輪 cold winner，8-byte hybrid 核心延遲降低 **5.1–11.9%**。
- 相對相同 height-2 的 16-byte 表示，核心延遲降低 **4.9–6.0%**。
- 完整 pipeline 為 **12.393–12.891 µs**，相對前輪 cold winner 降低 **0.24–2.14%**；收益很小且需在部署環境確認。
- Model p99 為 **459–625 ns**，前輪 cold winner 為 **500–583 ns**；**沒有一致的 p99 改善**。
- 同輪選出的 TL stock＋PGO 核心為 804.6–882.5 ns，f32 修改＋PGO 為 705.2–762.3 ns。這兩種 family 仍依 [既有比較契約](COLD_CACHE.md) 分開標示。

在 100×4 mixed-pressure，8-byte hybrid 雖然也被 tuning 選中，held-out 卻比前輪 cold winner **慢 4.9–9.3%**，完整 pipeline 慢約 0.12–0.20%。保留凍結的 selection 與結果，不用 evaluation 改寫本輪選模；部署建議仍是保留既有 mixed 配置。這也顯示三次 tuning 確認不保證排名能轉移到 evaluation。

大模型 8-byte height-2 表示在 code-pressure 比 16-byte 表示快 **5.7–15.4%**，但仍比前輪 cold winner 慢 **12.9–15.3%**；資料壓縮改善了 hybrid 自己的成本，沒有跨過現有引擎的門檻。

## 大小與解讀

| 模型／配置 | `__text` bytes | Hybrid node table bytes | dylib bytes |
|---|---:|---:|---:|
| 100×4 前輪 cold | 33,704 | 0 | 73,624 |
| 100×4 wide height-2 | 11,284 | 42,496 | 83,896 |
| 100×4 compact height-2 | 11,280 | 21,248 | 67,384 |
| 300×6 前輪 cold | 247,560 | 0 | 475,192 |
| 300×6 wide height-2 | 122,068 | 406,864 | 581,208 |
| 300×6 compact height-2 | 122,064 | 203,432 | 383,064 |

表中的 hybrid node table 不包括既有 rank／leaf 常數表。全部轉成共享遍歷的 300×6 候選，`__text` 能縮至 5,688 bytes，但 traversal 成本太高，沒有獲選。程式碼最小不等於延遲最低。

觀察符合「instruction 工作集與依賴資料讀取的取捨」這個假設，但尚未用硬體計數器確認原因。這一輪桌面執行時間有明顯變動，mixed 的整段 pipeline 在不同程序可從約 33 µs 到 39 µs；因此**不能把不同輪的絕對 ns 差異當成演算法改善或退步**。此處百分比只使用同輪、同程序的配對結果，仍不能保證生產環境效益。

## 驗證與使用決策

完整套件 **400 passed in 141.94s**，日誌 `results/hybrid-smoke/final-tests.log`。新增測試包含兩種 backend／兩種 node layout、threshold 與相鄰 float32、NaN／正負無限／正負零、rank／preload／hoisting／block 組合、未觀察路徑、不可編碼的 offset、以及固定參考配置在 selection／evaluation 的保留。原型在所有 benchmark 實際生成 features 上皆 `max_abs_error = 0`。

已凍結的研究 artifact 位於 `results/hybrid8-cold-macos-SHAPE/selected_models/PROFILE/`，包含 library／object／header／metadata。100×4 的 code-pressure 選定檔可由 C++ 直接連結，或用 Python `Predictor` 載入；native ns 未包含 ctypes 成本。`mixed_pressure` 的選定 artifact 仍保留原始 tuning 決定，**不代表建議部署它取代前輪 mixed winner**。

本輪的實際結論是新增一個對小模型 code-pressure 有用的選項，並排除目前這種共享逐節點遍歷作為通用加速方案。下一個不同方向可探索跨樹交錯執行／SIMD，降低相依資料讀取的等待；它尚未實作，也沒有預先保證更快。Ubuntu／perf 仍無可用結果，本輪全部在 Apple M3 macOS 執行。

後續更新（2026-09-14）：跨樹交錯與 SIMD 比較已完成兩輪，scalar16＋self-loop 在大模型 code／mixed 干擾有改善；詳見 [INTERLEAVED_TRAVERSAL.md](INTERLEAVED_TRAVERSAL.md)。上文的 400 項為當時階段紀錄，最新完整套件為 479 項。
