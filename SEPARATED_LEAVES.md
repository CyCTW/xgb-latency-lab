# Split／leaf 分離表示（2026-09-15）

本輪以獨立 float32 leaf 陣列與 8-byte split records 取代每個 node 一律 8 bytes 的表示。沒有補齊／展開缺少的子樹，也沒有壓縮浮點精度。兩個模型的原始表格 payload 已縮小約 23.7%／24.6%；完整六情境／四目標驗證已完成，兩模型都沒有選中新表示；同配置對照反而變慢。

## 索引推導

每個 split 都有左右兩個 child，因此含 k 個 internal nodes 的子樹一定有 k+1 個 leaves。若按 preorder 只排列 internal records，當前 split 的右 child 距離為 d=1+左子樹 internal 數；這也恰好是左子樹的 leaf 數。

每個 lane 保持 internal index 與 leaf prefix index：

- 走左邊：internal index 加 1，leaf prefix 不變；d=1 表示左 child 就是 leaf。
- 走右邊：internal index 與 leaf prefix 都加 d；右 child 是否為 leaf 存在 control 的一個 bit。
- 遇到 leaf：internal index 改為共享的 dummy record 0，leaf prefix 保留實際 leaf 值的位置。
- Dummy 的 control／threshold 都是 0，左右移動距離都是 0，因此已完成 lane 在剩餘迴圈中保持不動。

完成 group 的固定高度遍歷後，從 leaf 陣列讀出每個 lane 的值，仍按原始 tree 順序逐次 float32 加總。NaN default routing、嚴格 `<`、threshold 與 leaf 值保持精確。每個 lane 多維護一個索引，可能增加暫存器壓力與指令成本。

## API 與儲存

設定 `traversal_leaf_layout='separate'` 或 `--traversal-leaf-layout separate`，並指定非零 traversal lanes。支援 AoS／SoA／SoA8、scalar／vector 與 staged／direct／lane 排程。此輪 separate 不能和 prefetch 同時啟用；不支援的組合明確拒絕。預設仍是既有 sentinel，沒有自動改用新表示。

設 I 為 internal 數、L 為 leaf 數、T 為 tree 數。未計 linker 對齊間隙的 AoS／SoA 表格 payload：

- 原 self-loop：8(I+L)+4T bytes。
- 分離表示：8(I+1)+4L+8T bytes，含一個 dummy record 及每棵樹的 root／leaf-start indices。
- SoA8 另將 internal records 補到 8 的倍數；leaf 陣列不作此補齊。

| 模型 | Internal／leaf 數 | Self-loop payload | Separate payload | 縮小 |
|---|---:|---:|---:|---:|
| 100×4 | 1441／1541 | 24,256 B | 18,500 B | 23.73% |
| 300×6 | 15180／15480 | 246,480 B | 185,768 B | 24.63% |

這是原始表格大小，不是整個共享函式庫大小；constants 可能被 LLVM 消除，code／alignment 也影響實際 artifact。極小或全常數森林不保證變小。

## 驗證與效能方法

新增 128 個 separate 測試通過（57.40s，`results/separate-smoke/targeted-tests.log`），涵蓋 clang／llvmlite、binary／regression、三種布局、不同 widths／scalar／vector／排程、不同樹高、尾組、NaN／±inf／±0／threshold 邊界、空森林及不同值的常數樹。完整套件 **806 passed in 307.64s**（`results/separate-smoke/full-tests.log`）。

新增 `matched_controls` manifest 對照表。新表示若進入 shortlist 或最終 evaluation，其預先指定的同配置 self-loop 對照也必須一起執行。支援遞迴依賴去重，所有對照在 holdout 前凍結；benchmark 相關 18 項測試通過（3.60s），不將 targeted 數量相加稱作完整套件。

候選包括：三種布局的 scalar lanes 4／8／12／16／24／32（lane 排程）、AoS scalar 8／12／16 staged，以及 AoS／SoA vector 8／16 lane，共 25 個新表示；每個都編譯配對 self-loop。所有舊候選與 baseline 保留，object 去重後小模型 176、大模型 179 個候選，固定 references 各 17 個，另有 25 組 matched controls。

沿用四目標／六 profile 的共同量測：screen 256×4、5 次 shortlist 確認各 512×8、7 次 evaluation 各 1024×10。每對輪次使用相同 rows、反轉 engine order；所有 profile／objective 凍結後才讀 holdout。新 tuning／evaluation 各 4096×32 float32、3% NaN，seeds 32521／32522，runner seed 32520。

Builds：`results/separate-builds-{100x4,300x6}/builds.json`；儲存大小核對：`results/separate-smoke/layout-validation.json`。效能輸出為 `results/separate-cold-macos-{100x4,300x6}/`；可攜報告：[reports/separate-cold-20260915.json](reports/separate-cold-20260915.json)。已核對凍結 selection、binary／source hashes、正反配對與所有原型零誤差。

靜態機器碼診斷：`results/separate-smoke/register-diagnostics.json`。大模型 AoS lane 排程，8／12／16 lanes 的 compiler 標示 reload 指令分別由 self-loop 的 7／19／58 增為 separate 的 18／34／99。這是整份 assembly 的靜態指令數，包含 helper／prologue，不能當作動態存取次數或 PMU 計數；但可作下一步測試 packed 64-bit traversal state 的依據。

重現（輸出目錄須尚未存在）：

```sh
.venv/bin/python -m benchmarks.separate_candidates \
  --previous results/prefetch-cold-macos-100x4/report.json \
  --model results/controlled-100x4/model.json \
  --output results/separate-builds-100x4
.venv/bin/python -m benchmarks.multiobjective \
  --model results/controlled-100x4/model.json \
  --candidate-manifest results/separate-builds-100x4/candidates.json \
  --tuning results/optimization-inputs/100x4-separate-tuning.npy \
  --evaluation results/optimization-inputs/100x4-separate-evaluation.npy \
  --output results/separate-cold-macos-100x4 --seed 32520
```

大模型改為 `300x6`。Linux 需重新建置全部 native artifacts。

## 配對效能結果

下表為七個獨立程序內，相對相同 lanes／布局／排程 self-loop 對照的核心平均延遲增加率中位數；正數代表變慢。三個固定 references 在看 holdout 前已指定。

| 表示（lane 排程） | 100×4 code | 100×4 mixed | 300×6 code | 300×6 mixed |
|---|---:|---:|---:|---:|
| AoS scalar8 | +25.53% | +23.88% | +22.79% | +14.83% |
| AoS scalar12 | +43.79% | +31.30% | +36.35% | +30.88% |
| SoA scalar8 | +42.20% | +39.58% | +33.93% | +30.04% |

大模型 mixed 的完整 pipeline 平均增加率分別為 0.91%／1.56%／1.81%。新表示沒有在任一模型的六情境×四目標 tuning 勝出，不取代既有配置。完整報告保留每程序數值、p99、pipeline 與配對對照，未用 holdout 改寫 winner。

這輪支持「較小表格不足以補償額外狀態／解碼」的推論，尚未以動態硬體計數器證明原因。後續已加入 `traversal_leaf_state='packed'`，把 internal／leaf 兩個索引打包到單一 64-bit 狀態；完整套件 **935 passed in 368.63s**。機器碼靜態 spill／reload 數明顯下降，正式效能診斷尚未完成，並保留本輪 split 狀態與 self-loop 作固定對照。
