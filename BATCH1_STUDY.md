# Batch=1 方法總比較：VPred、節點布局、RapidScorer、區塊混用與自動調參（2026-09-30）

## 摘要

本輪加入五種新的 C lowering、區塊混用與自動調參器，並在五個模型上完成 tuning → 確認 → evaluation。所有新 lowering 的輸出都與既有 LLVM 原型**逐位元相同**：依原樹順序 float32 累加，NaN、±inf、門檻及其相鄰值都經過驗證。

| 模型 | 樹數×深度 | 自動調參選中 | evaluation 中位數 (ticks/row) | 前輪最佳 LLVM | 差異 |
|---|---|---|---:|---:|---:|
| 100×4 | 100×4 | `qs`（dense QuickScorer） | 821 | 842（`cost4_block32_rank4`） | −2.5%（持平） |
| 300×6 | 300×6 | `mix_k2`（兩區塊都是 VPred） | 5843 | 6623（`interleaved_self_scalar16`） | −11.8% |
| 1000×4 | 1000 淺樹 | `vpred:lanes=16,layout=level` | 12673 | 14902（`interleaved_self_scalar16`） | −15.0% |
| 300×8 | 最多 227 leaves | `vpred:lanes=16,layout=level` | 7539 | 9086（`interleaved_self_scalar16`） | −17.0% |
| 200lg128 | lossguide、128 leaves、高度 11–12 | `packed:lanes=16,layout=bfs` | 9234 | 8398（`interleaved_self_scalar16`） | **+10.0%（選錯）** |

- evaluation 值是三個獨立程序的中位數；TSC 是固定頻率的計數，這台機器約 2.1 ticks／ns。
- 300×6 的 evaluation 中，另一個單獨的 VPred 配置 `vpred:lanes=16,layout=tree,group_by_height=false`（5523 ticks）比選中的 `mix_k2` 更快。

結論：

- **VPred（隱式完全樹＋跨樹交錯）是本輪最重要的新 winner。** 在完整深度的模型（300×6、1000×4、300×8）上，它比既有 LLVM 交錯遍歷快 12–17%，比 TL2cgen 快 2.6–3.7×。
- 高度不齊的 lossguide 樹，補成完全樹的浪費太大，VPred 不適合。這個模型的 tuning 選出 `packed bfs`，但在 evaluation 輸給既有 LLVM 交錯遍歷 10%。依 protocol 不回寫選擇；這是 tuning 未能泛化的實例。
- QuickScorer 只在 100×4 與既有最佳配置持平；樹多或樹深時明顯落後。
- **RapidScorer（batch=1）在所有模型都最慢或接近最慢。** 等價節點合併可以把比較數減少 2–5.6 倍（見「新增的 lowering」），但剩下的迴圈次數仍隨輸入而變，epitome 範圍清除還是多筆記憶體寫入。
- **跨樹 SIMD（x86 AVX-512）** 在五個模型都比 scalar 交錯慢 2.6–3.8 倍，和 M3 的結論一致。
- 機率導向的布局（hot_dfs、frames、forest）沒有一致勝過 bfs／dfs，排名在模型之間互換，也在 VM 雜訊範圍內。
- 區塊混用從未明顯勝過最佳單一 lowering；mix 在 tuning 中最多和最佳單一 lowering 持平。

## 新增的 lowering

所有 lowering 都輸出 `void predict_row(const float *, float *)`，由 Clang 以 `-O3 -march=native -fno-fast-math -ffp-contract=off` 編譯。

| 模組 | 方法 | 要點 |
|---|---|---|
| `vpred.py` | VPred（Asadi 等，2014） | 每棵樹補成群組高度的完全二元樹，下一層為 `j = 2j + r`，其中 `r = (x >= t) \| (isnan(x) & default_right)`，恰為 XGBoost 路由的否定。提早出現的 leaf 會把值複製到其下所有 leaf 位置，所以補齊的 split 往哪走都不影響結果。支援 `lanes` 1–32 棵樹交錯、依高度分組（`group_by_height`），以及 `tree`（每棵樹 BFS 連續）或 `level`（同群組各樹的同層節點相鄰，Forest-Packing 式）兩種布局。 |
| `packed.py` | 顯式指標交錯遍歷＋布局 | 16-byte 節點，leaf 指向自己，固定步數、無分支。布局有五種：`bfs`、`dfs`、`hot_dfs`（常走的 child 先放）、`frames`（tree framing：每 64-byte frame 貪婪放入最常到達的相連節點）、`forest`（Forest Packing：整個 lane 群組的節點依到達次數排序）。 |
| `rapidscorer.py` | RapidScorer batch=1 子集（Ye 等，KDD 2018） | 每棵樹用 ⌈L/64⌉ 個 64-bit word，所以支援超過 64 個 leaf。左子樹的 leaf 範圍存成 epitome（首 word、尾 word、首 mask、尾 mask）。同一特徵、同一 float32 門檻的 split 跨樹合併成一個節點。每個特徵可選逐一掃描，或在預算內使用 dense 前綴表。去掉了跨樣本 SIMD。 |
| `direct.py` | 直接編譯的 if/else | 分支加上 `__builtin_expect_with_probability`，機率來自 calibration；高度不超過 `select_depth` 的子樹改用條件運算式。 |
| `blockmix.py` | 區塊混用 | 把樹切成連續區塊，每塊各自選策略，各在獨立的 translation unit，以 `acc = block_k(row, acc)` 串接累加值，所以結果仍然精確。策略用字串描述，例如 `vpred:lanes=8,layout=level`。 |
| `benchmarks/autotune.py` | 離線自動調參 | 見下節。 |

RapidScorer 的節點合併比例（合併後節點／原始 split）：

| 模型 | 比例 | 減少倍數 |
|---|---|---:|
| 100×4 | 669／1441 | 2.2× |
| 300×6 | 5282／15180 | 2.9× |
| 1000×4 | 4887／14555 | 3.0× |
| 300×8 | 7904／44687 | 5.7× |
| 200lg128 | 7247／25400 | 3.5× |

`quickscorer.py` 同樣改用共用的 `cgen.py`，並支援 chained 區塊。

## 自動調參流程

1. 建立全部 whole-model 候選，共 22–25 個（視 leaf 數而定）：
   - 5 個 LLVM 配置：`select1_profiled`、前輪 100×4 最佳 `cost4_block32_rank4`、scalar16 交錯遍歷，以及跨樹 SIMD vector8／vector16
   - QuickScorer（僅限 leaves ≤ 64）
   - VPred ×5
   - packed ×6
   - RapidScorer ×2
   - direct ×2

   每個候選都必須在 tuning rows 上與參考逐位元相同。
2. 把樹切成 K = 2、4 個連續區塊，每個區塊的 7 種策略各以獨立子模型計時；子模型以 XGBoost 的 `iteration_range` 驗證。每塊取最快策略，組成 `mix_kK`，並檢查逐位元相同。
3. 所有候選在 tuning rows 上計時；前 6 名再跑 3 次獨立程序確認，取中位數最小者。結果寫入 `selection.json` 並凍結。
4. 之後才讀取 evaluation rows。選中配置與 references（`select1_profiled`、`cost4_block32_rank4`、scalar16 交錯、QS）再跑 3 次獨立程序。

指標是 native harness 新增的 `block_median_ticks_per_row`：x86 用 `rdtsc`，ARM 用 `cntvct_el0`。TSC 固定頻率，**不是核心 cycles**；這台 VM 讀不到 PMU。

```sh
.venv/bin/python -m benchmarks.autotune --model model.json --calibration calibration.npy \
  --tuning tuning.npy --evaluation evaluation.npy --output results/autotune-mymodel
```

## Tuning 第一輪排名（ticks/row，各家族最佳）

| 家族 | 100×4 | 300×6 | 1000×4 | 300×8 | 200lg128 |
|---|---:|---:|---:|---:|---:|
| VPred | 1114 | **5881** | **11857** | **8486** | 10275 |
| packed 布局 | 1370 | 6076 | 14401 | 9561 | **8245** |
| LLVM scalar16 交錯 | 1435 | 6857 | 14737 | 11127 | 8941 |
| LLVM `cost4_block32_rank4` | 778 | 8717 | 14803 | 17620 | 10858 |
| QuickScorer | **706** | 13532 | 14066 | — | — |
| direct if/else | 925 | 10878 | 20995 | 21553 | 13438 |
| 區塊混用 | 825 | 5884 | 15023 | 10848 | 8872 |
| RapidScorer | 2038 | 24938 | 26578 | 110359 | 52095 |
| LLVM 跨樹 SIMD | 3684 | 23134 | 56406 | 37578 | 26083 |
| TL2cgen profiled＋LTO | 1934 | 14507 | 31863 | 28671 | 19497 |

每個模型的最快值以粗體標示。個別配置的完整數值見 [reports/batch1-study-x86-20260930.json](reports/batch1-study-x86-20260930.json)。

## 各項發現

**VPred 優於既有的交錯遍歷。** 兩者都是跨樹交錯、固定步數；VPred 用隱式索引，省去讀取 child offset，節點也只有 8 bytes，而 LLVM 版是 self-loop 顯式表示。
- 完整深度模型不需要補節點，VPred 明顯勝出：100×4 只補 59 個，300×6 補 3720 個（佔 split 的 25%），仍然勝出。
- lossguide 樹的高度 11–12、形狀很不平均，補成完全樹會大幅增加節點與 cache footprint，因此輸給顯式表示。
- 最佳 lanes 為 8–16；`level` 布局在 300×8／1000×4 較好，在 300×6 較差。

**依樹高分組。** 前三個 depthwise 模型所有樹同高，分組沒有作用。lossguide 高度只差 1，也看不出一致效果；VPred tree16 不分組反而較快，但在雜訊範圍內。要驗證這一點，需要高度差異更大的模型。

**機率導向布局。** 在 packed 家族內：
- 100×4 全部在 1370–1523 之間，模型可以整個放進快取，布局無差。
- 300×6 以 hot_dfs 最快（6076），但 bfs 也只有 6391。
- 300×8 以 dfs 與 forest 最快（9561／9575），hot_dfs 與 frames 反而最慢（14553／15718）。
- lossguide 以 bfs 與 frames 最快（8245／8248）。

沒有一種布局一致勝出。在這些模型大小下，L2 可以容納大部分表格，cache line 排列不是主要瓶頸。

**RapidScorer 與 QuickScorer。** batch=1 時兩者的成本都隨 split 數成長，又都有依輸入而變的掃描迴圈。等價節點合併確實把比較數大幅減少，但每次 AND 或清除仍是間接的讀改寫。dense 前綴表只在 100×4 裝得進合理預算，而且 dense QS 已經涵蓋這種情況。

**區塊混用。** 每塊策略依子模型單獨計時決定，但組合後不一定更快：
- QS 拆成區塊後，每個特徵的 rank 計算要在每塊重做，100×4 的 `mix_k4`（825）因此輸給整體 QS（706）。
- 大模型各區塊都選到 VPred，mix 只等於 VPred 本身加上呼叫開銷。

要讓混用有意義，需要依樹的性質把異質的樹分組；目前只支援連續區塊，改成任意分組後再用緩衝區依原順序加總。另一個方向是跨區塊共用 rank 計算。

**tuning 到 evaluation 的偏差。** 300×6 選中 `mix_k2`，evaluation 卻是單獨的 VPred 最快；lossguide 選中的 packed bfs 在 evaluation 輸給 LLVM 交錯遍歷。VM 雜訊約 ±10–30%，確認只跑 3 次程序還不夠。

## 正確性

- 新增 `tests/test_generators.py` 共 79 項測試：
  - 三個模型：regression d3、binary d6、lossguide（超過 64 leaves、高度不齊）
  - 所有 VPred／packed／RapidScorer／direct 參數組合、區塊混用與單一 spec
  - 子模型與 `iteration_range` 一致、epitome mask 單元測試、參數驗證
- 完整套件 **1049 passed**。

## 限制與下一步

- 只有一台 KVM VM，沒有 PMU，雜訊大；數值只適合同程序比較。模型都是合成資料，只有 lossguide 形狀不均。
- 下一步：
  - 加大確認與 evaluation 的程序數（≥5），並加入配對、交錯順序的確認，降低選錯率。
  - 讓區塊混用支援依樹高或 leaf 數的非連續分組，改用緩衝區依原順序加總，並跨區塊共用 rank。
  - 在 lossguide 等不平衡模型上，只對高度相近的子樹使用隱式索引，其餘用顯式表示。
  - 回到 M3 與實際部署 CPU，對 VPred 重跑自動調參。
