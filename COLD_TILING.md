# Tiling 延伸與干擾情境驗證（2026-10-01）

本輪完成 [TILING.md](TILING.md) 列出的三項：

1. 只對上層做 tiling 的混合表示（`top_levels`）
2. Treebeard 的機率導向 tiling（`probtiled.py`）
3. 把 VPred／tiling 系列放進六種同 thread 干擾情境重選模

兩項新 lowering 都與原型逐位元相同。完整套件 **1143 passed**。

## 結論

- **在干擾情境下 tiling 依然成立。** 300×6、1000×4、300×8 共 18 個情境，核心延遲都比前輪最佳 LLVM（`interleaved_self_scalar16`）低 11.7–46.6%。每個情境的 3 個 evaluation 程序全部勝出。資料與混合干擾下差距最大：300×6 為 −36%／−38%，1000×4 為 −41%／−47%。
- **100×4 依情境分化。**
  - 資料干擾與混合干擾選中 tiling，核心 756／883 ns，比 QuickScorer（860／1226）與 `cost4_block32_rank4`（990／1280）低 12–31%。這補上了 QS dense 表在資料干擾下的弱點。
  - hot／feature／code 三種情境仍選 QS，tiling 慢 4–23%。
- **lossguide（高度 11–12、不平衡）在干擾下反轉。** 六個情境都選既有 LLVM 顯式交錯遍歷；tiling 系列慢 3.5–43%。warm 時 tiling 快約 11%，但 7.5 MB 的完整 tile 表格在快取壓力下吃虧。
- **上層 tiling 混合（`top_levels`）**
  - warm 時從未勝過完整 tiling；lossguide 7549 對 7501 ticks，持平。
  - cold 的 lossguide tuning 中，T=6 在 5/6 情境勝過完整 tiling（例如 code_pressure 5155 對 8095 ns），也是唯一在 code_pressure 與 LLVM 持平的 tiling 變體（5155 對 5182）。
  - 它在其他情境仍輸 LLVM，因此沒有進入 evaluation。
  - 方向正確：縮小表格有幫助，但 bottom 段的顯式遍歷沒有比 LLVM 版快。
- **機率導向 tiling（probtiled）**
  - warm 只在 lossguide 勝出：7171 對 7501 ticks（−4.4%，5/5 程序）。
  - cold 下 lossguide 仍輸 LLVM 3.5–24%，其他模型慢 60–100%。
  - 每步多兩次相依載入（shape → LUT → child 指標），抵銷了熱路徑較短的好處。
  - 提前結束（early exit）是必要的：關掉時 300×6 從 7809 變 10617 ticks。

## 新 lowering

**`tiled.py` 的 `top_levels=T`**：只把上面 T 層補成完全樹並做 tiling。tile 的出口透過 `FRONT` 表對到 16-byte 顯式節點；leaf 指向自己；剩餘高度以固定步數、無分支的顯式指標遍歷。每棵樹的補齊上限是 2^T 格。lossguide 的 tile 表格從 7.5 MB 降到 T=6 時的 0.1 MB，另加 0.7 MB 顯式節點。

**`probtiled.py`**（Treebeard 的機率導向 tiling，batch=1 版）：
- 從 tile 根貪婪加入 calibration 到達次數最高的內部節點，最多 7 個。
- 每個 tile 有「節點數＋1」個出口，由左到右編號。同形狀的 tile 共用一張 128 格查表，每個 tile 另存每個出口的 child 指標。
- leaf 是 shape 0 的「leaf tile」，所有出口指回自己。
- 每個 lane 群組最多跑到最長的 tile 路徑；`early_exit` 在所有 lane 都到達 leaf 時提前結束，每步一個容易預測的分支。
- 各模型的 shape 數：100×4 為 54，300×8 為 480，lossguide 為 552。

## Warm 自動調參第三輪（5 次確認＋5 次 evaluation，新資料）

| 模型 | 選中 | 最佳完整 tiling | 最佳上層混合 | 最佳 probtiled | VPred | 前輪最佳 LLVM |
|---|---|---:|---:|---:|---:|---:|
| 100×4 | `qs`（708） | 1038 | 1230 | 1613 | 1115 | 719 |
| 300×6 | tiled k3 gather | **4582** | 4725 | 7477 | 5186 | 6468 |
| 1000×4 | tiled k2 insert | 11950 | 13725 | 19818 | **11782** | 14504 |
| 300×8 | tiled k3 gather | **6315** | 6748 | 12524 | 6931 | 8667 |
| 200lg128 | probtiled gather | 7501 | 7549 | **7171** | 9953 | 8412 |

evaluation 中位數，單位 ticks／row；1000×4 的 tiling 與 VPred 在雜訊範圍內持平。數值見 [reports/tiling-round3-x86-20261001.json](reports/tiling-round3-x86-20261001.json)。

## 六情境干擾（cold）

各模型的候選 manifest 取自第三輪 tuning pass：
- 自動調參的 winner
- 前輪 LLVM 兩個配置
- VPred／完整 tiling／上層混合／probtiled／packed 各家族在 tuning 的最佳者
- 100×4 另加 QS
- TL2cgen 作為獨立 family

新的 tuning／evaluation 檔；tuning shortlist 跑 3 次確認；全部凍結後才開 evaluation，每情境 3 個獨立程序。指標為 `model_mean_median_ns`。

下表是核心延遲（ns，3 程序中位數），與前輪最佳 LLVM 對照：

| 模型 | hot | feat 64K | feat 4M | data 2M | code | mixed |
|---|---:|---:|---:|---:|---:|---:|
| 300×6 tiling | 2272（−29%） | 2293（−27%） | 2395（−29%） | 3726（−36%） | 2314（−32%） | 3899（−38%） |
| 1000×4 tiling/VPred | 5665（−21%） | 5932（−17%） | 5687（−23%） | 6816（−41%） | 5660（−27%） | 7041（−47%） |
| 300×8 tiling | 3466（−17%） | 3448（−18%） | 3898（−12%） | 7447（−16%） | 4148（−25%） | 7317（−21%） |
| 100×4 最佳 tiling | 524（+23%） | 604（+16%） | 834（+17%） | 756（**−24%**） | 606（+4%） | 883（**−31%**） |
| lossguide 最佳 tiling | 4167（+4%） | 4937（+14%） | 5514（+24%） | 9740（+35%） | 6655（+43%） | 10437（+40%） |

- 前輪最佳 LLVM：100×4 為 `cost4_block32_rank4`，其餘為 `interleaved_self_scalar16`。
- 1000×4 在 features_64k 由 VPred 取得最低值（5932），但 tuning 選中的是 tiling（5946），兩者持平。
- 300×8 hot_control 的 tuning 選了 VPred，holdout 中 tiling 更快；依 protocol 不回寫。

完整 pipeline 在 hot／feature 情境跟著核心改善，例如 300×6 hot 為 2132 對 3210 ns。資料／程式碼／混合干擾下 pipeline 由前置工作主導，差異在 2–10% 內。lossguide 的 probtiled 在 hot 情境核心較慢，pipeline 卻較快（3398 對 4029），核心與 pipeline 的排名並不一致。

數值見 [reports/cold-tiling-x86-20261001.json](reports/cold-tiling-x86-20261001.json)。

## 建議的部署規則（這台 x86 上的經驗法則）

- 平衡樹、深度 ≥ 6，或樹數上千：**tiling**（深樹 k=3 gather，淺樹 k=2 insert），在 warm 與 cold 下都穩定領先。
- 小模型（約 100 棵、深度 4）：hot 情境用 **QS** 或 `cost4_block32_rank4`；有資料或混合干擾時用 **tiling**。
- 高度大、不平衡的樹：warm 用 **probtiled／tiling**；有快取壓力時用 **LLVM 顯式交錯遍歷**。實際選擇仍應交給自動調參，並在部署機上重跑。

## 限制與後續

- 仍然只有一台 KVM VM，軟體模擬的干擾，沒有 PMU，也沒有 ARM 數據。
- 不平衡樹還有空間：上層混合的 bottom 段可改用 LLVM 的交錯顯式遍歷，或 bottom 段也做 probtiled，需要能跨 lowering 組合的 bottom 段 codegen。
- probtiled 可以試著把 shape LUT 和 child 指標合併成一次載入：直接存「mask → child」表，每 tile 128 格。代價是表格更大。
