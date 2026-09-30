# Treebeard 式 tree tiling（2026-09-30）

參考 Treebeard（Prasad 等，PACT 2022）的 tree tiling，實作在 VPred 的隱式完全樹上：`xgb_latency/tiled.py`。只取 tiling 一項；Treebeard 的 loop interchange 與多執行緒只對 batch 推論有用，所以不做。

## 結論

在 x86 上，**tiling 是目前最快的 batch=1 遍歷方式**。它在五個模型的 evaluation 中都勝過 VPred，自動調參在其中四個選中 tiling。100×4 仍由 QuickScorer 和 LLVM `cost4_block32_rank4` 領先。

下表是 5 次獨立 evaluation 程序的中位數（ticks／row），括號內是 tiling 贏的程序數：

| 模型 | 最佳 tiling | ticks | 對 VPred | 對前輪最佳 LLVM | 對 TL2cgen | p99 ns（tiling／VPred／LLVM） |
|---|---|---:|---:|---:|---:|---:|
| 100×4 | k=2 insert, 8 lanes | 1039 | −5.9%（5/5） | +43.5%（0/5） | 1.8× | 834／1159／719 |
| 300×6 | k=3 gather, 8 lanes | 4609 | −14.1%（5/5） | −33.9%（5/5） | 3.4× | 3412／6097／8948 |
| 1000×4 | k=2 insert, 8 lanes | 11764 | −12.1%（5/5） | −21.8%（5/5） | 2.7× | 12963／16850／20501 |
| 300×8 | k=3 gather, 8 lanes | 6595 | −11.6%（4/5） | −27.3%（5/5） | 4.2× | 7272／8452／13378 |
| 200lg128 | k=3 gather, 8 lanes | 7600 | −29.0%（5/5） | −10.5%（3/5） | 2.1× | 8522／14438／11403 |

- 前輪最佳 LLVM：100×4 為 `cost4_block32_rank4`，其餘為 `interleaved_self_scalar16`。
- 在 300×6 與 1000×4，tiling 的 p99 也比 VPred 低 23–44%。
- lossguide 的 tiling 在 5 次程序中只贏 LLVM 交錯遍歷 3 次，這個勝出不穩定。

## 設計

- 每棵樹先補成群組高度的完全樹（同 VPred），每 k 層合成一個 tile。k=2 時 tile 有 3 個節點，放在 32-byte record；k=3 時有 7 個節點，放在一條 64-byte cache line。樹高不是 k 的倍數時，最後一段用較少層，例如深度 8、k=3 時為 [3, 3, 2]。
- 每個節點算出往右的 bit：`(x >= t) | (unordered & default_right)`，恰為 XGBoost 路由的否定，NaN、±0、±inf 都一致。
- tile 內各節點的 bit 組成 mask，查一張**與樹無關**的固定表（k=2 為 8 格，k=3 為 128 格）得到出口 child，下一個 tile 的索引為 `j = j·2^k + LUT[m]`。每棵樹的 dependent 步數從 h 降到 ⌈h/k⌉，代價是會多評估不在路徑上的節點。
- 評估方式有三種：
  - `scalar`：逐節點比較
  - `gather`：AVX2 `vgatherdps`
  - `insert`：逐個 scalar 載入後組成向量

  後兩者都只做一次向量比較加 `movemask`。
- 同 VPred，可以跨樹交錯多條 lanes、依樹高分組，leaf 值依原樹順序累加。
- 可以作為區塊混用的策略，例如 `tiled:lanes=8,tile_levels=3,mode=gather`；也已加入自動調參的候選。

## 觀察

- **向量評估是關鍵。** `scalar` 模式在所有模型都比 VPred 慢；例如 300×6 為 11422，VPred 為 5092。每個 tile 多做好幾次比較和 OR，收益被吃光。
- **深樹用 k=3 gather，淺樹用 k=2 insert。** 深度 4 的樹在 k=3 時切成 [3, 1]，第二步幾乎沒有收益，所以 k=2 較好。深樹用 k=3 能把一條 cache line 的 7 個節點一次比完，在 SPR 上硬體 gather 比逐個載入快。
- **lossguide（高度 11–12）**：補成完全樹後 tile 表達 7.5 MB，VPred 在這個模型輸給 LLVM 交錯遍歷 26%。tiling 把 dependent 步數從 12 降到 4，反而贏過兩者。所以這個模型的主要瓶頸是 dependent load 的延遲，不是表格大小。
- lanes 16 通常不如 8。每條 lane 的 tile 狀態較大，16 條會造成暫存器壓力。

## 正確性與流程

- 60 組 tile 層數 × 評估方式 × lanes 設定，在五個模型上都與既有原型逐位元相同。
- 新增 41 項測試，涵蓋三個模型（含 lossguide 超過 64 leaves）、查表與分段排程的單元測試、區塊混用與參數驗證。完整套件 **1090 passed**。
- 自動調參第二輪使用新的 tuning／evaluation 檔：前 6 名做 5 次程序確認，再做 5 次 evaluation 程序。VPred、tiling、packed 三個家族各自在 tuning 最快的配置，也強制帶進 evaluation 同場比較。數值見 [reports/tiling-x86-20260930.json](reports/tiling-x86-20260930.json)。

## 限制與下一步

- 只測了一台 x86 VM（AVX2／AVX-512），沒有 PMU，也沒測 ARM；M3 上需要 NEON 版本（沒有 gather，只能用 insert 方式）。
- 尚未嘗試：
  - Treebeard 的**機率導向 tiling**：常走的路徑放同一個 tile，tile 形狀不必是完全子樹，需要改用逐 tile 的查表。
  - AVX-512 的 16 節點 tile（4 層只有 15 個節點）。
  - 高度不齊的樹不補滿，只對上層做 tiling。
- 100×4 仍由 QuickScorer 和 `cost4_block32_rank4` 領先。
