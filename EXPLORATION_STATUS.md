# 優化探索狀態（2026-09-15 核對）

目前沒有探索完所有單筆推論方法。已完成 warm-cache、同 thread 干擾、混合樹遍歷，以及跨樹交錯／SIMD、資料布局／寬度實驗。已完成多目標 cold 選模，證實核心、pipeline 與 p99 可能需要不同配置；prefetch 已完成主實驗與新資料對照，沒有穩定整體收益。不能把某個 workload 的排名當成所有真實呼叫流程的排名。

## 已完成實作與效能實驗

- branch weights、height／profile／cost select 策略。
- 固定樹分塊、特徵 preload、重複比較 hoisting。
- truth table 與緊湊 leaf-index table。
- 精確 threshold rank：binary、Eytzinger、SIMD threshold counting。
- IEEE 位元前綴分桶，8／10／12／14／16-bit 前綴與不同特徵數。
- 正負值分開的 bucket directory：已通過新增測試與 warm-cache 比較，沒有取代原最快配置。
- 延後累加，保持原樹順序與 float32 加總；部分候選被 Clang 編回相同指令。
- O2／O3／Os／Oz；Os／Oz 限 Clang。
- 完整 instrumentation PGO：原型及 TL2cgen，TL 包含 dense／prepared 介面訓練對照。
- tuning／evaluation 分離、固定 binary 的獨立程序確認。
- Clang LLVM machine outliner，正確性與 code-size 實驗完成；沒有成為本輪 cold winner。
- 六種同 thread 工作 profile、每筆執行前置工作、model／pipeline 分別計時、獨立 oracle。
- Cold tuning shortlist 的三次獨立程序確認；六個 profile 全部凍結後才開 evaluation，再各跑三個獨立程序。
- 混合 code/data 樹表示：依子樹高度／校準到達比例，轉為共享 LLVM loop；完整保留未觀察分支。
- Hybrid node 的 16-byte／8-byte 表示；後者使用 preorder 隱式左 child 與精確右 child offset，沒有門檻或葉值近似。
- Manifest 固定 references，保留前輪 cold winner 與預先指定的 hybrid 對照至 evaluation。
- 同一筆資料的跨樹 scalar 交錯：1／2／4／8／12／16／24／32 lanes，保持原始 float32 加總順序。
- 跨樹 SIMD 比較：效能測過 4／8／16／32 lanes（API 另支援 12／24），確認生成 ARM 四路比較，但未成為本輪 winner。
- 固定高度遍歷的 sentinel／self-loop 葉節點表示；後者以零 child offset 保持 leaf，減少每層控制指令。

Warm 結果來源：`results/{exploration,combined,bucket,prefix,directory}-macos-{100x4,300x6}/report.json`。Cold 結果來源：`results/{cold,cold-refined}-macos-{100x4,300x6}/report.json`。Hybrid 結果來源：`results/{hybrid,hybrid8}-cold-macos-{100x4,300x6}/report.json`。交錯結果來源：`results/{interleaved,interleaved-self}-cold-macos-{100x4,300x6}/report.json`。布局階段完整套件 **543 passed in 191.15s**，不是將不同 targeted runs 的數量相加。

## Cold 結果與環境限制

- 完整報告：[COLD_CACHE.md](COLD_CACHE.md)；可攜數值：[reports/cold-cache-20260913.json](reports/cold-cache-20260913.json)。
- 混合干擾下 100×4 為 296.9–313.3 ns、300×6 為 1,653.6–1,756.3 ns；相對固定 warm winner，核心延遲分別降低 5.6–9.3%／2.3–5.5%。這些 per-call clock 指標不能直接和早期 hot block ns 比較。
- 相對本次 TL stock＋PGO，mixed 核心速度比 2.43–2.46×／1.98–2.05×，完整 pipeline 延遲降低約 1.3–1.5%／4.5–5.0%。
- 小模型 cold 配置偏向較少 rank features、較小 bucket 表與 8-tree blocks；大模型偏向 split directory 與較淺 select。更多 compact leaf、Os／Oz、outliner 的較小 code 未贏得最終 cold 選模。
- 300×6 的 hot／feature／data-only profile 仍選固定 warm winner；mixed pipeline 相對 warm winner 沒有穩定改善。預設編譯配置沒有強制改成 cold winner。
- 每個 profile 已保存可直接使用的 library／object／header／metadata，路徑 `results/cold-refined-macos-SHAPE/selected_models/PROFILE/`。
- 使用合成 feature producer；軟體干擾不是已驗證的硬體 cache flush。生成 code object 的 __TEXT 為 246,840 bytes，也擾動 branch predictor。
- Ubuntu ARM64：2026-09-14 machine exec 已恢復；實際 syscall 探測 task-clock 成功、cycles／instructions 回 ENOENT，沒有註冊 CPU PMU。沒有 Linux inference 效能結果，詳見 [UBUNTU_BENCHMARK.md](UBUNTU_BENCHMARK.md)。

## 後續混合遍歷結果

- 報告：[HYBRID_TRAVERSAL.md](HYBRID_TRAVERSAL.md)；數值：[reports/hybrid-traversal-20260913.json](reports/hybrid-traversal-20260913.json)。
- 小模型 8-byte height-2 hybrid 在 code-pressure 為 311.9–328.2 ns，比同程序前輪 cold winner 降低 5.1–11.9%；p99 沒有一致改善。
- 同一配置 mixed-pressure 雖被 tuning 選中，evaluation 卻比前輪 cold winner 慢 4.9–9.3%；保留原始凍結結果，不改寫 selection。
- 大模型 8-byte 表示比 16-byte 表示改善，但沒有比既有生成指令更快；本輪六個 profile 均選既有 `directory_seed`。
- 大幅縮小 code 不等於加速，預設 hybrid 仍關閉。小模型 code-pressure artifact 可作特定部署候選，不作通用替代。
- 不同輪的桌面絕對耗時有明顯變動；只用同輪／同程序配對數字評估配置差異。

## 最新跨樹交錯結果

- 報告：[INTERLEAVED_TRAVERSAL.md](INTERLEAVED_TRAVERSAL.md)；數值：[reports/interleaved-traversal-20260914.json](reports/interleaved-traversal-20260914.json)。
- 大模型 code-pressure 選 `interleaved_self_scalar16_O3`，為 1,402.8–1,443.1 ns；相較同程序原 cold 配置，核心延遲降低 16.2–19.4%、p99 降低 31.0–34.5%。
- 大模型 mixed-pressure 同樣選此配置，為 1,432.9–1,513.4 ns；核心延遲降低 13.5–19.8%、p99 降低 27.4–33.3%。
- 完整 pipeline 相對原 cold 配置只降低約 1.50–2.07%／0.12–0.78%，不能等同核心收益。
- 相對 TL stock＋PGO，mixed 核心速度比 2.30–2.50×，完整 pipeline 延遲降低約 5.0–5.6%。
- 小模型未選交錯遍歷，大模型 hot／feature／data-only 仍選 `directory_seed`；編譯器預設保持不變。
- 新大模型 code／mixed artifact 位於 `results/interleaved-self-cold-macos-300x6/selected_models/PROFILE/`，C++／Python 皆可載入。這些是 Apple M3 的合成 workload 結果，沒有跨 CPU／模型保證。

## 最新資料布局與排程進度

- 報告：[LAYOUT_TRAVERSAL.md](LAYOUT_TRAVERSAL.md)；數值：[reports/layout-traversal-20260914.json](reports/layout-traversal-20260914.json)。
- 23 個新布局／寬度／alignment 候選，六情境獨立 evaluation 完成。大模型沒有取代原 scalar16；小模型 mixed 選 AoS12，但核心平均改善不一致、pipeline 略退步。
- 發現 AoS 重構的等價 GEP 產生方式改變機器碼，已恢復原版 staging，兩個模型的組合語言與 object 都與前輪原版完全相同。
- 新 `traversal_load_schedule` 可比較 staged／direct／lane；排程輪完整套件 **592 passed in 215.16s**（`results/schedule-smoke/full-tests.log`）；16 個新候選、兩模型六情境的獨立量測已完成。詳見 [SCHEDULE_TRAVERSAL.md](SCHEDULE_TRAVERSAL.md)。
- 小模型 code 的 SoA12 lane 為 282.8–284.3 ns，比既有 hybrid 核心低 0.76–6.74%，但 pipeline 未改善。大模型 code 的 AoS12 lane 為 1384.6–1398.9 ns，比原 scalar16 核心低 2.77–3.33%、pipeline 低 0.25–0.43%，p99 未一致改善。
- 兩模型 mixed 排名仍不穩定；大模型選出的 direct AoS12 在 holdout 比原 scalar16 慢 0.83–6.69%。保留原始 selection，下一步加強 paired 確認，並以 pipeline／p99 重新選模。

## 多目標 cold 選模（已完成一輪）

- 新增 [MULTIOBJECTIVE_COLD.md](MULTIOBJECTIVE_COLD.md) 的共同量測／四目標選模，六個 profile 全部凍結後才讀取 holdout。
- 同 row sequence 的正反候選順序配對，5 次 shortlist 確認、7 次 evaluation；core 平均、core p99、pipeline 平均／p99 分開選模。
- benchmark 相關 17 項測試通過；引擎程式未改，不將 targeted 測試與前輪完整套件相加。
- 兩模型六情境的四目標選模／holdout 已完成；資料：[reports/multiobjective-cold-20260914.json](reports/multiobjective-cold-20260914.json)。
- 小模型 mixed 的 pipeline p99 配置比核心平均配置低中位數 3.10%，7/7 程序改善，但核心平均更慢。大模型 mixed 的 pipeline 平均低中位數 0.52%，6/7 程序改善。
- 大模型 data-pressure 的兩個 p99 配置在 holdout 全部輸給核心平均配置，沒有以 evaluation 回寫選模；重複量測不等於泛化保證。

## Prefetch 探索（已完成一輪）

- [PREFETCH_TRAVERSAL.md](PREFETCH_TRAVERSAL.md)：新增 roots／next／both 預取、root lookahead 距離與 locality hint。
- 32 個配置已編譯，4 個 disabled controls 與既有 object 完全相同，其餘 28 個都有 ARM prefetch 指令。
- 82 個新增 targeted 測試通過，完整套件 **677 passed in 269.92s**；兩模型六情境四目標的主實驗與所有被選中預取配置的新資料配對已完成。
- 主數據：[reports/prefetch-cold-20260914.json](reports/prefetch-cold-20260914.json)；補測：[reports/prefetch-attribution-20260914.json](reports/prefetch-attribution-20260914.json)。
- 大模型 mixed 的 AoS16 roots 在同配置對照中核心平均 7/7 次更慢、pipeline 平均中位增加 0.12%；小模型 mixed 的 pipeline p99 改善未一致重現。預設 none，未宣稱所有預取方法皆無效。

## Leaf 分離（已完成一輪）

- 新增 [SEPARATED_LEAVES.md](SEPARATED_LEAVES.md) 的 split／leaf 分表；精確值與原加總順序不變，不補齊稀疏子樹。
- 兩模型原始表格 payload 降低 23.73%／24.63%，但每 lane 增加 leaf prefix 索引，是否更快待測。
- 128 個新表示測試及 18 個 benchmark 測試通過，完整套件 **806 passed in 307.64s**。兩模型六情境／四目標已完成，沒有新 winner；固定 references 在 code／mixed 的核心平均比同配置 self-loop 慢約 14.8–43.8%（每配置七程序增加率中位數）。25 個新配置皆有預先綁定的同配置 self-loop 對照。

## QuickScorer 與 x86（2026-09-30，已完成一輪）

- 報告：[QUICKSCORER.md](QUICKSCORER.md)；數值：[reports/quickscorer-x86-20260930.json](reports/quickscorer-x86-20260930.json)。第一次在 x86-64 Linux（Xeon SPR KVM、clang 18）量測，不能與 M3 數字直接比較。
- QuickScorer 有 classic、dense rank 表與 checkpoint 三種策略，與 LLVM 原型逐位元相同；新增 35 項測試，完整套件 **970 passed**。
- 100×4：dense QS 與本機重建的前輪 winner 同場比較，hot／feature／code 核心低 5.5–23%，mixed 持平，data_pressure 高 20%；tuning 在 3/6 情境選中。300×6：六情境都選 interleaved scalar16，QS stride 8 比它高 4–75%。
- Pairwise 累加診斷在 LLVM 與 QS 都沒有一致收益：依序 float32 累加鏈不是主要瓶頸，維持精確契約。
- 路徑隱含冗餘 split 在兩個 hist 模型都是 0，不實作化簡。
- 原環境缺 `libclang_rt.profile` 導致 PGO 測試失敗，安裝 `libclang-rt-18-dev` 後恢復。

## Tree tiling（2026-09-30，已完成一輪）

- 報告：[TILING.md](TILING.md)；數值：[reports/tiling-x86-20260930.json](reports/tiling-x86-20260930.json)。
- `tiled.py`：k=2／3 tile、固定查表、scalar／gather／insert；五個模型逐位元相同，完整套件 **1090 passed**。
- 第二輪自動調參（5 次確認＋5 次 evaluation）：300×6、1000×4、300×8、lossguide 都選中 tiling，比 VPred 快 12–29%，比前輪最佳 LLVM 快 10–34%（lossguide 只在 5 次程序中贏 3 次）。100×4 仍由 QS／`cost4_block32_rank4` 領先。

## Batch=1 方法總比較（2026-09-30，已完成一輪）

- 報告：[BATCH1_STUDY.md](BATCH1_STUDY.md)；數值：[reports/batch1-study-x86-20260930.json](reports/batch1-study-x86-20260930.json)。
- 新增 C lowering：`vpred`、`packed`（bfs／dfs／hot_dfs／frames／forest）、`rapidscorer`、`direct`、`blockmix`；新增 79 項測試，完整套件 **1049 passed**。
- 自動調參器 `benchmarks.autotune`：tuning → 3 次程序確認 → 凍結 → evaluation，指標為 TSC ticks；五個模型已完成。
- VPred 在完整深度模型勝出（−12%～−17%）；QS 在 100×4 持平；lossguide 的 tuning 選擇在 holdout 輸 10%。
- RapidScorer、跨樹 SIMD（AVX-512）、機率導向布局、連續區塊混用都沒有一致收益。

## 尚未完成的方向

- Leaf 分離的 64-bit packed traversal state 已實作並通過完整套件；靜態 spill／reload 減少，尚待固定三方配對的 cold 效能診斷。

1. 接入真實 feature producer／模型／資料分布，以完整 pipeline 或 p99 重新選模；目前 cold runner 已支援這些 metric，但多目標輪已直接以 pipeline／p99 選模並增加 paired 確認；真實輸入及不同分布泛化仍未完成。顯式程式碼／資料大小限制仍未加入。
2. 改進資料表示與分組：高度／到達比例選子樹、8-byte node、跨樹交錯與 self-loop 已做；AoS／SoA／SoA8、交錯寬度至 32 已測；leaf 分表已實作、效能待驗證；高度分組與硬體成本驅動的分界仍未做。
3. Feature producer 與 inference 的 C++ caller LTO／內聯融合，避免重複存取中間 feature／rank 陣列；這不同於現有 TL 模型＋adapter 的 LTO。
4. 目標 CPU 特定的 gather／更完整的向量化；跨樹 scalar 交錯與 SIMD 比較已測，SIMD 仍有 scalar gather／lane packing 成本，沒有贏得本輪選模。
5. roots／next／both prefetch 與 1／2／4-group lookahead 已測，未有穩定整體收益；producer 內提前預取與 profile-guided 多層預取尚未測。更多 cache-line／page 布局（本輪只測 node alignment 16／64）、實際部署 profile 的 post-link layout（例如 Linux ELF 的 BOLT）。
6. 每個特徵／子樹不同的 encoder／prefix 精細混合，以及以測得硬體成本取代目前啟發式的成本估計。
7. 有 guard 與完整 fallback 的輸入特化，例如已確認不含 NaN 的快路徑。
8. Python 原生 extension／buffer protocol，以及完整 C++ direct-call 整合的單筆端到端成本。
9. 有條件的 stateful incremental inference：若相鄰請求只改少數特徵，可研究快取未受影響樹的葉值；若維持現有數值契約，仍須依原順序累加，不直接以總分減舊值再加新值。

這些是待驗證假設，不是保證更快的功能。已有的干擾量測顯示程式碼工作影響較大；需以目標機的硬體計數器區分 instruction cache、分支預測及其他原因，再決定實作次序。保持精確 raw margin 的目前目標，也限制了浮點重排、近似門檻／葉值及只回傳分類決策的 early-exit 方法。
