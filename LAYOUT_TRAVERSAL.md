# 節點布局與交錯寬度探索（2026-09-14）

本輪已完成兩個模型、六個 workload 的 tuning／獨立 evaluation。新增布局沒有取代大模型原 scalar16＋self-loop 的 code／mixed 最佳配置。小模型 mixed 選到 AoS12，但核心平均延遲沒有穩定改善，完整 pipeline 略退步。保持既有預設，不能宣稱布局普遍加速。

原始資料：`results/layout-cold-macos-{100x4,300x6}/report.json`。可攜摘要：[reports/layout-traversal-20260914.json](reports/layout-traversal-20260914.json)，包含每次量測、配對比較、凍結選模、候選與 artifact 雜湊。上一輪：[INTERLEAVED_TRAVERSAL.md](INTERLEAVED_TRAVERSAL.md)。

## 表示與方法

- AoS：每個 node 的 i32 control／float32 value 相鄰，8 bytes。
- SoA：control 與 value 分為兩個陣列，合計仍為每個 node 8 bytes。
- SoA8：每組 8 nodes 儲存 8 controls 與 8 values，64-byte tile；最後一組補齊。這不是保證與目標 CPU 的 cache line 大小相等。
- Scalar 交錯寬度 4／8／12／16／24／32；三種布局分別測 16-byte alignment，另測各布局 scalar16 的 64-byte alignment，以及 SoA vector16／32，共 23 個新配置。
- alignment 適用 node 陣列；root index 陣列保持 16-byte。API 另支援 128／4096，但本輪沒有量測這兩種 alignment。
- 原先 sentinel／self-loop 都保留；效能候選使用 self-loop。精確門檻、NaN routing 與 float32 原樹加總順序不變。

候選含舊配置及 TL2cgen 全部對照，object 去重後小模型 98、大模型 101 個。預先保留 6 個 references，包含前輪 tuning winner 與指定布局控制。每個 profile 三次 tuning shortlist 確認，全部六個選模先凍結才開 evaluation，再各跑三個獨立程序。

輸入各 4096×32 float32、3% NaN，tuning／evaluation seeds 28121／28122，runner seed 28120。完整 correctness suite **543 passed in 191.15s**，log：`results/layout-smoke/full-tests.log`。涵蓋兩個 LLVM backend、不同樹高、37 棵的尾組、常數／空森林、binary／regression、NaN／±inf／±0／threshold 邊界；所有 evaluation 原型對實際 producer 輸出的 max_abs_error 都為 0。

## 獨立驗證結果

以下為三次程序的核心平均延遲範圍（ns），非早期 warm block 計時。

| 模型／情境 | 凍結選出的配置 | 核心延遲 | 同程序 TL stock＋PGO |
|---|---|---:|---:|
| 100×4 code | hybrid8_h2_p1 | 292.0–297.8 | 714.8–773.7 |
| 100×4 mixed | layout_aos_scalar12_a16 | 304.9–352.4 | 727.9–779.6 |
| 300×6 code | interleaved_self_scalar16_O3 | 1389.2–1445.8 | 3397.1–3578.4 |
| 300×6 mixed | interleaved_self_scalar16_O3 | 1447.0–1495.8 | 3400.7–3548.1 |

小模型 mixed 的 AoS12 相對既有 `cold_compact4_cost8`，平均延遲變化介於降低 5.41% 與增加 8.77%，沒有一致勝出；p99 降低 16.8–27.2%，但選模目標是平均延遲。完整 pipeline 增加 0.02–0.26%，保留凍結結果，不依 evaluation 改選或宣稱成功。

大模型 code／mixed 仍選原 scalar16，其餘四個 profile 選 `directory_seed`；小模型其餘四個 profile 選 `cold_compact4_cost8`。大模型 mixed 相對 TL stock＋PGO 核心速度比 2.35–2.40×，完整 pipeline 僅降低 5.06–5.55%。

SoA 的固定 scalar16 對照在小模型比本輪 AoS16 較快，但仍未成為六個 profile 的最終選模；大模型的 SoA／SoA8 scalar16 未贏過既有 scalar16。加寬與加大 alignment 沒有帶來通用改善。

## 發現並修復的 IR 產生差異

加入通用 field-pointer helper 時，AoS 的 LLVM GEP 從「先計算各 node 位址，再取各欄位」變成「直接計算各欄位位址」。雖然資料與演算法相同，Clang 生成的暫存器配置與指令順序不同。本輪新 AoS16 比原先 AoS16 慢：code 情境原版低 7.95–9.64%，mixed 原版低 5.47–13.75%。這個差異不能誤歸因成記憶體布局收益。

已恢復原本 AoS staging 為預設，並新增 `traversal_load_schedule=staged|direct|lane` 用於後續排程實驗。恢復後兩個模型的 `.s` 與 `.o` 和前輪原版逐位元相同；證據：`results/schedule-smoke/aos-restoration.json`。`layout_candidates.py` 明確指定 `direct`，可重現本輪候選，不隨新的預設改變。後續排程 correctness／效能驗證另列，不回寫本輪結果。

## 限制

Apple M3、兩個合成回歸模型；沒有固定 CPU affinity／frequency。Code／data pressure 是每筆同 thread 的軟體干擾，不是硬體確認的 cache flush。核心包含每次 clock／dispatch；pipeline 另以整段計時。只比較同輪同程序的配置，不以不同輪的絕對 ns 計算改善。資料區大小不含 linker alignment gaps，不能只看 node bytes 推論 cache miss。Linux 硬體 PMU 尚不可用，詳見 [UBUNTU_BENCHMARK.md](UBUNTU_BENCHMARK.md)。
