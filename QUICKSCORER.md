# QuickScorer 特徵導向 lowering 與累加鏈診斷（2026-09-30）

本輪在 **x86-64 Linux** 上測試三個先前未做過的方向。之前的效能結果全部來自 Apple M3，本輪是第一個 x86 資料點：

1. **QuickScorer**（Lucchese 等，SIGIR 2015）：以特徵為主的 bitvector 遍歷，並加上本輪的 dense rank 表與 checkpoint 變體。
2. **float32 依序累加鏈**：這條相依鏈是否構成延遲下限。
3. **路徑隱含條件化簡**：同一路徑上被祖先比較決定的 split。

結論摘要：

- 小模型（100×4）的 **dense QuickScorer 在 x86 上具競爭力，但不是全面勝出**。
  - 與前輪最強配置（`directory_split16`、`clang_cost4_block32_rank4` 等，在本機重建）同場比較：hot／feature／code 情境的核心中位數低 5.5–23%；mixed 持平（−0.4%）；data_pressure 反而高 20%。六個情境中有 3 個被選中。
  - 只對 `select1_profiled_clang` 時，核心低 22–45%，p99 中位數低 34–59%；相對 TL2cgen 約 2.1–3.0×。
  - 完整 pipeline 只在 hot 情境穩定改善（−27%）。
- 大模型（300×6）的 QuickScorer 沒有勝出。六個情境都選既有的 `interleaved_self_scalar16`；QS stride 8 的核心比它高 4–75%，只比 `select1_profiled_clang` 低 1–10%，並快過 TL2cgen。
- 累加鏈不是主要瓶頸：兩種 lowering 的 pairwise 診斷都沒有一致改善。
- 路徑隱含冗餘 split 在兩個模型都是 **0**，不值得實作。

## 方法

每棵樹保存一個 leaf bitvector，每個 leaf 一個 bit，由左到右排列。XGBoost 的非 missing 值在 `!(x < threshold)` 時走右邊。對每個走右邊的 split，清掉它左子樹的所有 leaf bit；最後最低位的 1 就是 exit leaf。

**精確性證明**：
- exit leaf 不會被清掉。會清掉它的 split 必須是把資料送往左邊的祖先，但該祖先實際送往右邊，矛盾。
- exit leaf 左側的每個 leaf，都會被兩者的最近共同祖先清掉。
- Missing 值套用 `default_left == false` 的 split mask。
- 最後依原樹順序從 base margin 開始逐次 float32 相加，和 `compile_model` 的累加契約相同。

每個特徵有三種策略（`xgb_latency/quickscorer.py`）：

| 策略 | 參數 | 每筆工作 | 表格 |
|---|---|---|---|
| classic | `stride=0` | 依門檻排序，`while (thr <= x)` 逐一清 mask；結尾用 NaN sentinel。迴圈次數隨輸入而變 | 每 split 一筆 |
| dense | `stride=1` | 在 unique 門檻中求精確 rank `r`，NaN 使用專用列；再與預先 AND 好的整列 mask 做向量 AND，範圍只含用到該特徵的連續樹 | (unique+2)×span×word |
| checkpoint | `stride=s` | 每 s 個 split 存一列 prefix AND；剩餘不到 s 個 mask 用固定長度、無分支的 select 迴圈處理 | ⌊n/s⌋+1 列 |

Rank 有兩種實作：`two_level` 先以粗表（每 16 個門檻取一個）計數，再在 16 格內計數，兩層都可以 SIMD 化；`binary` 是無分支的 upper bound。兩者都只依賴嚴格比較，NaN 永遠比較為 false。100×4 的 dense kernel 在 x86 組合語言中**沒有任何分支指令**，使用 256-bit 向量。

`stride=None`（CLI 預設）會在 `--qs-budget-bytes`（預設 256 KiB）內選最小的 2 的冪次 stride。word 寬度依最大 leaf 數選 8／16／32／64 bits；超過 64 leaves 的樹明確拒絕。

```sh
.venv/bin/python -m xgb_latency.cli model.json build/qs --lowering quickscorer           # 依預算選 stride
.venv/bin/python -m xgb_latency.cli model.json build/qs --lowering quickscorer --qs-stride 1
```

```python
from xgb_latency import compile_quickscorer, Predictor
lib = compile_quickscorer("model.json", "build/qs", stride=1, rank_linear_max=1 << 20)
```

輸出同樣是 `void predict_row(const float *, float *)`，由 Clang 以 `-O3 -march=native -fno-fast-math -ffp-contract=off` 編譯 C 原始碼，不經 llvmlite。

## 正確性

- 新增 `tests/test_quickscorer.py` 共 35 項測試，涵蓋：
  - 三個模型（regression／binary／absolute error，深度 2／3／6，含 64-bit word）
  - 8 組 stride／rank 組合
  - NaN、±inf、−0、每個門檻及其相鄰可表示值
  - 無 split 的常數模型、超過 64 leaves 的拒絕、參數驗證、兩個 pairwise 診斷的 metadata 標記

  所有 QS 輸出都與既有 LLVM 原型**逐位元相同**，也與 XGBoost 一致。
- 完整套件 **970 passed**，即原有 935 項加上新增 35 項。
- Benchmark 腳本在計時前再次檢查：每個精確引擎都必須與第一個 LLVM 引擎逐位元相同。cold harness 則要求 engine family 對 XGBoost 的 `max_abs_error == 0`。

## 環境

- Intel Xeon（Sapphire Rapids 世代，model 207），KVM guest，4 vCPU，標稱 2.10 GHz，支援 AVX-512
- Ubuntu 24.04，clang 18.1.3
- 沒有綁核或頻率控制

VM 雜訊明顯，同一配置在不同程序間可差 20–30%；只比較同程序配對的數字。這些數字**不能和 M3 報告直接比較**。

原本環境缺少 `libclang_rt.profile-x86_64.a`，PGO 測試因此失敗；安裝 `libclang-rt-18-dev` 後，原套件 935 項全部通過。

模型與資料使用 `benchmarks.run` 的合成配方（seed 2026、32 特徵、3% NaN）。Cold 輪使用新產生、彼此獨立的 tuning／evaluation 檔案。數值見 [reports/quickscorer-x86-20260930.json](reports/quickscorer-x86-20260930.json)。

## Warm 結果（`benchmarks/quickscorer.py`）

100×4 的三次獨立程序（seed 3001–3003），block ns／row：

| 引擎 | r1 | r2 | r3 | p99 (r1/r2/r3) |
|---|---:|---:|---:|---:|
| `qs_stride1`，linear rank | 434.5 | 317.9 | 392.5 | 1434／549／720 |
| `qs_stride1`，two-level rank | 487.4 | 335.7 | 351.9 | 2660／587／756 |
| `qs_stride1`，binary rank | 406.3 | 333.6 | 444.6 | 702／637／794 |
| `llvm_select1_profiled_clang` | 579.6 | 443.0 | 463.4 | 1272／1159／1173 |
| `llvm_interleaved_self_scalar16` | 1203.5 | 674.6 | 933.0 | 1752／1450／3029 |
| `qs_stride0`（classic） | 1147.6 | 867.9 | 1081.2 | |
| `qs_stride8` | 1203.1 | 942.2 | 1020.4 | |
| `tl2cgen_profiled_lto_prepared` | 1096.1 | 877.9 | 1003.6 | 2063／1833／2060 |

Stride 掃描（`qs2` 輪）：stride 1／2／4／8／16 分別為 392／583／771／1074／1006 ns。

- Masked tail 的間接讀改寫會序列化，所以 checkpoint 越稀越慢。
- Classic QS 每個特徵都有一次難預測的迴圈結束，比 dense 慢約 2.5×。

300×6（單程序）：

| 引擎 | block ns／row | 表格 |
|---|---:|---:|
| `llvm_interleaved_self_scalar16` | 3719 | |
| `qs_stride8` | 4306 | 4.5 MB |
| `qs_stride32` | 4896 | 1.2 MB |
| `qs_stride1` | 5014 | 12.6 MB |
| `llvm_select1_profiled_clang` | 6042 | |
| `qs_stride0` | 7218 | 0 |
| `tl2cgen_profiled_lto_prepared` | 7285 | |

QS 的工作量隨 split 數成長。300×6 有 15,180 個 split，而遍歷只需 300×6 次比較；dense 表遠超 L2，所以大模型不適合 QS。

## Cold／同 thread 干擾（`benchmarks.interference`）

流程與前輪相同：
- 六個 profile 先在 tuning 上選模，並做 3 次 shortlist 確認。
- 全部凍結後才開啟 evaluation，每個 profile 跑 3 個獨立程序。
- 指標是 `model_mean_median_ns`。

下表是三程序的中位數：

**100×4**：

| Profile | 選中 | QS 核心／p99 | `select1_profiled` 核心／p99 | QS pipeline | `select1_profiled` pipeline | TL 核心 |
|---|---|---:|---:|---:|---:|---:|
| hot_control | `qs_stride1_lin0` | 545／807 | 992／1982 | 488 | 741 | 1509 |
| features_64k | `qs_stride1_lin0` | 422／639 | 741／1509 | 642 | 879 | 1207 |
| features_4m | `qs_stride1_lin0` | 510／939 | 751／1822 | 1310 | 1275 | 1274 |
| data_pressure_2m | `qs_stride1_lin16_bin` | 1100／2433 | 1403／3785 | 45759 | 49852 | 3304 |
| code_pressure | `qs_stride1_lin100000` | 699／1480 | 1047／2229 | 16348 | 16681 | 1693 |
| mixed_pressure | `qs_stride1_lin16_bin` | 1227／2635 | 1723／4372 | 64653 | 78282 | 2563 |

- 選中的 QS 在每個 profile 的每個 evaluation 程序，核心平均都低於本輪的 LLVM 與 TL 候選。p99 中位數也都較低，但 data_pressure 有一個程序例外。
- Pipeline 在 hot／features_64k 降低約 27–34%。有大量前置工作的情境由 producer 主導，差異落在雜訊內；features_4m 的中位數甚至略高。
- 本輪 LLVM 候選不含前輪的 rank／bucket winner，下一節補做這個比較。

**300×6**：六個 profile 都選 `llvm_interleaved_self_scalar16`。`qs_stride8` 的核心在六個 profile 都比 `select1_profiled_clang` 低 1–10%（例如 mixed 12120 對 12973 ns），但比 interleaved 高 4–75%。

## 與前輪最佳配置對照

從 `reports/` 的 build metadata 還原前輪 M3 在 100×4 的 winner 選項，並在本機以相同模型與 calibration 重建：
- `clang_cost4_block32_rank4`
- `prefix_bits12`
- `prefix_bits16_rank16`
- `directory_split16`：以 `prefix_bits16_rank16` 為基礎，改用 `bucket_split`

四者對 XGBoost 的誤差都是 0。接著使用**新的** tuning／evaluation 檔重跑六情境。`qs_stride1_lin100000`、`directory_split16`、`clang_cost4_block32_rank4` 在 holdout 前被指定為 references，因此一定會進入 evaluation。

輸出：`results/qs-cold-prior-x86-100x4/`。下表是三程序中位數：

| Profile | tuning 選中 | QS 核心／p99／pipeline | 最佳前輪配置 | 其核心／p99／pipeline | 核心差 | QS 核心勝 |
|---|---|---:|---|---:|---:|---:|
| hot_control | `qs_stride1_lin16_bin` | 462／1292／384 | `directory_split16` | 599／1239／528 | −23.0% | 3/3 |
| features_64k | `qs_stride1_lin0` | 419／761／646 | `directory_split16` | 472／1041／696 | −11.1% | 2/3 |
| features_4m | `qs_stride1_lin100000` | 487／977／1108 | `directory_split16` | 516／1190／1168 | −5.5% | 3/3 |
| data_pressure_2m | `clang_cost4_block32_rank4` | 1107／2287／50674 | `clang_cost4_block32_rank4` | 921／2446／48724 | +20.2% | 0/3 |
| code_pressure | `prefix_bits12` | 556／1484／16116 | `clang_cost4_block32_rank4` | 621／1262／15830 | −10.5% | 3/3 |
| mixed_pressure | `directory_split16` | 1299／2282／77737 | `clang_cost4_block32_rank4` | 1304／2612／76109 | −0.4% | 2/3 |

解讀：
- Dense QS 的 123 KB 表格需要常駐 L2。data_pressure 的 2 MB 驅逐直接打擊它；程式碼驅逐則對幾乎無分支、程式碼很小的 QS 較有利。
- Code_pressure 的 tuning 選了 `prefix_bits12`，但 holdout 顯示 QS 與 `cost4_rank4` 都更快。按照 protocol，不以 evaluation 改寫 selection。
- **結論**：QS 應作為 100×4 類小模型的正式候選，交給既有的選模流程決定，不應當成預設。

## 累加鏈診斷

`accumulation_order="pairwise_inexact"`（`compile_model`）與 `summation="pairwise_inexact"`（QS）改用 pairwise 樹狀加總，**不精確**，只用來估計依序累加鏈的成本。兩者 metadata 都標記 `exact_accumulation_order: false`，benchmark 也另列為 `diag_*`，不參與選模。

| 對照（100×4 warm，block ns） | r1 | r2 | r3 |
|---|---:|---:|---:|
| LLVM profiled：精確／pairwise | 579.6／587.6 | 443.0／439.6 | 463.4／553.0 |
| QS dense：精確／pairwise | 434.5／320.8 | 317.9／330.1 | 392.5／354.9 |

300×6 的 LLVM profiled 為 6042／6148 ns。

- LLVM 樹遍歷中，亂序執行已經把累加鏈與遍歷重疊，pairwise 沒有收益。
- QS dense 的累加鏈在所有 feature 處理完才開始，理論上會暴露在尾端，但三次結果方向不一致（−26%～+4%）。在本機雜訊下無法確認有穩定收益。
- **目前沒有理由為此放寬精確 raw margin 的契約。**

## 路徑隱含化簡

對每條根到葉路徑追蹤每個特徵的非 missing 區間 `[lo, hi)` 與「是否仍可能是 NaN」。若一個 split 的方向已被祖先決定，且 NaN 也走同一方向或已不可能出現，就視為冗餘。

兩個模型的冗餘 split 數都是 **0**。XGBoost `hist` 不會選零增益的隱含 split，因此不值得實作；其他 trainer 或手工合併的模型仍可能出現冗餘。

## 限制與後續

- 只測兩個合成模型與一台 VM。沒有 PMU 計數，因此不能區分 L1／L2、分支與前端瓶頸。
- M3 上尚未測 QS。ARM NEON 的向量寬度與 cache 大小不同，dense 表的取捨可能不同。
- Dense 表大小是 Σ(unique 門檻)×(樹範圍)。門檻多、樹多時會爆量；300×6 需要 checkpoint，而 checkpoint 的 masked tail 太慢。

值得嘗試的後續：
1. **依樹分組**：只把 leaf 數少、門檻少的樹交給 QS，其餘用 interleaved 遍歷，最後仍依樹順序累加。
2. 以 AVX-512 mask 暫存器處理 tail。
3. 對 unique 門檻很多的特徵，改用 bucket 前綴 rank（沿用 `rank.py` 的 IEEE prefix 設計）。
4. 把 QS 放進 `benchmarks.optimize` 的預設候選，並在目標機的實際特徵分布上重跑。
5. 縮小 dense 表以降低 data_pressure 的弱點：例如 16-bit word 只存 span 內實際用到的樹，或對高頻特徵共用 prefix 列。
