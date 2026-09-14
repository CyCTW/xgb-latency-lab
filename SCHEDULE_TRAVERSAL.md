# LLVM 載入排程探索（2026-09-14）

本輪發現等價 LLVM IR 的產生順序可以影響最終機器碼及 cold 情境延遲。小模型 code-pressure 選到 SoA12 lane，大模型 code-pressure 選到 AoS12 lane，核心平均延遲比各自先前配置降低。但 mixed-pressure 的選模不穩定，且核心、p99 與完整 pipeline 排名不一致。沒有將新候選設為通用預設。

結果：[reports/schedule-traversal-20260914.json](reports/schedule-traversal-20260914.json)。原始：`results/schedule-cold-macos-{100x4,300x6}/report.json`。前一輪：[LAYOUT_TRAVERSAL.md](LAYOUT_TRAVERSAL.md)。

## 改動與驗證

新增 `traversal_load_schedule`／`--traversal-load-schedule`：

- `staged`：先建立 AoS 各 node 位址，再分階段發出 control、threshold 與 feature loads。恢復原本交錯遍歷的 IR 形狀。
- `direct`：直接建立各欄位的 GEP，重現上一輪布局重構。資料、比較及加總語意相同，但會改變 LLVM 的最佳化與暫存器配置。
- `lane`：先對每個 lane 發出 control、threshold 與依賴它的 feature load，再處理下一個 lane；後續比較與 child 更新仍跨 lane 產生。這只控制輸入 IR 的順序，不限制 LLVM／CPU 最終排程。

預設 `staged` 的兩個 scalar16 模型與前輪原版 `.s`／`.o` 逐位元相同，證據 `results/schedule-smoke/aos-restoration.json`。這修復布局重構的效能退步，不把回復既有表現列作新加速。

完整套件 **592 passed in 215.16s**，log：`results/schedule-smoke/full-tests.log`。新增 48 個案例交叉測試 clang／llvmlite、三種布局、sentinel／self-loop、direct／lane、scalar／vector，以及一個非法 schedule 案例。使用 37 棵不同高度樹、尾組、獨立 float32 traversal oracle 與 XGBoost，涵蓋 threshold／nextafter／NaN／±inf／±0，檢查輸入不被改寫。

所有候選保留原樹加總順序，沒有 fastmath、閾值近似或葉值近似；效能 evaluation 的所有原型在實際 feature producer 輸出上 max_abs_error 都是 0。

## 實驗

新增 AoS widths 4／8／12／16／24／32 × staged／lane，以及 SoA widths 8／12／16／24 的 lane，共 16 個配置。保留上一輪所有候選與 TL2cgen baseline，object 去重後小模型 111、大模型 114 個；預先固定 references 分別 8／7 個。選模從 tuning 結果產生，不以 evaluation 挑 winner。

新 tuning／evaluation 各 4096×32 float32、3% NaN，seed 29221／29222；runner seed 29220。三次 shortlist 確認，六個 profile 全部凍結後才打開 evaluation，各跑三個獨立程序。正確性測試及候選編譯完成後才開始計時。

重現命令（輸出目錄須尚未存在）：

```sh
.venv/bin/python -m benchmarks.schedule_candidates \
  --previous results/layout-cold-macos-100x4/report.json \
  --model results/controlled-100x4/model.json \
  --output results/schedule-builds-100x4
.venv/bin/python -m benchmarks.interference \
  --shape 100x4 --seed 29220 --shortlist-runs 3 \
  --candidate-manifest results/schedule-builds-100x4/candidates.json \
  --model results/controlled-100x4/model.json \
  --tuning results/optimization-inputs/100x4-schedule-tuning.npy \
  --evaluation results/optimization-inputs/100x4-schedule-evaluation.npy \
  --output results/schedule-cold-macos-100x4
```

大模型改為 `300x6`。這些 manifests 使用 macOS 產物，Linux 要全部重建。

## 結果

以下為三個 evaluation 程序的範圍。核心平均指每輪 per-call clock 平均的中位數；pipeline 使用獨立整段計時。

| 模型／情境 | tuning 凍結配置 | 核心平均 ns | 核心 p99 ns | pipeline µs |
|---|---|---:|---:|---:|
| 100×4 code | schedule_soa_12_lane | 282.8–284.3 | 333–333 | 11.404–11.406 |
| 100×4 mixed | schedule_soa_12_lane | 287.8–338.1 | 375–416 | 29.196–29.289 |
| 300×6 code | schedule_aos_12_lane | 1384.6–1398.9 | 1500–2042 | 12.573–12.597 |
| 300×6 mixed | layout_aos_scalar12_a16 | 1439.4–1615.3 | 1708–1792 | 30.427–30.653 |

**小模型 code**：相對同程序 `hybrid8_h2_p1` 核心平均降低 0.76–6.74%、p99 降低 27.3–33.4%，但 pipeline 增加 0.002–0.114%。相對 `cold_compact4_cost8` 核心降低 6.24–8.39%、pipeline 降低 0.12–0.19%。相對 TL stock＋PGO 核心速度比 2.47–2.61×。

**小模型 mixed**：相對 `cold_compact4_cost8`，核心介於降低 2.65% 與增加 10.37%，pipeline 增加 0.038–0.056%。雖較前輪被選的 AoS12 核心低 2.74–6.75%，仍不能稱為更好的通用 cold 配置。

**大模型 code**：相對前輪 `interleaved_self_scalar16_O3` 核心降低 2.77–3.33%、pipeline 降低 0.25–0.43%，但其中一次 p99 增加 19.56%。相對 TL stock＋PGO 核心速度比 2.51–2.57×，pipeline 降低 13.04–13.44%。固定對照 AoS16 lane 相對原 scalar16 核心降低 2.73–6.55%、pipeline 降低 0.42–0.49%、p99 降低 2.46–9.32%；它不是本輪 tuning 凍結 winner，不依此回寫 selection。

**大模型 mixed**：本輪 tuning 選了前輪既有 direct AoS12，但 evaluation 比原 scalar16 核心慢 0.83–6.69%、p99 也較差。預先保留的 AoS16 lane 在三次程序中也有一次核心退步 6.22%，不能推論其 mixed 優勢已成立。

小模型其餘四種情境仍選 `cold_compact4_cost8`；大模型 hot／feature 選 `directory_seed`。大模型 data-pressure 選 `cold_depth3`，但 evaluation 核心為 965.5–984.2 ns，同程序 directory 為 945.0–948.6 ns，亦屬 tuning 排名未穩定重現的例子。

選定 library／object／header／metadata 位於 `results/schedule-cold-macos-SHAPE/selected_models/PROFILE/`，已驗證 library 雜湊與被測候選一致；這是研究候選，不是部署建議。

## 判斷與後續

指令排程仍有可測改善，尤其 code-pressure；但 mixed 干擾對平均延遲的微小排名不穩定。下一步需加強 paired 重複量測、直接以 pipeline／p99 做 tuning，並用新 holdout 驗證。只降低 inference clock 區間內的時間，不能保證完整呼叫流程改善。

`results/schedule-smoke/assembly-diagnostics.json` 保存編譯器標示的靜態 spill／reload 數量，只作機器碼診斷，並非實際執行次數或 PMU 證據。更多 lanes 會同時增加獨立工作與暫存器壓力，沒有單調的快慢關係。

仍是 Apple M3 的兩個合成模型，沒有 CPU affinity／frequency isolation，也沒有確認的硬體 cache flush。Cold 工作每筆都先在同 thread 執行；p99 含 clock／dispatch，不能與早期 warm block 結果直接比較。Linux PMU 狀態見 [UBUNTU_BENCHMARK.md](UBUNTU_BENCHMARK.md)。
