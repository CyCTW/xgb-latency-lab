# Cold 情境的多目標選模與配對確認（2026-09-14）

本輪檢驗核心平均、核心 p99、完整 pipeline 平均與 pipeline p99 是否需要不同編譯配置。先前排程輪顯示平均延遲改善未必同時改善尾端或完整流程，因此本輪使用既有候選，擴大量測並直接依不同目標選模。

## 量測與選模規則

- 兩個既有合成回歸模型 100×4／300×6，各保留 111／114 個引擎與 TL2cgen 候選；references 10／9 個，包含上一輪凍結選模與固定排程對照。
- 四個目標共用每次原生量測：`model_mean_median_ns`、`model_p99_ns`、`pipeline_block_median_ns`、`pipeline_p99_ns`。
- 每兩輪先隨機產生 row sequence 與 engine order，下一輪沿用同一 row sequence 並反轉 engine order。每個候選在一對輪次中位置對稱。下一對重新抽樣。
- 每筆仍先同 thread 執行 code／data 干擾與 feature producer；反向輪次沒有改成連續 hot inference。
- Screening 每個候選 256×4 次；每個 metric／family 的前兩名，加上固定 references，合成共同 shortlist。
- Shortlist 以 5 個獨立程序確認，每個候選 512×8 次。各目標依確認程序摘要的中位數，分別選每個 family 的配置。
- 六個 profile、四個目標的選模全部寫入 selection.json，才讀取任何 evaluation 資料，包括其 hash。Evaluation 不回寫選模。
- Evaluation 每個 profile 使用所有目標 winner 與 references 的聯集，7 個獨立程序，每個候選 1024×10 次。核心 per-call 樣本共 71,680 次，另有同等數量的完整 pipeline block 呼叫。
- tuning／evaluation 各 4096×32 float32、3% NaN，seed 30321／30322；runner seed 30320。與前輪資料分離。

核心平均是原生每輪 per-call 平均的中位數；pipeline 平均欄位是各輪整段總耗時除以樣本數，再取中位數。核心／pipeline p99 都來自有 per-call clocks 的測量，pipeline p99 包含這些 clock 開銷。它與沒有 inner clocks 的 pipeline 平均不是同一種測量。

p99 是直接量測整段呼叫的分位數，不是 preparation p99 加上 inference p99；分位數不能如此相加。Core p99 也受 clock 解析度影響，相同分數會依候選名稱確定性地解決平手，不表示兩個配置的真實延遲完全相同。

反向排序只平衡位置，不保證消除頻率變動、OS 中斷、cache／branch predictor 歷史或非線性時間漂移。同對輪次重用資料，不能將它們當作獨立觀察；比較以獨立程序摘要為單位，列出範圍、中位數與勝出程序數，沒有把 71,680 次重複呼叫當成獨立統計樣本。

## 實作與驗證

`benchmarks/interference.cc` 新增 `bench_paired`，舊 `bench` 仍可使用。輸出保存每輪候選順序與 row sequence hash，便於核對反向與相同輸入的要求；奇數 paired rounds 明確拒絕。

`benchmarks/multiobjective.py` 以共同 shortlist 執行多目標選模，保存每個目標的部署 artifact。每個原型都檢查 source model hash，整輪前後核對候選 library hash 及 selection hash；不允許測量中替換候選。

本輪 benchmark 相關測試 **17 passed in 3.36s**，log：`results/multiobjective-smoke/targeted-tests.log`。包括原有 feature producer／TL ABI／freeze 測試，新增反向輪次、row hash、拒絕奇數輪次、所有目標／profile 在 holdout 讀取與 hash 前凍結、不同 metric winner 保留、artifact 複製及自訂樣本數未被截斷。引擎編譯邏輯未改；前輪完整引擎套件 592 項通過，不將兩者相加稱作新的完整測試數。

## 重現

原始輸出位置：`results/multiobjective-cold-macos-{100x4,300x6}/`。使用既有 Mac artifacts 的候選 manifest：`results/multiobjective-builds-SHAPE/candidates.json`。

```sh
.venv/bin/python -m benchmarks.multiobjective \
  --model results/controlled-100x4/model.json \
  --candidate-manifest results/multiobjective-builds-100x4/candidates.json \
  --tuning results/optimization-inputs/100x4-multiobjective-tuning.npy \
  --evaluation results/optimization-inputs/100x4-multiobjective-evaluation.npy \
  --output results/multiobjective-cold-macos-100x4 --seed 30320
```

輸出目錄須尚未存在；大模型改 `300x6`。四個 metric、六個 profile 與上述樣本數均為此 runner 的預設。Linux 需重新建置所有模型與 harness。

## Holdout 結果

兩個模型、六種情境、四個目標均已完成。可攜數據：[reports/multiobjective-cold-20260914.json](reports/multiobjective-cold-20260914.json)，含每次量測、固定選模、所有候選對照、來源／library 雜湊。全部正反輪次與 row sequence hash 核對通過，原型在實際 producer 輸出上的 max_abs_error 都為 0；凍結 selection 與候選 library 沒有改動。

下表比較「直接以該指標選出的配置」與**同輪核心平均選模**在該指標上的表現。正值表示延遲降低；中位數取自七次程序的配對百分比，勝出次數不等同統計顯著性。

| 模型／情境 | 選模指標 | 凍結配置 | 延遲降低中位數 | 勝出程序 |
|---|---|---|---:|---:|
| 100×4 hot | pipeline 平均 | directory_seed | 4.84% | 7/7 |
| 100×4 code | pipeline 平均 | hybrid8_h2_p1 | 0.06% | 5/7 |
| 100×4 code | pipeline p99 | layout_aos_scalar12_a16 | −1.07% | 1/7 |
| 100×4 mixed | pipeline p99 | directory_seed | 3.10% | 7/7 |
| 300×6 data | 核心 p99 | layout_soa_scalar8_a16 | −4.74% | 0/7 |
| 300×6 data | pipeline p99 | layout_soa_scalar16_a16 | −2.83% | 0/7 |
| 300×6 code | pipeline 平均 | schedule_aos_16_lane | 0.19% | 6/7 |
| 300×6 mixed | 核心 p99 | layout_soa_scalar16_a16 | 2.34% | 5/7 |
| 300×6 mixed | pipeline 平均 | schedule_aos_16_lane | 0.52% | 6/7 |
| 300×6 mixed | pipeline p99 | layout_soa_scalar12_a16 | 1.16% | 5/7 |

**小模型 mixed 的具體取捨**：核心平均與 pipeline 平均都選 `schedule_aos_16_lane`，核心平均 284.7–324.8 ns，pipeline 平均 29.197–29.372 µs。Pipeline p99 則選 `directory_seed`，為 29.958–31.250 µs，比核心平均選出的配置低 0.54–4.50%，七次都改善。但 directory 核心平均慢 4.91–20.31%，核心 p99 也更差；pipeline 平均沒有一致改善。整體 p99 的改善不能解釋成每個組成部分都更快，目前沒有 PMU 證據確定硬體原因。

**大模型 mixed**：核心平均選 `schedule_aos_12_lane`；pipeline 平均選 `schedule_aos_16_lane`，為 30.321–30.492 µs，比前者降低中位數 0.52%，範圍為降低 0.59% 至增加 0.43%。Pipeline p99 選 `layout_soa_scalar12_a16`，為 31.750–32.625 µs，降低中位數 1.16%，範圍為降低 2.56% 至增加 1.56%。這兩個目標都還有反向結果，不能宣稱全面穩定加速。

**明確未通過的選模**：大模型 data-pressure 兩種 p99 目標都選了 SoA traversal，但在 holdout 相對核心平均選出的 directory，核心 p99 七次慢 2.34–12.80%、pipeline p99 七次慢 1.24–6.83%。增加重複測量並不能保證 tuning 排名能泛化到不同輸入。本輪保留這些失敗的 selection，不在 evaluation 後換成 directory。

小模型 features_64k／features_4m 四個目標皆選 `cold_compact4_cost8`；大模型 hot／features_64k／features_4m 四個目標皆選 `directory_seed`。小模型 data 的 directory pipeline p99 對核心平均配置 6 勝 1 平，平均 pipeline 則 5 勝 2 負。小模型 code 核心平均／p99 都選 SoA12 lane；大模型 code 核心平均／核心 p99／pipeline p99 都選 AoS12 lane。

## 相對 TL2cgen 與可用產物

比較各自依**同一目標**凍結選出的原型與 TL stock＋PGO，mixed 的 pipeline 平均延遲降低範圍：100×4 為 1.04–1.45%，300×6 為 5.44–7.85%，兩者皆 7/7 程序勝出。Code-pressure 的 pipeline 平均則降低 3.36–3.84%／12.48–13.67%。這是完整 producer＋interference＋inference 的結果，不等同核心速度比，也不能拿來與前輪絕對 ns 計算新增收益。TL 的 prepared packing 仍包含在 producer／pipeline 成本中，所有既有 baseline 選項保留。

每個 profile／objective 都保存 library、object、header 與 metadata：

`results/multiobjective-cold-macos-SHAPE/selected_models/PROFILE/OBJECTIVE/`

所有部署副本與實際量測 library 的 SHA256 一致。這些是可重現研究候選，沒有改動通用編譯預設；包含上述未通過 holdout 的 tuning 選擇，使用時需查閱驗證結果。

## 後續探索

多目標選模已完成一輪，之後的 lowering 實驗可使用這套流程直接追蹤完整 pipeline 與尾端代價。下一個獨立方向是模型感知 prefetch／leaf 分離表示，之後仍需探索 feature producer 與 C++ caller 融合，以及真實模型／輸入分布。不能只繼續增加相同合成資料的重複次數來宣稱泛化。

本輪仍是 Apple M3 的合成 workload，沒有 CPU／frequency isolation，軟體壓力不是已確認的 cache flush；Linux PMU 仍不可用。這些改善不代表跨模型、跨 CPU 或真實部署的保證。
