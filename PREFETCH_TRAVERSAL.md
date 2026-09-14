# 模型感知 prefetch 探索（2026-09-14）

本輪在精確交錯遍歷中探索根節點 lookahead 與下一節點 prefetch，檢查它們能否改善同 thread 前置工作後的推論／pipeline。主實驗與新資料的同配置對照均已完成，沒有找到可穩定取代既有配置的整體收益，預設保持關閉。

## 實作

`traversal_prefetch`／`--traversal-prefetch` 支援：

- `none`：不發出 prefetch，保留原本機器碼。
- `roots`：在入口先提示前 distance 組的 roots，再於處理每組前提示 distance 組之後的 roots。每棵樹的 root 只提示一次，尾組透過編譯時範圍檢查處理。
- `next`：每個 lane 算出實際下一節點後提示其資料；已結束 lane 留在有效 leaf，不猜測任意 child。最後一層的提示位址也是有效 leaf。
- `both`：合併上述兩者。

`traversal_prefetch_distance` 是 root lookahead 的組數，可設 1／2／4；next-only 不使用此距離。`traversal_prefetch_locality` 可設 0–3，是給 LLVM 的 temporal locality hint，不是可攜的指定 cache level 介面。提示為 read／data-cache 類型。AoS 提示 8-byte record 起點；SoA／SoA8 對 control 與 value 各提示一次，可能被硬體合併。

使用 `llvm.prefetch` intrinsic；它只提供效能提示，不回傳值，也不改變程式結果；不支援的 target 可以不產生對應指令。[LLVM LangRef](https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic)

只對已存在的 node 陣列與有效 index 建立 inbounds GEP。浮點比較、NaN default routing、leaf 數值及原樹 float32 加總順序保持不變。它可能增加 code size、指令數與記憶體流量，並非保證有效的優化。

## 候選與驗證

四組基礎配置：AoS12 lane、AoS16 lane、SoA12 lane、SoA16 direct。每組比較 8 個配置：none；roots distance 1／2／4、locality 3；next locality 2／3；both distance 1／2、locality 3。共 32 次編譯，其中 4 個 disabled controls 與舊 object 完全相同；28 個啟用候選均確認生成 ARM prefetch 指令。候選去重後小模型 139、大模型 142 個，固定 references 12／13 個。

Build metadata：`results/prefetch-builds-{100x4,300x6}/builds.json`。機器碼核對：`results/prefetch-smoke/assembly-validation.json`。例如大模型 AoS16 lane 的 roots 版本有 300 個靜態 prefetch 指令，next 版本有 28 個 helper 內指令；後者的動態執行次數會乘上 group 與樹高，不能直接比較靜態數量推論成本。

新增 targeted 測試 **82 passed in 35.62s**：`results/prefetch-smoke/targeted-tests.log`。涵蓋 clang／llvmlite、三種 data layout、兩種 leaf layout、scalar／vector、37 棵不同高度樹與尾組、NaN／±inf／±0／threshold 邊界，以及空／常數森林、非法參數。完整回歸套件 **677 passed in 269.92s**（`results/prefetch-smoke/full-tests.log`），正式效能實驗與補充配對均已完成，原型在實際 producer 輸出上的 max_abs_error 全部為 0。

## 效能方法

沿用 [MULTIOBJECTIVE_COLD.md](MULTIOBJECTIVE_COLD.md) 的四目標與六情境方法：screen 256×4，5 次 shortlist 確認各 512×8，7 次 evaluation 各 1024×10。正反輪次使用相同輸入；全部選模凍結後才讀取 holdout。

新 tuning／evaluation 各 4096×32 float32、3% NaN，seed 31421／31422，runner seed 31420。所有舊候選、TL2cgen baseline、前輪四目標選出的原型與指定 AoS16 root／next 控制保留。正式計時在編譯與測試完成後執行。

主結果目錄：`results/prefetch-cold-macos-{100x4,300x6}/`。可攜數據：[reports/prefetch-cold-20260914.json](reports/prefetch-cold-20260914.json)。

重現命令（輸出目錄須尚未存在，大模型把 `100x4` 改成 `300x6`）：

```sh
.venv/bin/python -m benchmarks.prefetch_candidates \
  --previous results/multiobjective-cold-macos-100x4/report.json \
  --model results/controlled-100x4/model.json \
  --output results/prefetch-builds-100x4
.venv/bin/python -m benchmarks.multiobjective \
  --model results/controlled-100x4/model.json \
  --candidate-manifest results/prefetch-builds-100x4/candidates.json \
  --tuning results/optimization-inputs/100x4-prefetch-tuning.npy \
  --evaluation results/optimization-inputs/100x4-prefetch-evaluation.npy \
  --output results/prefetch-cold-macos-100x4 --seed 31420
```


## 主實驗結果

沒有任何 prefetch 版本成為兩模型的核心平均選模結果。Hot／features 情境均未選中 prefetch；next 與 both 在本輪任何 profile／目標都沒有被選中。只有 roots 在部分 data／code／mixed 的尾端或 pipeline 目標被選中。

小模型 mixed 的 AoS16 roots distance 1 在 pipeline p99 相對本輪核心平均選模，7 次程序有 6 次較低，中位降低 1.11%，但範圍包含一次退步 0.11%。Code 的同一候選則只有 3 勝、1 平、3 負，中位降低為 0%。小模型 mixed 核心 p99 選 AoS12 roots distance 4，只有 2 勝、3 平、2 負，其中一次 p99 比核心平均配置增加 49.88%。

大模型 mixed 的 pipeline 平均選 AoS16 roots distance 1，相對核心平均選出的 AoS12 lane，只有 3/7 次改善，中位增加 0.22%。Pipeline p99 選 SoA12 roots distance 2，雖 5/7 次改善、中位降低 0.24%，仍有一次增加 23.88%。主選模如實保存，沒有因 evaluation 退步而改換候選。

這些比較是不同目標選模的部署取捨，不能單獨歸因為 prefetch 的效果，因此另做以下同配置對照。

## 新資料的同配置配對

主驗證中，小模型 AoS12 roots distance 4 缺少同寬度／布局／load schedule 的 disabled reference。其餘多數候選已有 matched control，但本輪統一對**所有 tuning 選中的 prefetch 候選**補做檢查；不依主 holdout 的好壞挑選補測對象。

`benchmarks/prefetch_attribution.py` 只讀取主實驗的 frozen selection.json 決定案例，不讀主 evaluation 分數。先保存 plan.json，再產生第三份 seed 31423 的 4096×32 float32／3% NaN 新資料，每組在原 native harness 執行 7 個獨立程序、每個 1024×10 次正反輪次。Disabled control 透過相同 backend／optimization／lanes／mode／leaf layout／data layout／load schedule／alignment 的 none build，按 object hash 找到原候選，避免夾帶布局變動。

結果：[reports/prefetch-attribution-20260914.json](reports/prefetch-attribution-20260914.json)，原始：`results/prefetch-attribution-{100x4,300x6}/report.json`。新 raw 資料與主 evaluation 不同；plan／source selection／library hash、正反 round order、row sequence hash 及零預測誤差均驗證通過。這是固定配置診斷，沒有新的 tuning 或部署選模。

下表以同配置、關閉 prefetch 的版本作基準；正值表示延遲降低，負值表示增加。中位數來自七個程序的配對百分比。

| 模型／情境 | 預取候選 | 原選模目標 | 延遲降低中位數 | 勝／平／負 |
|---|---|---|---:|---:|
| 100×4 data | AoS16 lane roots d1 | 核心 p99 | 0.00% | 3／4／0 |
| 100×4 code | AoS16 lane roots d1 | pipeline p99 | −0.71% | 2／0／5 |
| 100×4 mixed | AoS12 lane roots d4 | 核心 p99 | 0.00% | 1／6／0 |
| 100×4 mixed | AoS16 lane roots d1 | pipeline p99 | 1.03% | 4／0／3 |
| 300×6 mixed | AoS16 lane roots d1 | pipeline 平均 | −0.12% | 3／0／4 |
| 300×6 mixed | SoA12 lane roots d2 | pipeline p99 | −0.12% | 3／0／4 |

小模型 data 的 AoS16 roots 核心平均降低中位數 3.19%、6/7 次改善，但它在主實驗仍不是比 `cold_compact4_cost8` 更好的通用配置。小模型 code 的同一預取則核心平均及 pipeline 平均都 7/7 次變慢。

大模型 mixed AoS16 roots 的核心平均 7/7 次變慢（增加 0.16–5.54%，中位 1.50%），pipeline 平均沒有一致改善。SoA12 roots 的 pipeline p99 只有 3/7 次改善，中位退步 0.12%。補測沒有確認主實驗局部的尾端優勢能穩定重現。

## 結論與限制

預先知道 roots／下一節點並不足以保證 prefetch 有益；額外指令、code size、排程或 cache 壓力可能抵銷收益，本輪沒有 PMU 資料可將原因拆開。因此保留實驗 API，預設 none；下一輪優先探索 leaf 分離表示或 caller／feature producer 融合。

這只涵蓋兩個合成模型、四種基礎 lowering 與指定距離／locality。尚未測 production feature producer 內提前預取、profile-guided 多層子樹預取或其他 CPU，不能宣稱所有 prefetch 可能性都已排除。桌面背景、頻率與尾端波動仍存在；不將不同輪的絕對 ns 相減當成新增收益，也不將軟體干擾稱為已確認的 cache flush。
