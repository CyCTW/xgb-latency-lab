# Ubuntu 重現與 PMU 狀態

## 歷史連線問題

2026-09-13 本機 Apple container 1.4.1 服務可回應，`ubuntu24` 回報 running、ARM64、4 vCPU／8 GB，但 `container machine run -n ubuntu24 uname -a` 在 15 秒內未回應。直接 exec 也未成功。獨立 `ubuntu:24.04` 探測容器在 Starting container 停留 30 秒後逾時。沒有 Linux 測量結果，也沒有重啟既有 machine／服務。`container list --all` 只列出原有 buildkit，沒有本次實驗容器。

自動核准先前因用量限制拒絕程序檢查；超過重設時間後，同一項檢查已獲准完成。當時的阻塞是容器啟動／命令執行無回應；2026-09-14 已恢復，見下節。

## 2026-09-14 重新探測

`container machine run -n ubuntu24 uname -a` 已成功，kernel 6.18.35、aarch64。以實際保存的 [benchmarks/linux_pmu_probe.py](benchmarks/linux_pmu_probe.py) 在 UID 501 執行，task-clock 計數成功，hardware cycles／instructions 都回傳 errno 2（ENOENT），`/sys/bus/event_source/devices` 只有 software／uprobe／breakpoint／tracepoint／kprobe。`perf_event_paranoid=2`，perf／clang 尚未安裝。

可重現結果與探測程式 SHA256：[reports/ubuntu-pmu-readiness-20260914.json](reports/ubuntu-pmu-readiness-20260914.json)。這是環境可用性檢查，不是 inference benchmark。沒有改動 sysctl、重啟服務或安裝套件。

目前 VM 沒有可用 CPU 硬體事件，不能只靠安裝 perf 或提高權限補齊。要完成硬體分析，需要實際部署的 Linux 主機，或提供 vPMU 的 VM；先執行 `python3 benchmarks/linux_pmu_probe.py` 確認硬體事件成功，再確認該 CPU 支援的 cache／branch 事件。參考 [perf_event_open 手冊](https://www.man7.org/linux/man-pages/man2/perf_event_open.2.html)。

硬體分析尚需在原生 harness 加入量測區間控制；直接對整個 runner 執行 perf stat 會計入 oracle、載入與 feature 工作。須分開報告完整 pipeline 與推論區間，量化計數控制開銷，並以獨立的無計數執行取得延遲。不要為了降低單筆計數開銷而連續重複 inference，否則會失去每筆都先做其他工作的 cold 情境。

## Linux 建置與效能實驗

在 Ubuntu 內使用獨立 Linux venv，勿使用共享目錄的 macOS `.venv`：

```sh
sudo apt-get update
sudo apt-get install -y clang llvm llvm-dev lld python3-venv libgomp1
python3 -m venv /tmp/xgb-linux-venv
. /tmp/xgb-linux-venv/bin/activate
cd '/Users/cyctw/Documents/ChatGPT/ultra low latency xgboost inference engine'
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
uname -a
lscpu
clang --version
llvm-profdata --version
```

先確認套件可安裝、Clang LTO 可連結，以及 llvm-profdata 與 Clang 版本相容。共享家目錄路徑來自目前的 machine 設定；獨立容器可將專案掛載到 `/workspace`。以上安裝尚未在 Ubuntu 執行。

使用本次已保存且與 calibration 分離的輸入；輸出目錄必須不存在：

```sh
python -m benchmarks.optimize \
  --model results/controlled-100x4/model.json \
  --calibration results/controlled-100x4/calibration.npy \
  --tuning results/optimization-inputs/100x4-explore-tuning.npy \
  --evaluation results/optimization-inputs/100x4-explore-evaluation.npy \
  --output results/exploration-ubuntu-100x4 --preset explore --seed 13760
```

大模型改用 `300x6`。Linux 結果需重新選模，不能直接載入 macOS `.dylib` 或將 M3/macOS 排名當成 Linux 排名。ARM64 VM 也不代表 x86-64 裸機結果。

若 VM 支援 `perf stat`，可以對原生 harness 計數 cycles／instructions／branches／branch-misses／cache-misses。先固定 CPU affinity 並記錄 VM 配置；PMU 不可用時保留錯誤，不更改主機安全設定。獨立計時與硬體計數分開執行。
