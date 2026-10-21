# ansoncat/ds4 —— 本機補丁

[English](FORK.md)（以英文版為主，這份是翻譯，兩者不一致時以英文版為準）

這是 [antirez/ds4](https://github.com/antirez/ds4) 的個人 fork，用來在 **64 GB 的 M5 Pro MacBook** 上跑
Qwen3.8-Flash-Next Q2（262K context，供 Claude Code 經 llama-swap 使用）。補丁都只針對這個模型與 Metal，沒有送回上游。
量測與決策紀錄在另一個私人 repo。

## 分支

| 分支 | 用途 |
|---|---|
| `kv-grow` | **平常在用的版本**，以上游 `0aaea5a`（2026-09-20）為底，下面的 commit 依序疊上 |
| `opt-1154` | 上游 PR #1154（prefill 時預讀 PLE）＋#1062 的 acce8da（MTP 預熱）移植。輸出相同、速度沒差（本機是記憶體頻寬瓶頸），只留作紀錄，沒有併進 `kv-grow` |
| 其他 | 建 fork 當下的上游分支，沒動過 |

## `kv-grow` 上的改動

| commit | 內容 |
|---|---|
| d0d204c | attention cache 隨 context 成長（Metal），不必一開始就配滿 262K |
| 6851c1e | token 表不放進 Metal 的模型範圍，少釘一塊記憶體 |
| 35da5e9 | 讀 Q2_K down experts 時去掉列尾 padding（搭配 `gguf-tools/qwen4_q2k640.py` 重排的檔） |
| 737717f | indexer 原始 key 只留到所屬區塊 pool 完為止（環狀緩衝區）。磁碟 KV checkpoint 也只存未 pool 區塊的 key，並換了 payload tag，所以 `.kv` 檔和上游不通用 |
| 5fe6907 | indexer score 暫存區照 cache 實際列數配置 |
| a097c1e | `ds4_gpu_tensor_read`／`_write` 加 autorelease pool（不加的話，磁碟 KV checkpoint 碰過的 buffer 放不掉） |
| 0a3d8ef | down experts 讀成 640 寬的 IQ2_XXS 列（私有 tensor type 41；檔案用 `gguf-tools/qwen4_down_iq2.py convert` 產生）|
| e23581a | `gguf-tools/qwen4_dense_q4.py`：dense 轉 Q4_0。**實測沒用**（走通用 matmul 反而慢、agentic 時間 +84%），只留工具 |
| 5d7c601 | qwen4 dense 檢查接受 Q4_K，改走 GLM 的 Q4_K dense matvec；`gguf-tools/qwen4_dense_q4k.py` |
| 26968d9 | 2–4 token 的 Q4_K matvec（MTP 驗證時權重只讀一次）；`gguf-tools/qwen4_hc_q8.py`（每層的 hyper-connection 低秩矩陣 F16→Q8_0）；`tests/bench_qwen4_dense.c` |
| 07e6470 | `DS4_QWEN4_HC_Q8_GEMM=1`：讓 Q8_0 的 `hc_*_up` 在 prefill 也走 GEMM |
| c3b08cb | IQ2_XXS routed expert 的 decode kernel（`kernel_qwen4_moe_{mid,down}_iq2_*`）：activation 讀一次供 gate／up 與多個 row 共用、down 的權重讀取全部提前。輸出與通用路徑逐位元相同；decode +8–16%。M5 預設開，`DS4_QWEN4_MOE_IQ2=0` 關；`tests/bench_qwen4_moe.c` |
| 901e0f7 | indexer 分數與 top-k 選取改成分段算，一次只算 64 MiB 暫存放得下的列數（`DS4_QWEN4_IDX_SCORE_MB`）；原本整個 chunk 每個 token 一列（262K、chunk 2048 時 0.5 GiB）。輸出逐字相同、prefill 不變；灌滿峰值 −0.63 GiB |

## 目前跑法（檔案 2026-10-19 起；expert kernel、indexer 分段、`--prefill-chunk 2048` 2026-10-21 起）

轉好的檔案放在 Hugging Face：
[ansoncat/qwen3.8-flash-next-ds4-64gb-mac-gguf](https://huggingface.co/ansoncat/qwen3.8-flash-next-ds4-64gb-mac-gguf)
（`Qwen3.8-Flash-Next-Q2-IQ2down-Q4Kdense-Q8hc.gguf`）。

### 編譯與啟動

```sh
make ds4-server
M=~/models/ds4-gguf
DS4_METAL_NO_RESIDENCY=1 DS4_QWEN4_HC_Q8_GEMM=1 ./ds4-server \
  -m $M/Qwen3.8-Flash-Next-Q2-IQ2down-Q4Kdense-Q8hc.gguf --vision $M/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --ctx 262144 --prefill-chunk 2048 --mtp \
  --kv-disk-dir ~/.ds4/server-kv/Q2-IQ2down-Q4Kdense-Q8hc --kv-disk-space-mb 102400
```
- `DS4_QWEN4_HC_Q8_GEMM=1` **必帶**：hc 是 Q8_0 的檔沒有它，prefill 會掉 62%。
- `DS4_METAL_NO_RESIDENCY=1`：閒置時不釘住模型（wired 42 → 4 GiB），速度不變。
- `--vision` 可省略；投影檔來自 [ggml-org/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF)。
- 這台機器用 `sudo sysctl iogpu.wired_limit_mb=57344` 調高了 GPU wired 記憶體上限（短對話約 38 GiB、262K 灌滿約 45 GiB）。
  64 GB 機器用 macOS 預設值時，長 context 可能配置失敗；把 `--ctx` 調小或調高上限。
- 每個模型檔用自己的 `--kv-disk-dir` 子目錄：checkpoint 只記 model id 與 routed experts 的量化位元數
  （官方 Q2 與所有衍生檔都是 2），分不出不同的轉檔版本。
- 私有格式（type 41、Q4_K dense）**上游 ds4 讀不了**；這個分支仍讀得了官方原檔。

### 自己轉檔

從官方 Q2 經三步有損轉換。第 1 步從原始 BF16 權重搭配 Unsloth 的 importance matrix 重新量化 down experts；
第 2、3 步是對檔案裡現有的權重做 round-to-nearest（RTN）：
```
Qwen3.8-Flash-Next-Q2.gguf ─qwen4_down_iq2.py convert─▶ …-Q2-downiq2.gguf
                           ─qwen4_dense_q4k.py────────▶ …-Q2-downiq2-denseq4k.gguf
                           ─qwen4_hc_q8.py────────────▶ …-Q2-IQ2down-Q4Kdense-Q8hc.gguf
```
Python 要有 `numpy` 與 `huggingface_hub`（`--selftest` 另外要 `gguf`）。在 repo 根目錄執行：
```sh
M=~/models/ds4-gguf
DS4_GGUF_DIR=$M ./download_model.sh qwen38-q2    # 輸入檔，來自 antirez/qwen3.8-flash-next-gguf（137 GiB）
python3 gguf-tools/qwen4_down_iq2.py convert --model $M/Qwen3.8-Flash-Next-Q2.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2.gguf
python3 gguf-tools/qwen4_dense_q4k.py --model $M/Qwen3.8-Flash-Next-Q2-downiq2.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k.gguf
python3 gguf-tools/qwen4_hc_q8.py     --model $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k.gguf --out $M/Qwen3.8-Flash-Next-Q2-IQ2down-Q4Kdense-Q8hc.gguf
```
- 第 1 步會經 `huggingface_hub` 下載 Qwen/Qwen3.8-Flash-Next 的 BF16 down 投影分片與 Unsloth 的 imatrix（約 75 GiB），做完自己清掉中間檔。
- 每一步寫一個 130–140 GiB 的檔，寫完若剩不到 16 GiB 就拒絕開始。下一步的檔寫好後，上一步的輸入就可以刪，
  所以高峰約是兩個模型檔加上第 1 步的下載量。
- 某一步失敗時，先刪掉 `<out>.incomplete` 再重跑（第 1 步會從 `<out>.work` 接續）。
- 少了 `gguf-tools/libds4quants.dylib` 時，三支工具都會自己跑 `make -C gguf-tools quants-shared`。
- 三步都驗證過：用這個分支的工具重轉，輸出和線上檔逐位元組相同（第 1 步 2026-10-12、第 2–3 步 2026-10-19）。

## 實測（M5 Pro 64 GB）

每一步都和前一步在同一場次對照（這台機器的速度跨場次會漂移）。

### 記憶體（整機 wired，`--ctx 262144 --prefill-chunk 1024`）

| 步驟 | 約 9K token 的對話 | 262K 灌滿 |
|---|---|---|
| 上游 ds4＋官方 Q2 | 262K 的 KV（約 8 GiB）一用到就整塊 wire | 55.0–55.7 GiB |
| d0d204c attention cache 隨 context 成長 | 47.5 GiB | 不變 |
| 6851c1e token 表移出 Metal 範圍 | 46.2 | 54.1 |
| 35da5e9 down experts 去列尾 padding | 43.7 | 51.6 |
| 737717f＋5fe6907 indexer 環狀緩衝區 | 43.5 | 50.2 |
| 0a3d8ef down experts 改 IQ2_XXS（`IQ2down`） | 40.7 | 47.2 |
| 5d7c601–07e6470＋最終檔（`Q4Kdense`、`Q8hc`） | 約 38 | 44.1（扣掉閒置基線 2.65 後 41.4；上一列扣掉 3.77 後 43.4） |

- 閒置（模型已載入、沒有請求）：`DS4_METAL_NO_RESIDENCY=1` 讓 wired 從 42.2 降到 4.0 GiB。
- `--prefill-chunk 2048`（上面的指令，2026-10-21 起）讓 prefill 峰值多 0.6–1.0 GiB，context 越長越多：
  7K token 的 prompt 37.6 → 38.2、30K 38.1 → 38.8、62K 38.9 → 39.7、127K 40.7 → 41.6、262K 灌滿 44.3 → 45.2 GiB（1024 → 2048）。
  901e0f7（indexer 分數分段）把 2048 在長 context 的增量收回大半：127K 41.56 → 41.35、262K 灌滿 45.24 → 44.61 GiB；
  64K 以下不變。輸出逐字相同、prefill 不變。
- context：用 `--prefill-chunk 1024` 的話，上游在這台機器上也塞得下 262K（swap 0–35 MB）；
  這個分支在灌滿時多留約 8 GiB 餘裕，一般長度的對話餘裕更多。

### 速度（tokens／秒）

最終設定、單人、`--mtp`、沒命中 prompt cache：

| | decode | prefill |
|---|---|---|
| 8K token 的 prompt | 51 | 550 |
| 32K token 的 prompt | 51 | 563 |
| 16K token 的程式碼 prompt | 47 | 564 |
| 短 prompt（程式碼／散文／開 thinking） | 57／45／45 | — |
| 262K 灌滿（260K token 的 prompt，c3b08cb 之前） | 35 | 545（整段平均） |

- 前四列是 2026-10-21 對 07e6470 的同場次對照（同一個檔，A B B A A B B A，中位數）：decode 8K 46.8 → 51.2、
  32K 45.6 → 50.8、16K 程式碼 43.2 → 47.4、程式碼 49.7 → 57.4、散文 41.4 → 44.6、開 thinking 39.2 → 45.4；
  prefill 不變（prefill 走別的 kernel）。greedy 輸出逐字相同。
- 灌滿那一列是 2026-10-20 另一場次量的（`IQ2down` 在同條件是 prefill 469、decode 約 28，跨場次，不算乾淨的對照）。
- 對上一版檔案（只有 `IQ2down`）：長 context decode +10–13%（8K 41.0 → 46.1、32K 40.7 → 44.8、16K 程式碼 37.4 → 42.2），
  prefill 不變。那一輪短 prompt 雜訊太大，不引用。
- 上面那些記憶體補丁都不影響速度（輸出逐字相同、decode 差在 ±0.5% 內）。
- prefill 那欄是 `--prefill-chunk 1024` 量的。2048 快 7–12%（2026-10-21，同一個 binary，各 4 次取中位數，另一場次）：
  8K 624 → 670、32K 538 → 593、128K 518 → 569、254K 499 → 557（首字等待 510 → 457 秒）。
- agentic（Claude Code，3 題各 3 次）：每一步都 9/9；最終檔對 `IQ2down` 牆鐘 −5%。
- 這台機器上 decode 時 GPU 約 96% 時間在忙，所以藏 SSD 延遲、提高 MTP 接受率都沒效果。c3b08cb 之前兩顆 IQ2_XXS
  expert kernel 佔每步約 40%、頻寬只有 53–77 GB/s（dense Q4_K 約 250）；這次的加速來自把它們寫有效率，不是少讀位元組。

### 品質

| | 官方 Q2 | `IQ2down` | 最終檔 |
|---|---|---|---|
| agentic（3 題各 3 次） | 9/9 | 9/9 | 9/9 |
| HumanEval+ base／plus（164 題，greedy，thinking 關） | 156／152 | 154／149 | 156／151 |
| BFCL simple_python＋multiple（600 題，greedy，thinking 關） | 522 | 492 | 516 |
| 繁中回答裡的簡體字比例 | 0.42% | 0.40% | 與 `IQ2down` 差在雜訊內 |

會掉品質的是 `IQ2down` 那一步（thinking 關時 BFCL −30、顯著；thinking 開時 556 對 567、不顯著）。
這些 benchmark 不涵蓋真實的長 session。
