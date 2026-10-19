# ansoncat/ds4 —— 本機補丁

[English](FORK.md)（以英文版為主，這份是翻譯，兩者不一致時以英文版為準）

這是 [antirez/ds4](https://github.com/antirez/ds4) 的個人 fork，用來在 **64 GB 的 M5 Pro MacBook** 上跑
Qwen3.8-Flash-Next Q2（262K context，供 Claude Code 經 llama-swap 使用）。補丁都只針對這個模型與 Metal，沒有送回上游。
量測與決策紀錄在另一個私人 repo（llm-infra 的 `notes/TODO.md` TODO 96、`llm-bench/results/_ds4_opt/report.md`）。

## 分支

| 分支 | 用途 |
|---|---|
| `kv-grow` | **線上在跑的版本**，以上游 `0aaea5a`（2026-09-20）為底，下面的 commit 依序疊上 |
| `opt-1154` | 上游 PR #1154（prefill 時預讀 PLE）＋#1062 的 acce8da（MTP 預熱）移植。輸出相同、速度沒差（本機是記憶體頻寬瓶頸），只留作紀錄，沒有併進 `kv-grow` |
| 其他 | GitHub 建 fork 時從上游複製來的，沒動過 |

## `kv-grow` 上的改動

| commit | 內容 |
|---|---|
| d0d204c | attention cache 隨 context 成長（Metal），不必一開始就配滿 262K |
| 6851c1e | token 表不放進 Metal 的模型範圍，少釘一塊記憶體 |
| 35da5e9 | 讀 Q2_K down experts 時去掉列尾 padding（搭配 `gguf-tools/qwen4_q2k640.py` 重排的檔） |
| 737717f | indexer 原始 key 只留到所屬區塊 pool 完為止（環狀） |
| 5fe6907 | indexer score 暫存區照 cache 實際列數配置 |
| a097c1e | `ds4_gpu_tensor_read`／`_write` 加 autorelease pool（不加的話，磁碟 KV checkpoint 碰過的 buffer 放不掉） |
| 0a3d8ef | down experts 讀成 640 寬的 IQ2_XXS 列（私有 tensor type 41；檔案用 `gguf-tools/qwen4_down_iq2.py convert` 產生）|
| e23581a | `gguf-tools/qwen4_dense_q4.py`：dense 轉 Q4_0。**實測沒用**（走通用 matmul 反而慢、agentic 時間 +84%），只留工具 |
| 5d7c601 | qwen4 dense 白名單接受 Q4_K，改走 GLM 的 Q4_K dense matvec；`gguf-tools/qwen4_dense_q4k.py` |
| 26968d9 | 2–4 token 的 Q4_K matvec（MTP 驗證時權重只讀一次）；`gguf-tools/qwen4_hc_q8.py`（hyper-connection 低秩 F16→Q8_0）；`tests/bench_qwen4_dense.c` |
| 07e6470 | `DS4_QWEN4_HC_Q8_GEMM=1`：讓 Q8_0 的 hc up 在 prefill 也走 GEMM |

## 線上設定（2026-10-19 起）

模型檔從官方 Q2 依序轉出（RTN，有損）：
```
Q2.gguf ──qwen4_down_iq2.py convert──▶ Q2-downiq2.gguf
        ──qwen4_dense_q4k.py────────▶ Q2-downiq2-denseq4k.gguf
        ──qwen4_hc_q8.py────────────▶ Q2-downiq2-denseq4k-hcq8.gguf   ← 線上用這個
```
轉好的檔案放在 Hugging Face：
[ansoncat/qwen3.8-flash-next-ds4-gguf](https://huggingface.co/ansoncat/qwen3.8-flash-next-ds4-gguf)
（`Qwen3.8-Flash-Next-Q2-IQ2down-Q4Kdense-Q8hc.gguf`，和下面的 `Q2-downiq2-denseq4k-hcq8.gguf` 逐位元組相同）。

要自己轉的完整指令（在 repo 根目錄執行；Python 要有 numpy，`--selftest` 另外要 gguf-py）：
```sh
make -C gguf-tools quants-shared          # 產生 gguf-tools/libds4quants.dylib（量化器，三支工具共用）
M=~/models/ds4-gguf
python3 gguf-tools/qwen4_down_iq2.py convert --model $M/Qwen3.8-Flash-Next-Q2.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2.gguf
python3 gguf-tools/qwen4_dense_q4k.py --model $M/Qwen3.8-Flash-Next-Q2-downiq2.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k.gguf
python3 gguf-tools/qwen4_hc_q8.py     --model $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k-hcq8.gguf
```
第一步會下載 BF16 down_proj 與 imatrix（約 75 GiB），做完自己清掉中間檔；後兩步各寫一個約 130–140 GiB 的新檔，中間的 `denseq4k` 檔做完可以刪。
三步都驗證過：用這個分支的工具重轉，輸出和線上檔逐位元組相同（第一步 2026-10-12、後兩步 2026-10-19，各約 9 分鐘）。少了 `libds4quants.dylib` 時三支工具都會自己 `make`。

```sh
DS4_METAL_NO_RESIDENCY=1 DS4_QWEN4_HC_Q8_GEMM=1 ./ds4-server \
  -m Qwen3.8-Flash-Next-Q2-downiq2-denseq4k-hcq8.gguf --vision mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --ctx 262144 --prefill-chunk 1024 --mtp \
  --kv-disk-dir ~/.ds4/server-kv/Q2-downiq2-denseq4k-hcq8 --kv-disk-space-mb 102400
```
- `DS4_QWEN4_HC_Q8_GEMM=1` **必帶**：hc 是 Q8_0 的檔沒有它，prefill 會掉 62%。
- `DS4_METAL_NO_RESIDENCY=1`：閒置時不釘住模型（wired 42 → 4 GiB），速度不變。
- `--kv-disk-dir` 每個模型檔用自己的子目錄：checkpoint 只認 model id 與量化位元數，分不出不同的轉檔版本。
- 私有格式（type 41、Q4_K dense）**上游 ds4 讀不了**；這個分支仍讀得了官方原檔。

## 實測摘要（M5 Pro 64 GB，同場次對照）

- 最後三個 commit 加上 hcq8 檔，對比 `0a3d8ef`＋downiq2 檔：長 context decode +10–13%、prefill 不變、wired 少約 2 GiB；
  agentic 9/9 對 9/9、HumanEval+ 151 對 149、BFCL 516 對 492、繁中簡體字比例差別在雜訊內。
- 這台機器上 ds4 decode 卡在記憶體頻寬：藏 SSD 延遲、提高 MTP 接受率都沒效果，只有每個 token 少讀位元組才會變快。
