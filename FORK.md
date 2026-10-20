# ansoncat/ds4 — local patches

[繁體中文版](FORK.zh-TW.md)

A personal fork of [antirez/ds4](https://github.com/antirez/ds4) for running Qwen3.8-Flash-Next Q2 on a
**64 GB M5 Pro MacBook** (262K context, served to Claude Code through llama-swap). The patches target this model
and the Metal backend only and have not been sent upstream. Measurements and decisions live in a separate private repo.

## Branches

| Branch | Purpose |
|---|---|
| `kv-grow` | **The version in daily use.** Upstream `0aaea5a` (2026-09-20) plus the commits below, in order |
| `opt-1154` | Port of upstream PR #1154 (PLE prefetch during prefill) and acce8da from #1062 (MTP priming). Same output, no measurable speedup (this machine is memory-bandwidth bound); kept for reference, not merged into `kv-grow` |
| others | Unmodified upstream branches as of fork creation |

## Changes on `kv-grow`

| Commit | What it does |
|---|---|
| d0d204c | Attention caches grow with the context (Metal) instead of being allocated for 262K up front |
| 6851c1e | Keeps the token table out of the Metal model view, so less memory is pinned |
| 35da5e9 | Reads Q2_K down experts without their row padding (files repacked by `gguf-tools/qwen4_q2k640.py`) |
| 737717f | Keeps raw indexer keys only until their block is pooled (ring buffer). Disk KV checkpoints store only the open block's keys, with a new payload tag, so `.kv` files are not interchangeable with upstream's |
| 5fe6907 | Sizes the indexer score scratch for the rows the caches actually hold |
| a097c1e | Autorelease pool in `ds4_gpu_tensor_read`/`_write` (without it, buffers touched by disk KV checkpoints are never freed) |
| 0a3d8ef | Reads down experts as 640-wide IQ2_XXS rows (private tensor type 41; files made by `gguf-tools/qwen4_down_iq2.py convert`) |
| e23581a | `gguf-tools/qwen4_dense_q4.py`: dense tensors as Q4_0. **Not useful in practice** (takes the generic matmul, slower; agentic wall time +84%); only the tool is kept |
| 5d7c601 | The qwen4 dense validator accepts Q4_K, which then uses GLM's Q4_K dense matvec; `gguf-tools/qwen4_dense_q4k.py` |
| 26968d9 | Q4_K matvec for 2–4 tokens (MTP verify reads the weights once); `gguf-tools/qwen4_hc_q8.py` (per-layer hyper-connection low-rank tensors F16 → Q8_0); `tests/bench_qwen4_dense.c` |
| 07e6470 | `DS4_QWEN4_HC_Q8_GEMM=1` routes Q8_0 `hc_*_up` tensors through the prefill GEMM path as well |

## Current setup (since 2026-10-19)

The converted file is on Hugging Face:
[ansoncat/qwen3.8-flash-next-ds4-64gb-mac-gguf](https://huggingface.co/ansoncat/qwen3.8-flash-next-ds4-64gb-mac-gguf)
(`Qwen3.8-Flash-Next-Q2-IQ2down-Q4Kdense-Q8hc.gguf`).

### Build and run

```sh
make ds4-server
M=~/models/ds4-gguf
DS4_METAL_NO_RESIDENCY=1 DS4_QWEN4_HC_Q8_GEMM=1 ./ds4-server \
  -m $M/Qwen3.8-Flash-Next-Q2-IQ2down-Q4Kdense-Q8hc.gguf --vision $M/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --ctx 262144 --prefill-chunk 1024 --mtp \
  --kv-disk-dir ~/.ds4/server-kv/Q2-IQ2down-Q4Kdense-Q8hc --kv-disk-space-mb 102400
```
- `DS4_QWEN4_HC_Q8_GEMM=1` is **required**: with Q8_0 hc tensors and without it, prefill drops by 62%.
- `DS4_METAL_NO_RESIDENCY=1`: the model is not pinned while idle (wired memory 42 → 4 GiB), with no speed cost.
- `--vision` is optional; the projector comes from [ggml-org/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF).
- This machine raises the GPU wired-memory limit with `sudo sysctl iogpu.wired_limit_mb=57344`
  (about 41 GiB in use while computing, 47 GiB with a full 262K context). With the macOS default on 64 GB,
  long contexts may fail to allocate; use a smaller `--ctx` or raise the limit.
- Give each model file its own `--kv-disk-dir` subdirectory: a checkpoint records only the model id and the routed
  experts' quantization bits (2 for the official Q2 and every file derived from it), so it cannot tell conversions apart.
- The private formats (type 41, Q4_K dense) **cannot be read by upstream ds4**; this branch still reads the official files.

### Converting it yourself

Three lossy steps from the official Q2. Step 1 re-quantizes the down experts from the original BF16 weights with
Unsloth's importance matrix; steps 2 and 3 round to nearest from the weights already in the file:
```
Qwen3.8-Flash-Next-Q2.gguf ─qwen4_down_iq2.py convert─▶ …-Q2-downiq2.gguf
                           ─qwen4_dense_q4k.py────────▶ …-Q2-downiq2-denseq4k.gguf
                           ─qwen4_hc_q8.py────────────▶ …-Q2-IQ2down-Q4Kdense-Q8hc.gguf
```
Python needs `numpy` and `huggingface_hub` (plus `gguf` for `--selftest`). Run from the repo root:
```sh
M=~/models/ds4-gguf
DS4_GGUF_DIR=$M ./download_model.sh qwen38-q2    # the input, from antirez/qwen3.8-flash-next-gguf (137 GiB)
python3 gguf-tools/qwen4_down_iq2.py convert --model $M/Qwen3.8-Flash-Next-Q2.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2.gguf
python3 gguf-tools/qwen4_dense_q4k.py --model $M/Qwen3.8-Flash-Next-Q2-downiq2.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k.gguf
python3 gguf-tools/qwen4_hc_q8.py     --model $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k.gguf --out $M/Qwen3.8-Flash-Next-Q2-IQ2down-Q4Kdense-Q8hc.gguf
```
- Step 1 downloads the BF16 down-projection shards from Qwen/Qwen3.8-Flash-Next and the imatrix from Unsloth
  (about 75 GiB, via `huggingface_hub`), and deletes its own intermediate files when done.
- Each step writes a 130–140 GiB file and refuses to start unless 16 GiB stay free afterwards. The input of a step can be
  deleted once the next file is written, so the peak is about two model files plus step 1's download.
- If a step fails, delete `<out>.incomplete` before re-running (step 1 resumes from `<out>.work`).
- Each tool runs `make -C gguf-tools quants-shared` itself if `gguf-tools/libds4quants.dylib` is missing.
- All three steps are verified: re-running them with this branch's tools reproduces the files in use byte for byte
  (step 1 on 2026-10-12, steps 2–3 on 2026-10-19).

## Measured (M5 Pro 64 GB, same-session comparisons)

- The last three commits with the final file, against `0a3d8ef` with the downiq2 file: long-context decode +10–13%,
  prefill unchanged, about 2 GiB less wired memory; agentic 9/9 vs 9/9, HumanEval+ 151 vs 149, BFCL 516 vs 492, and the
  rate of Simplified characters leaking into Traditional Chinese answers within noise.
- On this machine ds4 decode is memory-bandwidth bound: hiding SSD latency or raising MTP acceptance does nothing;
  only reading fewer bytes per token makes it faster.
