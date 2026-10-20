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

## Measured (M5 Pro 64 GB)

Every step was measured against the step before it in the same session (the speed of this machine drifts between sessions).

### Memory (whole-machine wired memory, `--ctx 262144 --prefill-chunk 1024`)

| Step | ~9K-token conversation | Full 262K context |
|---|---|---|
| Upstream ds4 + official Q2 | the full 262K KV (about 8 GiB) is wired as soon as it is used | 55.0–55.7 GiB |
| d0d204c attention caches grow with the context | 47.5 GiB | unchanged |
| 6851c1e token table out of the Metal view | 46.2 | 54.1 |
| 35da5e9 down experts without row padding | 43.7 | 51.6 |
| 737717f + 5fe6907 indexer ring buffer | 43.5 | 50.2 |
| 0a3d8ef down experts as IQ2_XXS (`IQ2down`) | 40.7 | 47.2 |
| 5d7c601–07e6470 + final file (`Q4Kdense`, `Q8hc`) | about 38 | not measured |

- Idle (model loaded, no request): 42.2 → 4.0 GiB with `DS4_METAL_NO_RESIDENCY=1`.
- Context: with `--prefill-chunk 1024`, upstream also fits a full 262K context on this machine (0–35 MB of swap);
  this branch leaves about 8 GiB more headroom at full context and much more in normal-length conversations.

### Speed (tokens per second)

Final setup, single user, `--mtp`, prompt cache not hit:

| | Decode | Prefill |
|---|---|---|
| 8K-token prompt | 46 | 528 |
| 32K-token prompt | 45 | 544 |
| 16K-token code prompt | 42 | — |
| Short prompts (code / prose / with thinking) | 43 / 35 / 36 | — |
| Full 262K context (measured before the last three commits) | 30–31 | about 480 |

- Changes against the previous file (`IQ2down` only): long-context decode +10–13% (8K 41.0 → 46.1, 32K 40.7 → 44.8,
  16K code 37.4 → 42.2), prefill unchanged. Short prompts are too noisy in that round to quote.
- The memory patches above leave speed unchanged (outputs identical, decode within ±0.5%). `--prefill-chunk 1024`
  instead of 2048 trades about 10% of prefill speed for about 1 GiB of memory.
- Agentic coding (Claude Code, 3 fixtures × 3 runs): 9/9 in every step; wall time −5% for the final file against `IQ2down`.
- On this machine ds4 decode is memory-bandwidth bound: hiding SSD latency or raising MTP acceptance does nothing;
  only reading fewer bytes per token makes it faster.

### Quality

| | official Q2 | `IQ2down` | final file |
|---|---|---|---|
| Agentic coding (3 fixtures × 3 runs) | 9/9 | 9/9 | 9/9 |
| HumanEval+ base / plus (of 164, greedy, thinking off) | 156 / 152 | 154 / 149 | 156 / 151 |
| BFCL simple_python + multiple (of 600, greedy, thinking off) | 522 | 492 | 516 |
| Simplified characters in Traditional Chinese answers | 0.42% | 0.40% | within noise of `IQ2down` |

`IQ2down` is the step that costs quality (BFCL −30, significant with thinking off; with thinking on 556 vs 567, not significant).
These benchmarks do not cover long real-world sessions.
