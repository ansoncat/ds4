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
| others | Copied from upstream when the fork was created; untouched |

## Changes on `kv-grow`

| Commit | What it does |
|---|---|
| d0d204c | Attention caches grow with the context (Metal) instead of being allocated for 262K up front |
| 6851c1e | Keeps the token table out of the Metal model view, so less memory is pinned |
| 35da5e9 | Reads Q2_K down experts without their row padding (files repacked by `gguf-tools/qwen4_q2k640.py`) |
| 737717f | Keeps raw indexer keys only until their block is pooled (ring buffer) |
| 5fe6907 | Sizes the indexer score scratch for the rows the caches actually hold |
| a097c1e | Autorelease pool in `ds4_gpu_tensor_read`/`_write` (without it, buffers touched by disk KV checkpoints are never freed) |
| 0a3d8ef | Reads down experts as 640-wide IQ2_XXS rows (private tensor type 41; files made by `gguf-tools/qwen4_down_iq2.py convert`) |
| e23581a | `gguf-tools/qwen4_dense_q4.py`: dense tensors as Q4_0. **Not useful in practice** (takes the generic matmul, slower; agentic wall time +84%); only the tool is kept |
| 5d7c601 | The qwen4 dense validator accepts Q4_K, which then uses GLM's Q4_K dense matvec; `gguf-tools/qwen4_dense_q4k.py` |
| 26968d9 | Q4_K matvec for 2–4 tokens (MTP verify reads the weights once); `gguf-tools/qwen4_hc_q8.py` (hyper-connection low-rank tensors F16 → Q8_0); `tests/bench_qwen4_dense.c` |
| 07e6470 | `DS4_QWEN4_HC_Q8_GEMM=1`: lets Q8_0 hc up tensors take the prefill GEMM too |

## Current setup (since 2026-10-19)

The model file is derived from the official Q2 in three steps (round-to-nearest, lossy):
```
Q2.gguf ──qwen4_down_iq2.py convert──▶ Q2-downiq2.gguf
        ──qwen4_dense_q4k.py────────▶ Q2-downiq2-denseq4k.gguf
        ──qwen4_hc_q8.py────────────▶ Q2-downiq2-denseq4k-hcq8.gguf   ← the one in use
```
Commands (run from the repo root; Python needs numpy, and `--selftest` also needs gguf-py):
```sh
make -C gguf-tools quants-shared          # builds gguf-tools/libds4quants.dylib, the quantizer all three tools use
M=~/models/ds4-gguf
python3 gguf-tools/qwen4_down_iq2.py convert --model $M/Qwen3.8-Flash-Next-Q2.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2.gguf
python3 gguf-tools/qwen4_dense_q4k.py --model $M/Qwen3.8-Flash-Next-Q2-downiq2.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k.gguf
python3 gguf-tools/qwen4_hc_q8.py     --model $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k.gguf --out $M/Qwen3.8-Flash-Next-Q2-downiq2-denseq4k-hcq8.gguf
```
The first step downloads the BF16 down_proj tensors and an imatrix (about 75 GiB) and cleans up after itself; the other two each
write a new 130–140 GiB file, and the intermediate `denseq4k` file can be deleted afterwards.
All three steps are verified: re-running them with this branch's tools reproduces the files in use byte for byte
(step 1 on 2026-10-12, steps 2–3 on 2026-10-19, about 9 minutes each). Each tool runs `make` itself if `libds4quants.dylib` is missing.

```sh
DS4_METAL_NO_RESIDENCY=1 DS4_QWEN4_HC_Q8_GEMM=1 ./ds4-server \
  -m Qwen3.8-Flash-Next-Q2-downiq2-denseq4k-hcq8.gguf --vision mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --ctx 262144 --prefill-chunk 1024 --mtp \
  --kv-disk-dir ~/.ds4/server-kv/Q2-downiq2-denseq4k-hcq8 --kv-disk-space-mb 102400
```
- `DS4_QWEN4_HC_Q8_GEMM=1` is **required**: with Q8_0 hc tensors and without it, prefill drops by 62%.
- `DS4_METAL_NO_RESIDENCY=1`: the model is not pinned while idle (wired memory 42 → 4 GiB), with no speed cost.
- Give each model file its own `--kv-disk-dir` subdirectory: checkpoints are keyed only by model id and quantization bits,
  so they cannot tell different conversions apart.
- The private formats (type 41, Q4_K dense) **cannot be read by upstream ds4**; this branch still reads the official files.

## Measured (M5 Pro 64 GB, same-session comparisons)

- The last three commits with the hcq8 file, against `0a3d8ef` with the downiq2 file: long-context decode +10–13%, prefill unchanged,
  about 2 GiB less wired memory; agentic 9/9 vs 9/9, HumanEval+ 151 vs 149, BFCL 516 vs 492, and the rate of Simplified
  characters leaking into Traditional Chinese answers within noise.
- On this machine ds4 decode is memory-bandwidth bound: hiding SSD latency or raising MTP acceptance does nothing;
  only reading fewer bytes per token makes it faster.
