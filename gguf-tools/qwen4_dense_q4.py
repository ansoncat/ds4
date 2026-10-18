#!/usr/bin/env python3
"""Requantize a qwen4exp model's Q8_0 dense tensors as Q4_0 (2026-10-18, llm-infra TODO 96).

Decode reads every dense tensor once per token, and in the Q2 pack those are Q8_0: 5.65 GiB per token
against 3.75 GiB for Strata's GSQ-RCO Q2_0, whose dense tensors are 4-5 bit (that ratio matches the measured
1.55x decode gap).  The qwen4 dense validator and Metal matmul accept Q4_0, so this keeps the layout and halves
those bytes.  Q4_0 is ggml's reference round-to-nearest (quantize_row_q4_0_ref), computed from the Q8_0 values
(no BF16 source, no imatrix): a speed probe first, quality to be measured.  The MTP layer (blk.48) stays Q8_0.

  python3 gguf-tools/qwen4_dense_q4.py --model Q2-downiq2.gguf --out Q2-downiq2-denseq4.gguf [--work DIR]
  python3 gguf-tools/qwen4_dense_q4.py --selftest
"""

import argparse
from pathlib import Path
import re
import shutil

import numpy as np

from qwen4_pack_to_qwen4exp import Reader
from qwen4_q2k640 import repack

Q8_0, Q4_0 = 8, 2
SKIP = re.compile(r'blk\.48\.')   # the MTP layer


def q8_0_to_f32(raw):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


def f32_to_q4_0(x):
    """ggml quantize_row_q4_0_ref: d = (signed value of max |x|) / -8, q = min(15, (int8)(x/d + 8.5))."""
    x = x.reshape(-1, 32).astype(np.float32)
    idx = np.abs(x).argmax(axis=1)
    d = x[np.arange(len(x)), idx] / -8.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    q = np.minimum(15, np.trunc(x * inv[:, None] + 8.5).astype(np.int8)).astype(np.uint8)
    out = np.empty((len(x), 18), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()


def q4_0_to_f32(raw):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    lo, hi = (b[:, 2:] & 15).astype(np.int8) - 8, (b[:, 2:] >> 4).astype(np.int8) - 8
    return (np.concatenate([lo, hi], axis=1).astype(np.float32) * d).reshape(-1)


def selftest():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32 * 64).astype(np.float32)
    x[5] = -7.0                                        # the signed max sets d's sign: -7 / -8 > 0
    y = q4_0_to_f32(f32_to_q4_0(x))
    blocks = x.reshape(-1, 32); err = np.abs(y - x).reshape(-1, 32)
    step = np.abs(blocks).max(axis=1, keepdims=True) / 8   # |d|; values on the far side of the max clamp at +7*d,
    assert np.all(err <= step * 1.01 + 1e-6), 'error above one step'   # so ggml's Q4_0 errs up to one step there
    assert abs(y[5] + 7.0) < 1e-3                      # the max maps to -8 * d exactly
    assert q4_0_to_f32(f32_to_q4_0(np.zeros(32, np.float32))).max() == 0
    q8 = np.zeros((1, 34), np.uint8); q8[0, :2] = np.array([0.5], np.float16).view(np.uint8); q8[0, 2:] = np.arange(32).astype(np.int8).view(np.uint8)
    assert np.allclose(q8_0_to_f32(q8.tobytes()), np.arange(32) * 0.5)
    try:   # independent decoder, if llama.cpp's gguf-py is on the path
        from gguf import quants, GGMLQuantizationType
        ref = quants.dequantize(np.frombuffer(f32_to_q4_0(x), np.uint8).reshape(-1, 18), GGMLQuantizationType.Q4_0).reshape(-1)
        assert np.allclose(ref, y), 'gguf-py decodes our Q4_0 differently'
        print('gguf-py Q4_0 decode matches')
    except ImportError:
        pass
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path); ap.add_argument('--out', type=Path); ap.add_argument('--work', type=Path)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    work = a.work or Path(str(a.out) + '.work'); work.mkdir(parents=True, exist_ok=True)
    m = Reader(str(a.model)); swap = {}; before = after = 0
    try:
        for name, (kind, dims, old) in m.tensors.items():
            if kind != Q8_0 or SKIP.match(name):
                continue
            n = int(np.prod(dims)); size = n // 32 * 34
            m.f.seek(m.data_start + old); raw = m.f.read(size)
            q = f32_to_q4_0(q8_0_to_f32(raw))
            p = work / name; p.write_bytes(q)
            swap[name] = (Q4_0, list(dims), p); before += size; after += len(q)
    finally:
        m.f.close()
    print(f'{len(swap)} tensors Q8_0 -> Q4_0: {before / 2**30:.2f} -> {after / 2**30:.2f} GiB', flush=True)
    repack(a.model, a.out, swap)
    shutil.rmtree(work)


if __name__ == '__main__':
    main()
