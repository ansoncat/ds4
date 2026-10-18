#!/usr/bin/env python3
"""Requantize a qwen4exp model's Q8_0 dense tensors as Q4_K (2026-10-19, llm-infra TODO 96).

qwen4_dense_q4.py's Q4_0 halved the dense bytes but decoded slower: Q4_0 takes the generic mul_mv_ext path
(~220 GB/s on M5).  Q4_K takes the classic dense matvec ds4 built for GLM DenseQ4 (kernel_mul_mv_q4_K_dense_f32,
530-650 GB/s) once the qwen4 dense validator accepts it (branch dense-q4k).  Same 4.5 bits per weight as Q4_0.

Kept Q8_0: the MTP layer (blk.48, as in qwen4_dense_q4.py), token_embd (read row by row on the host by
qwen4_ref_row, which has no Q4_K decoder), and any tensor whose rows are not a multiple of 256 (Q4_K superblock).
Quantizer: gguf-tools/quants.c (make quants-shared), round-to-nearest unless --imatrix is given (not yet wired).

  python3 gguf-tools/qwen4_dense_q4k.py --model Q2-downiq2.gguf --out Q2-downiq2-denseq4k.gguf [--work DIR]
  python3 gguf-tools/qwen4_dense_q4k.py --selftest
"""

import argparse
import ctypes
from pathlib import Path
import re
import shutil

import numpy as np

from qwen4_pack_to_qwen4exp import Reader
from qwen4_q2k640 import repack
from qwen4_dense_q4 import q8_0_to_f32

Q8_0, Q4_K = 8, 12
SKIP = re.compile(r'blk\.48\.|token_embd\.')
LIB = Path(__file__).with_name('libds4quants.dylib')


def quantizer():
    lib = ctypes.CDLL(str(LIB))
    lib.ds4q_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
    lib.ds4q_quantize_chunk.restype = ctypes.c_size_t

    def f32_to_q4_k(x, ncols):
        x = np.ascontiguousarray(x.reshape(-1, ncols), dtype=np.float32)
        out = np.empty(x.shape[0] * ncols // 256 * 144, dtype=np.uint8)
        n = lib.ds4q_quantize_chunk(Q4_K, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.ctypes.data,
                                    0, x.shape[0], ncols, None)
        assert n == out.size, (n, out.size)
        return out.tobytes()
    return f32_to_q4_k


def q4_k_to_f32(raw):
    """Independent decoder: llama.cpp's gguf-py."""
    from gguf import quants, GGMLQuantizationType
    return quants.dequantize(np.frombuffer(raw, np.uint8).reshape(-1, 144), GGMLQuantizationType.Q4_K).reshape(-1)


def selftest():
    enc = quantizer()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 512)).astype(np.float32)
    raw = enc(x, 512)
    assert len(raw) == 8 * 2 * 144
    y = q4_k_to_f32(raw).reshape(8, 512)
    err = np.abs(y - x).reshape(-1, 32); step = np.abs(x).reshape(-1, 32).max(axis=1, keepdims=True) / 7
    assert np.all(err <= step * 1.5 + 1e-3), 'Q4_K error above 1.5 steps of its 32-weight sub-block'
    rel = np.linalg.norm(y - x) / np.linalg.norm(x)
    assert rel < 0.12, rel                                  # 4.5 bpw on Gaussian data: ~0.1
    print(f'selftest ok (relative error {rel:.3f}, gguf-py decodes our Q4_K)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path); ap.add_argument('--out', type=Path); ap.add_argument('--work', type=Path)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    enc = quantizer()
    work = a.work or Path(str(a.out) + '.work'); work.mkdir(parents=True, exist_ok=True)
    m = Reader(str(a.model)); swap = {}; before = after = 0; kept = []
    try:
        for name, (kind, dims, old) in m.tensors.items():
            if kind != Q8_0:
                continue
            if SKIP.match(name) or dims[0] % 256:
                kept.append(name); continue
            n = int(np.prod(dims)); size = n // 32 * 34
            m.f.seek(m.data_start + old); raw = m.f.read(size)
            q = enc(q8_0_to_f32(raw), dims[0])
            p = work / name; p.write_bytes(q)
            swap[name] = (Q4_K, list(dims), p); before += size; after += len(q)
    finally:
        m.f.close()
    print(f'{len(swap)} tensors Q8_0 -> Q4_K: {before / 2**30:.2f} -> {after / 2**30:.2f} GiB; '
          f'kept Q8_0: {len(kept)} ({", ".join(sorted({re.sub(r"blk[.][0-9]+[.]", "blk.N.", k) for k in kept}))})', flush=True)
    repack(a.model, a.out, swap)
    shutil.rmtree(work)


if __name__ == '__main__':
    main()
