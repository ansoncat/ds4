#!/usr/bin/env python3
"""Requantize a qwen4exp model's F16 hyper-connection low-rank tensors as Q8_0 (2026-10-19, llm-infra TODO 96).

After the dense Q4_K step, decode still reads about 1.2 GiB of F16 per token from blk.N.hc_{attn,ffn}_{down,up}
([10240,320] and [320,10240], 49 layers).  The Qwen3.8 graph already accepts Q8_0 for them: hc up goes to
kernel_qwen4_hc_gate_mix_q8 (the graph's type check allows F16/F32/Q8_0 for hc up and inject) and hc down is an
ordinary dense projection (qwen4_gemv).  Q8_0 halves those bytes; it is the conservative choice (upstream ships
them F16).  Left as they are: hc inject [10240,4], output_hc_*, the nextn hc head, everything else.

  python3 gguf-tools/qwen4_hc_q8.py --model IN.gguf --out OUT.gguf [--work DIR]
  python3 gguf-tools/qwen4_hc_q8.py --selftest
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

F16, Q8_0 = 1, 8
PICK = re.compile(r'blk\.[0-9]+\.hc_(attn|ffn)_(down|up)\.weight$')
LIB = Path(__file__).with_name('libds4quants.dylib')


def quantizer():
    lib = ctypes.CDLL(str(LIB))
    lib.ds4q_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
    lib.ds4q_quantize_chunk.restype = ctypes.c_size_t

    def f32_to_q8_0(x, ncols):
        x = np.ascontiguousarray(x.reshape(-1, ncols), dtype=np.float32)
        out = np.empty(x.size // 32 * 34, dtype=np.uint8)
        n = lib.ds4q_quantize_chunk(Q8_0, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.ctypes.data,
                                    0, x.shape[0], ncols, None)
        assert n == out.size, (n, out.size)
        return out.tobytes()
    return f32_to_q8_0


def selftest():
    enc = quantizer()
    x = np.random.default_rng(0).standard_normal((4, 320)).astype(np.float16).astype(np.float32)
    y = q8_0_to_f32(enc(x, 320)).reshape(4, 320)
    step = np.abs(x).reshape(-1, 32).max(axis=1, keepdims=True) / 127
    # half a step, plus the scale's own fp16 rounding (measured 0.526 here)
    assert np.all(np.abs(y - x).reshape(-1, 32) <= step * 0.55 + 1e-6), 'Q8_0 error above half a step'
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path); ap.add_argument('--out', type=Path); ap.add_argument('--work', type=Path)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    enc = quantizer()
    work = a.work or Path(str(a.out) + '.work'); work.mkdir(parents=True, exist_ok=True)
    m = Reader(str(a.model)); swap = {}; before = after = 0
    try:
        for name, (kind, dims, old) in m.tensors.items():
            if kind != F16 or not PICK.match(name):
                continue
            assert dims[0] % 32 == 0, (name, dims)
            n = int(np.prod(dims))
            m.f.seek(m.data_start + old); raw = m.f.read(n * 2)
            q = enc(np.frombuffer(raw, '<f2').astype(np.float32), dims[0])
            p = work / name; p.write_bytes(q)
            swap[name] = (Q8_0, list(dims), p); before += n * 2; after += len(q)
    finally:
        m.f.close()
    print(f'{len(swap)} tensors F16 -> Q8_0: {before / 2**30:.2f} -> {after / 2**30:.2f} GiB', flush=True)
    repack(a.model, a.out, swap)
    shutil.rmtree(work)


if __name__ == '__main__':
    main()
