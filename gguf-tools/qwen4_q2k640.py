#!/usr/bin/env python3
"""Repack Qwen Q2_K down experts without their unused padding.

Q2_K stores a 640-wide row as three 256-value blocks and half of the third is
never read.  Tensor type 40 keeps that block's 8 scales, 32 qs bytes, d and
dmin: 212-byte rows instead of 252.  Every other tensor is copied unchanged."""

import argparse
import fcntl
import io
import math
import os
from pathlib import Path
import re
import shutil
import struct

import numpy as np

from qwen4_native_ngrams import BLOCKS, NGRAM, copy_hash
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

SIZES = dict(BLOCKS)
SIZES[40] = (640, 212)
SIZES[41] = (640, 166)   # qwen4_down_iq2.py
DOWN = re.compile(r'blk\.\d+\.ffn_down_exps\.weight')


def size_of(kind, dims):
    block, size = SIZES[kind]
    if not dims or dims[0] % block:
        raise ValueError('Invalid tensor dimensions or block alignment')
    return math.prod(dims) // block * size


def pack(raw):
    """252-byte rows -> 212: blocks 0-1, then block 2's scales[0:8], qs[0:32], d, dmin."""
    rows = np.frombuffer(raw, np.uint8).reshape(-1, 252)
    return np.concatenate([rows[:, :176], rows[:, 184:216], rows[:, 248:]], axis=1).tobytes()


def nocache(fp):
    if hasattr(fcntl, 'F_NOCACHE'):   # 147 GB through the page cache evicts everything else
        fcntl.fcntl(fp.fileno(), fcntl.F_NOCACHE, 1)


def repack(source, output, swap=None):
    """swap: tensor name -> (type, dims, file holding its payload), written instead of the source tensor."""
    swap = swap or {}
    row = bytes(range(252))
    assert pack(row * 2) == (row[:176] + row[184:216] + row[248:]) * 2
    pending = Path(str(output) + '.incomplete')
    if output.exists() or pending.exists():
        raise ValueError('Output already exists; choose a new path or inspect the incomplete file')
    model = Reader(str(source))
    try:
        if model.kv.get('general.architecture') != 'qwen4exp':
            raise ValueError('Expected a qwen4exp GGUF')
        alignment = model.kv.get('general.alignment', 32)
        align = lambda n, a=alignment: (n + a - 1) // a * a
        plan, offset, packed = [], 0, 0   # name, kind, dims, new offset, old size, old offset, pack?
        for name, (kind, dims, old) in model.tensors.items():
            if name == NGRAM:
                continue
            size, new_kind, new_dims = size_of(kind, dims), kind, dims
            pad = DOWN.fullmatch(name) and kind == 10 and dims[0] == 768 and name not in swap
            if name in swap:
                new_kind, new_dims = swap[name][:2]
            elif pad:
                new_kind, new_dims, packed = 40, [640] + dims[1:], packed + 1
            plan.append((name, new_kind, new_dims, offset, size, old, pad))
            offset = align(offset + size_of(new_kind, new_dims))
        if not (packed or swap) or NGRAM not in model.tensors:
            raise ValueError('No padded Q2_K down experts or no n-gram table to repack')
        ngram_kind, ngram_dims, ngram_old = model.tensors[NGRAM]
        ngram_size = size_of(ngram_kind, ngram_dims)

        prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan) + 1, len(model.kv))
        prefix += b''.join(kv_bytes(k, model.kv_types[k], v) for k, v in model.kv.items())

        def header(ngram_offset):
            data = bytearray(prefix)
            for name, kind, dims, pos in [p[:4] for p in plan] + [(NGRAM, ngram_kind, ngram_dims, ngram_offset)]:
                data += w_str(name) + struct.pack('<I', len(dims))
                data += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, pos)
            return data

        data_start = align(len(header(0)))
        table_start = align(data_start + offset, max(65536, alignment))
        total = table_start + ngram_size
        if model.data_start + ngram_old + ngram_size > source.stat().st_size:
            raise ValueError('Truncated input')
        if shutil.disk_usage(output.parent).free < total + (16 << 30):
            raise ValueError('Not enough disk space with a 16 GiB reserve')
        nocache(model.f)
        records = []
        with pending.open('xb') as dst:
            nocache(dst)
            dst.write(header(table_start - data_start))
            for name, kind, dims, pos, size, old, pad in plan:
                dst.seek(data_start + pos)
                model.f.seek(model.data_start + old)
                src = model.f
                if name in swap:
                    size = size_of(kind, dims)
                    if swap[name][2].stat().st_size != size:
                        raise ValueError('Replacement payload has the wrong size: ' + name)
                    src = swap[name][2].open('rb')
                elif pad:
                    raw = model.f.read(size)
                    if len(raw) != size:
                        raise ValueError('Truncated tensor payload')
                    src, size = io.BytesIO(pack(raw)), size // 252 * 212
                records.append((name, data_start + pos, size, copy_hash(src, dst, size)))
            print('Copying n-grams', flush=True)
            dst.seek(table_start)
            model.f.seek(model.data_start + ngram_old)
            records.append((NGRAM, table_start, ngram_size, copy_hash(model.f, dst, ngram_size)))
            if dst.tell() != total:
                raise ValueError('Incorrect assembled size')
            dst.flush()
            os.fsync(dst.fileno())
        print('Verifying every output tensor', flush=True)
        with pending.open('rb') as check:
            nocache(check)
            for name, pos, size, digest in records:
                check.seek(pos)
                if copy_hash(check, None, size) != digest:
                    raise ValueError('Output payload verification failed: ' + name)
        pending.rename(output)
        print(f'Verified {output}: {total} bytes, {packed} tensors packed, {len(swap)} replaced, '
              f'{source.stat().st_size - total} bytes smaller', flush=True)
    finally:
        model.f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        repack(args.model, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
