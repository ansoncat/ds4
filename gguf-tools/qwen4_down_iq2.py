#!/usr/bin/env python3
"""Requantize Qwen trunk down experts as 640-wide IQ2_XXS rows.

A row is quantized as three IQ2_XXS blocks with its 640 inputs zero-padded to
768, like the Q2_K down pack, and only the bytes that are read are kept: two
blocks, then d and four of the third block's eight groups.  That is 166 bytes
per row instead of Q2_K's 212 (tensor type 41, local).

`convert` is the whole thing in one command and leaves only the output:

  python3 gguf-tools/qwen4_down_iq2.py convert \
    --model Qwen3.8-Flash-Next-Q2.gguf --out Qwen3.8-Flash-Next-Q2-downiq2.gguf

It needs NumPy, downloads 75 GiB (the BF16 down_proj of the pinned official
checkpoint, dropped layer by layer) plus the pinned imatrix, and wants room
for the output and about 20 GiB of work files next to it.  --model is any
qwen4exp Q2 file: the released one or its q2k640 repack give the same bytes.

Its two steps can be run alone.  `quantize` range-reads each layer's BF16
down_proj (kept in --bf16) and writes one tensor per layer to --experts-dir.
`assemble` copies a qwen4exp model with those tensors in place of its own."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

from qwen4_q2k640 import repack
from qwen4_pack import (QwenGGUFImatrix, GGMLQuantizer, Q2_IMATRIX_REPOSITORY, Q2_IMATRIX_REVISION,
                        Q2_IMATRIX_SHA256, encode_weighted_experts,
                        hub_file_url, read_hub_range, read_hub_safetensors_header)

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / ('libds4quants.dylib' if sys.platform == 'darwin' else 'libds4quants.so')

REPO, REVISION = 'Qwen/Qwen3.8-Flash-Next', 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
SHAPE = (512, 2560, 640)
NBYTES = 512 * 2560 * 640 * 2


def fetch(layer, directory, index):
    path = directory / f'down_proj.{layer:02d}.bf16'
    if path.is_file() and path.stat().st_size == NBYTES:
        return path
    name = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
    header, data_offset, _ = read_hub_safetensors_header(REPO, REVISION, index[name], None)
    info = header[name]
    begin, end = (data_offset + o for o in info['data_offsets'])
    if info['dtype'] != 'BF16' or tuple(info['shape']) != SHAPE or end - begin != NBYTES:
        raise SystemExit(f'Unexpected source layout: {name}: {info}')
    part = path.with_suffix('.part')
    with part.open('ab') as f:   # resumes a partial download
        pos = begin + f.tell()
        while pos < end:
            data, _ = read_hub_range(REPO, REVISION, index[name], pos, min(pos + (64 << 20), end) - 1, None)
            f.write(data)
            pos += len(data)
    part.replace(path)
    return path


def quantize(args):
    args.bf16.mkdir(parents=True, exist_ok=True)
    args.experts_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(hub_file_url(REPO, REVISION, 'model.safetensors.index.json'), timeout=120) as r:
        index = json.load(r)['weight_map']
    imatrix = QwenGGUFImatrix(args.imatrix, Q2_IMATRIX_SHA256)
    quant = GGMLQuantizer(args.library)
    quant.lib.ds4q_quantize_init(16)   # shared lookup tables, before the workers start
    manifest = args.experts_dir / 'experts.json'
    report = json.loads(manifest.read_text()) if manifest.exists() else {}
    first, last = (int(v) for v in args.layers.split(':'))
    todo = []
    for layer in range(first, last):
        name = f'blk.{layer}.ffn_down_exps.weight'
        out = args.experts_dir / name
        if not (name in report and out.is_file() and out.stat().st_size == report[name]['bytes']):
            todo.append(layer)
    # The downloads run ahead of the quantizer.  With --drop-bf16 the disk use stays near
    # --fetch-jobs layers only while the link is slower than ~130 MB/s; a faster one can
    # pile up to the full 75 GiB.
    pool = ThreadPoolExecutor(args.fetch_jobs)
    for layer, source in zip(todo, pool.map(lambda n: fetch(n, args.bf16, index), todo)):
        name = f'blk.{layer}.ffn_down_exps.weight'
        out = args.experts_dir / name
        start = time.monotonic()
        values = np.fromfile(source, dtype='<u2').reshape(SHAPE)
        raw = encode_weighted_experts(values, 'IQ2_XXS', imatrix, name, quant, args.threads, pad_last_to=768)
        rows = np.frombuffer(raw, np.uint8).reshape(512 * 2560, 198)[:, :166]
        tmp = out.with_suffix('.incomplete')
        tmp.write_bytes(rows.tobytes())
        tmp.replace(out)
        report[name] = {'sha256': hashlib.sha256(rows.tobytes()).hexdigest(), 'bytes': rows.size,
                        'source_sha256': hashlib.sha256(values).hexdigest(),
                        'seconds': round(time.monotonic() - start, 1)}
        manifest.write_text(json.dumps(report, indent=1) + '\n')
        if args.drop_bf16:
            source.unlink()
        print(time.strftime('%T'), name, report[name]['seconds'], flush=True)
    pool.shutdown()
    imatrix.close()


def assemble(args):
    report = json.loads((args.experts_dir / 'experts.json').read_text())
    swap = {}
    for layer in range(48):
        name = f'blk.{layer}.ffn_down_exps.weight'
        path = args.experts_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[name]['sha256']:
            raise SystemExit(f'{name} does not match experts.json')
        swap[name] = (41, [640, 2560, 512], path)
    repack(args.model, args.out, swap)


def convert(args):
    work = args.work or Path(str(args.out) + '.work')
    work.mkdir(parents=True, exist_ok=True)
    if not args.library.is_file():
        subprocess.run(['make', '-C', str(HERE), 'quants-shared'], check=True)
    args.imatrix = work / 'imatrix_unsloth.gguf_file'
    if not args.imatrix.is_file():   # 553 MB; QwenGGUFImatrix checks its sha256
        url = hub_file_url(Q2_IMATRIX_REPOSITORY, Q2_IMATRIX_REVISION, args.imatrix.name)
        part = args.imatrix.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=120) as r, part.open('wb') as f:
            shutil.copyfileobj(r, f, 16 << 20)
        part.replace(args.imatrix)
    args.bf16, args.experts_dir, args.layers, args.drop_bf16 = work / 'bf16', work / 'experts', '0:48', not args.keep_work
    quantize(args)
    assemble(args)
    if not args.keep_work:
        shutil.rmtree(work)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    q = sub.add_parser('quantize')
    q.add_argument('--bf16', type=Path, required=True)
    q.add_argument('--imatrix', type=Path, required=True)
    q.add_argument('--library', type=Path, default=LIBRARY)
    q.add_argument('--experts-dir', type=Path, required=True)
    q.add_argument('--threads', type=int, default=8)
    q.add_argument('--fetch-jobs', type=int, default=4)
    q.add_argument('--layers', default='0:48')
    q.add_argument('--drop-bf16', action='store_true', help='delete each BF16 layer once it is quantized')
    q.set_defaults(run=quantize)
    a = sub.add_parser('assemble')
    a.add_argument('--model', type=Path, required=True)
    a.add_argument('--experts-dir', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.set_defaults(run=assemble)
    c = sub.add_parser('convert')
    c.add_argument('--model', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--work', type=Path, help='work directory, default <out>.work')
    c.add_argument('--keep-work', action='store_true', help='keep the BF16 layers, imatrix and quantized tensors')
    c.add_argument('--library', type=Path, default=LIBRARY)
    c.add_argument('--threads', type=int, default=8)
    c.add_argument('--fetch-jobs', type=int, default=4)
    c.set_defaults(run=convert)
    args = ap.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
