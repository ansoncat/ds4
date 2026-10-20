/* Decode-shape dense matvec bandwidth: Flash-Next's Q8_0 matvec against the Q4_K dense path (llm-infra TODO 96).
 *
 * The dense Q4_K model reads 1.64 GiB less per token than Q8_0 yet decodes only ~10% faster.  This times the two
 * kernels qwen4_gemv_rows picks on the model's own dense shapes (n_tok 1 = plain decode, 3 = MTP verify) with the
 * weights resident, so the gap between bytes saved and time saved can be read per shape.  Weights are random but
 * well formed (finite fp16 scales); timing does not depend on the values.
 *
 *   make ds4_metal.o ds4_image.o && cc -O2 -Wall -o tests/bench_qwen4_dense tests/bench_qwen4_dense.c ds4_metal.o ds4_image.o \
 *     -lm -lpthread -framework Foundation -framework Metal && ./tests/bench_qwen4_dense
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "../ds4_gpu.h"

#include <stdbool.h>
/* ds4_metal.o logs through this (the kernel test defines it the same way) */
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

enum { T_Q8_0 = 8, T_Q4_K = 12 };

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static uint64_t row_bytes(int type, uint64_t in) { return type == T_Q8_0 ? in / 32 * 34 : in / 256 * 144; }

/* random quants, scales set to a small finite fp16 (0x2000 = 2^-7) */
static void fill(uint8_t *w, int type, uint64_t rows, uint64_t in) {
    const uint64_t n = rows * row_bytes(type, in);
    for (uint64_t i = 0; i < n; i++) w[i] = (uint8_t)(rand() & 0x3f);
    const uint64_t blk = type == T_Q8_0 ? 34 : 144, nb = n / blk;
    for (uint64_t b = 0; b < nb; b++) {
        uint16_t h = 0x2000;
        memcpy(w + b * blk, &h, 2);
        if (type == T_Q4_K) memcpy(w + b * blk + 2, &h, 2);   /* d and dmin */
    }
}

int main(void) {
    static const struct { const char *name; uint64_t in, out, per_token; } S[] = {
        { "attn_qkv",    2560,  10240, 36 }, { "attn_gate",   2560,  6144, 36 }, { "ssm_out",     6144, 2560, 36 },
        { "attn_q",      2560,  12288, 12 }, { "attn_output", 6144,  2560, 12 }, { "output",      2560, 248320, 1 },
    };
    const int types[2] = { T_Q8_0, T_Q4_K };
    const uint64_t arena_size = (uint64_t)2 << 30;
    uint8_t *arena = mmap(NULL, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (arena == MAP_FAILED || !ds4_gpu_init() || !ds4_gpu_set_model_map(arena, arena_size)) { fprintf(stderr, "init failed\n"); return 1; }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(3u * 6144 * sizeof(float));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc(3u * 248320 * sizeof(float));
    float xs[3 * 6144];
    for (int i = 0; i < 3 * 6144; i++) xs[i] = (float)((i % 97) - 48) / 64.0f;
    if (!x || !y || !ds4_gpu_tensor_write(x, 0, xs, sizeof(xs))) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* correctness first: Q4_K with 3 tokens at once (multi-token kernel) == the same 3 tokens one at a time */
    for (size_t s = 0; s < sizeof(S) / sizeof(S[0]); s++) {
        const uint64_t in = S[s].in, out = S[s].out, bytes = out * row_bytes(T_Q4_K, in);
        fill(arena, T_Q4_K, out, in);
        float *a = malloc(3 * out * sizeof(float)), *b = malloc(3 * out * sizeof(float));
        if (!ds4_gpu_matmul_quant_tensor(y, arena, arena_size, 0, T_Q4_K, in, out, x, 3) ||
            !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(y, 0, a, 3 * out * sizeof(float))) return 1;
        for (int t = 0; t < 3; t++) {
            ds4_gpu_tensor *xv = ds4_gpu_tensor_view(x, (uint64_t)t * in * sizeof(float), in * sizeof(float));
            if (!xv || !ds4_gpu_matmul_quant_tensor(y, arena, arena_size, 0, T_Q4_K, in, out, xv, 1) ||
                !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(y, 0, b + (uint64_t)t * out, out * sizeof(float))) return 1;
            ds4_gpu_tensor_free(xv);
        }
        double maxd = 0, maxv = 0;
        for (uint64_t i = 0; i < 3 * out; i++) { maxd = fmax(maxd, fabs(a[i] - b[i])); maxv = fmax(maxv, fabs(b[i])); }
        printf("check %-12s 3 tokens at once vs one by one: max |diff| %.3g (max |value| %.3g)\n", S[s].name, maxd, maxv);
        if (!(maxd <= 1e-4 * maxv)) { fprintf(stderr, "MISMATCH\n"); return 2; }
        free(a); free(b); (void)bytes;
    }
    printf("\n");

    printf("%-12s %5s %5s | %9s %9s | %9s %9s | %s\n", "shape", "in", "n_tok", "Q8 us", "Q8 GB/s", "Q4K us", "Q4K GB/s", "Q4K/Q8 time");
    double tok_ms[2][2] = {{0}};   /* [n_tok index][type] */
    for (size_t s = 0; s < sizeof(S) / sizeof(S[0]); s++) {
        for (int ti = 0; ti < 2; ti++) {
            const uint32_t n_tok = ti ? 3u : 1u;
            double us[2];
            for (int k = 0; k < 2; k++) {
                const int type = types[k];
                const uint64_t bytes = S[s].out * row_bytes(type, S[s].in);
                /* enough copies that the set is far larger than the SLC: every call streams from DRAM */
                const uint64_t copies = bytes * 8 < arena_size ? 8 : 1;
                for (uint64_t c = 0; c < copies; c++) fill(arena + c * bytes, type, S[s].out, S[s].in);
                const int iters = 64;
                for (int pass = 0; pass < 2; pass++) {   /* pass 0 warms the pipelines */
                    if (!ds4_gpu_begin_commands()) return 1;
                    const double t0 = now();
                    for (int i = 0; i < iters; i++) {
                        const uint64_t off = (uint64_t)(i % copies) * bytes;
                        const int ok = type == T_Q8_0
                            ? ds4_gpu_qwen4_matmul_q8_0_tensor(y, arena, arena_size, off, S[s].in, S[s].out, x, n_tok)
                            : ds4_gpu_matmul_quant_tensor(y, arena, arena_size, off, type, S[s].in, S[s].out, x, n_tok);
                        if (!ok) { fprintf(stderr, "%s type %d failed\n", S[s].name, type); return 1; }
                    }
                    if (!ds4_gpu_end_commands()) return 1;
                    us[k] = (now() - t0) * 1e6 / iters;
                }
                tok_ms[ti][k] += us[k] * S[s].per_token / 1e3;
            }
            const double b8 = S[s].out * row_bytes(T_Q8_0, S[s].in), b4 = S[s].out * row_bytes(T_Q4_K, S[s].in);
            printf("%-12s %5llu %5u | %9.1f %9.0f | %9.1f %9.0f | %.2f\n", S[s].name, (unsigned long long)S[s].in, n_tok,
                   us[0], b8 / us[0] / 1e3, us[1], b4 / us[1] / 1e3, us[1] / us[0]);
        }
    }
    printf("\nper token over these shapes (x per-layer counts): n_tok 1: Q8 %.2f ms, Q4K %.2f ms; n_tok 3: Q8 %.2f ms, Q4K %.2f ms\n",
           tok_ms[0][0], tok_ms[0][1], tok_ms[1][0], tok_ms[1][1]);
    return 0;
}
