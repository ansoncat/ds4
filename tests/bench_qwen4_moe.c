/* Decode-shape routed expert kernels on Flash-Next's C' layout (llm-infra TODO 96): IQ2_XXS gate/up 2560->640,
 * 640-wide IQ2_XXS down 640->2560 (type 41), shared expert Q4_K gate/up + Q8_0 down in the same dispatch.
 *
 * The decode profile has kernel_qwen4_moe_mid / _down at ~40% of a verify step yet only 53-77 GB/s.  This times
 * moe_mid + moe_down for n_tok 1 and 2 (MTP verify) under each env variant and checks every variant's output is
 * bit-identical to the default dispatch.  Expert picks rotate over 64 random sets so calls stream from DRAM.
 *
 *   make ds4_metal.o ds4_image.o && cc -O2 -Wall -o tests/bench_qwen4_moe tests/bench_qwen4_moe.c ds4_metal.o ds4_image.o \
 *     -lm -lpthread -framework Foundation -framework Metal && ./tests/bench_qwen4_moe
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "../ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

enum { E = 2560, F = 640, NE = 512, SLOTS = 10, NSETS = 64 };
enum { GU_ROW = E / 256 * 66, DN_ROW = 166, SH_GU_ROW = E / 256 * 144, SH_DN_ROW = F / 32 * 34 };

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* random quants; every block's fp16 scale(s) set to 2^-7 */
static void fill(uint8_t *w, uint64_t rows, uint64_t row, uint64_t blk, int dmin) {
    for (uint64_t i = 0; i < rows * row; i++) w[i] = (uint8_t)rand();
    const uint16_t h = 0x2000;
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t o = 0; o < row; o += blk) {
            memcpy(w + r * row + o, &h, 2);
            if (dmin) memcpy(w + r * row + o + 2, &h, 2);
        }
}

static const char *VARS[] = { "", "DS4_QWEN4_MOE_MV_SPECIALIZE=1", "DS4_QWEN4_MOE_MV_SPECIALIZE=1 DS4_QWEN4_MOE_MV_NSG=4",
                              "DS4_QWEN4_MOE_MV_SPECIALIZE=1 DS4_QWEN4_MOE_MV_NSG=16",
                              "DS4_QWEN4_MOE_MV_SPECIALIZE=1 DS4_QWEN4_MOE_MV_NR=2 DS4_QWEN4_MOE_MV_NSG=4",
                              "DS4_QWEN4_MOE_IQ2=1",
                              "DS4_QWEN4_MOE_IQ2=1 DS4_QWEN4_MOE_IQ2_MID_NR=1 DS4_QWEN4_MOE_IQ2_DOWN_NR=1",
                              "DS4_QWEN4_MOE_IQ2=1 DS4_QWEN4_MOE_IQ2_MID_NR=4 DS4_QWEN4_MOE_IQ2_DOWN_NR=4",
                              "DS4_QWEN4_MOE_IQ2=1 DS4_QWEN4_MOE_IQ2_MID_NR=1 DS4_QWEN4_MOE_IQ2_DOWN_NR=8",
                              "DS4_QWEN4_MOE_IQ2=1 DS4_QWEN4_MOE_IQ2_MID_NR=1 DS4_QWEN4_MOE_IQ2_DOWN_NR=4 DS4_QWEN4_MOE_IQ2_DOWN_NSG=8",
                              "DS4_QWEN4_MOE_IQ2=1 DS4_QWEN4_MOE_IQ2_MID_NSG=2 DS4_QWEN4_MOE_IQ2_DOWN_NSG=2",
                              "DS4_QWEN4_MOE_IQ2=1 DS4_QWEN4_MOE_IQ2_MID_NSG=8 DS4_QWEN4_MOE_IQ2_DOWN_NSG=8" };

static void set_env(const char *spec) {
    unsetenv("DS4_QWEN4_MOE_MV_SPECIALIZE"); unsetenv("DS4_QWEN4_MOE_MV_NSG"); unsetenv("DS4_QWEN4_MOE_MV_NR");
    const char *iq2v[] = { "DS4_QWEN4_MOE_IQ2_MID_NR", "DS4_QWEN4_MOE_IQ2_MID_NSG", "DS4_QWEN4_MOE_IQ2_DOWN_NR", "DS4_QWEN4_MOE_IQ2_DOWN_NSG" };
    for (int i = 0; i < 4; i++) unsetenv(iq2v[i]);
    setenv("DS4_QWEN4_MOE_IQ2", "0", 1);   /* the reference is the generic dispatch */
    char buf[256]; snprintf(buf, sizeof buf, "%s", spec);
    for (char *t = strtok(buf, " "); t; t = strtok(NULL, " ")) { char *eq = strchr(t, '='); *eq = 0; setenv(t, eq + 1, 1); }
}

int main(int argc, char **argv) {
    const uint64_t gu = (uint64_t)NE * F * GU_ROW, dn = (uint64_t)NE * E * DN_ROW;
    const uint64_t off_g = 0, off_u = gu, off_d = 2 * gu, off_sg = off_d + dn, off_su = off_sg + (uint64_t)F * SH_GU_ROW,
                   off_sd = off_su + (uint64_t)F * SH_GU_ROW, arena_size = (uint64_t)1 << 30;
    uint8_t *arena = mmap(NULL, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (arena == MAP_FAILED || !ds4_gpu_init() || !ds4_gpu_set_model_map(arena, arena_size)) { fprintf(stderr, "init failed\n"); return 1; }
    srand(7);
    fill(arena + off_g, (uint64_t)NE * F, GU_ROW, 66, 0);
    fill(arena + off_u, (uint64_t)NE * F, GU_ROW, 66, 0);
    fill(arena + off_d, (uint64_t)NE * E, DN_ROW, 66, 0);
    fill(arena + off_sg, F, SH_GU_ROW, 144, 1);
    fill(arena + off_su, F, SH_GU_ROW, 144, 1);
    fill(arena + off_sd, E, SH_DN_ROW, 34, 0);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(8u * E * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((uint64_t)NSETS * 8 * SLOTS * sizeof(int32_t));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(8u * (SLOTS + 1) * F * sizeof(float));
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(8u * (SLOTS + 1) * E * sizeof(float));
    static float xs[8 * E];
    for (int i = 0; i < 8 * E; i++) xs[i] = (float)((rand() % 2001) - 1000) / 1000.0f;
    static int32_t ss[NSETS * 8 * SLOTS];
    for (int s = 0; s < NSETS * 8; s++)
        for (int k = 0; k < SLOTS; k++) {
            int e, dup;
            do { e = rand() % NE; dup = 0; for (int j = 0; j < k; j++) dup |= ss[s * SLOTS + j] == e; } while (dup);
            ss[s * SLOTS + k] = e;
        }
    if (!x || !sel || !mid || !part || !ds4_gpu_tensor_write(x, 0, xs, sizeof xs) || !ds4_gpu_tensor_write(sel, 0, ss, sizeof ss)) return 1;
    ds4_gpu_tensor *sv[NSETS];
    for (int s = 0; s < NSETS; s++) sv[s] = ds4_gpu_tensor_view(sel, (uint64_t)s * 8 * SLOTS * sizeof(int32_t), 8 * SLOTS * sizeof(int32_t));

#define MID(T, S) ds4_gpu_qwen4_moe_mid_tensor(mid, x, sv[S], arena, arena_size, off_g, off_u, 16u, NE, T, SLOTS, E, F, off_sg, off_su, sh ? 12u : UINT32_MAX)
#define DOWN(T, S) ds4_gpu_qwen4_moe_down_tensor(part, mid, sv[S], arena, arena_size, off_d, 41u, NE, T, SLOTS, F, E, off_sd, sh ? 8u : UINT32_MAX)
    const int nv = argc > 1 ? atoi(argv[1]) : (int)(sizeof VARS / sizeof VARS[0]);
    static const uint32_t TS[] = { 1, 2, 3, 4, 8, 4 };   /* the last runs without the shared slot (T > 8 batches have none) */
    enum { NT = sizeof TS / sizeof TS[0] };
    static float ref_mid[NT][8 * (SLOTS + 1) * F], ref_part[NT][8 * (SLOTS + 1) * E], got_mid[8 * (SLOTS + 1) * F], got_part[8 * (SLOTS + 1) * E];
    printf("%-72s %5s | %8s %8s | %8s\n", "variant", "n_tok", "mid us", "down us", "GB/s");
    for (int v = 0; v < nv; v++) {
        set_env(VARS[v]);
        for (int ti = 0; ti < NT; ti++) {
            const uint32_t T = TS[ti]; const int sh = ti < NT - 1;
            /* correctness on set 0 against the default dispatch */
            const uint64_t nm = (uint64_t)T * (SLOTS + sh) * F, np = (uint64_t)T * (SLOTS + sh) * E;
            float *cm = v ? got_mid : ref_mid[ti], *cp = v ? got_part : ref_part[ti];
            if (!MID(T, 0) || !DOWN(T, 0) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(mid, 0, cm, nm * sizeof(float)) || !ds4_gpu_tensor_read(part, 0, cp, np * sizeof(float))) {
                printf("%-72s %5u | dispatch failed\n", VARS[v][0] ? VARS[v] : "(default)", (unsigned)T); continue;
            }
            int bad = v && (memcmp(cm, ref_mid[ti], nm * sizeof(float)) || memcmp(cp, ref_part[ti], np * sizeof(float)));
            double us[2] = {0};
            for (int k = 0; k < 2; k++) {
                const int iters = 256;
                for (int pass = 0; pass < 2; pass++) {
                    if (!ds4_gpu_begin_commands()) return 1;
                    const double t0 = now();
                    for (int i = 0; i < iters; i++) if (!(k ? DOWN(T, i % NSETS) : MID(T, i % NSETS))) return 1;
                    if (!ds4_gpu_end_commands()) return 1;
                    us[k] = (now() - t0) * 1e6 / iters;
                }
            }
            const double bytes = (double)T * SLOTS * (2.0 * F * GU_ROW + (double)E * DN_ROW) + (sh ? 2.0 * F * SH_GU_ROW + (double)E * SH_DN_ROW : 0);
            printf("%-72s %4u%s | %8.1f %8.1f | %8.0f%s\n", VARS[v][0] ? VARS[v] : "(default)", T, sh ? " " : "n", us[0], us[1],
                   bytes / (us[0] + us[1]) / 1e3, bad ? "  MISMATCH" : "");
        }
    }
    return 0;
}
