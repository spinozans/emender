// e97_quantize_q4_refcheck.c — dumps quantize_row_q4_0_ref output for a raw
// f32 file so scripts/e97_quantize_q4_0.py can prove its numpy codec is
// byte-identical to the pinned fork's C reference.
//
// Build+run is orchestrated by the python script (g++ against the fork's
// build-plain static ggml). Usage: refcheck <x.f32> <out.bin>

#include "ggml.h"

// block_q4_0 and the quantize_row_*_ref prototypes live in the internal
// ggml headers (not the public ggml.h); pull the common block declarations in
// the C++ flavor since this TU is compiled with g++.
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-quants.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: refcheck <x.f32> <out.bin>\n"); return 1; }
    FILE * fx = fopen(argv[1], "rb");
    FILE * fo = fopen(argv[2], "wb");
    if (!fx || !fo) { fprintf(stderr, "open failed\n"); return 1; }
    std::vector<float> x;
    float v;
    while (fread(&v, sizeof(float), 1, fx) == 1) x.push_back(v);
    const int64_t n = (int64_t) x.size();
    const int64_t nb = n / 32;
    std::vector<uint8_t> out(nb * 18);
    quantize_row_q4_0_ref(x.data(), (block_q4_0 *) out.data(), n);
    fwrite(out.data(), 1, out.size(), fo);
    fclose(fx); fclose(fo);
    printf("quantized %lld values -> %lld blocks (%lld bytes)\n", (long long) n, (long long) nb, (long long) out.size());
    return 0;
}
