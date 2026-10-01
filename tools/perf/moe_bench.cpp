// Isolated T=1 MoE matvec cost for q2_0 experts at this model's shapes (512 experts, n_embd 2560, n_ff 640):
// MUL_MAT_ID gate/up (2560 -> 640) and down (640 -> 2560) with n_used ids, R chained per graph so the per-call
// figure includes the in-order queue cost a layer graph pays. Kernel choice follows the env
// (GGML_SYCL_MOE_Q2W=1 for the wide kernel).
//   moe_bench BACKEND_DIR [iters]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static const int64_t N_EMBD = 2560, N_FF = 640, N_EXP = 512;
static const int R = 16;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s BACKEND_DIR [iters]\n", argv[0]); return 1; }
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no GPU\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    std::mt19937 rng(7);

    for (int n_used : { 10, 2 }) {
        for (int down = 0; down <= 1; down++) {
            const int64_t cols = down ? N_FF : N_EMBD, rows = down ? N_EMBD : N_FF;
            ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * wctx = ggml_init(wp);
            ggml_tensor * w = ggml_new_tensor_3d(wctx, GGML_TYPE_Q2_0, cols, rows, N_EXP);
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
            {   // random quant bytes are fine for timing; keep the fp16 scales finite
                std::vector<uint8_t> blob(ggml_nbytes(w));
                for (auto & b : blob) b = (uint8_t) rng();
                const size_t bs = ggml_type_size(GGML_TYPE_Q2_0);
                for (size_t off = 0; off + bs <= blob.size(); off += bs) { blob[off] = 0x00; blob[off + 1] = 0x3c; }   // d = 1.0
                ggml_backend_tensor_set(w, blob.data(), 0, blob.size());
            }
            ggml_init_params gp = { ggml_tensor_overhead() * (6 * R + 8) + ggml_graph_overhead(), nullptr, true };
            ggml_context * gctx = ggml_init(gp);
            ggml_tensor * x   = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, cols, down ? n_used : 1, 1); ggml_set_input(x);
            // a disjoint expert set per chained call: every call reads its weights from memory, as in the engine
            std::vector<ggml_tensor *> idss;
            ggml_cgraph * gf = ggml_new_graph(gctx);
            for (int r = 0; r < R; r++) {
                ggml_tensor * ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, n_used, 1); ggml_set_input(ids);
                idss.push_back(ids);
                ggml_build_forward_expand(gf, ggml_mul_mat_id(gctx, w, x, ids));
            }
            ggml_gallocr_t ga = ggml_gallocr_new(buft);
            if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("alloc failed\n"); return 1; }
            std::vector<float> xv((size_t) ggml_nelements(x)); std::normal_distribution<float> nd(0, 1);
            for (auto & v : xv) v = nd(rng);
            ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
            std::vector<int32_t> iv(n_used); std::vector<int> perm(N_EXP);
            for (int e = 0; e < N_EXP; e++) perm[e] = e;
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int r = 0; r < R; r++) {
                for (int k = 0; k < n_used; k++) iv[k] = perm[(r * n_used + k) % N_EXP];
                ggml_backend_tensor_set(idss[r], iv.data(), 0, ggml_nbytes(idss[r]));
            }
            for (int i = 0; i < 10; i++) ggml_backend_graph_compute(be, gf);
            double best = 1e30;
            for (int rep = 0; rep < 5; rep++) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, gf);
                best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters / R);
            }
            const double mb = (double) ggml_nbytes(w) / N_EXP * n_used / 1e6;
            printf("moebench q2_0 %-4s n_used=%-2d %7.1f us/call  (%.2f MB of weights, %4.0f GB/s)\n", down ? "down" : "gate",
                   n_used, best, mb, mb / best * 1e3);
            fflush(stdout);
            ggml_gallocr_free(ga); ggml_free(gctx); ggml_backend_buffer_free(wbuf); ggml_free(wctx);
        }
    }
    ggml_backend_free(be);
    return 0;
}
