// Decode (T = 1) matvec cost at this model's small-matmul shapes, per weight type.
// Each graph holds R independent matmuls with distinct weights (so nothing is L2-resident across them),
// timed as one submission: the per-matmul figure includes the in-order queue cost a layer graph pays.
//   mv_bench BACKEND_DIR [iters]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static const int R = 16;

struct bench {
    ggml_backend_t be;
    ggml_backend_buffer_type_t buft;
    int iters;
    std::mt19937 rng{42};

    std::vector<uint8_t> make_matrix(ggml_type type, int64_t cols, int64_t rows) {
        std::normal_distribution<float> nd(0.0f, 0.02f);
        std::vector<float> src((size_t) (cols * rows));
        for (float & v : src) v = nd(rng);
        std::vector<uint8_t> dst(ggml_row_size(type, cols) * rows);
        if (type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) dst.data(), (int64_t) src.size());
        } else if (type == GGML_TYPE_BF16) {
            ggml_fp32_to_bf16_row(src.data(), (ggml_bf16_t *) dst.data(), (int64_t) src.size());
        } else if (type == GGML_TYPE_F32) {
            std::copy((uint8_t *) src.data(), (uint8_t *) src.data() + dst.size(), dst.data());
        } else {
            ggml_quantize_chunk(type, src.data(), dst.data(), 0, rows, cols, nullptr);
        }
        return dst;
    }

    double time_graph(ggml_cgraph * gf) {
        for (int i = 0; i < 10; i++) ggml_backend_graph_compute(be, gf);
        double best = 1e30;
        for (int r = 0; r < 5; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, gf);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            best = std::min(best, us);
        }
        return best;
    }

    // out = W[rows x cols] @ x[cols x ncols_x]; R copies with distinct W
    void mv(const char * name, ggml_type type, int64_t cols, int64_t rows, int64_t ncols_x = 1) {
        if (cols % ggml_blck_size(type) != 0) return;
        ggml_init_params wp = { ggml_tensor_overhead() * (R + 1), nullptr, true };
        ggml_context * wctx = ggml_init(wp);
        std::vector<ggml_tensor *> ws;
        for (int i = 0; i < R; i++) ws.push_back(ggml_new_tensor_2d(wctx, type, cols, rows));
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        for (int i = 0; i < R; i++) { auto m = make_matrix(type, cols, rows); ggml_backend_tensor_set(ws[i], m.data(), 0, m.size()); }

        ggml_init_params gp = { ggml_tensor_overhead() * (4 * R + 8) + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, cols, ncols_x); ggml_set_input(x);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        for (int i = 0; i < R; i++) ggml_build_forward_expand(gf, ggml_mul_mat(gctx, ws[i], x));
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("mvbench %s: alloc failed\n", name); return; }
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xv((size_t) ggml_nelements(x));
        for (float & v : xv) v = nd(rng);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        const double us = time_graph(gf) / R;
        const double mb = ggml_row_size(type, cols) * rows / 1e6;
        printf("mvbench %-12s %5lldx%-5lld x%-4lld %-6s %7.1f us/matmul  (%6.3f MB, %5.0f GB/s)\n", name, (long long) rows, (long long) cols,
               (long long) ncols_x, ggml_type_name(type), us, mb, mb / us * 1e3);
        fflush(stdout);
        ggml_gallocr_free(ga); ggml_free(gctx); ggml_backend_buffer_free(wbuf); ggml_free(wctx);
    }

    // reference: R chained SCALE ops on a 10240 vector (the cost of a trivial kernel in the queue)
    void trivial() {
        ggml_init_params gp = { ggml_tensor_overhead() * (2 * R + 8) + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * x = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 10240); ggml_set_input(x);
        ggml_tensor * y = x;
        for (int i = 0; i < R; i++) y = ggml_scale(gctx, y, 1.0001f);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        ggml_gallocr_alloc_graph(ga, gf);
        std::vector<float> xv(10240, 1.0f);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        printf("mvbench trivial SCALE[10240] chained: %.1f us/kernel\n", time_graph(gf) / R);
        ggml_gallocr_free(ga); ggml_free(gctx);
    }
};

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s BACKEND_DIR [iters]\n", argv[0]); return 1; }
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no GPU device\n"); return 1; }
    printf("mvbench device %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    bench b;
    b.be = ggml_backend_dev_init(dev, nullptr);
    b.buft = ggml_backend_get_default_buffer_type(b.be);
    b.iters = argc > 2 ? atoi(argv[2]) : 50;

    b.trivial();
    struct shape { const char * name; int64_t rows, cols, ncx; };
    const shape shapes[] = {
        { "hc_down",   320, 10240, 1 }, { "hc_up",   10240,  320, 1 }, { "hc_inject",  4, 10240, 1 },
        { "hc_down+inj", 324, 10240, 1 }, { "router",  512,  2560, 1 }, { "router+sg", 513, 2560, 1 },
        { "shexp_gate",  1,  2560, 1 }, { "alpha",     48,  2560, 1 }, { "alpha+beta", 96, 2560, 1 },
        { "dense_2560",  2560, 2560, 1 },
    };
    const ggml_type types[] = { GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_F32 };
    for (const shape & s : shapes)
        for (ggml_type t : types) b.mv(s.name, t, s.cols, s.rows, s.ncx);
    // the two f32 GEMMs the engine builds at T = 1
    b.mv("hc_mean", GGML_TYPE_F32, 4, 1, 2560);       // (1/hc)[4] against [4, 2560]
    b.mv("moe_wsum", GGML_TYPE_F32, 10, 2560, 1);     // expert rows [10, 2560] against weights [10]
    ggml_backend_free(b.be);
    printf("mvbench done\n");
    return 0;
}
