// qwfn-opbench -- per-op-group correctness and speed of one decode layer's graph pieces,
// in the spirit of DeepGEMM's tests: each piece runs on the GPU backend and on the CPU
// backend with the same real weights (one layer of the checkpoint) and the same inputs,
// is compared with calc_diff (1 - 2<x,y>/(<x,x>+<y,y>)), and is timed on the GPU.
//
//   qwfn-opbench MODEL.gguf [--layer N] [--reps N]
//
// It loads only the chosen layer's dense tensors (~150 MB), applies the engine's load-time
// transforms that change the graph (the hc norm gammas shaped [n_embd, hc], the 1/hc fold
// into the inject weights), and builds the pieces with the engine's graph_builder at T = 1:
//   hc_mix.attn   the attention-side hyper-connection mixer (+ its inject)
//   hc_combine    residual + block * 2 sigmoid(inject)
//   ffn_front     the FFN-side mixer, the router (softmax, top-10, renormalise), the shared expert
// A fusion in ggml-cuda shows up here as a lower GPU time at an unchanged calc_diff.
#include "qwfn_graph.h"
#include "qwfn_model.h"
#include "qwfn_weights.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace qwfn;

static double calc_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double xy = 0, xx = 0, yy = 0;
    for (size_t i = 0; i < a.size(); i++) { xy += (double) a[i] * b[i]; xx += (double) a[i] * a[i]; yy += (double) b[i] * b[i]; }
    return xx + yy > 0 ? 1.0 - 2.0 * xy / (xx + yy) : 0.0;
}

static std::vector<float> read_f32(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> iv(v.size()); ggml_backend_tensor_get(t, iv.data(), 0, iv.size() * sizeof(int32_t));
        for (size_t i = 0; i < iv.size(); i++) v[i] = (float) iv[i];
    }
    return v;
}

// One side (GPU or CPU): the layer's weights, the 1/hc stream-mean vector, and the transforms the engine applies.
struct side {
    weights w;
    ggml_context * ectx = nullptr; ggml_backend_buffer_t ebuf = nullptr;
    ggml_tensor * hc_mean = nullptr;
    bool load(const model_index & mi, bool gpu, uint32_t il, const std::string & bdir, std::string & err) {
        const std::string pre = "blk." + std::to_string(il) + ".";
        if (!w.init(&mi, gpu, bdir, err)) return false;
        if (!w.declare_dense_core(err, [&](const std::string & n) { return n.rfind(pre, 0) == 0; })) return false;
        if (!w.commit(err)) return false;
        const int64_t hc = mi.hp().hc_count, n_embd = mi.hp().n_embd;
        for (const char * n : { "hc_attn_norm", "hc_ffn_norm" }) {   // engine::init: the gamma as [n_embd, hc], metadata only
            ggml_tensor * t = w.get(pre + n + ".weight");
            if (!t || ggml_nelements(t) != hc * n_embd || ggml_blck_size(t->type) != 1) continue;
            t->ne[0] = n_embd; t->ne[1] = hc; t->ne[2] = 1; t->ne[3] = 1;
            t->nb[1] = t->nb[0] * n_embd; t->nb[2] = t->nb[1] * hc; t->nb[3] = t->nb[2];
        }
        for (const char * n : { "hc_attn_inject", "hc_ffn_inject" }) {   // engine::init: 1/hc folded into F32 inject weights
            ggml_tensor * t = w.get(pre + n + ".weight");
            if (!t || t->type != GGML_TYPE_F32) { err = "inject weights are not F32"; return false; }
            std::vector<float> v(ggml_nelements(t)); ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
            for (float & x : v) x *= 1.0f / (float) hc;
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
        }
        ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
        ectx = ggml_init(p);
        hc_mean = ggml_new_tensor_1d(ectx, GGML_TYPE_F32, hc);
        ebuf = ggml_backend_alloc_ctx_tensors(ectx, w.backend());
        std::vector<float> m(hc, 1.0f / (float) hc); ggml_backend_tensor_set(hc_mean, m.data(), 0, hc * 4);
        return true;
    }
};

struct piece {
    const char * name;
    // Builds the piece on `gb` from the inputs; returns the outputs to compare.
    std::function<std::vector<ggml_tensor *>(graph_builder & gb, ggml_context * c, const std::vector<ggml_tensor *> & in)> build;
    std::vector<std::vector<int64_t>> in_shapes;
};

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: qwfn-opbench MODEL.gguf [--layer N] [--reps N]\n"); return 2; }
    uint32_t il = 0; int reps = 300;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--layer" && i + 1 < argc) il = (uint32_t) atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
    }
    std::string err;
    model_index mi;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    hparams hp = mi.hp();
    hp.hc_inject_prescaled = true;    // as the engine runs it (folded above)
    hp.hc_down_prescaled   = false;   // Q8_0 down weights are not folded
    const std::string bdir = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.unsloth/llama.cpp/build/bin";
    side G, C;
    if (!G.load(mi, true, il, bdir, err) || !C.load(mi, false, il, bdir, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!G.w.on_gpu()) { fprintf(stderr, "error: no GPU backend\n"); return 1; }
    const int64_t E = hp.n_embd, H = hp.hc_count;
    printf("layer %u (%s), %s vs CPU, %d reps\n", il, hp.is_attn_layer(il) ? "attention" : "recurrent", G.w.dev_name(), reps);

    const std::vector<piece> pieces = {
        { "hc_mix.attn", [&](graph_builder & gb, ggml_context *, const std::vector<ggml_tensor *> & in) {
              ggml_tensor * inject = nullptr;
              ggml_tensor * m = gb.hc_mix(in[0], (int) il, false, &inject);
              return std::vector<ggml_tensor *>{ m, inject }; },
          { { E, H, 1 } } },
        { "hc_combine", [&](graph_builder & gb, ggml_context *, const std::vector<ggml_tensor *> & in) {
              return std::vector<ggml_tensor *>{ gb.hc_combine(in[0], in[1], in[2]) }; },
          { { E, H, 1 }, { E, 1 }, { H, 1 } } },
        { "ffn_front", [&](graph_builder & gb, ggml_context *, const std::vector<ggml_tensor *> & in) {
              ggml_tensor * inject = nullptr;
              ggml_tensor * cur2 = gb.hc_mix(in[0], (int) il, true, &inject);
              ggml_tensor * sl = nullptr, * wt = nullptr;
              gb.moe_route(cur2, (int) il, &sl, &wt);
              ggml_tensor * sh = gb.shared_expert(cur2, (int) il);
              return std::vector<ggml_tensor *>{ cur2, inject, wt, sl, sh }; },
          { { E, H, 1 } } },
    };

    std::mt19937 rng(42); std::normal_distribution<float> nd(0.0f, 1.0f);
    int failures = 0;
    for (const piece & pc : pieces) {
        // The same random inputs on both sides.
        std::vector<std::vector<float>> data;
        for (const auto & s : pc.in_shapes) {
            int64_t n = 1; for (int64_t d : s) n *= d;
            std::vector<float> v(n); for (float & x : v) x = nd(rng);
            data.push_back(std::move(v));
        }
        std::vector<std::vector<float>> outs[2];
        double gpu_us = 0; int n_kernels = 0;
        for (int sidx = 0; sidx < 2; sidx++) {
            side & S = sidx == 0 ? G : C;
            ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
            ggml_context * ictx = ggml_init(ip);
            std::vector<ggml_tensor *> in;
            for (const auto & s : pc.in_shapes) {
                ggml_tensor * t = s.size() == 3 ? ggml_new_tensor_3d(ictx, GGML_TYPE_F32, s[0], s[1], s[2])
                                                : ggml_new_tensor_2d(ictx, GGML_TYPE_F32, s[0], s[1]);
                ggml_set_input(t); in.push_back(t);
            }
            ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors(ictx, S.w.backend());
            for (size_t k = 0; k < in.size(); k++) ggml_backend_tensor_set(in[k], data[k].data(), 0, data[k].size() * 4);

            ggml_init_params gp = { ggml_tensor_overhead() * 4096 + ggml_graph_overhead(), nullptr, true };
            ggml_context * c = ggml_init(gp);
            ggml_cgraph * g = ggml_new_graph(c);
            graph_builder gb(c, &hp, &S.w);
            gb.bind(nullptr, g, 0);
            gb.set_gpu_fusion(sidx == 0, S.hc_mean);   // the GPU graph takes the engine's GPU-only fusions
            std::vector<ggml_tensor *> o = pc.build(gb, c, in);
            for (ggml_tensor * t : o) { ggml_set_output(t); ggml_build_forward_expand(g, t); }
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(S.w.backend()));
            if (!ggml_gallocr_alloc_graph(ga, g)) { fprintf(stderr, "%s: alloc failed\n", pc.name); return 1; }
            ggml_backend_graph_compute(S.w.backend(), g);
            for (ggml_tensor * t : o) outs[sidx].push_back(read_f32(t));
            if (sidx == 0) {
                for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
                    const ggml_op op = ggml_graph_node(g, i)->op;
                    if (op != GGML_OP_VIEW && op != GGML_OP_RESHAPE && op != GGML_OP_PERMUTE && op != GGML_OP_TRANSPOSE && op != GGML_OP_NONE) n_kernels++;
                }
                std::vector<double> ts;
                for (int r = 0; r < reps + 20; r++) {
                    const auto t0 = std::chrono::steady_clock::now();
                    ggml_backend_graph_compute(S.w.backend(), g);   // synchronous: launch + run + wait
                    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (r >= 20) ts.push_back(dt);
                }
                std::sort(ts.begin(), ts.end());
                gpu_us = ts[ts.size() / 2] * 1e6;
            }
            ggml_gallocr_free(ga); ggml_free(c);
            ggml_backend_buffer_free(ibuf); ggml_free(ictx);
        }
        printf("%-12s  %3d ops  GPU %7.1f us (median)  calc_diff:", pc.name, n_kernels, gpu_us);
        for (size_t k = 0; k < outs[0].size(); k++) {
            const double d = calc_diff(outs[0][k], outs[1][k]);
            const bool ids = k < outs[0].size() && pc.name == std::string("ffn_front") && k == 3;
            printf(" %s%.2e", ids ? "ids:" : "", d);
            if (!(d < (ids ? 1e-12 : 1e-5))) failures++;
        }
        printf("\n");
    }
    printf("%s\n", failures ? "MISMATCH beyond tolerance" : "all outputs match the CPU reference");
    return failures ? 1 : 0;
}
