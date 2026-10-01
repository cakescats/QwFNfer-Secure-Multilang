// Quick re-encode of selected tensors of a split GGUF, for speed tests only (quantizing already-quantized
// weights keeps their error). The output shards hold only the converted tensors; qwfn's loader keeps the
// first tensor of a name across shards, so a variant is: a metadata head, the converted shards, then the
// original shards (symlinks).
//   gguf_requant meta  HEAD_IN OUT SPLIT_COUNT                 metadata-only head with split.count replaced
//   gguf_requant err   RULE - THREADS SHARD_IN...          round-trip error the re-encode adds, no file written
//   gguf_requant conv  RULE OUT THREADS SHARD_IN...           RULE: gateup (ffn_gate/up_exps -> Q4_K)
//                                                              dense  (2D Q8_0 except token_embd -> Q4_K, or Q4_0
//                                                                      when the row width is not a multiple of 256)
//                                                              dense5 (Unsloth's selection: attn/shexp gate+up -> Q5_K,
//                                                                      ssm_out -> Q6_K, the rest left alone)
//                                                              down41 (expert down -> Q4_1)
#include "ggml.h"
#include "gguf.h"

#include <atomic>
#include <cmath>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

static ggml_type target_type(const std::string & rule, const char * name, ggml_type type, int n_dims, int64_t ne0) {
    const std::string nm = name;
    if (rule == "gateup") {
        if (nm.find("ffn_gate_exps.weight") == std::string::npos && nm.find("ffn_up_exps.weight") == std::string::npos) return GGML_TYPE_COUNT;
        return ne0 % 256 == 0 ? GGML_TYPE_Q4_K : GGML_TYPE_Q4_0;
    }
    if (rule == "dense") {
        if (type != GGML_TYPE_Q8_0 || n_dims != 2 || nm == "token_embd.weight") return GGML_TYPE_COUNT;
        return ne0 % 256 == 0 ? GGML_TYPE_Q4_K : GGML_TYPE_Q4_0;
    }
    if (rule == "dense5") {
        // Unsloth's own dense selection at UD-Q2_K_XL/IQ1_S: input-side projections to Q5_K, ssm_out to Q6_K,
        // hyper-connection mixers and the shared expert's down projection left at Q8_0.
        if (type != GGML_TYPE_Q8_0 || n_dims != 2 || ne0 % 256 != 0) return GGML_TYPE_COUNT;
        static const char * to_q5[] = { "attn_qkv", "attn_gate", "attn_q.", "attn_k.", "attn_v.", "attn_output",
                                        "ffn_gate_shexp", "ffn_up_shexp" };
        for (const char * k : to_q5) if (nm.find(k) != std::string::npos) return GGML_TYPE_Q5_K;
        if (nm.find("ssm_out") != std::string::npos) return GGML_TYPE_Q6_K;
        return GGML_TYPE_COUNT;
    }
    if (rule == "down41") {
        // Q4_1 is 3x faster than IQ4_NL for this shape on the B70, at 5 bits instead of 4.5.
        if (nm.find("ffn_down_exps.weight") == std::string::npos) return GGML_TYPE_COUNT;
        return GGML_TYPE_Q4_1;
    }
    fprintf(stderr, "unknown rule %s\n", rule.c_str()); exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: see the header comment\n"); return 1; }
    const std::string mode = argv[1];

    if (mode == "meta") {
        gguf_init_params p = { true, nullptr };
        gguf_context * in = gguf_init_from_file(argv[2], p);
        if (!in) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        gguf_context * out = gguf_init_empty();
        gguf_set_kv(out, in);
        gguf_set_val_u16(out, "split.count", (uint16_t) atoi(argv[4]));
        if (!gguf_write_to_file(out, argv[3], false)) { fprintf(stderr, "write failed\n"); return 1; }
        printf("meta head %s: %lld kv, split.count %s\n", argv[3], (long long) gguf_get_n_kv(out), argv[4]);
        return 0;
    }
    const bool err_mode = mode == "err";
    if (mode != "conv" && !err_mode) { fprintf(stderr, "unknown mode\n"); return 1; }

    const std::string rule = argv[2];
    const char * out_path  = argv[3];
    const int n_threads    = atoi(argv[4]);

    struct job { int shard; const ggml_tensor * t; size_t in_off; ggml_type in_type; ggml_type out_type; };
    std::vector<gguf_context *> gin;
    std::vector<ggml_context *> tin;
    std::vector<int> fds;
    std::vector<job> jobs;
    gguf_context * gout = gguf_init_empty();
    ggml_init_params ip = { ggml_tensor_overhead() * 4096, nullptr, true };
    ggml_context * cout_ = ggml_init(ip);

    for (int s = 5; s < argc; s++) {
        ggml_context * meta = nullptr;
        gguf_init_params p = { true, &meta };
        gguf_context * g = gguf_init_from_file(argv[s], p);
        if (!g) { fprintf(stderr, "cannot open %s\n", argv[s]); return 1; }
        int fd = open(argv[s], O_RDONLY);
        if (fd < 0) { perror(argv[s]); return 1; }
        gin.push_back(g); tin.push_back(meta); fds.push_back(fd);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
            const char * name = gguf_get_tensor_name(g, i);
            const ggml_tensor * t = ggml_get_tensor(meta, name);
            const ggml_type tt = target_type(rule, name, t->type, ggml_n_dims(t), t->ne[0]);
            if (tt == GGML_TYPE_COUNT) continue;
            ggml_tensor * o = ggml_new_tensor(cout_, tt, ggml_n_dims(t), t->ne);
            ggml_set_name(o, name);
            gguf_add_tensor(gout, o);
            jobs.push_back({ (int) gin.size() - 1, t, gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i), t->type, tt });
        }
    }
    printf("rule %s: %zu tensors to convert\n", rule.c_str(), jobs.size());
    if (jobs.empty()) return 1;

    FILE * fo = err_mode ? nullptr : fopen(out_path, "wb");
    if (!err_mode && !fo) { perror(out_path); return 1; }
    const size_t meta_size = gguf_get_meta_size(gout);
    std::vector<uint8_t> zeros(meta_size, 0);
    if (fo) fwrite(zeros.data(), 1, meta_size, fo);
    const size_t align = gguf_get_alignment(gout);
    double err_ss = 0, src_ss = 0;   // error the re-encode adds on top of the source quantization
    size_t err_n = 0;

    for (int ft = 0; ft < (int) GGML_TYPE_COUNT; ft++) ggml_quantize_init((ggml_type) ft);

    size_t written_in = 0, written_out = 0;
    for (size_t j = 0; j < jobs.size(); j++) {
        const job & jb = jobs[j];
        const ggml_tensor * t = jb.t;
        const int64_t ne0 = t->ne[0], ne1 = t->ne[1], n_slices = t->ne[2] * t->ne[3];
        const size_t in_slice  = ggml_row_size(jb.in_type, ne0) * ne1;
        const size_t out_slice = ggml_row_size(jb.out_type, ne0) * ne1;
        std::vector<uint8_t> out((size_t) n_slices * out_slice);
        const auto to_float = ggml_get_type_traits(jb.in_type)->to_float;
        std::atomic<int64_t> next{0};
        std::atomic<bool> fail{false};
        std::mutex err_mu;
        double t_err_ss = 0, t_src_ss = 0;
        size_t t_err_n = 0;
        auto worker = [&]() {
            std::vector<uint8_t> src(in_slice);
            std::vector<float> f((size_t) (ne0 * ne1));
            for (int64_t e; (e = next++) < n_slices;) {
                const ssize_t r = pread(fds[jb.shard], src.data(), in_slice, (off_t) (jb.in_off + e * in_slice));
                if (r != (ssize_t) in_slice) { fail = true; return; }
                to_float(src.data(), f.data(), ne0 * ne1);
                ggml_quantize_chunk(jb.out_type, f.data(), out.data() + e * out_slice, 0, ne1, ne0, nullptr);
                if (err_mode && e < 2) {   // two slices per tensor is plenty for an RMS
                    std::vector<float> back((size_t) (ne0 * ne1));
                    ggml_get_type_traits(jb.out_type)->to_float(out.data() + e * out_slice, back.data(), ne0 * ne1);
                    double es = 0, ss = 0;
                    for (int64_t k = 0; k < ne0 * ne1; k++) {
                        const double d = (double) back[k] - (double) f[k];
                        es += d * d; ss += (double) f[k] * (double) f[k];
                    }
                    std::lock_guard<std::mutex> lk(err_mu);
                    err_ss += es; src_ss += ss; err_n += (size_t) (ne0 * ne1);
                }
            }
        };
        err_ss = 0; src_ss = 0; err_n = 0;
        std::vector<std::thread> th;
        for (int k = 0; k < std::max(1, std::min<int>(n_threads, (int) n_slices)); k++) th.emplace_back(worker);
        for (auto & x : th) x.join();
        if (fail) { fprintf(stderr, "read failed on %s\n", ggml_get_name(t)); return 1; }
        if (fo) {
            fwrite(out.data(), 1, out.size(), fo);
            const size_t pad = GGML_PAD(out.size(), align) - out.size();
            if (pad) fwrite(zeros.data(), 1, pad, fo);
        }
        written_in += in_slice * n_slices; written_out += out.size();
        if (err_mode) {
            printf("[%zu/%zu] %-36s %-8s -> %-8s  added RMS error %.4f%% of the source weights\n", j + 1, jobs.size(),
                   ggml_get_name(t), ggml_type_name(jb.in_type), ggml_type_name(jb.out_type),
                   100.0 * std::sqrt(err_ss / std::max<size_t>(err_n, 1)) / std::sqrt(src_ss / std::max<size_t>(err_n, 1)));
        } else {
            printf("[%zu/%zu] %-36s %s -> %s  %.1f -> %.1f MB\n", j + 1, jobs.size(), ggml_get_name(t), ggml_type_name(jb.in_type),
                   ggml_type_name(jb.out_type), in_slice * n_slices / 1e6, out.size() / 1e6);
        }
        fflush(stdout);
    }
    // Tail slack: qwfn's prefill reader rounds each expert range up to the I/O alignment, which runs
    // past the end of a file whose last tensor is an expert tensor (a short read, reported as an error).
    std::vector<uint8_t> tail(1u << 20, 0);
    fwrite(tail.data(), 1, tail.size(), fo);

    if (err_mode) { printf("err mode: no file written\n"); return 0; }

    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(gout, meta.data());
    fseek(fo, 0, SEEK_SET);
    fwrite(meta.data(), 1, meta_size, fo);
    fclose(fo);

    gguf_init_params p = { true, nullptr };
    gguf_context * chk = gguf_init_from_file(out_path, p);
    if (!chk || gguf_get_n_tensors(chk) != (int64_t) jobs.size()) { fprintf(stderr, "re-open check failed\n"); return 1; }
    // The reader must find the data exactly where it was written: a mismatch shifts every tensor.
    if (gguf_get_data_offset(chk) != meta_size) {
        fprintf(stderr, "data offset mismatch: wrote %zu, reader expects %zu\n", meta_size, (size_t) gguf_get_data_offset(chk));
        return 1;
    }
    printf("done %s: %zu tensors, %.2f GB in -> %.2f GB out\n", out_path, jobs.size(), written_in / 1e9, written_out / 1e9);
    return 0;
}
