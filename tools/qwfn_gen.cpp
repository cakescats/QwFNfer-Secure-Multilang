// qwfn-gen -- prefill a prompt, then decode greedily, tracking position.
//
// Works in token ids so it can be compared against llama.cpp directly without
// needing a tokenizer.

#include "qwfn_engine.h"
#include "qwfn_model.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace qwfn;

// --ckpt-test: a prefix checkpoint must continue exactly like the sequence it was
// taken from. Three runs over the same prompt:
//   base   prefill prompt[:-1], save, then feed the last token and decode greedily
//   fresh  reset, the same prefill without saving, then replay base's tokens
//   ckpt   reset, prefill and decode another prompt (with a different image
//          override at the same position), restore, then replay base's tokens
// GPU decode is not bit-reproducible (experts move between tiers), so fresh sets
// the noise floor: per replayed step, the largest logit difference from base and
// the NLL of base's tokens. ckpt must sit inside that floor; a checkpoint that
// missed a field lands near the other prompt's distance instead.
// The other prompt carries a synthetic embedding override just past the checkpoint,
// where the replay will evaluate: a restore must drop it (QWFN_CKPT_BREAK=ov puts
// it back after the restore, a negative control that must fail).
static int ckpt_test(engine & eng, const engine_config & cfg, const std::vector<int32_t> & prompt,
                     const std::vector<int32_t> & other, int n_gen) {
    std::string err;
    if (prompt.size() < 16 || other.size() < 16) { fprintf(stderr, "ckpt-test: prompts need 16+ tokens\n"); return 1; }
    const int64_t V = eng.n_vocab();
    const int32_t d = 2560;
    auto override_at = [&](int32_t pos) {
        std::vector<float> e(d);
        for (int32_t k = 0; k < d; k++) e[k] = 0.5f * std::sin(0.37f * k);
        eng.set_embeddings(pos, e.data(), 4);
    };
    auto prefill = [&](std::vector<int32_t> & hist, int32_t n) {
        for (int32_t done = eng.n_past(); done < n; ) {
            const int32_t take = std::min<int32_t>(cfg.n_batch, n - done);
            if (!eng.eval(hist.data(), done + take, take, err)) { fprintf(stderr, "ckpt-test: prefill failed: %s\n", err.c_str()); exit(1); }
            done += take;
        }
    };
    auto argmax = [&](const float * l) { int b = 0; for (int64_t v = 1; v < V; v++) if (l[v] > l[b]) b = (int) v; return b; };
    auto nll = [&](const float * l, int32_t tok) {
        float mx = l[0]; for (int64_t v = 1; v < V; v++) mx = std::max(mx, l[v]);
        double z = 0.0; for (int64_t v = 0; v < V; v++) z += std::exp((double) l[v] - mx);
        return -((double) l[tok] - mx - std::log(z));
    };
    std::vector<std::vector<float>> base_l;   // base's logits at every step
    std::vector<int32_t> base_t;
    struct cmp { double max_d = 0, mean_d = 0, nll = 0; int agree = 0; };
    // Feed the prompt's last token, then n_gen more: greedy when base_t is empty
    // (recording base), otherwise base's tokens, compared step by step.
    auto run = [&](std::vector<int32_t> & hist, int steps) {
        cmp c; bool agreeing = true;
        const bool record = base_t.empty();
        const float * lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
        for (int i = 0; lg && i < steps; i++) {
            if (record) { base_l.emplace_back(lg, lg + V); base_t.push_back(argmax(lg)); }
            else {
                double m = 0; for (int64_t v = 0; v < V; v++) m = std::max(m, (double) std::fabs(lg[v] - base_l[i][v]));
                c.max_d = std::max(c.max_d, m); c.mean_d += m / steps;
                c.nll += nll(lg, base_t[i]) / steps;
                if (agreeing && argmax(lg) == base_t[i]) c.agree++; else agreeing = false;
            }
            hist.push_back(base_t[i]);
            lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
        }
        if (!lg) { fprintf(stderr, "ckpt-test: eval failed: %s\n", err.c_str()); exit(1); }
        return c;
    };
    const int32_t np = (int32_t) prompt.size() - 1;

    std::vector<int32_t> h = prompt;
    eng.reset(); eng.clear_embeddings();
    prefill(h, np);
    engine::checkpoint ck;
    auto t0 = std::chrono::steady_clock::now();
    if (!eng.checkpoint_save(ck, err)) { fprintf(stderr, "ckpt-test: %s\n", err.c_str()); return 1; }
    const double t_save = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    run(h, n_gen);
    double base_nll = 0; for (int i = 0; i < n_gen; i++) base_nll += nll(base_l[i].data(), base_t[i]) / n_gen;

    h = prompt;
    eng.reset(); eng.clear_embeddings();
    prefill(h, np);
    const cmp fresh = run(h, n_gen);

    std::vector<int32_t> ho = other;
    eng.reset(); eng.clear_embeddings(); override_at(np + 2);
    prefill(ho, (int32_t) other.size() - 1);
    ho.resize(other.size() - 1);
    ho.push_back(prompt.back());   // same last token, different history: the "missed everything" distance
    const cmp wrong = run(ho, 1);
    for (int i = 0; i < 8; i++) {   // move the other sequence along a little before switching back
        ho.push_back(base_t[i]);
        if (!eng.eval(ho.data(), (int32_t) ho.size(), 1, err)) { fprintf(stderr, "ckpt-test: %s\n", err.c_str()); return 1; }
    }
    // QWFN_CKPT_BREAK: negative controls, the gate must FAIL on these.
    //   ov  keep the other prompt's override   state  zero a stretch in the middle of the state bytes
    const char * br = getenv("QWFN_CKPT_BREAK");
    if (br && *br) {
        if (!strcmp(br, "state")) std::fill(ck.st.begin() + ck.st.size() / 2, ck.st.begin() + ck.st.size() / 2 + ck.st.size() / 20, 0);
        printf("ckpt-test: negative control: %s\n", br);
    }
    h = prompt;
    t0 = std::chrono::steady_clock::now();
    if (!eng.checkpoint_restore(ck, err)) { fprintf(stderr, "ckpt-test: %s\n", err.c_str()); return 1; }
    const double t_restore = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Round trip: what the restore wrote must read back byte for byte.
    engine::checkpoint ck2;
    if (!eng.checkpoint_save(ck2, err)) { fprintf(stderr, "ckpt-test: %s\n", err.c_str()); return 1; }
    const bool exact = ck2.n_past == ck.n_past && ck2.st == ck.st;
    if (br && !strcmp(br, "ov")) override_at(np + 2);
    const cmp ckpt = run(h, n_gen);

    printf("ckpt-test: prompt %d tokens, other %zu, checkpoint %.1f MB (%d positions)\n",
           np, other.size() - 1, ck.bytes() / 1e6, ck.n_past);
    printf("ckpt-test: save %.1f ms, restore %.1f ms, round trip %s\n", t_save * 1e3, t_restore * 1e3, exact ? "byte-exact" : "DIFFERS");
    printf("ckpt-test: base  NLL %.4f over %d greedy tokens\n", base_nll, n_gen);
    auto show = [&](const char * name, const cmp & c) {
        printf("ckpt-test: %-5s max|dlogit| %.3f, mean per step %.3f, NLL %.4f, greedy agrees for %d/%d\n",
               name, c.max_d, c.mean_d, c.nll, c.agree, n_gen);
    };
    show("fresh", fresh); show("ckpt", ckpt);
    printf("ckpt-test: other first-step max|dlogit| %.3f (must be far above both)\n", wrong.max_d);
    // Inside the floor, with room for its own run-to-run spread (measured ~1.2 logits
    // per step at 4K on the B70, so a small loss hides in it: the round trip and the
    // negative controls carry the rest).
    const bool pass = exact && ckpt.mean_d <= 1.5 * fresh.mean_d + 0.05 &&
                      std::fabs(ckpt.nll - base_nll) <= std::max(0.02, 1.5 * std::fabs(fresh.nll - base_nll)) &&
                      wrong.max_d > ckpt.max_d;
    printf("ckpt-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 2;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: qwfn-gen <shard.gguf> [--prompt id,id,...] [--gen N] [--ctx N]\n"
            "                [--ram GB] [--vram GB] [--batch N] [--threads N] [--cpu] [--no-qsa]\n");
        return 1;
    }
    std::vector<int32_t> prompt, replay, other;
    std::string cold_path, save_replay;   // --save-replay FILE: the generated ids, one per line, for a later --replay-file
    bool want_ppl = false;   // with --replay-file: mean NLL of the replayed tokens (a quality number)
    bool pair_test = false, rollback_test = false;   // exercise the multi-token decode step without the head
    int  multi_test = 2;                              // tokens per step for --multi-test / --rollback-test
    int n_gen = 16;
    engine_config cfg;
    cfg.n_ctx = 4096; cfg.n_batch = 128; cfg.ram_bytes = 8e9; cfg.vram_bytes = 0;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (a == "--gen"     && i + 1 < argc) { n_gen = atoi(next()); continue; }
        if (a == "--ctx"     && i + 1 < argc) { cfg.n_ctx = (uint32_t) atoi(next()); continue; }
        if (a == "--batch"   && i + 1 < argc) { cfg.n_batch = (uint32_t) atoi(next()); continue; }
        if (a == "--ram"     && i + 1 < argc) { cfg.ram_bytes = (size_t)(atof(next()) * 1e9); continue; }
        if (a == "--vram"    && i + 1 < argc) { cfg.vram_bytes = (size_t)(atof(next()) * 1e9); continue; }
        if (a == "--threads" && i + 1 < argc) { cfg.n_threads = atoi(next()); continue; }
        if (a == "--cpu")    { cfg.use_gpu = false; continue; }
        if (a == "--io-threads" && i + 1 < argc) { cfg.io_threads = true; cfg.io_workers = (unsigned) atoi(next()); continue; }
        if (a == "--io-uring") { cfg.io_threads = false; continue; }
        if (a == "--state-host" && i + 1 < argc) {   // none | idx | kv | kv,idx
            std::string v = next();
            cfg.idx_host = v.find("idx") != std::string::npos;
            cfg.kv_host  = v.find("kv")  != std::string::npos;
            continue;
        }
        if (a == "--kv" && i + 1 < argc) { std::string v = next();
            cfg.type_k = cfg.type_v = (v == "q8_0") ? GGML_TYPE_Q8_0 :
                                      (v == "q4_0") ? GGML_TYPE_Q4_0 : GGML_TYPE_F16; continue; }
        if (a == "--no-reuse") { cfg.reuse_graphs = false; continue; }
        if (a == "--no-speculate") { cfg.speculate = false; continue; }
        if (a == "--skip-miss") { cfg.skip_miss = true; continue; }
        if (a == "--ppl") { want_ppl = true; continue; }
        if (a == "--reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t) atof(next()) * (1ull << 20); continue; }
        if (a == "--predictor" && i + 1 < argc) { cfg.predictor_path = next(); continue; }
        if (a == "--spec-depth" && i + 1 < argc) { cfg.speculate_depth = (uint32_t) atoi(next());
            if (cfg.speculate_depth == 0) cfg.speculate = false; continue; }
        if (a == "--spec-ahead" && i + 1 < argc) { cfg.speculate_ahead = (uint32_t) atoi(next()); continue; }
        if (a == "--spec-depth2" && i + 1 < argc) { cfg.speculate_depth2 = (uint32_t) atoi(next()); continue; }
        if (a == "--spec-margin" && i + 1 < argc) { cfg.spec_margin = (float) atof(next()); continue; }
        if (a == "--spec-gate-inflight" && i + 1 < argc) { cfg.spec_gate_inflight = (uint32_t) atoi(next()); continue; }
        if (a == "--spec-block") { cfg.spec_block = true; continue; }
        if (a == "--spec-block-layers" && i + 1 < argc) { cfg.spec_block = true; cfg.spec_block_layers = next(); continue; }
        if (a == "--mtp" && i + 1 < argc) { cfg.mtp_path = next(); cfg.rollback_snapshots = true; continue; }
        if (a == "--pair-test") { pair_test = true; continue; }          // decode the replay two tokens per step
        if (a == "--multi-test" && i + 1 < argc) { pair_test = true; multi_test = atoi(next()); cfg.mtp_drafts = (uint32_t) std::max(1, multi_test - 1); continue; }   // K tokens per step from the replay
        if (a == "--mtp-drafts" && i + 1 < argc) { cfg.mtp_drafts = (uint32_t) std::max(1, std::min(3, atoi(next()))); continue; }
        if (a == "--rollback-test") { rollback_test = true; cfg.rollback_snapshots = true; if (cfg.mtp_drafts < (uint32_t) std::max(1, multi_test - 1)) cfg.mtp_drafts = (uint32_t) std::max(1, multi_test - 1); continue; }  // every token as the first of a pair with a wrong second, then roll back
        if (a == "--ram-frac" && i + 1 < argc) { cfg.ram_frac = atof(next()); continue; }
        // 0 forces the batched prefill path at every size. Reference runs want
        // this: token-by-token prefill is a different (equally valid) summation
        // order, and this model turns that into different tokens.
        if (a == "--prefill-decode-max" && i + 1 < argc) { cfg.prefill_decode_max = (uint32_t) atoi(next()); continue; }
        if (a == "--gate-drop" && i + 1 < argc) { cfg.gate_drop = (float) atof(next()); continue; }
        if (a == "--prefill-chunk" && i + 1 < argc) { cfg.prefill_chunk = (uint32_t) atoi(next()); continue; }
        if (a == "--vram-reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t)(atof(next()) * 1e6); continue; }
        if (a == "--no-prefill-overlap") { cfg.prefill_overlap = false; continue; }
        if (a == "--save-replay" && i + 1 < argc) { save_replay = next(); continue; }
        if (a == "--replay-file" && i + 1 < argc) {
            // Feed these ids as the "generated" tokens instead of sampling, so
            // every configuration sees identical routing. GPU decode is
            // nondeterministic, and comparing different generated texts was
            // measured to produce 1.5 tok/s of phantom variance.
            FILE * f = fopen(next(), "rb");
            if (!f) { fprintf(stderr, "error: cannot open replay file\n"); return 1; }
            int v; while (fscanf(f, "%d%*[ ,\n\t\r]", &v) == 1) replay.push_back(v);
            fclose(f);
            continue;
        }
        if (a == "--promote" && i + 1 < argc) { cfg.promote_per_layer = (uint32_t) atoi(next()); continue; }
        if (a == "--evict" && i + 1 < argc) { std::string v = next();
            cfg.evict_policy = v == "lfu" ? 1 : v == "hybrid" ? 2 : 0; continue; }
        if (a == "--no-qsa") { cfg.use_qsa = false; continue; }
        if (a == "--prefill-cpu") { cfg.prefill_on_gpu = false; continue; }
        if (a == "--prefill-gpu") { cfg.prefill_on_gpu = true; continue; }  // FAST BUT BROKEN >T~200
        if (a == "--ubatch-kv" && i + 1 < argc) { cfg.ubatch_kv_product = (uint64_t)(atof(next()) * 1e6); continue; }
        if (a == "--indexer-top-k" && i + 1 < argc) { cfg.indexer_top_k = (uint32_t) atoi(next()); continue; }
        if (a == "--cold" && i + 1 < argc) { cold_path = next(); cfg.use_cold_tier = true; continue; }
        if (a == "--ckpt-test" && i + 1 < argc) {   // a second prompt file to interpose between save and restore
            FILE * f = fopen(next(), "rb");
            if (!f) { fprintf(stderr, "error: cannot open checkpoint test prompt\n"); return 1; }
            int v; while (fscanf(f, "%d%*[ ,\n\t\r]", &v) == 1) other.push_back(v);
            fclose(f);
            continue;
        }
        if (a == "--prompt-file" && i + 1 < argc) {
            // Long contexts blow past ARG_MAX on the command line.
            FILE * f = fopen(next(), "rb");
            if (!f) { fprintf(stderr, "error: cannot open prompt file\n"); return 1; }
            int v; while (fscanf(f, "%d%*[ ,\n\t\r]", &v) == 1) prompt.push_back(v);
            fclose(f);
            continue;
        }
        if (a == "--prompt"  && i + 1 < argc) {
            std::string t = next(); size_t p = 0;
            while (p < t.size()) {
                size_t c = t.find(',', p); if (c == std::string::npos) c = t.size();
                prompt.push_back(atoi(t.substr(p, c - p).c_str())); p = c + 1;
            }
            continue;
        }
    }
    if (prompt.empty()) prompt = { 9707, 11, 1879, 0 };

    model_index mi;
    std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    model_index cold;
    model_index * coldp = nullptr;
    if (!cold_path.empty()) {
        if (cold.load(cold_path, err)) { coldp = &cold; printf("cold tier: %s\n", cold_path.c_str()); }
        else fprintf(stderr, "cold tier unavailable: %s\n", err.c_str());
    }

    engine eng;
    if (!eng.init(&mi, coldp, cfg, std::string(getenv("HOME")) + "/.unsloth/llama.cpp/build/bin", err)) {
        fprintf(stderr, "engine init: %s\n", err.c_str()); return 1;
    }
    printf("%s\n\n", eng.memory_summary().c_str());

    if (!other.empty()) return ckpt_test(eng, cfg, prompt, other, n_gen);

    std::vector<int32_t> hist = prompt;
    const float * lg = nullptr;

    // ---- prefill in ubatches ----------------------------------------------
    {
        const auto t0 = std::chrono::steady_clock::now();
        int32_t done = 0;
        while (done < (int32_t) prompt.size()) {
            const int32_t take = std::min<int32_t>(cfg.n_batch, (int32_t) prompt.size() - done);
            lg = eng.eval(hist.data(), done + take, take, err);
            if (!lg) { fprintf(stderr, "prefill failed: %s\n", err.c_str()); return 1; }
            done += take;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("prefill: %zu tokens in %.2f s  (%.1f tok/s)\n", prompt.size(), dt, prompt.size() / dt);
    }

    // ---- greedy decode ------------------------------------------------------
    printf(replay.empty() ? "generated:" : "replaying:");
    if (!replay.empty()) n_gen = std::min<int>(n_gen, (int) replay.size());
    const auto t0 = std::chrono::steady_clock::now();
    double nll_sum = 0.0; int nll_n = 0;
    auto argmax = [&](const float * l) { int b = 0; for (int64_t v = 1; v < eng.n_vocab(); v++) if (l[v] > l[b]) b = (int) v; return b; };
    auto score  = [&](const float * l, int32_t tok) {   // -log softmax(l)[tok]
        float mx = l[0]; for (int64_t v = 1; v < eng.n_vocab(); v++) mx = std::max(mx, l[v]);
        double z = 0.0; for (int64_t v = 0; v < eng.n_vocab(); v++) z += std::exp((double) l[v] - mx);
        const double v = -((double) l[tok] - mx - std::log(z));
        nll_sum += v; nll_n++;
        static const bool verbose = getenv("QWFN_PPL_VERBOSE") != nullptr;
        if (verbose) fprintf(stderr, "[nll] #%d tok %d: %.6f\n", nll_n, tok, v);
    };
    const bool use_mtp = !cfg.mtp_path.empty() && !getenv("QWFN_MTP_NOVERIFY");   // NOVERIFY: the head loaded (its VRAM taken) but the plain loop
    if ((pair_test || rollback_test) && replay.empty()) { fprintf(stderr, "--pair-test / --rollback-test need --replay-file\n"); return 1; }
    uint64_t sp_steps = 0, sp_acc = 0, sp_single = 0;
    std::vector<uint64_t> sp_accepted_at(engine::MTP_MAX_DRAFTS + 1, 0);   // verify steps that kept exactly j drafts
    // Decode progress by segment (QWFN_SEGMENTS=N): tok/s and the tiers' share over
    // each N tokens, to see how long the decode after a prompt takes to warm up.
    const int seg_n = getenv("QWFN_SEGMENTS") ? atoi(getenv("QWFN_SEGMENTS")) : 0;
    int seg_next = seg_n; auto seg_t = t0; expert_cache_stats seg_st = eng.cache_stats();
    auto segment = [&](int done) {
        if (seg_n <= 0 || done < seg_next) return;
        const auto now = std::chrono::steady_clock::now(); const expert_cache_stats & s = eng.cache_stats();
        const double dts = std::chrono::duration<double>(now - seg_t).count();
        const uint64_t lk = s.lookups - seg_st.lookups, gh = s.gpu_hits - seg_st.gpu_hits, hh = s.hits - seg_st.hits;
        fprintf(stderr, "[segment] tokens %d-%d: %.2f tok/s, %.1f%% from VRAM, %.1f%% hit, %.2f GB read\n",
                seg_next - seg_n, done, (done - (seg_next - seg_n)) / dts, 100.0 * gh / std::max<uint64_t>(1, lk),
                100.0 * hh / std::max<uint64_t>(1, lk), (s.bytes_from_disk - seg_st.bytes_from_disk) / 1e9);
        seg_t = now; seg_st = s; seg_next += seg_n;
    };
    if (!use_mtp && !pair_test && !rollback_test) {
        for (int i = 0; i < n_gen; i++) {
            int best = 0;
            if (replay.empty()) best = argmax(lg);
            else { best = replay[i]; if (want_ppl) score(lg, best); }
            printf(" %d", best);
            fflush(stdout);
            hist.push_back(best);
            segment(i + 1);
            lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
            if (!lg) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return 1; }
        }
    } else {
        // Speculative loop. `next` is the token to feed; the head's draft for the
        // token after it (or, in the tests, the replay's own next token / a wrong
        // one) rides along as the second of a pair. Position 0's logits verify the
        // draft; position 1's are the next token's if it is accepted. The replay
        // is the ground truth for acceptance when replaying; argmax when not.
        // NLL covers replay[0..n_gen) as in the plain loop.
        int i = 0;
        int32_t next = replay.empty() ? argmax(lg) : replay[0];
        if (want_ppl && !replay.empty()) score(lg, replay[0]);
        if (use_mtp && !eng.mtp_step(&next, 1, err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return 1; }
        int produced = 0;
        auto next_after = [&](const float * l, int k) -> int32_t {   // the token at replay index k, or argmax
            if (replay.empty()) return argmax(l);
            return k < (int) replay.size() ? replay[k] : -1;
        };
        // QWFN_MTP_DRAFT2_TEST=1: after every draft, draft once more from the head's
        // own residual and score both against the replay -- the second draft's
        // accuracy given the first was right is what a longer draft would earn.
        static const bool draft2_test = getenv("QWFN_MTP_DRAFT2_TEST") != nullptr;
        uint64_t d2_n1 = 0, d2_hit1 = 0, d2_n2 = 0, d2_hit2 = 0;
        auto draft2_probe = [&](int idx_next) {   // idx_next: replay index of the token that follows the drafted-from position
            if (!draft2_test || !use_mtp || replay.empty()) return;
            const int32_t d1 = eng.mtp_draft_id();
            if (d1 < 0 || idx_next + 1 >= (int) replay.size()) return;
            if (!eng.mtp_draft_next(err)) { fprintf(stderr, "\ndraft2: %s\n", err.c_str()); return; }
            const int32_t d2 = eng.mtp_draft2_id();
            d2_n1++;
            if (d1 == replay[idx_next]) { d2_hit1++; d2_n2++; if (d2 == replay[idx_next + 1]) d2_hit2++; }
        };
        // Layer 0's reads for the tokens the next eval will take, issued as soon as
        // they are known: the bonus token before the head runs, the pair once it has.
        auto spec_l0 = [&](const int32_t * toks, int n) {
            if (!use_mtp) return;   // no lead without the head: the eval follows at once
            for (int k = 0; k < n; k++) hist.push_back(toks[k]);
            eng.spec_layer0(hist.data(), (int32_t) hist.size(), n, err);
            for (int k = 0; k < n; k++) hist.pop_back();
        };
        // The verify step: a token and K drafts (the head's, or the replay's for the
        // tests), evaluated as one step of K+1 positions; the trunk's pick after each
        // position decides how many drafts stand, the rest is rolled back.
        std::vector<int32_t> drafts; drafts.reserve(engine::MTP_MAX_DRAFTS);
        const int rb_at = getenv("QWFN_RB_AT") ? atoi(getenv("QWFN_RB_AT")) : 1;   // --rollback-test: the draft that is wrong (1-based)
        while (produced < n_gen) {
            drafts.clear();
            int K = 0;
            if (pair_test || rollback_test) {
                K = std::max(1, multi_test - 1);
                for (int k = 1; k <= K; k++) {
                    if (i + k >= (int) replay.size()) break;
                    const int32_t d = (rollback_test && k == rb_at)
                        ? (int32_t) ((replay[i + k] + (getenv("QWFN_RB_WRONG") ? atoi(getenv("QWFN_RB_WRONG")) : 1)) % eng.n_vocab())
                        : replay[i + k];
                    drafts.push_back(d);
                }
            } else if (use_mtp && eng.mtp_draft_id() >= 0) {
                if (!eng.mtp_draft_more((int) cfg.mtp_drafts, err)) { fprintf(stderr, "\ndraft: %s\n", err.c_str()); return 1; }
                for (int k = 0; k < eng.mtp_draft_count(); k++) drafts.push_back(eng.mtp_draft_k(k));
            }
            // Room for the drafts: within the requested count and the replay.
            while (!drafts.empty() && (produced + (int) drafts.size() >= n_gen || (!replay.empty() && i + (int) drafts.size() >= (int) replay.size())))
                drafts.pop_back();
            K = (int) drafts.size();
            if (K == 0) {
                hist.push_back(next);
                lg = eng.eval_decode(hist.data(), (int32_t) hist.size(), 1, err);
                if (!lg) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return 1; }
                printf(" %d", next); fflush(stdout); produced++; sp_single++;
                const int32_t y = next_after(lg, i + 1);
                if (want_ppl && !replay.empty() && i + 1 < n_gen) score(lg, replay[i + 1]);
                i += 1;
                if (y < 0) break;
                spec_l0(&y, 1);
                if (use_mtp && !eng.mtp_step(&y, 1, err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return 1; }
                next = y;
                continue;
            }
            {
                std::vector<int32_t> step; step.push_back(next); for (int32_t d : drafts) step.push_back(d);
                spec_l0(step.data(), (int) step.size());
                for (int32_t s : step) hist.push_back(s);
            }
            if (!eng.eval_decode(hist.data(), (int32_t) hist.size(), K + 1, err)) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return 1; }
            printf(" %d", next); fflush(stdout); produced++;
            int j = 0; int32_t y = -1;
            for (j = 0; j < K; j++) {
                const float * lj = eng.logits_pos(j);
                y = next_after(lj, i + 1 + j);
                if (want_ppl && !replay.empty() && i + 1 + j < n_gen) score(lj, replay[i + 1 + j]);
                const bool ok = rollback_test ? (j + 1 != rb_at) : pair_test ? true : (y == drafts[j]);
                if (!ok) break;
                printf(" %d", drafts[j]); fflush(stdout); produced++;
            }
            sp_steps++; sp_acc += j; sp_accepted_at[j]++;
            std::vector<int32_t> fed(drafts.begin(), drafts.begin() + j);
            if (j == K) {
                const float * lK = eng.logits_pos(K);
                y = next_after(lK, i + 1 + K);
                if (want_ppl && !replay.empty() && i + 1 + K < n_gen) score(lK, replay[i + 1 + K]);
                i += K + 1;
            } else {
                if (!eng.rollback_n(K - j, err)) { fprintf(stderr, "\nrollback: %s\n", err.c_str()); return 1; }
                hist.resize(hist.size() - (size_t) (K - j));
                i += j + 1;
            }
            segment(i);
            if (y < 0) break;
            fed.push_back(y);
            spec_l0(&y, 1);
            if (use_mtp && !eng.mtp_step(fed.data(), (int) fed.size(), err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return 1; }
            next = y;
        }
        n_gen = produced;
        if (draft2_test && d2_n1)
            printf("\ndraft-2 probe: first draft right %llu of %llu (%.1f%%); second draft right %llu of %llu when the first was (%.1f%%)\n",
                   (unsigned long long) d2_hit1, (unsigned long long) d2_n1, 100.0 * d2_hit1 / d2_n1,
                   (unsigned long long) d2_hit2, (unsigned long long) d2_n2, d2_n2 ? 100.0 * d2_hit2 / d2_n2 : 0.0);
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("\n\ndecode: %d tokens in %.2f s  (%.2f tok/s), n_past=%d\n", n_gen, dt, n_gen / dt, eng.n_past());
    if (eng.n_replay)
        printf("cached graph replays: %llu; per replay alloc %.0f us, launch %.0f us, wait %.0f us\n", (unsigned long long) eng.n_replay,
               eng.t_replay_alloc / eng.n_replay * 1e6, eng.t_replay_launch / eng.n_replay * 1e6, eng.t_replay_wait / eng.n_replay * 1e6);
    if (const char * pl = getenv("QWFN_PROFILE_LAYERS")) {   // "3,4,5": per-node timing of those layers' cached decode graphs; "all": every layer's whole graph
        std::string s(pl); size_t p = 0;
        if (s == "all") { eng.profile_all_graphs(); s.clear(); }
        while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) eng.profile_layer_graph((uint32_t) atoi(s.substr(p, q - p).c_str())); p = q + 1; }
    }
    if (!save_replay.empty()) {
        FILE * f = fopen(save_replay.c_str(), "wb");
        if (f) { for (size_t i = hist.size() - n_gen; i < hist.size(); i++) fprintf(f, "%d\n", hist[i]); fclose(f);
                 printf("saved %d generated ids to %s\n", n_gen, save_replay.c_str()); }
        else fprintf(stderr, "cannot write %s\n", save_replay.c_str());
    }
    if (nll_n) printf("replay NLL: %.4f per token (ppl %.2f) over %d tokens\n", nll_sum / nll_n, std::exp(nll_sum / nll_n), nll_n);
    if (eng.n_exp_skipped) printf("skipped experts: %llu (misses computed without)\n", (unsigned long long) eng.n_exp_skipped);
    if (eng.n_spec_l0) printf("layer-0 speculation: %llu calls, %.2f ms each\n", (unsigned long long) eng.n_spec_l0, 1e3 * eng.t_spec_l0 / eng.n_spec_l0);
    if (eng.t_mtp_pre + eng.t_mtp_moe + eng.t_mtp_post > 0)
        printf("head split: dense half %.2f s | CPU experts + transfers %.2f s | second half + LM head + argmax %.2f s\n", eng.t_mtp_pre, eng.t_mtp_moe, eng.t_mtp_post);
    if (getenv("QWFN_IO_PROFILE") && !eng.prof_io_end.empty() && n_gen > 0) {
        // Where the decode loop waits for reads, layer by layer, per decoded token.
        printf("io profile (ms per token; begin = wait when the layer's fetch is issued, end = wait for its reads after the ready pass, reads = demand reads issued by the fetch):\n");
        double tb = 0, te = 0; uint64_t tr = 0;
        for (size_t il = 0; il < eng.prof_io_end.size(); il++) {
            tb += eng.prof_io_begin[il]; te += eng.prof_io_end[il]; tr += eng.prof_reads[il];
            printf("  L%-2zu %s begin %5.2f end %5.2f reads %5.2f%s", il, eng.is_attn_layer((uint32_t) il) ? "A" : "R",
                   1e3 * eng.prof_io_begin[il] / n_gen, 1e3 * eng.prof_io_end[il] / n_gen, (double) eng.prof_reads[il] / n_gen, (il % 3 == 2) ? "\n" : " |");
        }
        printf("\n  total begin %.1f ms, end %.1f ms, %.1f demand reads per token; layer 0 begin %.2f ms\n",
               1e3 * tb / n_gen, 1e3 * te / n_gen, (double) tr / n_gen, 1e3 * eng.prof_io_begin[0] / n_gen);
    }
    if (getenv("QWFN_VRAM_AUDIT")) {
        size_t ab = 0, mb = 0; int ag = 0, mg = 0; eng.graph_buffer_bytes(ab, ag, mb, mg);
        printf("VRAM audit: cached decode graphs hold %.0f MB of activations in %d layer graphs (%.1f MB each), MoE graphs %.0f MB in %d\n",
               ab / 1e6, ag, ag ? ab / 1e6 / ag : 0.0, mb / 1e6, mg);
    }
    if (eng.n_exp_dropped) printf("dropped experts: %llu (gate below --gate-drop; %.1f%% of routed)\n", (unsigned long long) eng.n_exp_dropped,
                                  100.0 * eng.n_exp_dropped / std::max<uint64_t>(1, eng.n_exp_dropped + eng.n_exp_gpu + eng.n_exp_cpu + eng.n_exp_skipped));

    const auto & s = eng.cache_stats();
    printf("expert cache: %.1f%% hit, %.1f%% from VRAM, %.2f GB from disk\n",
           100.0 * s.hit_rate(), 100.0 * s.gpu_rate(), s.bytes_from_disk / 1e9);
    printf("              tiers: %llu promotions, %llu RAM evictions, %llu cold-file reads, %llu upgrades\n",
           (unsigned long long) s.promotions, (unsigned long long) s.evictions,
           (unsigned long long) s.cold_tier_reads, (unsigned long long) s.upgrades);
    {
        const auto c = eng.ram_census();
        printf("              RAM tier at end: %llu slots = %llu hot + %llu cold (%llu of them hot-worthy) + %llu empty; %llu prefetched unused, %llu in flight; %llu experts marked hot-worthy\n",
               (unsigned long long) c.slots, (unsigned long long) c.hot, (unsigned long long) c.cold, (unsigned long long) c.cold_hotw,
               (unsigned long long) c.empty, (unsigned long long) c.speculative, (unsigned long long) c.inflight, (unsigned long long) c.hotw_marked);
    }
    printf("decode split: graphA(GPU) %.2f s | MoE gpu %.2f s (%llu experts, sync-wait %.2f s) | MoE cpu %.2f s (%llu experts) | io %.2f s\n",
           eng.t_layerA, eng.t_moe_gpu, (unsigned long long) eng.n_exp_gpu, eng.t_moe_gpu_sync,
           eng.t_moe_cpu, (unsigned long long) eng.n_exp_cpu, eng.t_io);
    if (eng.n_exp_cpu && eng.n_exp_gpu)
        printf("              per expert: gpu %.0f us, cpu %.0f us  (%.1fx)\n",
               eng.t_moe_gpu / eng.n_exp_gpu * 1e6, eng.t_moe_cpu / eng.n_exp_cpu * 1e6,
               (eng.t_moe_cpu / eng.n_exp_cpu) / (eng.t_moe_gpu / eng.n_exp_gpu));
    if (eng.n_layerA_rec && eng.n_layerA_attn)
        printf("              graphA per decode layer: recurrent %.0f us (x%llu), attention %.0f us (x%llu)"
               " [attention: build+alloc %.0f us, gpu %.0f us; per-token inputs %.0f us]\n",
               eng.t_layerA_rec / eng.n_layerA_rec * 1e6, (unsigned long long) eng.n_layerA_rec,
               eng.t_layerA_attn / eng.n_layerA_attn * 1e6, (unsigned long long) eng.n_layerA_attn,
               eng.t_attn_build / eng.n_layerA_attn * 1e6, eng.t_attn_compute / eng.n_layerA_attn * 1e6,
               eng.n_decode ? eng.t_inputs / eng.n_decode * 1e6 : 0.0);
    printf("prefetch: %.1f%% of the next layer's experts predicted correctly",
           eng.pred_total ? 100.0 * eng.pred_hits / eng.pred_total : 0.0);
    if (eng.pred2_total)
        printf(" (two ahead: %.1f%%)", 100.0 * eng.pred2_hits / eng.pred2_total);
    printf(" | %llu issued, %llu used (%.1f%%), %llu wasted\n",
           (unsigned long long) s.pf_issued, (unsigned long long) s.pf_used,
           s.pf_issued ? 100.0 * s.pf_used / s.pf_issued : 0.0,
           (unsigned long long) s.pf_wasted);
    if (sp_steps) {
        printf("verify steps by drafts kept:");
        for (size_t j = 0; j < sp_accepted_at.size(); j++) if (sp_accepted_at[j]) printf(" %zu:%llu", j, (unsigned long long) sp_accepted_at[j]);
        printf("  (tokens per step %.2f)\n", (double) (sp_steps + sp_acc) / sp_steps);
    }
    if (sp_steps || sp_single)
        printf("speculative: %llu pair steps, %llu accepted (%.1f%%), %llu single steps, %llu rollbacks (%.3f s), head %.2f s\n",
               (unsigned long long) sp_steps, (unsigned long long) sp_acc, sp_steps ? 100.0 * sp_acc / sp_steps : 0.0,
               (unsigned long long) sp_single, (unsigned long long) eng.n_rollback, eng.t_rollback, eng.t_mtp);
    if (eng.mtp_n || eng.mtp_prompt_n)
        printf("mtp draft: %llu decode drafts scored, %llu accepted (%.1f%%), top-3 %.1f%% | prompt: %llu scored, %.1f%% accepted | %.2f s in the head\n",
               (unsigned long long) eng.mtp_n, (unsigned long long) eng.mtp_acc,
               eng.mtp_n ? 100.0 * eng.mtp_acc / eng.mtp_n : 0.0, eng.mtp_n ? 100.0 * eng.mtp_top3 / eng.mtp_n : 0.0,
               (unsigned long long) eng.mtp_prompt_n, eng.mtp_prompt_n ? 100.0 * eng.mtp_prompt_acc / eng.mtp_prompt_n : 0.0, eng.t_mtp);
    if (eng.pf_gated)
        printf("prefetch gate: %llu predicted candidates not read (margin < %.2f%s)\n",
               (unsigned long long) eng.pf_gated, cfg.spec_margin,
               cfg.spec_gate_inflight ? ", only with reads in flight" : "");
    {   // precision by predicted rank, then by confidence margin, then recall by layer
        printf("prediction by rank, precision %%:");
        for (int k = 0; k < (int) QWFN_SPEC_MAX; k++)
            if (eng.rank_total[k]) printf(" r%d %.0f", k + 1, 100.0 * eng.rank_hits[k] / eng.rank_total[k]);
        printf("\n");
        unsigned long long mt = 0; for (int b = 0; b < engine::SPEC_MARGIN_BUCKETS; b++) mt += eng.margin_total[b];
        printf("prediction by margin [edge+) P(correct)%% / share%%:");
        for (int b = 0; b < engine::SPEC_MARGIN_BUCKETS; b++)
            if (eng.margin_total[b])
                printf(" [%.2g) %.0f/%.1f", engine::margin_edge(b), 100.0 * eng.margin_hits[b] / eng.margin_total[b],
                       mt ? 100.0 * eng.margin_total[b] / mt : 0.0);
        printf("\n");
        std::vector<std::pair<double, int>> acc;
        for (size_t l = 0; l < eng.pred_total_layer.size(); l++)
            if (eng.pred_total_layer[l]) acc.push_back({ 100.0 * eng.pred_hits_layer[l] / eng.pred_total_layer[l], (int) l });
        std::sort(acc.begin(), acc.end());
        printf("prediction by layer, worst 10:");
        for (size_t i = 0; i < acc.size() && i < 10; i++) printf(" L%d %.0f", acc[i].second, acc[i].first);
        if (acc.size() > 10) printf("  | best: L%d %.0f", acc.back().second, acc.back().first);
        printf("\n");
    }
    printf("io breakdown: submit %.2f s, promote %.2f s, wait %.2f s | %llu bursts, %llu reads, "
           "%.1f reads/burst, %.0f KiB/read\n",
           s.t_submit, s.t_promote, s.t_wait,
           (unsigned long long) s.n_bursts, (unsigned long long) s.n_reads,
           s.n_bursts ? (double) s.n_reads / s.n_bursts : 0.0,
           s.n_reads ? s.bytes_from_disk / (double) s.n_reads / 1024.0 : 0.0);
    printf("             wait-only bandwidth: %.2f GB/s   (device peak measured 7.0 GB/s)\n",
           s.t_wait > 0 ? s.bytes_from_disk / s.t_wait / 1e9 : 0.0);
    {
        const auto & io = eng.cache_io();
        printf("             io_engine: backend=%s, O_DIRECT=%d, %llu reads, %.2f GB actually read, "
               "prep %.2f s, io_uring_submit %.2f s\n",
               io.which() == io_engine::backend::threads ? "threads" : "uring",
               (int) io.direct_io(), (unsigned long long) io.stat_reads,
               io.stat_bytes / 1e9, io.stat_t_prep, io.stat_t_submit_syscall);
    }
    printf("engine time: prefill %.2f s / %lld tok, decode %.2f s / %lld tok, expert io %.2f s\n",
           eng.t_prefill, (long long) eng.n_prefill, eng.t_decode, (long long) eng.n_decode, eng.t_io);
    if (eng.n_prefill > 0 && (eng.t_pf_graphA > 0 || eng.t_pf_moe > 0))
        printf("prefill split: dense graphs %.2f s | expert reads (blocking) %.2f s | MoE %.2f s | warm-up %.2f s  (of %.2f s)\n",
               eng.t_pf_graphA, eng.t_pf_read, eng.t_pf_moe, eng.t_warm, eng.t_prefill);
    if (s.warm_admitted || s.warm_promoted)
        printf("cache warm-up from prefill: %llu blocks into RAM, %llu on to VRAM, %.2f s\n",
               (unsigned long long) s.warm_admitted, (unsigned long long) s.warm_promoted, eng.t_warm);
    if (eng.prefill_bytes_read() || eng.prefill_bytes_from_ram())
        printf("streamed sweeps: %.2f GB of experts read, %.2f GB taken from the RAM tier\n",
               eng.prefill_bytes_read() / 1e9, eng.prefill_bytes_from_ram() / 1e9);
    return 0;
}
