// qwfn-pack -- build an expert pack (src/qwfn_pack.h) from a GGUF checkpoint, or check one.
//
//   qwfn-pack MODEL.gguf OUT.qwpk [--verify N]    build, then compare N random experts (default 64)
//   qwfn-pack --verify MODEL.gguf PACK.qwpk [N]   compare N random experts of an existing pack
//
// In the pack every decode miss or prefetch of an expert can be one read of its
// whole block instead of three reads of its parts (measured on the reference NVMe
// with qwfn-iobench: ~+10% read throughput over io_uring). The engine-side reader
// is being measured and is not part of this tool. The GGUF stays where it is and
// is still needed: the dense core, the prefill and everything else come from it.
//
// Both files are read and written with O_DIRECT in large sequential pieces, so the
// page cache is left alone; the write of one group overlaps the read of the next.
// The pack is written to OUT.partial and renamed when complete; its header goes
// in last, so an interrupted run never leaves a file the engine would accept.
#include "qwfn_model.h"
#include "qwfn_pack.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace qwfn;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static void * aligned(size_t n) {
    void * p = nullptr;
    if (posix_memalign(&p, QWFN_PACK_ALIGN, (n + QWFN_PACK_ALIGN - 1) / QWFN_PACK_ALIGN * QWFN_PACK_ALIGN)) return nullptr;
    return p;
}

static int open_direct(const char * path, int flags, mode_t mode = 0) {
    int fd = open(path, flags | O_DIRECT, mode);
    if (fd < 0 && errno == EINVAL) fd = open(path, flags, mode);   // a filesystem without O_DIRECT
    return fd;
}

// Full read; a short count only at end of file. -1 on error.
static ssize_t read_full(int fd, void * dst, size_t n, uint64_t off) {
    size_t got = 0;
    while (got < n) {
        const ssize_t k = pread(fd, (uint8_t *) dst + got, n - got, (off_t) (off + got));
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) break;
        got += (size_t) k;
    }
    return (ssize_t) got;
}

static bool write_full(int fd, const void * src, size_t n, uint64_t off) {
    size_t put = 0;
    while (put < n) {
        const ssize_t k = pwrite(fd, (const uint8_t *) src + put, n - put, (off_t) (off + put));
        if (k < 0) { if (errno == EINTR) continue; return false; }
        put += (size_t) k;
    }
    return true;
}

static std::string dir_of(const std::string & p) {
    const size_t s = p.find_last_of('/');
    return s == std::string::npos ? "." : s == 0 ? "/" : p.substr(0, s);
}

// Compare n random (layer, expert) payloads of the pack against the GGUF.
static bool verify(const model_index & mi, const pack_layout & L, const std::string & pack_path, int n) {
    const int pfd = open_direct(pack_path.c_str(), O_RDONLY);
    if (pfd < 0) { fprintf(stderr, "error: cannot open %s: %s\n", pack_path.c_str(), strerror(errno)); return false; }
    std::vector<int> sfd;
    for (const std::string & p : mi.shard_paths()) sfd.push_back(open_direct(p.c_str(), O_RDONLY));
    uint32_t max_block = 0, max_part = 0;
    for (const pack_layer & pl : L.layers) {
        max_block = std::max(max_block, pl.stride);
        for (int q = 0; q < EXPERT_NPARTS; q++) max_part = std::max(max_part, pl.part_bytes[q]);
    }
    uint8_t * blk = (uint8_t *) aligned(max_block);
    uint8_t * src = (uint8_t *) aligned(max_part + QWFN_PACK_ALIGN);
    std::mt19937_64 g((uint64_t) time(nullptr));
    int bad = 0;
    for (int i = 0; i < n; i++) {
        // Always include the first and the last block: the two edges of the file.
        const uint32_t il = i == 0 ? 0 : i == 1 ? L.n_layer - 1 : (uint32_t) (g() % L.n_layer);
        const uint32_t e  = i == 0 ? 0 : i == 1 ? L.n_expert - 1 : (uint32_t) (g() % L.n_expert);
        const pack_layer & pl = L.layers[il];
        if (read_full(pfd, blk, pl.stride, pl.offset + (uint64_t) e * pl.stride) != (ssize_t) pl.stride) {
            fprintf(stderr, "  layer %u expert %u: short read from the pack\n", il, e); bad++; continue;
        }
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range r = mi.expert_range(il, e, (expert_part) q);
            const uint64_t a0 = r.offset / QWFN_PACK_ALIGN * QWFN_PACK_ALIGN;
            const size_t   pad = (size_t) (r.offset - a0);
            const size_t   len = (pad + r.nbytes + QWFN_PACK_ALIGN - 1) / QWFN_PACK_ALIGN * QWFN_PACK_ALIGN;
            if (sfd[r.shard] < 0 || read_full(sfd[r.shard], src, len, a0) < (ssize_t) (pad + r.nbytes) ||
                memcmp(blk + pl.part_off[q] + pl.part_pay[q], src + pad, r.nbytes) != 0) {
                fprintf(stderr, "  layer %u expert %u part %d: MISMATCH\n", il, e, q); bad++;
            }
        }
    }
    free(blk); free(src);
    close(pfd);
    for (int f : sfd) if (f >= 0) close(f);
    printf("verify: %d random experts (3 parts each) against the GGUF: %s\n", n, bad ? "FAILED" : "identical");
    return bad == 0;
}

static int usage() {
    fprintf(stderr, "usage: qwfn-pack MODEL.gguf OUT.qwpk [--verify N]\n"
                    "       qwfn-pack --verify MODEL.gguf PACK.qwpk [N]\n");
    return 2;
}

int main(int argc, char ** argv) {
    if (argc < 3) return usage();
    std::string err;

    if (std::string(argv[1]) == "--verify") {
        if (argc < 4) return usage();
        model_index mi;
        if (!mi.load(argv[2], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        pack_layout L;
        if (!pack_open(argv[3], mi, L, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        return verify(mi, L, argv[3], argc > 4 ? atoi(argv[4]) : 256) ? 0 : 1;
    }

    const std::string model_path = argv[1], out_path = argv[2];
    int n_verify = 64;
    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--verify" && i + 1 < argc) n_verify = atoi(argv[++i]);
        else return usage();
    }

    model_index mi;
    if (!mi.load(model_path, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    pack_layout L;
    if (!pack_compute_layout(mi, 0, L, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    uint64_t payload = 0;
    for (uint32_t il = 0; il < L.n_layer; il++)
        for (int q = 0; q < EXPERT_NPARTS; q++) payload += (uint64_t) L.layers[il].part_bytes[q] * L.n_expert;
    printf("model     : %s (%zu shards)\n", model_path.c_str(), mi.shard_paths().size());
    printf("pack      : %s\n", out_path.c_str());
    printf("layout    : %u layers x %u experts, %u-byte slots (the engine's direct-I/O alignment for this file), block %.2f-%.2f MiB\n"
           "            %.2f GB (%.2f GB of weights, %.2f%% padding)\n",
           L.n_layer, L.n_expert, L.dio_align,
           L.layers[0].block_bytes / 1048576.0, L.layers.back().block_bytes / 1048576.0,
           L.file_bytes / 1e9, payload / 1e9, 100.0 * (L.file_bytes - payload) / L.file_bytes);

    struct statvfs sv{};
    if (statvfs(dir_of(out_path).c_str(), &sv) == 0) {
        const uint64_t avail = (uint64_t) sv.f_bavail * sv.f_frsize;
        if (avail < L.file_bytes + (2ull << 30)) {
            fprintf(stderr, "error: %.1f GB free in %s, the pack needs %.1f GB (+2 GB margin)\n",
                    avail / 1e9, dir_of(out_path).c_str(), L.file_bytes / 1e9);
            return 1;
        }
    }

    const std::string tmp_path = out_path + ".partial";
    const int ofd = open_direct(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) { fprintf(stderr, "error: cannot create %s: %s\n", tmp_path.c_str(), strerror(errno)); return 1; }
    // Reserve the whole file up front: one allocation, as contiguous as the
    // filesystem can make it, so the pack's blocks are as sequential on the
    // drive as they are in the file.
    if (posix_fallocate(ofd, 0, (off_t) L.file_bytes) != 0)
        fprintf(stderr, "note: could not preallocate %.1f GB; writing without it\n", L.file_bytes / 1e9);

    std::vector<int> sfd;
    for (const std::string & p : mi.shard_paths()) {
        const int f = open_direct(p.c_str(), O_RDONLY);
        if (f < 0) { fprintf(stderr, "error: cannot open %s: %s\n", p.c_str(), strerror(errno)); return 1; }
        sfd.push_back(f);
    }

    // Groups of experts, ~256 MB of output each; two output buffers so the write
    // of one group runs while the next is read.
    uint32_t max_block = 0, max_part = 0;
    for (const pack_layer & pl : L.layers) {
        max_block = std::max(max_block, pl.stride);
        for (int q = 0; q < EXPERT_NPARTS; q++) max_part = std::max(max_part, pl.part_bytes[q]);
    }
    const uint32_t G = std::max<uint32_t>(1, std::min<uint32_t>(L.n_expert, (256u << 20) / max_block));
    uint8_t * outb[2] = { (uint8_t *) aligned((size_t) G * max_block), (uint8_t *) aligned((size_t) G * max_block) };
    uint8_t * inb = (uint8_t *) aligned((size_t) G * max_part + 2 * QWFN_PACK_ALIGN);
    if (!outb[0] || !outb[1] || !inb) { fprintf(stderr, "error: out of memory for the group buffers\n"); return 1; }

    std::thread writer;
    bool write_ok = true;
    int cur = 0;
    uint64_t done_bytes = 0;
    const auto t0 = clk::now();
    for (uint32_t il = 0; il < L.n_layer; il++) {
        const pack_layer & pl = L.layers[il];
        for (uint32_t e0 = 0; e0 < L.n_expert; e0 += G) {
            const uint32_t n = std::min(G, L.n_expert - e0);
            uint8_t * out = outb[cur];
            memset(out, 0, (size_t) n * pl.stride);
            for (int q = 0; q < EXPERT_NPARTS; q++) {
                // n consecutive experts of one tensor are one contiguous range.
                const byte_range r = mi.expert_range(il, e0, (expert_part) q);
                const uint64_t a0   = r.offset / QWFN_PACK_ALIGN * QWFN_PACK_ALIGN;   // O_DIRECT: page-aligned window
                const size_t   pad  = (size_t) (r.offset - a0);
                const size_t   want = pad + (size_t) n * r.nbytes;
                const size_t   len  = (want + QWFN_PACK_ALIGN - 1) / QWFN_PACK_ALIGN * QWFN_PACK_ALIGN;
                const ssize_t  got  = read_full(sfd[r.shard], inb, len, a0);
                if (got < (ssize_t) want) {
                    fprintf(stderr, "error: short read on layer %u part %d (%s)\n", il, q, got < 0 ? strerror(errno) : "end of file");
                    return 1;
                }
                for (uint32_t i = 0; i < n; i++)
                    memcpy(out + (size_t) i * pl.stride + pl.part_off[q] + pl.part_pay[q], inb + pad + (size_t) i * r.nbytes, r.nbytes);
            }
            if (writer.joinable()) writer.join();
            if (!write_ok) { fprintf(stderr, "error: write to %s failed: %s\n", tmp_path.c_str(), strerror(errno)); return 1; }
            const uint64_t woff = pl.offset + (uint64_t) e0 * pl.stride;
            const size_t   wlen = (size_t) n * pl.stride;
            writer = std::thread([&, out, woff, wlen] { if (!write_full(ofd, out, wlen, woff)) write_ok = false; });
            cur ^= 1;
            done_bytes += wlen;
        }
        const double dt = secs(t0, clk::now());
        printf("\r  layer %2u/%u   %6.1f / %.1f GB   %.2f GB/s   ", il + 1, L.n_layer, done_bytes / 1e9, L.file_bytes / 1e9, done_bytes / 1e9 / dt);
        fflush(stdout);
    }
    if (writer.joinable()) writer.join();
    printf("\n");
    if (!write_ok) { fprintf(stderr, "error: write to %s failed: %s\n", tmp_path.c_str(), strerror(errno)); return 1; }
    if (fdatasync(ofd) != 0) { fprintf(stderr, "error: fdatasync: %s\n", strerror(errno)); return 1; }

    // Header last: until it is on disk the file has no magic and nothing accepts it.
    uint8_t * hdr = (uint8_t *) aligned(QWFN_PACK_HEADER);
    pack_encode_header(L, hdr);
    if (!write_full(ofd, hdr, QWFN_PACK_HEADER, 0) || fsync(ofd) != 0) {
        fprintf(stderr, "error: writing the header: %s\n", strerror(errno)); return 1;
    }
    free(hdr);
    close(ofd);
    for (int f : sfd) close(f);
    free(outb[0]); free(outb[1]); free(inb);
    if (rename(tmp_path.c_str(), out_path.c_str()) != 0) {
        fprintf(stderr, "error: rename %s -> %s: %s\n", tmp_path.c_str(), out_path.c_str(), strerror(errno)); return 1;
    }
    printf("written   : %.1f GB in %.0f s (%.2f GB/s)\n", L.file_bytes / 1e9, secs(t0, clk::now()), L.file_bytes / 1e9 / secs(t0, clk::now()));

    pack_layout chk;
    if (!pack_open(out_path, mi, chk, err)) { fprintf(stderr, "error: the written pack does not open: %s\n", err.c_str()); return 1; }
    if (n_verify > 0 && !verify(mi, chk, out_path, n_verify)) return 1;
    printf("check it  : qwfn-pack --verify %s %s\n", model_path.c_str(), out_path.c_str());
    return 0;
}
