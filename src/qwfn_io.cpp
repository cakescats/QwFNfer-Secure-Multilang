#include "qwfn_io.h"
#include "qwfn_plat.h"

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <liburing.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#if !defined(_WIN32)
#  include <dlfcn.h>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace qwfn {

static uint64_t g_dio_align = 512;
uint64_t dio_align() { return g_dio_align; }
void     set_dio_align(uint64_t a) { g_dio_align = a == QWFN_DIO_PAGE ? QWFN_DIO_PAGE : 512; }

// Always a page, whatever the slot layout is: a destination that satisfies the
// page also satisfies any sector size below it, and the RAM arena is one of
// these, so over-aligning it costs one page across fifteen gigabytes.
void * dio_alloc(size_t bytes) {
    return plat_alloc_aligned(dio_align_up(bytes), QWFN_DIO_PAGE);
}

void dio_free(void * p) { plat_free_aligned(p); }

#if !defined(_WIN32)
bool host_lock_requested() {
    static const bool on = [] { const char * e = getenv("QWFN_LOCK_HOST"); return e && *e && strcmp(e, "0") != 0; }();
    return on;
}

namespace {
long vmlck_kb() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "VmLck: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

// The Level Zero import calls, looked up through the loader SYCL has already loaded, so the
// engine carries no build dependency on Level Zero. Experimental functions are not exported
// symbols: the driver hands them out by name.
struct ze_import {
    void * drv = nullptr;
    int (*import_fn)(void *, void *, size_t) = nullptr;
    int (*release_fn)(void *, void *)        = nullptr;
};
ze_import & ze() {
    static ze_import z;
    static std::once_flag once;
    std::call_once(once, [] {
        void * lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!lib) return;   // no Level Zero in this process (CPU or CUDA build): lock only
        auto get = (int (*)(uint32_t *, void **)) dlsym(lib, "zeDriverGet");
        auto ext = (int (*)(void *, const char *, void **)) dlsym(lib, "zeDriverGetExtensionFunctionAddress");
        if (!get || !ext) return;
        uint32_t n = 0;
        if (get(&n, nullptr) != 0 || n == 0) return;
        std::vector<void *> h(n);
        if (get(&n, h.data()) != 0) return;
        for (void * d : h) {
            void * fi = nullptr, * fr = nullptr;
            if (ext(d, "zexDriverImportExternalPointer", &fi) == 0 && fi &&
                ext(d, "zexDriverReleaseImportedPointer", &fr) == 0 && fr) {
                z.drv = d;
                z.import_fn  = (int (*)(void *, void *, size_t)) fi;
                z.release_fn = (int (*)(void *, void *)) fr;
                return;
            }
        }
    });
    return z;
}
}

bool host_block_alloc(host_block & b, size_t bytes, const char * what, bool huge_pages) {
    const size_t align = huge_pages ? (2u << 20) : 4096u;
    const size_t sz = (bytes + align - 1) / align * align;
    void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[qwfn] %s: %.2f GB of host memory unavailable (%s)\n", what, sz / 1e9, strerror(errno));
        return false;
    }
    if (huge_pages) madvise(p, sz, MADV_HUGEPAGE);
    madvise(p, sz, MADV_DONTFORK);   // a child's copy-on-write would move pages under the driver's import
    b = host_block{};
    b.p = p; b.bytes = sz;
    // mlock faults every page in: the whole block is committed now, at startup, rather than
    // during the first long prefill. Success is measured, not assumed from the return code.
    const long before = vmlck_kb();
    const int  rc = mlock(p, sz);
    const int  lerr = errno;
    const long after = vmlck_kb();
    b.locked = rc == 0 && after - before >= (long) (sz / 1024) - 4096;
    ze_import & z = ze();
    int irc = -1;
    if (z.import_fn) {
        irc = z.import_fn(z.drv, p, sz);
        b.imported = irc == 0;
    }
    char lock_note[160];
    if (b.locked) snprintf(lock_note, sizeof lock_note, "locked (VmLck +%ld MB)", (after - before) / 1024);
    else if (rc != 0) snprintf(lock_note, sizeof lock_note, "NOT locked: mlock %s (raise RLIMIT_MEMLOCK / LimitMEMLOCK)", strerror(lerr));
    else snprintf(lock_note, sizeof lock_note, "NOT locked: mlock returned 0 but VmLck grew only %ld MB", (after - before) / 1024);
    char imp_note[96];
    if (!z.import_fn) snprintf(imp_note, sizeof imp_note, "no Level Zero import available, not used");
    else if (b.imported) snprintf(imp_note, sizeof imp_note, "registered with the GPU driver");
    else snprintf(imp_note, sizeof imp_note, "GPU driver import FAILED (0x%x), not used", (unsigned) irc);
    fprintf(stderr, "[qwfn] %s: %.2f GB anonymous host memory, %s, %s%s\n", what, sz / 1e9, lock_note, imp_note,
            huge_pages ? ", transparent huge pages requested" : "");
    return true;
}

void host_block_free(host_block & b) {
    if (!b.p) return;
    if (b.imported) { ze_import & z = ze(); if (z.release_fn) z.release_fn(z.drv, b.p); }
    munmap(b.p, b.bytes);   // also drops the lock
    b = host_block{};
}

#else
// Locked host memory (QWFN_LOCK_HOST) is the Linux + Level Zero path: mlock and the
// driver import. On Windows it is not requested, so the callers keep their pageable
// or ggml-pinned buffers.
bool host_lock_requested() { return false; }
bool host_block_alloc(host_block &, size_t, const char *, bool) { return false; }
void host_block_free(host_block & b) { b = host_block{}; }
#endif

uint64_t mem_available_bytes() { return plat_mem_available(); }

size_t clamp_to_available(size_t want, double frac, size_t headroom) {
    const uint64_t avail = mem_available_bytes();
    if (avail == 0) return want;                       // unknown: trust the caller
    const uint64_t budget = (uint64_t) ((double) avail * frac);
    const uint64_t safe   = budget > headroom ? budget - headroom : 0;
    if (safe == 0) return 0;
    if ((uint64_t) want <= safe) return want;
    fprintf(stderr,
            "[qwfn] requested %.1f GB RAM tier but only %.1f GB is available; "
            "clamping to %.1f GB (%.0f%% of MemAvailable minus %.1f GB headroom)\n",
            want / 1e9, avail / 1e9, safe / 1e9, frac * 100, headroom / 1e9);
    return (size_t) safe;
}

io_engine::~io_engine() { shutdown(); }

bool io_engine::init(const std::vector<std::string> & paths, unsigned queue_depth,
                     bool direct_io, std::string & err, backend be) {
    shutdown();
    direct_ = direct_io;
    be_     = be;
    qd_     = queue_depth ? queue_depth : 256;

    for (const auto & p : paths) {
        bool got_direct = false;
        // Falls back to buffered by itself when the filesystem refuses
        // unbuffered I/O; one shard refusing demotes the whole engine, because
        // the alignment rules below have to hold for all of them alike.
        const file_handle h = plat_open_read(p.c_str(), direct_, &got_direct);
        if (h == FILE_NONE) {
            err = "open failed for " + p + ": " + plat_last_error();
            shutdown();
            return false;
        }
        if (direct_ && !got_direct) direct_ = false;
        fds_.push_back(h);
    }

#if defined(_WIN32)
    // Windows fails a misaligned unbuffered read outright, so ask the volume
    // what it wants rather than assuming.
    dev_align_ = paths.empty() ? QWFN_DIO_PAGE : plat_sector_size(paths[0].c_str());
    if (dev_align_ == 0) dev_align_ = QWFN_DIO_PAGE;
#else
    // The historical rule, kept exactly: the case this exists for is btrfs's
    // 4096 sectorsize, and ext4 and xfs report 4096 as well, so nothing that
    // was measured on Linux changes here.
    dev_align_ = QWFN_DIO_PAGE;
#endif

    // With a 512-byte layout a direct read of the exact window would be served
    // buffered on a 4096-sector filesystem (and refused outright on Windows):
    // the workers read an aligned window into their own buffer and copy the
    // payload into the slot instead.
    bounce_ = direct_ && dio_align() < dev_align_;

    if (be_ == backend::threads) {
        // The platform read is positional and thread-safe, so the shard handles
        // are shared between every worker.
        const unsigned n = qd_ ? std::min(qd_, 32u) : 8u;
        stop_ = false;
        for (unsigned i = 0; i < n; i++) workers_.emplace_back([this] { worker_loop(); });
        return true;
    }

#if defined(_WIN32)
    return iocp_init(err);
#else
    ring_ = (io_uring *) calloc(1, sizeof(io_uring));
    if (!ring_) { err = "out of memory allocating io_uring"; shutdown(); return false; }

    int rc = io_uring_queue_init(qd_, ring_, 0);
    if (rc < 0) {
        free(ring_);
        ring_ = nullptr;
        err = std::string("io_uring_queue_init failed: ") + strerror(-rc);
        shutdown();
        return false;
    }

    // Registering the fds removes a per-op file table lookup. If it fails we must
    // fall back to real fds: submitting with IOSQE_FIXED_FILE against an
    // unregistered table makes every read fail with -EBADF, and the destination
    // buffer then keeps whatever malloc left there.
    const int rr = io_uring_register_files(ring_, fds_.data(), (unsigned) fds_.size());
    registered_files = rr == 0;
    if (!registered_files) {
        fprintf(stderr, "[qwfn] io_uring_register_files failed (%s); using plain fds\n", strerror(-rr));
    }
    return true;
#endif
}

void io_engine::shutdown() {
    if (!workers_.empty()) {
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_work_.notify_all();
        for (auto & t : workers_) if (t.joinable()) t.join();
        workers_.clear();
        q_.clear(); done_.clear();
        stop_ = false;
    }
#if defined(_WIN32)
    iocp_shutdown();
#else
    if (ring_) {
        io_uring_queue_exit(ring_);
        free(ring_);
        ring_ = nullptr;
    }
#endif
    for (file_handle h : fds_) if (h != FILE_NONE) plat_close(h);
    fds_.clear();
    rejected_.clear();
    in_flight_ = 0;
    min_expect_ = 0;
}

size_t io_engine::submit(const io_request * reqs, size_t n) {
    if (be_ == backend::threads) {
        const auto t0 = std::chrono::steady_clock::now();
        size_t queued_jobs = 0;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (size_t i = 0; i < n; i++) {
                const io_request & r = reqs[i];
                if (r.shard < 0 || (size_t) r.shard >= fds_.size() || !r.dst || r.nbytes == 0) {
                    // Completed here as an error rather than dropped: see rejected_.
                    // Returning n while quietly queueing fewer jobs is what used to
                    // leave fetch_end() waiting on a completion nothing would produce.
                    stat_errors++;
                    done_.push_back(r.tag);
                    in_flight_++;
                    continue;
                }
                uint64_t off = r.offset;
                uint32_t len = r.nbytes;
                if (direct_) { off = dio_align_down(r.offset); len = dio_padded_size(r.offset, r.nbytes); }
                q_.push_back(job{ r.shard, off, len, r.dst, r.tag, r.offset, r.nbytes });
                in_flight_++;
                queued_jobs++;
            }
        }
        // notify_all(), not one notify_one() per job. Waking exactly as many
        // workers as there are jobs looks cheaper, but it was measured slower:
        // k notify_one() calls are k futex syscalls made serially by the
        // submitting thread, and the workers they wake start one after another,
        // while a single notify_all() starts them together. On an i9-12900H /
        // RTX 3080 Ti Laptop, UD-Q4_K_XL, 128-token decode, the per-job version
        // raised the per-token read wait from 63 to 68 ms (7.92 -> 7.75 tok/s).
        if (queued_jobs) cv_work_.notify_all();
        if (queued_jobs < n) cv_done_.notify_all();   // the rejected ones are already in done_
        stat_t_prep += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return n;
    }

#if defined(_WIN32)
    return iocp_submit(reqs, n);
#else
    if (!ring_) return 0;
    // `prepped` counts SQEs; `accepted` counts requests this call takes
    // responsibility for completing, which includes the rejected ones.
    size_t prepped = 0, accepted = 0;
    const auto t_prep0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < n; i++) {
        const io_request & r = reqs[i];
        if (r.shard < 0 || (size_t) r.shard >= fds_.size() || !r.dst || r.nbytes == 0) {
            // Completed here as an error rather than dropped: see rejected_.
            // Skipping it used to shift every later request's position in the
            // caller's retry loop, which then resubmitted the same bad request
            // for ever.
            stat_errors++;
            rejected_.push_back(r.tag);
            in_flight_++;
            accepted++;
            continue;
        }

        io_uring_sqe * sqe = io_uring_get_sqe(ring_);
        if (!sqe) break;   // ring full; caller should reap and retry

        uint64_t off = r.offset;
        uint32_t len = r.nbytes;
        if (direct_) {
            off = dio_align_down(r.offset);
            len = dio_padded_size(r.offset, r.nbytes);
        }

        io_uring_prep_read(sqe, registered_files ? r.shard : fds_[r.shard], r.dst, len, off);
        if (registered_files) sqe->flags |= IOSQE_FIXED_FILE;
        expect_[prepped & 1023] = len;
        if (min_expect_ == 0 || len < min_expect_) min_expect_ = len;
        io_uring_sqe_set_data64(sqe, r.tag);
        prepped++;
        accepted++;
    }

    const auto t_prep1 = std::chrono::steady_clock::now();
    stat_t_prep += std::chrono::duration<double>(t_prep1 - t_prep0).count();

    if (prepped) {
        const int rc = io_uring_submit(ring_);
        stat_t_submit_syscall += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_prep1).count();
        // A short or failed submit does NOT lose the entries: liburing has
        // already advanced the SQ tail, so whatever the kernel did not take
        // stays in the ring and goes out on the next io_uring_submit(). They
        // will therefore all complete, and all of them count as in flight --
        // what used to hang the decode loop was that nothing called submit()
        // again before the wait, so reap() flushes the ring before waiting.
        if (rc < 0 || (size_t) rc < prepped) stat_errors++;
        in_flight_ += prepped;
    }
    return accepted;
#endif
}

size_t io_engine::reap(uint64_t * tags_out, size_t max_tags, size_t min_complete) {
    if (be_ == backend::threads) {
        size_t got = 0;
        std::unique_lock<std::mutex> lk(mtx_);
        while (got < max_tags) {
            if (done_.empty()) {
                if (got >= min_complete) break;
                // Nothing in flight and nothing done: a caller whose count has
                // drifted would wait here forever. Return short instead; the
                // caller reports a failed read, which beats a silent hang.
                if (in_flight_ == 0) break;
                cv_done_.wait(lk, [this] { return !done_.empty() || in_flight_ == 0; });
                if (done_.empty()) break;
            }
            tags_out[got++] = done_.front();
            done_.pop_front();
        }
        return got;
    }

#if defined(_WIN32)
    return iocp_reap(tags_out, max_tags, min_complete);
#else
    if (!ring_ || in_flight_ == 0) return 0;

    size_t got = 0;
    if (min_complete > in_flight_) min_complete = in_flight_;

    // Requests rejected at submit: complete in the only sense the caller tracks.
    while (got < max_tags && !rejected_.empty()) {
        tags_out[got++] = rejected_.back();
        rejected_.pop_back();
        in_flight_--;
    }
    if (in_flight_ == 0) return got;

    // io_uring_submit() takes as many SQEs as the kernel will accept and leaves
    // the rest in the ring for the next submit. Nothing else calls submit()
    // between the caller's last one and the wait below, so a short submit would
    // park entries the kernel never saw and io_uring_wait_cqe() would block on
    // completions that cannot arrive. Flush first.
    if (io_uring_sq_ready(ring_) > 0) {
        const int rc = io_uring_submit(ring_);
        if (rc <= 0 && io_uring_sq_ready(ring_) > 0) {
            // The ring will not drain. Hand back what is already there rather
            // than waiting for ever; the caller reports a short read, which is
            // recoverable, where a hang is not.
            stat_errors++;
            min_complete = 0;
        }
    }

    while (got < max_tags) {
        io_uring_cqe * cqe = nullptr;
        int rc;
        if (got < min_complete) {
            rc = io_uring_wait_cqe(ring_, &cqe);
        } else {
            rc = io_uring_peek_cqe(ring_, &cqe);
            if (rc == -EAGAIN || !cqe) break;
        }
        if (rc < 0) { stat_errors++; break; }

        if (cqe->res < 0) {
            stat_errors++;
        } else {
            stat_reads++;
            stat_bytes += (uint64_t) cqe->res;
            if ((uint32_t) cqe->res < min_expect_) stat_short++;
        }
        tags_out[got++] = io_uring_cqe_get_data64(cqe);
        io_uring_cqe_seen(ring_, cqe);
        in_flight_--;
        if (in_flight_ == 0) break;
    }
    return got;
#endif
}


void io_engine::worker_loop() {
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_work_.wait(lk, [this] { return stop_ || !q_.empty(); });
            if (stop_ && q_.empty()) return;
            j = q_.front();
            q_.pop_front();
        }
        int64_t got = 0;   // int64_t, not ssize_t: there is no such type on the MSVC CRT
        if (bounce_) {
            // The page-aligned window around the requested range, into this
            // worker's buffer; the payload then goes where the 512-byte layout
            // expects it. A window past the end of a shard reads short, which is
            // fine as long as the payload arrived.
            static thread_local uint8_t * scratch = nullptr;
            static thread_local size_t    scratch_bytes = 0;
            const uint64_t w0 = j.ooff & ~((uint64_t) dev_align_ - 1);
            const uint64_t w1 = (j.ooff + j.onb + dev_align_ - 1) & ~((uint64_t) dev_align_ - 1);
            const size_t   wl = (size_t) (w1 - w0);
            if (scratch_bytes < wl) {
                if (scratch) dio_free(scratch);
                scratch_bytes = wl + (1u << 20);
                scratch = (uint8_t *) dio_alloc(scratch_bytes);
            }
            const int64_t need = (int64_t) (j.ooff - w0 + j.onb);
            if (scratch) {
                while (got < (int64_t) wl) {
                    const int64_t r = plat_pread(fds_[j.shard], scratch + got, wl - (size_t) got, w0 + (uint64_t) got);
                    if (r <= 0) break;
                    got += r;
                }
            }
            if (got >= need) {
                memcpy((char *) j.dst + dio_pad(j.ooff), scratch + (j.ooff - w0), j.onb);
                got = (int64_t) j.len;   // the caller's notion of a complete read
            } else {
                got = 0;
            }
        } else {
            while (got < (int64_t) j.len) {
                const int64_t r = plat_pread(fds_[j.shard], (char *) j.dst + got,
                                             j.len - (size_t) got, j.off + (uint64_t) got);
                if (r <= 0) break;
                got += r;
            }
        }
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (got < (int64_t) j.len) stat_errors++;
            else { stat_reads++; stat_bytes += (uint64_t) got; }
            done_.push_back(j.tag);
            in_flight_--;
        }
        cv_done_.notify_all();
    }
}

#if defined(_WIN32)
// ------------------------------------------------------------ IOCP backend --
// The Windows counterpart of the io_uring path: one completion port, every
// shard handle bound to it, and a fixed pool of OVERLAPPED blocks so that
// submitting a burst allocates nothing.
//
// Two differences from io_uring are worth knowing, because both would otherwise
// desynchronise the submitted/completed counts the callers rely on:
//
//   A read that fails synchronously -- a misaligned offset under NO_BUFFERING
//   is the one to expect -- posts NO completion packet at all, so it has to be
//   accounted for at submit time. That is exactly what `rejected_` already does
//   for a malformed request, so it is reused rather than given a second path.
//
//   A read that SUCCEEDS synchronously still posts a packet, because
//   FILE_SKIP_COMPLETION_PORT_ON_SUCCESS is deliberately not set: one
//   completion path is worth more here than the microsecond it would save on
//   the rare unbuffered read that the drive answers immediately.
//
// Completions are drained with GetQueuedCompletionStatusEx, a batch per call.
// Taking them one at a time would put a kernel transition on every expert slice.

struct iocp_state {
    HANDLE port = nullptr;
    struct req {
        OVERLAPPED ov;          // must stay first: a completion hands back &ov
        uint64_t   tag = 0;
        uint32_t   len = 0;
    };
    std::vector<req>   store;   // sized once; freelist points into it
    std::vector<req *> freelist;
};

bool io_engine::iocp_init(std::string & err) {
    iocp_ = new iocp_state();
    iocp_->port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (!iocp_->port) {
        err = std::string("CreateIoCompletionPort failed: ") + plat_last_error();
        iocp_shutdown();
        return false;
    }
    for (size_t i = 0; i < fds_.size(); i++) {
        if (!CreateIoCompletionPort((HANDLE) fds_[i], iocp_->port, (ULONG_PTR) i, 0)) {
            err = std::string("binding a shard to the completion port failed: ") + plat_last_error();
            iocp_shutdown();
            return false;
        }
        // The handle reports through the port now, so signalling the handle
        // itself on every completion is pure overhead. (Not
        // FILE_SKIP_COMPLETION_PORT_ON_SUCCESS -- see the note above.)
        SetFileCompletionNotificationModes((HANDLE) fds_[i], FILE_SKIP_SET_EVENT_ON_HANDLE);
    }
    iocp_->store.resize(qd_ ? qd_ : 256);
    iocp_->freelist.reserve(iocp_->store.size());
    for (auto & r : iocp_->store) iocp_->freelist.push_back(&r);
    registered_files = true;
    return true;
}

void io_engine::iocp_shutdown() {
    if (!iocp_) return;
    // The handles are closed by the caller right after this, which cancels any
    // read still outstanding; the packets are then dropped with the port.
    if (iocp_->port) CloseHandle(iocp_->port);
    delete iocp_;
    iocp_ = nullptr;
}

size_t io_engine::iocp_submit(const io_request * reqs, size_t n) {
    if (!iocp_) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    size_t accepted = 0;

    for (size_t i = 0; i < n; i++) {
        const io_request & r = reqs[i];
        if (r.shard < 0 || (size_t) r.shard >= fds_.size() || !r.dst || r.nbytes == 0) {
            stat_errors++;
            rejected_.push_back(r.tag);
            in_flight_++;
            accepted++;
            continue;
        }
        if (iocp_->freelist.empty()) break;   // queue full: the caller reaps and retries

        uint64_t off = r.offset;
        uint32_t len = r.nbytes;
        if (direct_) {
            off = dio_align_down(r.offset);
            len = dio_padded_size(r.offset, r.nbytes);
        }

        iocp_state::req * q = iocp_->freelist.back();
        memset(&q->ov, 0, sizeof q->ov);
        q->ov.Offset     = (DWORD) (off & 0xFFFFFFFFull);
        q->ov.OffsetHigh = (DWORD) (off >> 32);
        q->tag = r.tag;
        q->len = len;

        if (!ReadFile((HANDLE) fds_[r.shard], r.dst, (DWORD) len, nullptr, &q->ov)) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                // Nothing will ever complete for this read, so complete it here.
                // ERROR_INVALID_PARAMETER at this point means the offset, the
                // length or the destination violated the volume's unbuffered
                // alignment -- which is a layout bug, not a transient failure.
                // Reported once: if the layout is wrong every read is wrong, and
                // fifty thousand identical lines a second help nobody.
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    fprintf(stderr, "[qwfn] the volume rejected an unbuffered read (error %lu): "
                                    "offset %llu, length %u, destination %p, required alignment %u. "
                                    "Further failures are counted, not printed.\n",
                            (unsigned long) e, (unsigned long long) off, len, r.dst, dev_align_);
                }
                stat_errors++;
                rejected_.push_back(r.tag);
                in_flight_++;
                accepted++;
                continue;                      // q was never taken off the free list
            }
        }
        iocp_->freelist.pop_back();
        if (min_expect_ == 0 || len < min_expect_) min_expect_ = len;
        in_flight_++;
        accepted++;
    }

    stat_t_prep += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return accepted;
}

size_t io_engine::iocp_reap(uint64_t * tags_out, size_t max_tags, size_t min_complete) {
    if (!iocp_ || in_flight_ == 0) return 0;

    size_t got = 0;
    if (min_complete > in_flight_) min_complete = in_flight_;

    // Reads that failed at submit: complete in the only sense the caller tracks.
    while (got < max_tags && !rejected_.empty()) {
        tags_out[got++] = rejected_.back();
        rejected_.pop_back();
        in_flight_--;
    }
    if (in_flight_ == 0) return got;

    OVERLAPPED_ENTRY entries[64];
    while (got < max_tags && in_flight_ > 0) {
        const ULONG want = (ULONG) std::min<size_t>(max_tags - got, 64);
        ULONG removed = 0;
        // Wait only while the caller still has a minimum to make; past that,
        // take whatever has already landed and return.
        const DWORD timeout = got < min_complete ? INFINITE : 0;
        if (!GetQueuedCompletionStatusEx(iocp_->port, entries, want, &removed, timeout, FALSE)) {
            if (GetLastError() != WAIT_TIMEOUT) stat_errors++;
            break;
        }
        for (ULONG k = 0; k < removed; k++) {
            iocp_state::req * q = (iocp_state::req *) entries[k].lpOverlapped;
            const DWORD bytes = entries[k].dwNumberOfBytesTransferred;
            // Internal carries the NTSTATUS of the operation; 0 is success.
            if (entries[k].Internal != 0 || bytes == 0) {
                stat_errors++;
            } else {
                stat_reads++;
                stat_bytes += bytes;
                if (bytes < q->len) stat_short++;
            }
            tags_out[got++] = q->tag;
            iocp_->freelist.push_back(q);
            in_flight_--;
        }
    }
    return got;
}
#endif // _WIN32

} // namespace qwfn
