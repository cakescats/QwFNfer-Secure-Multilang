// POSIX half of qwfn_plat.h. Behaviour here is exactly what the engine did
// before the platform split: the same open flags, the same fallback, the same
// madvise. Nothing in this file is new.

#if !defined(_WIN32)

#include "qwfn_plat.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace qwfn {

static thread_local char g_err[256] = {0};

static void set_err(const char * what) {
    snprintf(g_err, sizeof g_err, "%s: %s", what, strerror(errno));
}

const char * plat_last_error() { return g_err; }

file_handle plat_open_read(const char * path, bool direct, bool * direct_ok) {
    if (direct_ok) *direct_ok = direct;
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (direct) flags |= O_DIRECT;
#else
    if (direct && direct_ok) *direct_ok = false;   // no such thing on this libc
#endif
    int fd = ::open(path, flags);
#ifdef O_DIRECT
    if (fd < 0 && direct) {
        // Some filesystems refuse O_DIRECT outright. Fall back rather than fail:
        // the page cache costs the RAM tier capacity, it does not break the run.
        fd = ::open(path, O_RDONLY);
        if (fd >= 0 && direct_ok) *direct_ok = false;
    }
#endif
    if (fd < 0) { set_err("open failed"); return FILE_NONE; }
    return fd;
}

void plat_close(file_handle h) { if (h >= 0) ::close(h); }

int64_t plat_pread(file_handle h, void * dst, size_t nbytes, uint64_t offset) {
    const ssize_t n = ::pread(h, dst, nbytes, (off_t) offset);
    if (n < 0) set_err("pread failed");
    return (int64_t) n;
}

int64_t plat_file_size(file_handle h) {
    struct stat st;
    if (::fstat(h, &st) != 0) { set_err("fstat failed"); return -1; }
    return (int64_t) st.st_size;
}

uint32_t plat_sector_size(const char * path) {
    // What a direct read must be aligned to is a property of the FILESYSTEM,
    // not of the device: NVMe reports 512, but btrfs with a 4096 sectorsize
    // serves anything not 4096-aligned through the page cache. statvfs's
    // f_bsize is that filesystem block size.
    struct statvfs vfs;
    if (::statvfs(path, &vfs) != 0 || vfs.f_bsize == 0) { set_err("statvfs failed"); return 0; }
    return (uint32_t) vfs.f_bsize;
}

void * plat_alloc_aligned(size_t bytes, size_t align) {
    void * p = nullptr;
    if (align < sizeof(void *)) align = sizeof(void *);
    if (posix_memalign(&p, align, bytes) != 0) { set_err("posix_memalign failed"); return nullptr; }
    return p;
}

void plat_free_aligned(void * p) { free(p); }

void * plat_map_read(const char * path, size_t * size_out) {
    if (size_out) *size_out = 0;
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) { set_err("open failed"); return nullptr; }
    const off_t sz = ::lseek(fd, 0, SEEK_END);
    if (sz <= 0) { set_err("lseek failed"); ::close(fd); return nullptr; }
    void * base = ::mmap(nullptr, (size_t) sz, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) { set_err("mmap failed"); return nullptr; }
    // Expert access is scattered by the router; sequential read-ahead would
    // only evict pages we still want.
    ::madvise(base, (size_t) sz, MADV_RANDOM);
    if (size_out) *size_out = (size_t) sz;
    return base;
}

void plat_unmap(void * base, size_t size) { if (base) ::munmap(base, size); }

uint64_t plat_mem_available() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { found = true; break; }
    }
    fclose(f);
    return found ? kb * 1024ull : 0;
}

} // namespace qwfn

#endif // !_WIN32
