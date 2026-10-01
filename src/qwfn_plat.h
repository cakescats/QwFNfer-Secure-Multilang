#pragma once
// The operating system surface of the engine, and all of it.
//
// Everything below the engine that is not ggml goes through this header: opening
// a model shard for unbuffered reads, a positional read, an aligned allocation,
// a read-only mapping, and what the machine will admit about its free memory.
// Two implementations -- qwfn_plat_posix.cpp and qwfn_plat_win32.cpp -- and
// nothing above this header knows which one it is linked against.
//
// The asynchronous read engines are NOT here: io_uring and IOCP are different
// enough in shape that hiding them behind one set of calls would cost more than
// it explains. They live behind io_engine::backend in qwfn_io.cpp instead, which
// is the abstraction that was already there.

#include <cstddef>
#include <cstdint>

namespace qwfn {

#if defined(_WIN32)
// HANDLE, without dragging windows.h into every translation unit that reads a
// file. INVALID_HANDLE_VALUE is (HANDLE)-1 rather than null, so plat_open_read
// normalises a failure to nullptr and callers only ever test against FILE_NONE.
using file_handle = void *;
inline constexpr file_handle FILE_NONE = nullptr;
#else
using file_handle = int;
inline constexpr file_handle FILE_NONE = -1;
#endif

// Open a model shard read-only.
//
// `direct` asks for the page cache to be bypassed: O_DIRECT on Linux,
// FILE_FLAG_NO_BUFFERING on Windows. It can be refused -- some filesystems do
// not implement it at all -- so *direct_ok reports whether it was granted and
// the caller falls back to buffered reads rather than failing the run. The
// engine cares because the RAM tier it manages itself would otherwise be
// mirrored by the page cache, halving its effective capacity.
//
// The handle is always safe for concurrent positional reads from several
// threads (on Windows that means it is opened overlapped; see plat_pread).
file_handle plat_open_read(const char * path, bool direct, bool * direct_ok);
void        plat_close(file_handle h);

// Positional, thread-safe read: bytes read, 0 at end of file, -1 on error.
// Never moves a shared file position -- several workers read one handle at once.
int64_t     plat_pread(file_handle h, void * dst, size_t nbytes, uint64_t offset);
int64_t     plat_file_size(file_handle h);

// The alignment an unbuffered read of a file on this path requires of its
// offset, its length AND its destination address. 0 when it cannot be
// determined, which the caller should read as "assume a page".
//
// The two platforms punish a violation differently, and it matters. Linux
// serves a misaligned O_DIRECT read through the page cache, silently: measured
// on btrfs on 2026-09-09, every expert read of this engine was buffered and
// nobody noticed until the throughput was traced. Windows fails the read with
// ERROR_INVALID_PARAMETER instead, which is louder and easier to trust.
uint32_t    plat_sector_size(const char * path);

// Aligned allocation for unbuffered read destinations. `align` is what
// plat_sector_size asked for, rounded up to a page by the caller.
void *      plat_alloc_aligned(size_t bytes, size_t align);
void        plat_free_aligned(void * p);

// Whole-file read-only mapping, hinted for random access: the router scatters
// expert access, so sequential read-ahead would only evict pages still wanted.
// Returns nullptr on failure. `size_out` receives the mapped length, which
// plat_unmap needs back.
void *      plat_map_read(const char * path, size_t * size_out);
void        plat_unmap(void * base, size_t size);

// What the system says it can hand out without swapping; 0 when unknown.
// Linux reads MemAvailable, which is the kernel's own estimate. Windows has no
// exact equivalent, so this is ullAvailPhys -- free plus the standby list --
// which is the nearest thing and errs in the same direction.
uint64_t    plat_mem_available();
// Installed physical memory; 0 when unknown.
uint64_t    plat_mem_total();

// The last failure on this thread, as text, for error messages. The pointer is
// valid until the next platform call on the same thread.
const char * plat_last_error();

} // namespace qwfn
