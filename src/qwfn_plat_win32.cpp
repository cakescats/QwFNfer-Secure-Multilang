// Win32 half of qwfn_plat.h.
//
// The three things that differ from POSIX in ways that matter here:
//
//   Unbuffered reads. FILE_FLAG_NO_BUFFERING is the O_DIRECT of this platform,
//   and it is stricter: offset, length AND destination address must all be
//   multiples of the volume's sector size, and a violation fails the read with
//   ERROR_INVALID_PARAMETER instead of quietly falling back to the cache. That
//   is an improvement -- the Linux side lost 23x of read bandwidth to exactly
//   that silent fallback once, on btrfs, and it took a bandwidth trace to find.
//
//   Positional reads. There is no pread. A handle opened FILE_FLAG_OVERLAPPED
//   takes the offset in the OVERLAPPED itself and keeps no shared file pointer,
//   so several workers can read one handle at once -- which is what the thread
//   pool backend needs. Each read then has to be waited for individually, which
//   is what plat_pread does with a per-thread event.
//
//   Paths. Model shards live in the Hugging Face cache, whose directory names
//   are long, so a path can pass MAX_PATH. The \\?\ prefix lifts that limit but
//   only accepts a fully qualified path with backslashes, so it is applied
//   deliberately rather than always.

#if defined(_WIN32)

#include "qwfn_plat.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <malloc.h>     // _aligned_malloc / _aligned_free; NOT in <cstdlib> on the MS CRT
#include <string>

namespace qwfn {

static thread_local char g_err[512] = {0};

static void set_err(const char * what) {
    const DWORD e = GetLastError();
    char * msg = nullptr;
    const DWORD n = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, e, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &msg, 0, nullptr);
    // FormatMessage leaves a trailing CRLF on almost everything.
    if (msg) { for (DWORD i = n; i > 0 && (msg[i - 1] == '\r' || msg[i - 1] == '\n'); i--) msg[i - 1] = 0; }
    snprintf(g_err, sizeof g_err, "%s: (%lu) %s", what, (unsigned long) e, msg ? msg : "unknown error");
    if (msg) LocalFree(msg);
}

const char * plat_last_error() { return g_err; }

// ---------------------------------------------------------------- paths -----

static std::wstring widen(const char * utf8) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w((size_t) n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, &w[0], n);
    return w;
}

// A path Win32 will take at any length. Forward slashes are accepted by the
// normal path parser but NOT by the \\?\ form, so they are converted first;
// GetFullPathNameW then resolves any relative component, which \\?\ also
// refuses to do. Left alone when it is already a device or UNC \\?\ path.
static std::wstring win_path(const char * utf8) {
    std::wstring w = widen(utf8);
    if (w.empty()) return w;
    if (w.size() >= 4 && w[0] == L'\\' && w[1] == L'\\' && w[2] == L'?' && w[3] == L'\\') return w;
    if (w.size() < MAX_PATH - 12) return w;   // short enough that none of this is needed

    for (wchar_t & c : w) if (c == L'/') c = L'\\';
    std::wstring full((size_t) 32768, L'\0');
    const DWORD n = GetFullPathNameW(w.c_str(), (DWORD) full.size(), &full[0], nullptr);
    if (n == 0 || n >= full.size()) return w;   // give up and let CreateFileW report it
    full.resize(n);
    if (full.size() >= 2 && full[0] == L'\\' && full[1] == L'\\') return L"\\\\?\\UNC\\" + full.substr(2);
    return L"\\\\?\\" + full;
}

// ---------------------------------------------------------------- files -----

static HANDLE as_handle(file_handle h) { return (HANDLE) h; }

file_handle plat_open_read(const char * path, bool direct, bool * direct_ok) {
    if (direct_ok) *direct_ok = direct;
    const std::wstring w = win_path(path);
    if (w.empty()) { SetLastError(ERROR_INVALID_NAME); set_err("path conversion failed"); return FILE_NONE; }

    // FILE_FLAG_OVERLAPPED unconditionally: it is what makes the handle safe for
    // concurrent positional reads, whether the caller uses plat_pread or binds it
    // to a completion port. FILE_FLAG_RANDOM_ACCESS turns off the read-ahead that
    // would evict cached pages the router still wants (it is a no-op under
    // NO_BUFFERING, and the right hint when NO_BUFFERING was refused).
    DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS;
    if (direct) flags |= FILE_FLAG_NO_BUFFERING;

    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, flags, nullptr);
    if (h == INVALID_HANDLE_VALUE && direct) {
        // A filesystem that will not do unbuffered I/O (some network redirectors,
        // some virtual filesystems). Fall back rather than fail the run.
        flags &= ~(DWORD) FILE_FLAG_NO_BUFFERING;
        h = CreateFileW(w.c_str(), GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        nullptr, OPEN_EXISTING, flags, nullptr);
        if (h != INVALID_HANDLE_VALUE && direct_ok) *direct_ok = false;
    }
    if (h == INVALID_HANDLE_VALUE) { set_err("CreateFile failed"); return FILE_NONE; }
    return (file_handle) h;
}

void plat_close(file_handle h) { if (h != FILE_NONE) CloseHandle(as_handle(h)); }

// One manual-reset event per thread, reused for every read that thread makes:
// creating one per read would put a kernel object allocation on the path of
// every expert slice. Closed when the thread ends rather than leaked, which
// matters because the io engines start and stop pools of these.
namespace {
struct thread_event {
    HANDLE h = nullptr;
    ~thread_event() { if (h) CloseHandle(h); }
};
}

int64_t plat_pread(file_handle h, void * dst, size_t nbytes, uint64_t offset) {
    static thread_local thread_event te;
    if (!te.h) {
        te.h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!te.h) { set_err("CreateEvent failed"); return -1; }
    }
    HANDLE ev = te.h;

    int64_t done = 0;
    while (done < (int64_t) nbytes) {
        const DWORD want = (DWORD) ((nbytes - (size_t) done) > 0xF0000000u
                                        ? 0xF0000000u : (nbytes - (size_t) done));
        const uint64_t at = offset + (uint64_t) done;
        OVERLAPPED ov{};
        ov.Offset     = (DWORD) (at & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) (at >> 32);
        ov.hEvent     = ev;
        ResetEvent(ev);

        DWORD got = 0;
        if (!ReadFile(as_handle(h), (char *) dst + done, want, nullptr, &ov)) {
            const DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            if (e != ERROR_IO_PENDING) { set_err("ReadFile failed"); return done > 0 ? done : -1; }
        }
        if (!GetOverlappedResult(as_handle(h), &ov, &got, TRUE)) {
            const DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            set_err("GetOverlappedResult failed");
            return done > 0 ? done : -1;
        }
        if (got == 0) break;                       // end of file
        done += (int64_t) got;
        if (got < want) break;                     // short read: the caller decides
    }
    return done;
}

int64_t plat_file_size(file_handle h) {
    LARGE_INTEGER li{};
    if (!GetFileSizeEx(as_handle(h), &li)) { set_err("GetFileSizeEx failed"); return -1; }
    return (int64_t) li.QuadPart;
}

uint32_t plat_sector_size(const char * path) {
    const std::wstring w = widen(path);
    if (w.empty()) return 0;
    // The alignment NO_BUFFERING demands is the volume's, so resolve the path to
    // its volume root first -- the file itself may not even exist yet.
    wchar_t root[MAX_PATH] = {0};
    if (!GetVolumePathNameW(w.c_str(), root, MAX_PATH)) { set_err("GetVolumePathName failed"); return 0; }
    DWORD spc = 0, bps = 0, freec = 0, totalc = 0;
    if (!GetDiskFreeSpaceW(root, &spc, &bps, &freec, &totalc) || bps == 0) {
        set_err("GetDiskFreeSpace failed");
        return 0;
    }
    return (uint32_t) bps;
}

// --------------------------------------------------------------- memory -----

void * plat_alloc_aligned(size_t bytes, size_t align) {
    if (align < sizeof(void *)) align = sizeof(void *);
    void * p = _aligned_malloc(bytes, align);
    if (!p) { SetLastError(ERROR_NOT_ENOUGH_MEMORY); set_err("_aligned_malloc failed"); }
    return p;
}

// _aligned_malloc'd memory must go back through _aligned_free; free() on it is
// undefined behaviour on the Microsoft CRT, not merely wasteful.
void plat_free_aligned(void * p) { if (p) _aligned_free(p); }

void * plat_map_read(const char * path, size_t * size_out) {
    if (size_out) *size_out = 0;
    const std::wstring w = win_path(path);
    if (w.empty()) { SetLastError(ERROR_INVALID_NAME); set_err("path conversion failed"); return nullptr; }

    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) { set_err("CreateFile failed"); return nullptr; }

    LARGE_INTEGER li{};
    if (!GetFileSizeEx(h, &li) || li.QuadPart <= 0) {
        set_err("GetFileSizeEx failed");
        CloseHandle(h);
        return nullptr;
    }
    HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { set_err("CreateFileMapping failed"); CloseHandle(h); return nullptr; }

    void * base = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    // The view holds its own references, so both handles can go now.
    CloseHandle(m);
    CloseHandle(h);
    if (!base) { set_err("MapViewOfFile failed"); return nullptr; }

    if (size_out) *size_out = (size_t) li.QuadPart;
    return base;
}

// UnmapViewOfFile takes only the base: the length is remembered by the kernel.
void plat_unmap(void * base, size_t /*size*/) { if (base) UnmapViewOfFile(base); }

uint64_t plat_mem_available() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) { set_err("GlobalMemoryStatusEx failed"); return 0; }
    return (uint64_t) ms.ullAvailPhys;
}

} // namespace qwfn

#endif // _WIN32
