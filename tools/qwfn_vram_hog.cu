// qwfn-vram-hog -- занять всю свободную VRAM, кроме <leave_mb>, и держать до завершения.
// Нужна для проверки, как сервер переживает нехватку памяти посреди запроса
// (scripts/oom_survival.sh).
//
//   nvcc -O2 -o build/qwfn-vram-hog tools/qwfn_vram_hog.cu    (при gcc > 13: -ccbin g++-13)
//   build/qwfn-vram-hog 40
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

int main(int argc, char ** argv) {
    const size_t leave = (size_t) (argc > 1 ? atol(argv[1]) : 64) << 20;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    const size_t want = fr > leave ? fr - leave : 0, chunk = 64u << 20;
    size_t got = 0; void * p = nullptr;
    while (want - got >= chunk && cudaMalloc(&p, chunk) == cudaSuccess) got += chunk;
    cudaMemGetInfo(&fr, &tot);
    printf("hog: занято %zu МБ, свободно %zu МБ\n", got >> 20, fr >> 20); fflush(stdout);
    pause();
}
