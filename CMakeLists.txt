cmake_minimum_required(VERSION 3.20)
project(qwfnfer LANGUAGES C CXX)

# qwfnfer -- a purpose-built inference engine for Qwen3.8-Flash-Next (GGUF arch "qwen4exp").
#
# We own the model graph and the memory hierarchy; ggml supplies the quantized
# math kernels. The entire performance thesis of this engine lives in
# qwfn_io / qwfn_expert_cache / qwfn_ple: fetch expert slices at their natural
# 627KB-921KB granularity over the platform's asynchronous read engine instead
# of 4KiB demand paging.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Where llama.cpp/ggml lives. Override with -DLLAMA_CPP_ROOT=...
if(WIN32)
  # %USERPROFILE%, unless a HOME is set (Git Bash, MSYS)
  if("$ENV{HOME}" STREQUAL "")
    set(LLAMA_CPP_ROOT "$ENV{USERPROFILE}/.unsloth/llama.cpp" CACHE PATH "llama.cpp source root")
  else()
    set(LLAMA_CPP_ROOT "$ENV{HOME}/.unsloth/llama.cpp" CACHE PATH "llama.cpp source root")
  endif()
else()
  set(LLAMA_CPP_ROOT "$ENV{HOME}/.unsloth/llama.cpp" CACHE PATH "llama.cpp source root")
endif()
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build/bin"   CACHE PATH "llama.cpp built libraries")

# QWFN_PORTABLE builds the release bundle (scripts/package.sh): baseline x86-64-v3 code
# instead of this machine's -march=native, a glibc 2.34 floor (cmake/glibc_compat.h keeps
# the pre-C23 strtol/sscanf symbols, -fno-math-errno the pre-2.43 sqrtf), and the ggml/CUDA
# libraries found next to the binaries ($ORIGIN). Every target the bundle ships takes
# QWFN_ARCH_FLAGS, the -O2 ones included: without it they are built for whatever -march the
# build machine's compiler happens to default to, which is nobody's guarantee of x86-64-v3.
#
# -static-libstdc++ -static-libgcc used to be here as well, to keep the bundle off the user's
# C++ runtime. They also copy the build machine's runtime into the binaries, and the linker
# ORs the ISA property of every object it links into the output: on a distribution that
# compiles libstdc++.a and libgcc.a for AVX-512 that leaves the binaries marked "x86-64-v4
# needed" whatever -march the engine itself was built with, after which glibc's loader
# refuses to start them on any CPU without AVX-512. It bought little in return -- libllama
# and the ggml libraries link libstdc++ dynamically, so the bundle needs the user's copy
# either way -- and the C++ runtime is one thing every distribution has.
option(QWFN_PORTABLE "Relocatable release build" OFF)

# ---------------------------------------------------------------- toolchain --
#
# Two flag dialects, not two compilers: MSVC and clang-cl share an ABI, a
# runtime and a command line, and CMake's MSVC variable is true for both. That
# is the ABI the engine has to speak on Windows whichever one is chosen, because
# nvcc accepts only cl.exe as a host compiler, so the CUDA half of ggml is built
# with it and everything that links against ggml has to match.
#
# The choice barely shows up in throughput -- the arithmetic is all inside ggml,
# built by its own CMake, and what this project compiles is orchestration. It
# shows up in whether the link works at all.
if(MSVC)
  # MSVC has no -march=native: AVX2 is the x86-64-v3 baseline the Linux bundle
  # also targets. clang-cl additionally understands /clang:-march=native, which
  # is worth trying on a machine that will only ever run its own build.
  set(QWFN_ARCH_FLAGS /arch:AVX2 CACHE STRING "ISA flags for MSVC-dialect compilers")
  set(QWFN_OPT_FLAGS  /O2)
  set(QWFN_WARN_FLAGS /W3 /wd4244 /wd4267 /wd4305)   # narrowing conversions, by the thousand, all deliberate
  add_compile_options(
    /utf-8              # the sources are UTF-8 and say so, rather than hoping for the active code page
    /EHsc
    /Zc:preprocessor    # the conforming preprocessor; the legacy one mangles __VA_ARGS__
    /bigobj             # qwfn_engine.cpp is 3,300 lines of lambdas and passes the 65,536-section limit
  )
  add_compile_definitions(
    NOMINMAX                  # windows.h defines min/max as macros and breaks <algorithm>
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    _WINSOCK_DEPRECATED_NO_WARNINGS
    # GetQueuedCompletionStatusEx and SetFileCompletionNotificationModes are
    # Vista and later. The SDK's default is high enough today but is not a
    # contract, so say it.
    _WIN32_WINNT=0x0A00       # Windows 10
    WINVER=0x0A00
  )
else()
  if(QWFN_PORTABLE)
    set(QWFN_ARCH_FLAGS -march=x86-64-v3 -mtune=generic)
    add_compile_options(-include "${CMAKE_SOURCE_DIR}/cmake/glibc_compat.h" -fno-math-errno)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
  else()
    set(QWFN_ARCH_FLAGS -march=native)
  endif()
  set(QWFN_OPT_FLAGS  -O3)
  set(QWFN_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter)
endif()

if(NOT EXISTS "${LLAMA_CPP_ROOT}/ggml/include/ggml.h")
  message(FATAL_ERROR "ggml headers not found under ${LLAMA_CPP_ROOT}. Pass -DLLAMA_CPP_ROOT=<path>.")
endif()

# ------------------------------------------------------------- ggml imports --
#
# One shared library per name, found where each platform's build actually leaves
# it. On Linux that is a versioned .so next to the built binaries. On Windows a
# shared library is two files -- the .dll to load and the .lib to link against --
# and which directory holds which depends on the generator, so look in the ones
# Visual Studio and Ninja use rather than assuming.
function(qwfn_import_shared target basename)
  add_library(${target} SHARED IMPORTED)
  if(WIN32)
    find_library(${target}_IMPLIB
      NAMES ${basename}
      PATHS "${LLAMA_CPP_BUILD}" "${LLAMA_CPP_BUILD}/Release"
            "${LLAMA_CPP_ROOT}/build/lib" "${LLAMA_CPP_ROOT}/build/lib/Release"
            "${LLAMA_CPP_ROOT}/build/${basename}" "${LLAMA_CPP_ROOT}/build"
            # where a Ninja / MSVC build of llama.cpp leaves the import libraries
            # (found on a real Windows build by the Rafiekuntest port)
            "${LLAMA_CPP_BUILD}/../ggml/src" "${LLAMA_CPP_BUILD}/../ggml/src/Release"
            "${LLAMA_CPP_BUILD}/../src" "${LLAMA_CPP_BUILD}/../src/Release"
      NO_DEFAULT_PATH)
    find_file(${target}_DLL
      NAMES ${basename}.dll
      PATHS "${LLAMA_CPP_BUILD}" "${LLAMA_CPP_BUILD}/Release" "${LLAMA_CPP_ROOT}/build/bin"
      NO_DEFAULT_PATH)
    if(NOT ${target}_IMPLIB)
      message(FATAL_ERROR
        "${basename}.lib not found under ${LLAMA_CPP_BUILD}. Build llama.cpp first "
        "(cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release), "
        "then pass -DLLAMA_CPP_BUILD=<the directory holding the .lib files>.")
    endif()
    set_target_properties(${target} PROPERTIES
      IMPORTED_IMPLIB   "${${target}_IMPLIB}"
      IMPORTED_LOCATION "${${target}_DLL}")
  else()
    set_target_properties(${target} PROPERTIES
      IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/lib${basename}.so.0")
  endif()
endfunction()

qwfn_import_shared(ggml_base ggml-base)
qwfn_import_shared(ggml_all  ggml)

# Two things need libllama: qwfn-refdump drives llama.cpp's own graph to produce
# the reference intermediates our graph is bisected against, and qwfn-chat
# borrows its tokenizer (opened vocab_only, so no weights are loaded).
qwfn_import_shared(llama_ref llama)

find_package(Threads REQUIRED)

# ----------------------------------------------------------------- read I/O --
#
# The asynchronous read engine, which is the whole point of qwfn_io.cpp. Linux
# reads through io_uring, so liburing is mandatory there. Windows reads through
# an I/O completion port, which is in the OS.
if(WIN32)
  set(QWFN_IO_LIBS "")
  message(STATUS "async read engine: Win32 I/O completion port")
else()
  find_path(URING_INCLUDE_DIR liburing.h)
  find_library(URING_LIBRARY NAMES uring liburing)
  if(NOT URING_INCLUDE_DIR OR NOT URING_LIBRARY)
    message(FATAL_ERROR "liburing not found. Install liburing (Arch: pacman -S liburing).")
  endif()
  set(QWFN_IO_LIBS "${URING_LIBRARY}")
  message(STATUS "async read engine: io_uring (${URING_LIBRARY})")
endif()

# Both platform halves are always listed: each is an empty translation unit on
# the other platform, which keeps the file list identical everywhere and stops a
# new source from being added to one build and forgotten in the other.
add_library(qwfn_core STATIC
  src/qwfn_model.cpp
  src/qwfn_pack.cpp
  src/qwfn_plat_posix.cpp
  src/qwfn_plat_win32.cpp
  src/qwfn_io.cpp
  src/qwfn_expert_cache.cpp
  src/qwfn_ple.cpp
  src/qwfn_weights.cpp
  src/qwfn_graph.cpp
  src/qwfn_state.cpp
  src/qwfn_engine.cpp
  src/qwfn_prefill.cpp
  src/qwfn_vision.cpp
)
target_include_directories(qwfn_core PUBLIC
  "${LLAMA_CPP_ROOT}/vendor/stb"
  src
  "${LLAMA_CPP_ROOT}/ggml/include"
  "${LLAMA_CPP_ROOT}/ggml/src"       # ggml-impl.h: the graph struct, for the per-node profiler only
)
if(NOT WIN32)
  target_include_directories(qwfn_core PUBLIC "${URING_INCLUDE_DIR}")
endif()
target_link_libraries(qwfn_core PUBLIC ggml_base ggml_all Threads::Threads ${QWFN_IO_LIBS} ${CMAKE_DL_LIBS})
target_compile_options(qwfn_core PRIVATE ${QWFN_OPT_FLAGS} ${QWFN_ARCH_FLAGS} ${QWFN_WARN_FLAGS})

# The ggml-sycl patch series (patches/ggml-sycl, Intel Arc) adds two weight layouts as
# new ggml types. A stock ggml -- the CUDA build -- does not have them, so the engine
# compiles their use in only when the headers it builds against declare them.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${LLAMA_CPP_ROOT}/ggml/include")
check_cxx_source_compiles("#include <ggml.h>
int main() { return (int) GGML_TYPE_Q2_0_SOA + (int) GGML_TYPE_IQ4_NL_SOA; }" QWFN_HAVE_GGML_SOA)
unset(CMAKE_REQUIRED_INCLUDES)
if(QWFN_HAVE_GGML_SOA)
  target_compile_definitions(qwfn_core PUBLIC QWFN_HAVE_GGML_SOA=1)
else()
  target_compile_definitions(qwfn_core PUBLIC QWFN_HAVE_GGML_SOA=0)
endif()

# Collected once at configure time, so build llama.cpp before configuring this.
# If the list comes up empty the copy is skipped and the DLLs have to be on PATH
# instead, which is why it says so.
if(WIN32)
  file(GLOB QWFN_RUNTIME_DLLS "${LLAMA_CPP_BUILD}/*.dll" "${LLAMA_CPP_BUILD}/Release/*.dll")
  list(LENGTH QWFN_RUNTIME_DLLS _n_dlls)
  if(_n_dlls EQUAL 0)
    message(WARNING
      "no .dll found under ${LLAMA_CPP_BUILD}: the tools will need llama.cpp's DLLs on PATH. "
      "Build llama.cpp with -DBUILD_SHARED_LIBS=ON before configuring this project.")
  else()
    message(STATUS "runtime DLLs to stage next to each tool: ${_n_dlls}")
  endif()
endif()

# A helper so every executable gets the same treatment and a new one cannot
# quietly be built at the compiler's default -march.
function(qwfn_tool name)
  cmake_parse_arguments(T "" "" "SOURCES;LIBS;INCLUDES;OPT" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  if(T_INCLUDES)
    target_include_directories(${name} PRIVATE ${T_INCLUDES})
  endif()
  target_link_libraries(${name} PRIVATE ${T_LIBS})
  if(T_OPT)
    target_compile_options(${name} PRIVATE ${T_OPT} ${QWFN_ARCH_FLAGS})
  else()
    target_compile_options(${name} PRIVATE ${QWFN_OPT_FLAGS} ${QWFN_ARCH_FLAGS})
  endif()
  # Windows resolves a DLL from the executable's own directory first, so put
  # llama.cpp's there after every build and a tool runs straight out of the
  # tree. All of them, not just the two linked against: ggml.dll loads
  # ggml-cpu.dll and ggml-cuda.dll by name at run time, and those would
  # otherwise have to be on PATH.
  if(WIN32 AND QWFN_RUNTIME_DLLS)
    add_custom_command(TARGET ${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QWFN_RUNTIME_DLLS} "$<TARGET_FILE_DIR:${name}>")
  endif()
endfunction()

qwfn_tool(qwfn-iobench  SOURCES tools/qwfn_iobench.cpp        LIBS qwfn_core)
if(NOT WIN32)   # POSIX file calls; the pack format itself (src/qwfn_pack.cpp) is portable
  qwfn_tool(qwfn-pack   SOURCES tools/qwfn_pack.cpp           LIBS qwfn_core)
endif()
qwfn_tool(qwfn-inspect  SOURCES tools/qwfn_inspect.cpp        LIBS qwfn_core)
qwfn_tool(qwfn-test-hc  SOURCES tools/qwfn_test_hc.cpp        LIBS qwfn_core)
qwfn_tool(qwfn-logits   SOURCES tools/qwfn_logits.cpp         LIBS qwfn_core)
qwfn_tool(qwfn-run      SOURCES tools/qwfn_run.cpp            LIBS qwfn_core)
qwfn_tool(qwfn-gen      SOURCES tools/qwfn_gen.cpp            LIBS qwfn_core)
# qwfn_vocab.cpp is deliberately NOT in qwfn_core: only the binaries that need a
# tokenizer should carry a libllama dependency.
qwfn_tool(qwfn-vtest    SOURCES tools/qwfn_vtest.cpp          LIBS qwfn_core)

qwfn_tool(qwfn-refdump
  SOURCES  tools/qwfn_refdump.cpp
  INCLUDES "${LLAMA_CPP_ROOT}/include" "${LLAMA_CPP_ROOT}/ggml/include"
  LIBS     llama_ref ggml_base ggml_all
  OPT      ${QWFN_OPT_FLAGS})

qwfn_tool(qwfn-chat
  SOURCES  tools/qwfn_chat.cpp src/qwfn_vocab.cpp
  INCLUDES "${LLAMA_CPP_ROOT}/include"
  LIBS     qwfn_core llama_ref)

qwfn_tool(qwfn-tok
  SOURCES  tools/qwfn_tok.cpp src/qwfn_vocab.cpp
  INCLUDES "${LLAMA_CPP_ROOT}/include"
  LIBS     qwfn_core llama_ref)

# Overlay tooling (docs/dense-overlay.md): metadata heads and local re-encodes of split GGUFs.
# POSIX only (unistd.h).
if(NOT WIN32)
  qwfn_tool(gguf-requant SOURCES tools/overlay/gguf_requant.cpp LIBS ggml_base ggml_all Threads::Threads
            INCLUDES "${LLAMA_CPP_ROOT}/ggml/include")
endif()

# Isolated kernel benches at this model's decode shapes (tools/perf).
foreach(b mv_bench moe_bench)
  qwfn_tool(${b} SOURCES tools/perf/${b}.cpp LIBS ggml_base ggml_all INCLUDES "${LLAMA_CPP_ROOT}/ggml/include")
endforeach()

qwfn_tool(qwfn-train-predictor SOURCES tools/qwfn_train_predictor.cpp LIBS qwfn_core)
target_compile_definitions(qwfn-train-predictor PRIVATE LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD}")

# ------------------------------------------------------------------ server --
#
# On Windows the stall watchdog logs the stall without the stuck thread's
# stack (no pthread_kill/backtrace there); the Rafiekuntest port built and
# ran the server that way, so it is on for every platform.
option(QWFN_BUILD_SERVER "Build qwfn-server" ON)
if(QWFN_BUILD_SERVER)
  # cpp-httplib 0.54 is split header/source, so its .cpp joins the target.
  qwfn_tool(qwfn-server
    SOURCES  tools/qwfn_server.cpp src/qwfn_vocab.cpp
             "${LLAMA_CPP_ROOT}/vendor/cpp-httplib/httplib.cpp"
    INCLUDES "${LLAMA_CPP_ROOT}/include"
             "${LLAMA_CPP_ROOT}/vendor"
             "${LLAMA_CPP_ROOT}/vendor/cpp-httplib"
    LIBS     qwfn_core llama_ref
    OPT      ${QWFN_OPT_FLAGS})
  if(WIN32)
    target_link_libraries(qwfn-server PRIVATE ws2_32 crypt32)   # cpp-httplib: sockets; crypt32 as in the tested Windows port
  endif()
endif()
