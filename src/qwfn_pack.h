#pragma once
// Expert pack: every (layer, expert) block of a checkpoint, gate+up+down back to
// back, byte for byte in the layout of a RAM-tier slot.
//
// In the GGUF the three parts of one expert sit in three different tensors, so a
// miss costs three reads in three places on the drive. Measured on the reference
// NVMe (2026-10-01, O_DIRECT, random offsets): 0.6-1 MiB reads top out at
// 6.0-6.1 GB/s and the engine's three-part pattern at 5.3-5.5 GB/s, while one
// 3 MiB read per expert reaches 6.45-6.5 GB/s. With the pack every decode miss can be
// that one read: it lands at the slot's base and every part pointer the
// cache already computes (part_off + part_pay) is valid without a copy.
//
// The pack holds only the routed experts. The prefill keeps reading the GGUF:
// it already streams each tensor as one sequential range.
//
// The slot layout depends on the direct-I/O alignment the engine picks for the
// file (engine::init): 4096 when every expert slice and its stride are page
// multiples, else 512. The Q4 checkpoint is 512: its strides are page multiples
// but GGUF aligns tensors to 32 bytes only, so each part's payload starts
// `offset % 512` bytes into its slot part. The pack records that alignment and
// a reader may use it only when the engine's layout is the same, byte for byte.
//
// File layout:
//   [0, QWFN_PACK_HEADER)  header (written last, so an interrupted pack has no magic)
//   per layer il:          n_expert blocks, expert e at offset[il] + e * stride[il]
//                          (stride = block_bytes rounded up to QWFN_PACK_ALIGN: every block starts on a page)
//   per block:             block_bytes as the slot holds them: part q at part_off[q], its payload at
//                          part_off[q] + part_pay[q], zeros elsewhere
#include "qwfn_model.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace qwfn {

static constexpr uint32_t QWFN_PACK_ALIGN   = 4096;
static constexpr uint32_t QWFN_PACK_HEADER  = 16384;
static constexpr uint32_t QWFN_PACK_VERSION = 1;

struct pack_layer {
    uint64_t offset      = 0;   // file offset of expert 0
    uint32_t block_bytes = 0;   // one expert as a slot holds it: all three parts, padded to dio_align
    uint32_t stride      = 0;   // block_bytes rounded up to QWFN_PACK_ALIGN
    std::array<uint32_t, EXPERT_NPARTS> part_off   {};   // part start inside a block
    std::array<uint32_t, EXPERT_NPARTS> part_pay   {};   // payload start inside its part
    std::array<uint32_t, EXPERT_NPARTS> part_bytes {};   // payload size
};

struct pack_layout {
    uint32_t n_layer  = 0;
    uint32_t n_expert = 0;
    uint32_t dio_align = 0;     // the engine's slot alignment this layout mirrors (512 or 4096)
    uint64_t fingerprint = 0;   // of the source checkpoint's expert tensors, see pack_fingerprint
    uint64_t file_bytes  = 0;   // header + every block
    std::vector<pack_layer> layers;
};

// The direct-I/O alignment engine::init picks for `mi` alone (no cold tier):
// 4096 when every expert slice offset and stride is a page multiple, else 512.
uint32_t pack_engine_dio_align(const model_index & mi);

// The slot layout the expert cache computes for `mi` at `dio_align` (0 = the
// one the engine would pick), as an expert pack lays it out.
bool pack_compute_layout(const model_index & mi, uint32_t dio_align, pack_layout & out, std::string & err);

// FNV-1a over the shard sizes and, for every expert tensor, its shard, file
// offset, type and shape: a pack built from another checkpoint never matches.
uint64_t pack_fingerprint(const model_index & mi);

// Header (de)serialisation; `buf` is QWFN_PACK_HEADER bytes.
void pack_encode_header(const pack_layout & L, uint8_t * buf);
bool pack_decode_header(const uint8_t * buf, pack_layout & L, std::string & err);

// Open-and-check for the engine: reads the header of `path`, and accepts it only
// if it was built from `mi` (fingerprint) and the file is complete.
bool pack_open(const std::string & path, const model_index & mi, pack_layout & out, std::string & err);

} // namespace qwfn
