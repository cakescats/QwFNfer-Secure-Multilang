#include "qwfn_pack.h"

#include "qwfn_plat.h"

#include <cstdlib>
#include <cstring>

namespace qwfn {

static const char PACK_MAGIC[8] = { 'Q', 'W', 'F', 'N', 'P', 'A', 'K', '1' };

static const char * part_tensor_suffix(int q) {
    return q == EXPERT_GATE ? "ffn_gate_exps.weight" : q == EXPERT_UP ? "ffn_up_exps.weight" : "ffn_down_exps.weight";
}

uint32_t pack_engine_dio_align(const model_index & mi) {
    if (getenv("QWFN_DIO_512")) return 512;
    for (uint32_t il = 0; il < mi.hp().n_layer; il++)
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range a = mi.expert_range(il, 0, (expert_part) q);
            const byte_range b = mi.expert_range(il, 1, (expert_part) q);
            if (a.valid() && (a.offset % QWFN_PACK_ALIGN) != 0) return 512;
            if (a.valid() && b.valid() && ((b.offset - a.offset) % QWFN_PACK_ALIGN) != 0) return 512;
        }
    return QWFN_PACK_ALIGN;
}

bool pack_compute_layout(const model_index & mi, uint32_t align, pack_layout & L, std::string & err) {
    const hparams & hp = mi.hp();
    if (align == 0) align = pack_engine_dio_align(mi);
    if (align != 512 && align != QWFN_PACK_ALIGN) { err = "direct-I/O alignment must be 512 or 4096"; return false; }
    L = pack_layout{};
    L.n_layer   = hp.n_layer;
    L.n_expert  = hp.n_expert;
    L.dio_align = align;
    L.fingerprint = pack_fingerprint(mi);
    L.layers.resize(hp.n_layer);
    const uint64_t A = align;
    uint64_t off = QWFN_PACK_HEADER;
    for (uint32_t il = 0; il < hp.n_layer; il++) {
        pack_layer & pl = L.layers[il];
        uint32_t boff = 0;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range r0 = mi.expert_range(il, 0, (expert_part) q);
            const byte_range r1 = mi.expert_range(il, 1, (expert_part) q);
            if (!r0.valid()) { err = "missing expert tensor on layer " + std::to_string(il); return false; }
            // expert_cache::init refuses the same thing: the read-around pad must be one constant per (layer, part).
            if (r1.valid() && ((r1.offset - r0.offset) % A) != 0) {
                err = "expert slice stride is not " + std::to_string(A) + "-byte aligned on layer " + std::to_string(il);
                return false;
            }
            // expert_cache::init: part_pay = dio_pad(offset), the part's slot = dio_padded_size(offset, nbytes).
            const uint32_t pad = (uint32_t) (r0.offset % A);
            pl.part_off[q]   = boff;
            pl.part_pay[q]   = pad;
            pl.part_bytes[q] = r0.nbytes;
            boff += (uint32_t) ((pad + (uint64_t) r0.nbytes + A - 1) / A * A);
        }
        pl.block_bytes = boff;
        pl.stride      = (boff + QWFN_PACK_ALIGN - 1) / QWFN_PACK_ALIGN * QWFN_PACK_ALIGN;
        pl.offset      = off;
        off += (uint64_t) pl.stride * hp.n_expert;
    }
    L.file_bytes = off;
    return true;
}

uint64_t pack_fingerprint(const model_index & mi) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { for (int i = 0; i < 8; i++) { h ^= (v >> (8 * i)) & 0xff; h *= 1099511628211ull; } };
    for (const std::string & p : mi.shard_paths()) {
        bool direct = false;
        const file_handle f = plat_open_read(p.c_str(), false, &direct);
        mix(f == FILE_NONE ? 0 : (uint64_t) plat_file_size(f));
        if (f != FILE_NONE) plat_close(f);
    }
    for (uint32_t il = 0; il < mi.hp().n_layer; il++)
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const tensor_ref * t = mi.find("blk." + std::to_string(il) + "." + part_tensor_suffix(q));
            if (!t) { mix(~0ull); continue; }
            mix((uint64_t) t->shard); mix(t->file_offset); mix((uint64_t) t->type);
            for (int d = 0; d < 4; d++) mix((uint64_t) t->ne[d]);
        }
    return h;
}

// Little-endian fixed fields; the per-layer records follow at byte 64.
namespace {
struct wr { uint8_t * p; void u32(uint32_t v) { memcpy(p, &v, 4); p += 4; } void u64(uint64_t v) { memcpy(p, &v, 8); p += 8; } };
struct rd { const uint8_t * p; uint32_t u32() { uint32_t v; memcpy(&v, p, 4); p += 4; return v; } uint64_t u64() { uint64_t v; memcpy(&v, p, 8); p += 8; return v; } };
constexpr size_t LAYER_REC = 8 + 4 + 4 + 3 * 4 * 3;
}

void pack_encode_header(const pack_layout & L, uint8_t * buf) {
    memset(buf, 0, QWFN_PACK_HEADER);
    memcpy(buf, PACK_MAGIC, 8);
    wr w{ buf + 8 };
    w.u32(QWFN_PACK_VERSION); w.u32(QWFN_PACK_ALIGN); w.u32(L.n_layer); w.u32(L.n_expert); w.u32(L.dio_align);
    w.u64(L.fingerprint); w.u64(L.file_bytes);
    w.p = buf + 64;
    for (const pack_layer & pl : L.layers) {
        w.u64(pl.offset); w.u32(pl.block_bytes); w.u32(pl.stride);
        for (int q = 0; q < EXPERT_NPARTS; q++) { w.u32(pl.part_off[q]); w.u32(pl.part_pay[q]); w.u32(pl.part_bytes[q]); }
    }
}

bool pack_decode_header(const uint8_t * buf, pack_layout & L, std::string & err) {
    if (memcmp(buf, PACK_MAGIC, 8) != 0) { err = "not an expert pack, or an incomplete one (no header)"; return false; }
    rd r{ buf + 8 };
    const uint32_t ver = r.u32(), align = r.u32();
    if (ver != QWFN_PACK_VERSION) { err = "expert pack version " + std::to_string(ver) + ", this build reads " + std::to_string(QWFN_PACK_VERSION); return false; }
    if (align != QWFN_PACK_ALIGN) { err = "expert pack alignment " + std::to_string(align); return false; }
    L.n_layer = r.u32(); L.n_expert = r.u32(); L.dio_align = r.u32();
    L.fingerprint = r.u64(); L.file_bytes = r.u64();
    if (L.n_layer == 0 || 64 + (size_t) L.n_layer * LAYER_REC > QWFN_PACK_HEADER) { err = "expert pack header is corrupt"; return false; }
    r.p = buf + 64;
    L.layers.assign(L.n_layer, pack_layer{});
    for (pack_layer & pl : L.layers) {
        pl.offset = r.u64(); pl.block_bytes = r.u32(); pl.stride = r.u32();
        for (int q = 0; q < EXPERT_NPARTS; q++) { pl.part_off[q] = r.u32(); pl.part_pay[q] = r.u32(); pl.part_bytes[q] = r.u32(); }
    }
    return true;
}

bool pack_open(const std::string & path, const model_index & mi, pack_layout & L, std::string & err) {
    bool direct = false;
    const file_handle f = plat_open_read(path.c_str(), false, &direct);
    if (f == FILE_NONE) { err = "cannot open expert pack " + path + ": " + plat_last_error(); return false; }
    std::vector<uint8_t> buf(QWFN_PACK_HEADER);
    const int64_t got = plat_pread(f, buf.data(), buf.size(), 0);
    const int64_t size = plat_file_size(f);
    plat_close(f);
    if (got != (int64_t) buf.size()) { err = "expert pack " + path + " is too short"; return false; }
    if (!pack_decode_header(buf.data(), L, err)) { err = path + ": " + err; return false; }
    if (L.fingerprint != pack_fingerprint(mi) || L.n_layer != mi.hp().n_layer || L.n_expert != mi.hp().n_expert) {
        err = path + " was built from another checkpoint (fingerprint mismatch); rebuild it with qwfn-pack";
        return false;
    }
    if (size < 0 || (uint64_t) size < L.file_bytes) { err = path + " is truncated"; return false; }
    return true;
}

} // namespace qwfn
