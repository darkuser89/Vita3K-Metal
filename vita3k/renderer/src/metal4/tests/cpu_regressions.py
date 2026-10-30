#!/usr/bin/env python3
"""CPU-only Metal regressions; generated sources/binaries live in a temp directory.

Extract production functions verbatim so the GXM export can be tested without
constructing the emulator. Only its environment/logging boundary is stubbed.
This does not test a GPU, game, or Vita hardware equivalence.
"""
from pathlib import Path
import os
import re
import shlex
import subprocess
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[5]


def function(path, signature, source=None):
    if source is None:
        source = (ROOT / path).read_text()
    begin = source.index(signature)
    body = source.index('{', begin)
    while source.find(';', begin, body) != -1:
        begin = source.index(signature, begin + len(signature))
        body = source.index('{', begin)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def run(name, source, directory, extra=(), arguments=(), sanitizer='address,undefined'):
    path = directory / (name + '.cpp')
    path.write_text(source)
    binary = directory / name
    command = ['xcrun', 'clang++', '-std=c++23', '-O1', '-g',
               '-fsanitize='+sanitizer, '-fno-omit-frame-pointer',
               '-I' + str(ROOT / 'vita3k/gxm/include'),
               '-I' + str(ROOT / 'vita3k/util/include'), str(path), *extra, '-o', str(binary)]
    subprocess.run(command, check=True)
    subprocess.run([str(binary), *arguments], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


COMMON = r'''
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <gxm/types.h>
'''


def uniform_source():
    size = function('vita3k/gxm/src/gxp.cpp', 'int get_parameter_type_size(')
    half = function('vita3k/util/src/float_to_half.cpp', 'void float_to_half(')
    convert = function('vita3k/modules/SceGxm/SceGxm.cpp', 'template <typename T>\nstatic void convert_uniform_data(')
    export = function('vita3k/modules/SceGxm/metal4/SceGxm_metal.inl', 'int export_sceGxmSetUniformDataF_metal3(')
    return COMMON + r'''
#define TRACY_FUNC(...)
#define RET_ERROR(value) static_cast<int>(value)
namespace renderer { enum class Backend { Metal }; }
struct Renderer { renderer::Backend current_backend = renderer::Backend::Metal; };
struct EmuEnvState { Renderer *renderer; };
namespace gxp {
''' + size + '\n}\n' + half + '\n' + convert + '\n' + export + r'''
int main() {
    Renderer renderer; EmuEnvState env{&renderer};
    size_t cases = 0;
    for (auto type : {SCE_GXM_PARAMETER_TYPE_S8, SCE_GXM_PARAMETER_TYPE_U8,
            SCE_GXM_PARAMETER_TYPE_S16, SCE_GXM_PARAMETER_TYPE_U16,
            SCE_GXM_PARAMETER_TYPE_S32, SCE_GXM_PARAMETER_TYPE_U32,
            SCE_GXM_PARAMETER_TYPE_F16, SCE_GXM_PARAMETER_TYPE_F32}) {
        const size_t bytes = gxp::get_parameter_type_size(type);
        for (unsigned components = 1; components <= 4; ++components)
        for (unsigned arrays : {1u, 2u, 5u})
        for (unsigned offset = 0; offset <= components * arrays; ++offset)
        for (unsigned count = 0; count <= components * arrays - offset; ++count) {
            SceGxmProgramParameter p{};
            p.category = SCE_GXM_PARAMETER_CATEGORY_UNIFORM;
            p.type = type; p.component_count = components; p.array_size = arrays;
            p.resource_index = 3;
            std::array<uint8_t, 512> actual, expected;
            actual.fill(0xA5); expected = actual;
            std::vector<float> input(std::max(count, 1u));
            const bool signed_type = type == SCE_GXM_PARAMETER_TYPE_S8
                || type == SCE_GXM_PARAMETER_TYPE_S16 || type == SCE_GXM_PARAMETER_TYPE_S32;
            for (unsigned i = 0; i < count; ++i) input[i] = signed_type ? float(int(i) - 12) : float(i + 2);
            const size_t align = (type == SCE_GXM_PARAMETER_TYPE_F32 || type == SCE_GXM_PARAMETER_TYPE_F16)
                && components != 1 ? 8 : 4;
            const size_t stride = arrays == 1 ? components * bytes
                : (components * bytes + align - 1) / align * align;
            for (unsigned i = 0; i < count; ++i) {
                const unsigned component = offset + i;
                auto *dest = expected.data() + 12 + (component / components) * stride + (component % components) * bytes;
                switch (type) {
#define CONVERT(kind, T) case kind: { T value = static_cast<T>(input[i]); std::memcpy(dest, &value, sizeof(value)); break; }
                CONVERT(SCE_GXM_PARAMETER_TYPE_S8, int8_t)
                CONVERT(SCE_GXM_PARAMETER_TYPE_U8, uint8_t)
                CONVERT(SCE_GXM_PARAMETER_TYPE_S16, int16_t)
                CONVERT(SCE_GXM_PARAMETER_TYPE_U16, uint16_t)
                CONVERT(SCE_GXM_PARAMETER_TYPE_S32, int32_t)
                CONVERT(SCE_GXM_PARAMETER_TYPE_U32, uint32_t)
                CONVERT(SCE_GXM_PARAMETER_TYPE_F16, __fp16)
                CONVERT(SCE_GXM_PARAMETER_TYPE_F32, float)
#undef CONVERT
                default: assert(false);
                }
            }
            assert(export_sceGxmSetUniformDataF_metal3(env, 0, "test", actual.data(), &p, offset, count, input.data()) == 0);
            assert(actual == expected);
            const auto before = actual;
            assert(export_sceGxmSetUniformDataF_metal3(env, 0, "test", actual.data(), &p, components * arrays, 1, input.data())
                == static_cast<int>(SCE_GXM_ERROR_INVALID_VALUE));
            assert(actual == before);
            ++cases;
        }
    }
    std::cout << "PASS uniform: " << cases << " type/array/partial-write cases + rejected ranges; padding preserved\n";
}
'''


def ranges_source():
    source = function('vita3k/renderer/src/metal4/textures.mm', 'bool surface_memory_ranges(')
    # Use the actual range enumerator; size/format stubs provide valid storage.
    return r'''
#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
enum { SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_COLOR_SURFACE_TILED, SCE_GXM_COLOR_SURFACE_SWIZZLED };
struct SceGxmColorSurface { uint32_t width, height, strideInPixels, colorFormat, surfaceType; };
struct SurfaceRect { uint32_t x, y, width, height; };
struct SurfaceMemoryRange { size_t offset, size; bool operator==(const SurfaceMemoryRange &) const = default; };
namespace gxm { uint32_t get_base_format(uint32_t f) { return f; } uint32_t bits_per_pixel(uint32_t f) { return f * 8; } }
size_t surface_memory_size(const SceGxmColorSurface &s) { return size_t(s.width) * s.height * s.colorFormat; }
''' + source + r'''
uint32_t compact(uint32_t value) {
    uint32_t result = 0;
    for (unsigned bit = 0; bit < 16; ++bit) result |= ((value >> (2 * bit)) & 1) << bit;
    return result;
}
std::vector<SurfaceMemoryRange> reference(const SceGxmColorSurface &s, SurfaceRect r) {
    std::vector<SurfaceMemoryRange> result;
    const uint32_t side = std::min(s.width, s.height), k = std::bit_width(side) - 1;
    for (uint32_t pixel = 0; pixel < s.width * s.height; ++pixel) {
        const auto upper = (pixel >> (2 * k)) << k;
        const auto x = (compact(pixel >> 1) & (side - 1)) | (s.width >= s.height ? upper : 0);
        const auto y = (compact(pixel) & (side - 1)) | (s.width < s.height ? upper : 0);
        if (x < r.x || y < r.y || uint64_t(x) >= uint64_t(r.x) + r.width || uint64_t(y) >= uint64_t(r.y) + r.height) continue;
        const size_t offset = size_t(pixel) * s.colorFormat;
        if (!result.empty() && result.back().offset + result.back().size == offset) result.back().size += s.colorFormat;
        else result.push_back({offset, s.colorFormat});
    }
    return result;
}
int main() {
    size_t cases = 0;
    for (uint32_t w : {1, 2, 4, 8, 16}) for (uint32_t h : {1, 2, 4, 8, 16})
    for (uint32_t bytes : {1, 2, 4, 8, 16})
    for (uint32_t x = 0; x <= w; ++x) for (uint32_t y = 0; y <= h; ++y)
    for (uint32_t rw = 0; rw <= w + 1; ++rw) for (uint32_t rh = 0; rh <= h + 1; ++rh) {
        SceGxmColorSurface s{w,h,w,bytes,SCE_GXM_COLOR_SURFACE_SWIZZLED}; SurfaceRect r{x,y,rw,rh};
        std::vector<SurfaceMemoryRange> result;
        assert(surface_memory_ranges(s, r, result)); assert(result == reference(s, r)); ++cases;
    }
    SceGxmColorSurface large{16384,16384,16384,16,SCE_GXM_COLOR_SURFACE_SWIZZLED};
    std::vector<SurfaceMemoryRange> result;
    assert(surface_memory_ranges(large, {16383,16383,1,1}, result));
    assert((result == std::vector<SurfaceMemoryRange>{{size_t(16384)*16384*16-16,16}}));
    std::cout << "PASS Morton: " << cases << " exhaustive small rectangles + 16384-square one-pixel edge\n";
}
'''


def bc7_source():
    return r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#define BCDEC_IMPLEMENTATION
''' + '#include "' + str(ROOT / 'vita3k/renderer/src/metal4/bcdec.h') + '"\n' + r'''
int main() {
    std::mt19937_64 rng(0xBC7);
    for (unsigned mode = 0; mode < 8; ++mode) {
        for (unsigned i = 0; i < 10000; ++i) {
            alignas(8) std::array<uint64_t,2> block{rng(),rng()};
            block[0] = (block[0] & ~((1ull << (mode + 1)) - 1)) | (1ull << mode);
            std::array<uint8_t,80> out; out.fill(0xA5);
            bcdec_bc7(block.data(), out.data() + 8, 16);
            for (unsigned j = 0; j < 8; ++j) assert(out[j] == 0xA5 && out[72+j] == 0xA5);
            if (mode <= 3) for (unsigned j = 0; j < 16; ++j) assert(out[8 + j*4 + 3] == 255);
        }
    }
    std::cout << "PASS BC7: 80000 mode-specific blocks, bounds and opaque-alpha checks (not a hardware oracle)\n";
}
'''


def bc7_reference_test(directory):
    source = (ROOT / 'vita3k/renderer/src/metal4/tests/bc7_reference.cpp').read_text()
    source = source.replace('#include "../bcdec.h"',
        '#include "' + str(ROOT / 'vita3k/renderer/src/metal4/bcdec.h') + '"')
    run('bc7_reference', source, directory, ('-Wno-deprecated-declarations',))


def bcn_upload_source():
    path = 'vita3k/renderer/src/metal4/texture_format_metal.cpp'
    formats = function(path, 'SceGxmTextureBaseFormat get_matching_decompressed_format_metal3(')
    decoder = function(path, 'uint32_t decompress_compressed_texture_metal3(')
    bpp = function('vita3k/gxm/src/textures.cpp', 'uint32_t bits_per_pixel(')
    upload = (ROOT / 'vita3k/renderer/src/metal4/texture_cache_metal.cpp').read_text()
    start = upload.index('            // decompress the texture\n')
    branch = upload[start:upload.index('\n        }', start)]
    return COMMON + r'''
#include <random>
#define LOG_ERROR(...)
#define BCDEC_IMPLEMENTATION
''' + '#include "' + str(ROOT / 'vita3k/renderer/src/metal4/bcdec.h') + '"\n' + r'''
// Unvisited legacy BC/PVRTC boundaries are intentionally not emulated here.
void decompress_bc_image(...) { assert(false); }
namespace pvr { void PVRTDecompressPVRTC(...) { assert(false); } }
uint32_t get_compressed_size(SceGxmTextureBaseFormat, uint32_t w, uint32_t h) { return ((w+3)/4)*((h+3)/4)*16; }
namespace gxm {
''' + bpp + '\n}\n' + formats + '\n' + decoder + r'''
int main() {
    std::mt19937_64 rng(0xBC6);
    size_t cases=0;
    for (auto base_format : {SCE_GXM_TEXTURE_BASE_FORMAT_UBC6H, SCE_GXM_TEXTURE_BASE_FORMAT_SBC6H, SCE_GXM_TEXTURE_BASE_FORMAT_UBC7})
    for (uint32_t pixels_per_stride : {1,3,4,7,8,16})
    for (uint32_t memory_height : {1,3,4,5,8})
    for (unsigned variant=0;variant<32;++variant) {
        std::vector<uint64_t> compressed(get_compressed_size(base_format,pixels_per_stride,memory_height)/8);
        for (auto &value : compressed) value=rng();
        const void *pixels=compressed.data();
        std::vector<uint8_t> texture_data_decompressed;
        uint32_t bytes_per_pixel=0,bpp=0;
        SceGxmTextureBaseFormat upload_format{};
''' + branch + r'''
        const bool hdr=base_format!=SCE_GXM_TEXTURE_BASE_FORMAT_UBC7;
        assert(bytes_per_pixel==(hdr?8:4));
        assert(texture_data_decompressed.size()==size_t(pixels_per_stride)*memory_height*bytes_per_pixel);
        if(hdr) for(size_t i=0;i<size_t(pixels_per_stride)*memory_height;++i) {
            uint16_t alpha; std::memcpy(&alpha, texture_data_decompressed.data()+i*8+6,2); assert(alpha==0x3c00);
        }
        ++cases;
    }
    std::cout << "PASS BC upload: " << cases << " actual allocation/decoder paths, including partial blocks and RGBA16F alpha\n";
}
'''


def dds_import_test(directory):
    source = Path(__file__).with_name('dds_import.cpp').read_text()
    replacement = 'vita3k/renderer/src/metal4/texture_replacement_metal.cpp'
    shared = 'vita3k/renderer/src/texture/format.cpp'
    private = 'vita3k/renderer/src/metal4/texture_format_metal.cpp'
    gxm = '\n'.join(function('vita3k/gxm/src/textures.cpp', signature) for signature in (
        'SceGxmTextureFormat get_format(', 'SceGxmTextureBaseFormat get_base_format(',
        'uint32_t get_num_components(', 'bool is_bcn_format(', 'uint32_t bits_per_pixel(',
        'std::pair<uint32_t, uint32_t> get_block_size('))
    decoders = '\n'.join(function(shared, signature) for signature in (
        'bool is_astc_format(', 'uint32_t get_compressed_size(',
        'static void decompress_block_bc1(', 'static void decompress_block_alpha(',
        'static void decompress_block_alpha_signed(', 'static void decompress_block_bc2(',
        'static void decompress_block_bc3(', 'static void decompress_block_bc4u(',
        'static void decompress_block_bc4s(', 'static void decompress_block_bc5u(',
        'static void decompress_block_bc5s(', 'void decompress_bc_image('))
    decoders += '\n' + '\n'.join(function(private, signature) for signature in (
        'SceGxmTextureBaseFormat get_matching_decompressed_format_metal3(',
        'uint32_t decompress_compressed_texture_metal3('))
    formats = '\n'.join(function(replacement, signature) for signature in (
        'static SceGxmTextureBaseFormat dxgi_to_gxm(', 'static bool dds_swap_rb(',
        'static bool software_bcn_format('))
    formats += '\n' + function(replacement, 'static uint64_t dds_mip_size(')
    formats += '\n' + function(replacement, 'static bool valid_dds_import(')
    imports = '\n'.join(function(replacement, signature) for signature in (
        'bool metal::MetalTextureCache::import_configure_texture(',
        'void metal::MetalTextureCache::import_upload_texture('))
    for marker, text in {'@GXM@': gxm, '@DECODERS@': decoders,
                         '@FORMATS@': formats, '@IMPORT@': imports,
                         '@BCDEC@': str(ROOT / 'vita3k/renderer/src/metal4/bcdec.h'),
                         '@ASTC@': str(ROOT / 'vita3k/renderer/src/texture/astc_formats.inc')}.items():
        source = source.replace(marker, text)
    source = re.sub(r'#include "(?:\.\./)*[^"\n]*astc_formats.inc"',
        '#include "' + str(ROOT / 'vita3k/renderer/src/texture/astc_formats.inc') + '"', source)
    run('dds_import', source, directory,
        ('-I' + str(ROOT / 'external/ddspp'), '-I' + str(ROOT / 'external/fmt/include')),
        (str(directory / 'dds-input'),))


def cache_read_concurrency_test(directory, implementation=None):
    path = 'vita3k/renderer/src/metal4/cache.mm'
    text = implementation if implementation is not None else (ROOT / path).read_text()
    source = Path(__file__).with_name('cache_read_concurrency.mm').read_text()
    source = source.replace('// STATS_HOOK', '\n'.join(function(
        'vita3k/renderer/include/renderer/metal4/cache.h', signature) + ';'
        for signature in ('struct CacheWriteWaitStats', 'struct CacheStats')))
    source = source.replace('// HELPERS_HOOK', '\n'.join(function(path, signature, text)
        for signature in ('bool valid_guest_hash(', 'NSURL *url(', 'std::string error_text(',
                          'void put32(', 'uint32_t get32(')))
    # Older baselines do not call reject_file; its unused definition is harmless.
    source = source.replace('// REJECT_HOOK', function(path, '    void reject_file('))
    source = source.replace('// ARCHIVE_HOOK', function(path, '    id<MTL4Archive> load_archive(', text))
    source = source.replace('// READERS_HOOK', '\n'.join(function(path, signature, text)
        for signature in ('std::optional<shader::metal::Program> PersistentCache::load_program(',
                          'std::vector<CachedVariant> PersistentCache::variants(',
                          'std::vector<CachedPipeline> PersistentCache::render_pipeline_templates('))
        .replace('std::ifstream', 'ObservedInput'))
    for marker, signature in (('// RENDER_OPEN_HOOK', 'id<MTLRenderPipelineState> PersistentCache::create_render_pipeline('),
                              ('// COMPUTE_OPEN_HOOK', 'id<MTLComputePipelineState> PersistentCache::create_compute_pipeline(')):
        body = function(path, signature, text)
        source = source.replace(marker, body[body.index('    id<MTL4Archive> archive;'):body.index('    NSError *native_error = nil;')])
    generated = directory / 'cache_read_concurrency.mm'; generated.write_text(source)
    binary = directory / 'cache_read_concurrency'
    sanitizer = os.environ.get('VITA3K_CACHE_READ_SANITIZER', 'address,undefined')
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize='+sanitizer, '-fno-omit-frame-pointer', '-Wno-deprecated-declarations',
                    '-I'+str(ROOT/'vita3k/shader/include'), str(generated),
                    '-framework', 'Foundation', '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(directory/'cache-reader-files')], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1', 'TSAN_OPTIONS':'halt_on_error=1'})


def cache_write_slots_test(directory, implementation=None):
    path = 'vita3k/renderer/src/metal4/cache.mm'
    source = Path(__file__).with_name('cache_write_slots.mm').read_text()
    source = source.replace('// STATS_HOOK', '\n'.join(function('vita3k/renderer/include/renderer/metal4/cache.h', x) + ';'
        for x in ('struct CacheWriteWaitStats', 'struct CacheStats')))
    source = source.replace('// ATOMIC_HOOK', function(path, 'struct TemporaryFile') + ';\n' + function(path, 'bool atomic_write('))
    for marker, signature in (('// SLOT_HOOK', '    void acquire_write_slot('),
                              ('// SAVE_HOOK', '    void save('),
                              ('// FILE_HOOK', '    void queue_file('),
                              ('// DRAIN_HOOK', '    void drain_writes(')):
        # Keep the body verbatim. Spell out empty default constructors so the
        # brace-based extractor does not mistake an argument initializer for it.
        text = (implementation if implementation is not None else (ROOT / path).read_text()).replace(
            'const std::filesystem::path &index_path = {}, std::string index_bytes = {})',
            'const std::filesystem::path &index_path = std::filesystem::path(), std::string index_bytes = std::string())')
        source = source.replace(marker, function(path, signature, text))
    drain_index = function(path, '    void drain_index_writes(', text) if 'void drain_index_writes(' in text else (
        'void drain_index_writes() { drain_writes(); }')
    source = source.replace('// INDEX_DRAIN_HOOK', drain_index)
    source = source.replace('// RETIRE_HOOK', function(path, '    void retire_archive(', text)
                            if 'void retire_archive(' in text else '')
    generated = directory / 'cache_write_slots.mm'; generated.write_text(source)
    binary = directory / 'cache_write_slots'
    sanitizer = os.environ.get('VITA3K_CACHE_WRITE_SANITIZER', 'address,undefined')
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize='+sanitizer, '-fno-omit-frame-pointer',
                    str(generated), '-framework', 'Foundation', '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(directory / 'cache-slot-files')], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1', 'TSAN_OPTIONS':'halt_on_error=1'})


def guest_publication_test(directory):
    path = 'vita3k/renderer/src/metal4/state.mm'
    source = Path(__file__).with_name('guest_publication.mm').read_text()
    source = source.replace('// METHODS_HOOK', '\n'.join(function(path, x) for x in (
        'void MetalState::drain_guest_publications(', 'void MetalState::check_guest_publications(',
        'bool MetalState::defer_scene_completion(')))
    source = source.replace('// FREE_HOOK', function('vita3k/renderer/src/metal4/memory_metal.cpp', 'void free_metal3('))
    generated = directory / 'guest_publication.mm'; generated.write_text(source)
    binary = directory / 'guest_publication'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        '-I' + str(ROOT / 'vita3k/renderer/include'), str(generated),
        '-framework', 'Foundation', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1'})


def readback_buffers_test(directory):
    source = Path(__file__).with_name('readback_buffers.mm').read_text()
    header = 'vita3k/renderer/include/renderer/metal4/textures.h'
    path = 'vita3k/renderer/src/metal4/textures.mm'
    source = source.replace('// STRUCT_HOOK', '\n'.join(function(header, name) + ';' for name in (
        'struct DepthMemoryLayout {', 'struct DepthStoreReadback {')))
    source = source.replace('// GEOMETRY_HOOK', '\n'.join(function(path, name) for name in (
        'static size_t depth_sample_offset(', 'static bool contiguous_depth_store(')))
    source = source.replace('// METHODS_HOOK', '\n'.join(function(path, name) for name in (
        'bool SurfaceCaster::enqueue_depth_store(', 'void SurfaceCaster::finish_depth_store(')))
    generated = directory / 'readback_buffers.mm'; generated.write_text(source)
    binary = directory / 'readback_buffers'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        '-I' + str(ROOT / 'vita3k/renderer/include'), str(generated),
        '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1'})


def color_publication_test(directory):
    path = 'vita3k/renderer/src/metal4/state.mm'
    source = Path(__file__).with_name('color_publication.mm').read_text()
    source = source.replace('// STRUCTS_HOOK', '\n'.join(function(path, x)+';' for x in (
        'struct SurfaceWriteBounds {', 'struct Surface {', 'enum class SurfacePublication {',
        'struct SurfacePublicationPlan {', 'struct SurfacePublicationReadback {')))
    source = source.replace('// FUNCTIONS_HOOK', '\n'.join(function(path, x) for x in (
        'static id<MTLTexture> raw_surface_color(', 'static bool surface_cpu_bytes_changed(',
        'static SurfacePublicationPlan plan_surface_publication(', 'static id<MTLTexture> prepare_surface_publication_image(',
        'static SurfacePublicationReadback prepare_surface_publication(', 'static SurfacePublication finish_surface_publication(',
        'static SurfacePublication read_surface_publication(')))
    generated = directory/'color_publication.mm'; generated.write_text(source)
    binary = directory/'color_publication'
    subprocess.run(['xcrun','clang++','-std=c++23','-fobjc-arc','-O1','-g',
        '-fsanitize=address,undefined','-fno-omit-frame-pointer',
        *['-I'+str(ROOT/('vita3k/'+p+'/include')) for p in ('renderer','gxm','mem','util')],
        str(generated),'-framework','Foundation','-framework','Metal','-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1'})


def guest_memory_publication_test(directory):
    path='vita3k/renderer/src/metal4/memory_metal.cpp'
    source=Path(__file__).with_name('guest_memory_publication.cpp').read_text()
    source=source.replace('// PROTECTION_HOOK','\n'.join(function(path,x) for x in (
        'static void align_to_page(', 'static void consume_protection(', 'static bool add_protect_impl(',
        'bool add_protect_metal3(', 'bool add_write_watch(', 'bool handle_access_violation_metal3(')))
    generated=directory/'guest_memory_publication.cpp';generated.write_text(source)
    binary=directory/'guest_memory_publication'
    subprocess.run(['xcrun','clang++','-std=c++23','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',
        *['-I'+str(ROOT/('vita3k/'+p+'/include')) for p in ('renderer','gxm','mem','util')],str(generated),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1'})


def async_color_publication_test(directory):
    path='vita3k/renderer/src/metal4/state.mm';memory='vita3k/renderer/src/metal4/memory_metal.cpp'
    source=Path(__file__).with_name('color_publication.mm').read_text().split('static void policy_cases(){')[0]
    source=source.replace('struct SurfaceWriteStamp {bool dirty=false;bool changed()const{return dirty;}};',
        '#include <renderer/metal4/guest_memory_publication.h>\n#include <renderer/metal4/guest_publication_queue.h>')
    source=source.replace('// STRUCTS_HOOK','\n'.join(function(path,x)+';' for x in (
        'struct SurfaceWriteBounds {','struct Surface {','enum class SurfacePublication {',
        'struct SurfacePublicationPlan {','struct SurfacePublicationReadback {')))
    source=source.replace('// FUNCTIONS_HOOK','\n'.join(function(path,x) for x in (
        'static id<MTLTexture> raw_surface_color(','static bool surface_cpu_bytes_changed(',
        'static SurfacePublicationPlan plan_surface_publication(','static id<MTLTexture> prepare_surface_publication_image(',
        'static SurfacePublicationReadback prepare_surface_publication(','static SurfacePublication finish_surface_publication(',
        'static SurfacePublication read_surface_publication(')))
    source=source.replace('@property unsigned waits;','@property unsigned waits;\n@property(strong) NSCondition *gate;\n@property bool released;')
    source=source.replace('- (void)waitUntilCompleted{++_waits;',
        '- (void)waitUntilCompleted{[_gate lock];while(_gate&&!_released)[_gate wait];[_gate unlock];++_waits;')
    memory_source=Path(__file__).with_name('guest_memory_publication.cpp').read_text().split('static void ordinary_cases(){')[0]
    memory_source=memory_source.replace('// PROTECTION_HOOK','\n'.join(function(memory,x) for x in (
        'static void align_to_page(','static void consume_protection(','static bool add_protect_impl(',
        'bool add_protect_metal3(','bool add_write_watch(','bool handle_access_violation_metal3(')))
    fixture=Path(__file__).with_name('async_color_publication.inl').read_text()
    fixture=fixture.replace('// PENDING_HOOK',function(path,'struct PendingColorPublication {')+';')
    fixture=fixture.replace('// ASYNC_METHODS_HOOK','\n'.join(function(path,x) for x in (
        'void MetalState::collect_color_publications(','void MetalState::drain_guest_publications(',
        'void MetalState::check_guest_publications(','bool MetalState::defer_color_scene_completion(')))
    generated=directory/'async_color_publication.mm';generated.write_text(source+'\n'+memory_source+'\n'+fixture)
    binary=directory/'async_color_publication'
    subprocess.run(['xcrun','clang++','-std=c++23','-fobjc-arc','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',
        *['-I'+str(ROOT/('vita3k/'+p+'/include')) for p in ('renderer','gxm','mem','util')],str(generated),
        '-framework','Foundation','-framework','Metal','-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1'})


def buffer_test(directory):
    binary = directory / 'buffer_lifetime'
    # Exercise the exact arena algorithm with deliberately colliding hashes.
    # The separately linked production buffers.mm still uses real XXH3.
    collision_class = function('vita3k/renderer/include/renderer/metal4/buffers.h', 'class UploadBufferArena') + ';'
    collision_methods = '\n'.join(function('vita3k/renderer/src/metal4/buffers.mm', signature)
        for signature in ('UploadBufferSlice UploadBufferArena::allocate(',
                          'ReadonlyBufferUpload UploadBufferArena::copy_readonly('))
    collision_code = (collision_class + collision_methods).replace('UploadBufferArena', 'CollisionUploadArena')
    collision_code = collision_code.replace('XXH3_64bits(bytes.data(), bytes.size())', '0')
    source = Path(__file__).with_name('buffer_lifetime.mm').read_text().replace('// COLLISION_ARENA_HOOK',
        'namespace renderer::metal {\n' + collision_code + '\n}')
    source = source.replace('// COMMAND_DEPENDENCY_HOOK', Path(__file__).with_name('command_dependencies.inl').read_text())
    fixture = Path(__file__).with_name('vertex_uploads.inl').read_text()
    fixture = fixture.replace('// COPY_VERTEX_HOOK', function('vita3k/renderer/src/metal4/state.mm',
        'template <size_t Size>\nvoid copy_vertex_attribute_records('))
    reference = re.search(r'R"baseline\((.*?)\)baseline"', fixture, re.S).group(1)
    current = (ROOT / 'vita3k/renderer/src/metal4/state.mm').read_text()
    loops = []
    for name, state in [('original', reference), ('current', current)]:
        if name == 'original':
            loop = state
        else:
            first = state.index('        size_t vertex_elements = 0;')
            end = state.index('        for (uint32_t index = 0; index < SCE_GXM_MAX_TEXTURE_UNITS * 2;', first)
            loop = state[first:end]
        loop = loop.replace('fmt::format', 'vertex_test_fmt::format')
        owner = 'return std::vector<uint8_t>(owner->begin(), owner->end());' if name == 'original' else 'return owner;'
        result_type = 'std::vector<uint8_t>' if name == 'original' else 'renderer::metal::OwnedVertexSnapshot'
        loops.append('static VertexUploadResult ' + name + '_vertex_uploads(renderer::metal::Device &device, '
                     'const VertexUploadFixture &f, VertexUploadContext &ctx) {\n' + r'''
    using namespace renderer::metal;
    struct { Device *device; } storage{&device}; auto *impl = &storage;
    const auto *vp = &f.program; const auto &record = f.record;
    const auto &vertex_stream_layouts = f.layouts; const auto &vertex_attributes = f.attributes;
    const auto &repack_stream = f.repack;
    const auto align = [](size_t n, size_t a) { return (n + a - 1) & ~(a - 1); };
    const auto require = [](bool ok, const char *message) { if (!ok) throw std::runtime_error(message); };
    int mem = 0; constexpr unsigned SCE_GXM_INDEX_FORMAT_U16 = 0, index_format = 0;
    const void *indices = f.indices.data(); const uint32_t count = f.indices.size(), instances = f.instances;
    const uint32_t streams_used = (1u << f.program.streams.size()) - 1;
    auto encoder = [VertexUploadEncoder new]; const bool capture_draw = true;
    std::ostringstream draw_metadata;
    VertexUploadResult result; size_t draw_upload_bytes = 0;
    const auto dump_bytes = [&](const std::string &, const void *bytes, size_t size) {
        const auto *p = static_cast<const uint8_t *>(bytes); result.captured.emplace_back(p, p + size);
    };
''' + '    const auto vertex_stream_snapshot = [&](uint64_t handle, size_t size) -> ' + result_type + r''' {
        if (handle >= f.owners.size() || !f.owners[handle]) return {};
        const auto &owner = f.owners[handle]; assert(owner->size() == size);
        ''' + owner + '\n    };\n' + loop + '\n    result.copied = draw_upload_bytes; return result;\n}')
    fixture = fixture.replace('// VERTEX_LOOPS_HOOK', '\n'.join(loops))
    source = source.replace('// VERTEX_UPLOAD_HOOK', fixture)
    source_path = directory / 'buffer_lifetime.mm'
    source_path.write_text(source)
    command = ['xcrun', 'clang++', '-std=c++23', '-O1', '-g', '-fobjc-arc',
               '-fsanitize='+os.environ.get('VITA3K_BUFFER_SANITIZER', 'address,undefined'), '-fno-omit-frame-pointer',
               '-Wno-nullability-completeness', '-mmacosx-version-min=26.0']
    command += ['-I' + str(ROOT / 'vita3k' / module / 'include')
                for module in ('renderer', 'shader', 'gxm', 'util')]
    command += ['-I' + str(ROOT / 'external/xxHash')]
    command += [str(source_path),
                str(ROOT / 'vita3k/renderer/src/metal4/buffers.mm'),
                str(ROOT / 'vita3k/renderer/src/metal4/metal4_commands.mm'),
                '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'QuartzCore', '-o', str(binary)]
    subprocess.run(command, check=True)
    subprocess.run([str(binary)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def capture_pass_test(directory):
    path = 'vita3k/renderer/src/metal4/state.mm'
    production = (ROOT / path).read_text()
    source = Path(__file__).with_name('capture_pass.mm').read_text()
    helpers = '\n'.join(function(path, signature) for signature in (
        'static void end_pass(', 'static void expand_color_samples(',
        'static bool needs_raw_color_resolve(', 'static void resolve_raw_color_samples(',
        'static void prepare_pass_completion(', 'static bool prepare_face_replay('))
    source = source.replace('// PASS_HELPERS', helpers)
    def capture_loop(marker):
        start = production.index('for (uint32_t instance = 0;', production.index(marker))
        at = production.index('{', start) + 1
        depth = 1
        while depth:
            depth += (production[at] == '{') - (production[at] == '}')
            at += 1
        return production[start:at]
    source = source.replace('// ROUTED_CAPTURE_LOOP', capture_loop('if (routed_geometry) {'))
    source = source.replace('// SPLIT_CAPTURE_LOOP', capture_loop('} else if (capture_split_faces) {'))
    source = source.replace('// WATCH_COMPLETE', function(
        'vita3k/renderer/include/renderer/metal4/surface_write_watch.h', '    void complete_gpu_writes('))
    start = production.index('        const auto prepare_shader_guest_writes = [&] {')
    end = production.index('        // Capture finishes and resets', start)
    source = source.replace('// WRITE_PREPARE', production[start:end])
    start = production.index('            if (!prepare_face_replay(ctx, *impl->device, impl->caster)) {')
    end = production.index('            split_replay_indices =', start)
    source = source.replace('// FACE_TRANSITION', production[start:end])
    start = production.index('            // finish rearmed watched guest pages after capture.')
    end = production.index('            ctx.impl->commands =', start)
    source = source.replace('// CPU_REPLAY_PREPARE', production[start:end])
    file = directory / 'capture_pass.mm'
    file.write_text(source)
    binary = directory / 'capture_pass'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                    '-I' + str(ROOT / 'vita3k/shader/include'), str(file),
                    '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def vertex_snapshot_test(directory, implementation=None):
    source = Path(__file__).with_name('vertex_snapshots.cpp').read_text()
    methods = '\n'.join(function('vita3k/renderer/src/metal4/state.mm', signature, implementation)
        for signature in ('void MetalState::snapshot_vertex_streams(',
                          'std::shared_ptr<const std::vector<uint8_t>> MetalState::vertex_stream_snapshot(',
                          'void MetalState::disable_vertex_stream_snapshots(',
                          'void MetalState::reset_vertex_stream_snapshots('))
    run('vertex_snapshots', source.replace('// SNAPSHOT_METHODS_HOOK', methods), directory,
        ('-I' + str(ROOT / 'vita3k/renderer/include'),),
        sanitizer=os.environ.get('VITA3K_VERTEX_SANITIZER', 'address,undefined'))


def mip_source():
    return '#include "' + str(ROOT / 'vita3k/renderer/include/renderer/metal4/overlay_mips.h') + '"\n' + r'''
#include <cassert>
#include <iostream>
#include <numeric>
int main() {
    using renderer::metal::overlay_next_mip;
    assert((overlay_next_mip(std::vector<uint8_t>{0,0,255},3,1,1) == std::vector<uint8_t>{85}));
    assert((overlay_next_mip(std::vector<uint8_t>{0,0,255},1,3,1) == std::vector<uint8_t>{85}));
    std::vector<uint8_t> ramp(15); std::iota(ramp.begin(),ramp.end(),0);
    assert((overlay_next_mip(ramp,5,3,1) == std::vector<uint8_t>{6,8}));
    assert((overlay_next_mip(std::vector<uint8_t>{0,255,255,0},2,2,1) == std::vector<uint8_t>{128}));
    size_t levels = 0;
    for (uint32_t w : {1,2,3,63,256,257,512}) for (uint32_t h : {1,2,3,63,256,257})
    for (uint32_t channels : {1,4}) {
        std::vector<uint8_t> data(size_t(w)*h*channels);
        for (size_t i=0;i<data.size();++i) data[i] = uint8_t(31+41*(i%channels));
        auto width=w, height=h;
        while(width>1 || height>1) {
            data=overlay_next_mip(data,width,height,channels);
            width=std::max(1u,width/2); height=std::max(1u,height/2);
            assert(data.size()==size_t(width)*height*channels);
            for (size_t i=0;i<data.size();++i) assert(data[i]==31+41*(i%channels));
            ++levels;
        }
    }
    bool rejected=false;
    try { (void)overlay_next_mip(std::vector<uint8_t>{1},2,2,1); } catch(const std::invalid_argument &) { rejected=true; }
    assert(rejected);
    std::cout << "PASS overlay: " << levels << " rectangular/odd mip levels, channel preservation and weighted edge fixtures\n";
}
'''


def caster_test(directory, implementation=None):
    path = 'vita3k/renderer/src/metal4/textures.mm'
    text = implementation if implementation is not None else (ROOT / path).read_text()
    constructor = function(path, 'SurfaceCaster::SurfaceCaster(', text)
    getter = function(path, 'id<MTLComputePipelineState> SurfaceCaster::compute_pipeline(', text)
    factory = function(path, 'static id<MTLComputePipelineState> cached_compute(', text)
    library = function(path, 'id<MTLLibrary> SurfaceCaster::surface_library(', text) if 'SurfaceCaster::surface_library(' in text else (
        'id<MTLLibrary> SurfaceCaster::surface_library() { return multisample_library; }')
    source = r"""
#include <Metal/Metal.h>
#include <cassert>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <utility>
static size_t libraries, pipelines;
static bool reject_next;
static unsigned library_failure;
static id<MTLFunction> metal4_function(id<MTLLibrary> library, NSString *name) {
    assert(library); return (id<MTLFunction>)name;
}
struct Device {
    id<MTLLibrary> compile_library(NSString *source, id, NSError **error) {
        assert(source.length > 1000); ++libraries;
        switch (std::exchange(library_failure, 0)) {
        case 1: *error = [NSError errorWithDomain:@"test" code:1 userInfo:nil]; return nil;
        case 2: return nil;
        case 3: throw std::runtime_error("test library exception");
        }
        return (id<MTLLibrary>)[NSObject new];
    }
    id<MTLComputePipelineState> create_compute_pipeline(id<MTLFunction> function, std::string &error) {
        ++pipelines;
        if (reject_next) { reject_next = false; error = "test compiler rejection"; return nil; }
        return (id<MTLComputePipelineState>)[NSObject new];
    }
};
struct SurfaceCaster {
    Device &device;
    id<MTLLibrary> multisample_library;
    explicit SurfaceCaster(Device &);
    id<MTLLibrary> surface_library();
    id<MTLComputePipelineState> compute_pipeline(id<MTLComputePipelineState> __strong &, NSString *);
};
""" + factory + constructor + library + getter + r"""
int main() {
    @autoreleasepool {
        Device device;
        SurfaceCaster caster(device);
        assert(libraries == 0 && pipelines == 0);
        // Owners used for blits, deferred readback and rejected requests must
        // not depend on the shader compiler merely to exist or be destroyed.
        {
            std::vector<std::unique_ptr<SurfaceCaster>> owners;
            library_failure = 1;
            for (size_t i = 0; i < 32; ++i) owners.push_back(std::make_unique<SurfaceCaster>(device));
            assert(library_failure == 1 && libraries == 0);
            library_failure = 0;
        }
        std::array<id<MTLComputePipelineState>,26> slots{};
        for (auto &slot : slots) {
            auto first = caster.compute_pipeline(slot, @"helper");
            assert(first);
            for (size_t i = 0; i < 1000; ++i) assert(caster.compute_pipeline(slot, @"helper") == first);
        }
        assert(pipelines == 26 && libraries == 1);
        auto shared = caster.surface_library();
        for (size_t i = 0; i < 1000; ++i) assert(caster.surface_library() == shared);
        assert(libraries == 1);
        id<MTLComputePipelineState> retry = nil;
        reject_next = true;
        bool rejected = false;
        try { (void)caster.compute_pipeline(retry, @"retry"); }
        catch (const std::runtime_error &e) { rejected = std::string(e.what()) == "test compiler rejection"; }
        assert(rejected && !retry && pipelines == 27);
        assert(caster.compute_pipeline(retry, @"retry") && pipelines == 28);
        assert(caster.compute_pipeline(retry, @"retry") == retry && pipelines == 28);
        // Compiler failures leave the lazy library/PSO unpopulated; a later
        // request retries. A failed PSO never recompiles a successful library.
        for (unsigned mode : {1u, 2u, 3u}) {
            SurfaceCaster retry_caster(device);
            const auto before_libraries = libraries, before_pipelines = pipelines;
            library_failure = mode;
            id<MTLComputePipelineState> slot = nil;
            bool failed = false;
            try { (void)retry_caster.compute_pipeline(slot, @"retry-library"); }
            catch (const std::runtime_error &) { failed = true; }
            assert(failed && !slot && !retry_caster.multisample_library);
            assert(libraries == before_libraries + 1 && pipelines == before_pipelines);
            assert(retry_caster.compute_pipeline(slot, @"retry-library"));
            assert(libraries == before_libraries + 2 && pipelines == before_pipelines + 1);
            assert(retry_caster.compute_pipeline(slot, @"retry-library") == slot);
        }
    }
    std::cout << "PASS surface helpers: 33 actual constructors create zero libraries/PSOs; first helper builds one shared library, 26 lazy PSOs reuse 26000 calls, direct library reuse, three library error/retry modes and PSO retry; ASAN/UBSAN, no device/GPU/app\n";
}
"""
    file = directory / 'caster.mm'; file.write_text(source)
    binary = directory / 'caster'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', str(file), '-framework', 'Foundation',
                    '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def depth_bounds_source():
    # Compile the actual MSL seed_axis as scalar C++ and compare every sample
    # lane against the inverse used for the render scissor. No GPU is created.
    seed = function('vita3k/renderer/src/metal4/textures.mm', 'uint2 seed_axis(')
    return r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <array>
#include <renderer/metal4/depth_patch_bounds.h>
using uint = uint32_t;
using std::min;
using std::max;
struct uint2 { uint x, y; uint2(uint a, uint b) : x(a), y(b) {} };
''' + seed + r'''
uint64_t cases = 0;
void check(uint n, uint g, uint extent, uint origin, uint length) {
    const auto [first, last] = renderer::metal::depth_patch_bounds(n, g, extent, origin, length);
    assert(first <= last && last <= n);
    uint expected_first = n, expected_last = 0;
    const auto end = min(uint64_t(origin) + length, uint64_t(g));
    for (uint x = 0; x < n; ++x) {
        bool touched = false;
        for (uint sample = 0; sample < max(extent, 1u); ++sample) {
            const uint guest = extent ? seed_axis(x, n, g, g, extent, sample).y : x * g / n;
            touched |= guest >= origin && guest < end;
        }
        assert(touched == (x >= first && x < last));
        if (touched) { expected_first = min(expected_first, x); expected_last = x + 1; }
    }
    if (expected_first == n) assert(first == last);
    else assert(first == expected_first && last == expected_last);
    ++cases;
}
int main() {
    for (uint n = 1; n <= 32; ++n)
        for (uint g = 1; g <= 32; ++g)
            for (uint e = 0; e <= 2; ++e)
                for (uint origin = 0; origin <= g; ++origin)
                    for (uint length = 0; length <= g - origin + 1; ++length)
                        check(n, g, e, origin, length);
    for (uint n : {1u, 2u, 127u, 4096u, 16384u})
        for (uint g : {1u, 3u, 127u, 4096u, 16383u})
            for (uint e = 0; e <= 2; ++e)
                for (uint origin : {0u, g / 2, g - 1, g})
                    for (uint length : {0u, 1u, g, std::numeric_limits<uint>::max()})
                        check(n, g, e, origin, length);
    for (const auto bad : {std::array<uint,3>{0,1,0}, {1,0,0}, {1,1,3}}) {
        bool rejected = false;
        try { renderer::metal::depth_patch_bounds(bad[0],bad[1],bad[2],0,1); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    std::cout << "PASS depth patch bounds: " << cases
        << " tight inverse regions against actual shader seed_axis, single/2x/4x MSAA axes, odd/up/downscaled edges\n";
}
'''


def presentation_test(directory):
    binary = directory / 'presentation_scratch'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                    '-I' + str(ROOT / 'vita3k/renderer/include'),
                    str(Path(__file__).with_name('presentation_scratch.mm')),
                    '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def surface_cache_budget_test(directory):
    source = Path(__file__).with_name('surface_cache_budget.mm').read_text()
    source = source.replace('// SWAP_HOOK', function('vita3k/renderer/src/metal4/state.mm',
                                                   'void MetalState::swap_window('))
    path = directory / 'surface_cache_budget.mm'
    path.write_text(source)
    for release in (False, True):
        binary = directory / ('surface_cache_budget-release' if release else 'surface_cache_budget-debug')
        subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                        *(['-DNDEBUG'] if release else []),
                        '-I' + str(ROOT / 'vita3k/renderer/include'), str(path),
                        '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def fsr_presentation_test(directory):
    screen = 'vita3k/renderer/src/metal4/screen.mm'
    source = Path(__file__).with_name('fsr_presentation.mm').read_text()
    for hook, signature in (
            ('LOAD', 'std::vector<uint32_t> load_spirv('),
            ('INITIALIZE', 'void ScreenRenderer::initialize_fsr('),
            ('DISABLE', 'void ScreenRenderer::disable_fsr('),
            ('PREPARE', 'id<MTLTexture> ScreenRenderer::prepare('),
            ('RENDER', 'void ScreenRenderer::render(')):
        source = source.replace('// ' + hook + '_HOOK', function(screen, signature))
    path = directory / 'fsr_presentation.mm'
    path.write_text(source)
    assets = ROOT / 'vita3k'
    unknown = directory / 'fsr-unknown/shaders-builtin/vulkan'
    unknown.mkdir(parents=True)
    # A different generator word is still valid SPIR-V, but has not been
    # explicitly accepted by the optimization's complete-module fingerprint.
    words = bytearray((assets / 'shaders-builtin/vulkan/fsr_filter_rcas.comp.spv').read_bytes())
    words[8] ^= 1
    (unknown / 'fsr_filter_rcas.comp.spv').write_bytes(words)
    binary = directory / 'fsr_presentation'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                    '-I' + str(ROOT / 'vita3k/renderer/include'),
                    '-I' + str(ROOT / 'vita3k/shader/include'),
                    '-I' + str(ROOT / 'vita3k/util/include'), str(path),
                    '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(assets), str(unknown.parent.parent)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})

    # Translate the unchanged bundled shaders using the production converter.
    # Append the real wrapper without replacing any original helper/math code.
    converter = r'''
#include <spirv_msl.hpp>
#include <shader/metal4/msl_recompiler.h>
#include <renderer/metal4/fsr_presentation.h>
#include <fstream>
#include <stdexcept>
namespace shader::metal {
''' + function('vita3k/shader/src/metal4/msl_recompiler_metal.cpp',
               'Program convert_builtin_compute_spirv(') + r'''
}
int main(int argc, char **argv) {
    if (argc != 4) return 1;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) return 2;
    std::vector<uint32_t> words(file.tellg() / 4); file.seekg(0);
    file.read(reinterpret_cast<char *>(words.data()), words.size() * 4);
    const auto original = shader::metal::convert_builtin_compute_spirv(words);
    std::ofstream(argv[2]) << original.source;
    std::ofstream(argv[3]) << original.source << renderer::metal::fsr_rcas_fragment;
}
'''
    path = directory / 'fsr_convert.cpp'
    path.write_text(converter)
    build = Path(os.environ.get('VITA3K_BUILD', '/Users/jbw/Documents/Vita3K-Metal-Build/build'))
    binary = directory / 'fsr_convert'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-O1',
                    '-I' + str(ROOT / 'external/SPIRV-Cross'),
                    '-I' + str(ROOT / 'vita3k/shader/include'),
                    '-I' + str(ROOT / 'vita3k/renderer/include'), str(path),
                    *[str(build / ('external/SPIRV-Cross/Release/libspirv-cross-' + lib + '.a'))
                      for lib in ('msl', 'glsl', 'core')], '-o', str(binary)], check=True)
    original, fused = directory / 'fsr-original.metal', directory / 'fsr-fused.metal'
    subprocess.run([str(binary), str(assets / 'shaders-builtin/vulkan/fsr_filter_rcas.comp.spv'),
                    str(original), str(fused)], check=True)
    assert fused.read_text().startswith(original.read_text())
    # Preserve signatures referenced by the appended entry point. The offline
    # compiler independently checks the exact argument and resource ABI.
    assert 'void FsrRcasCon(' in original.read_text() and 'void FsrRcasH(' in original.read_text()
    for shader in (original, fused):
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-fno-fast-math',
                        '-c', str(shader), '-o', str(shader.with_suffix('.air'))], check=True)
    print('PASS FSR offline: original RCAS and fused fragment entry compiled to AIR with unchanged original MSL helpers; no GPU execution', flush=True)


def packed_surface_cast_test(directory):
    path = 'vita3k/renderer/src/metal4/textures.mm'
    source = Path(__file__).with_name('packed_surface_cast.mm').read_text()
    source = source.replace('// GXM_BASE_HOOK', function('vita3k/gxm/src/color.cpp', 'SceGxmColorBaseFormat get_base_format(') + '\n' + function('vita3k/gxm/src/textures.cpp', 'SceGxmTextureBaseFormat get_base_format('))
    source = source.replace('// X8_DECODE_HOOK', function('vita3k/renderer/src/metal4/texture_format_metal.cpp', 'void convert_x8s8s8u8_to_f16f16f16f16('))
    signatures = [
        'Map surface_memory_mapping(', 'SurfaceComponents surface_components(', 'SurfaceComponents texture_components(',
        'static bool raw_storage_matches(', 'static bool packed_color_cast_source(', 'static bool packed_texture_cast_target(',
        'bool surface_format_cast_supported(', 'bool surface_format_cast_enqueueable(', 'void copy_rgba8_row(',
        'size_t surface_memory_size(', 'static uint16_t swap_565_red_blue(', 'static uint16_t swap_5551_red_blue(',
        'static uint32_t swap_e5_red_blue(', 'static uint32_t swap_10_red_blue(', 'static uint32_t packed_10_guest_word(',
        'static uint16_t permute_4444(', 'static size_t packed_float_guest_pixel(', 'static uint16_t packed_float10_from_half(',
        'static uint32_t packed_float_word(', 'static void unpack_packed_float(', 'static uint32_t pack_rgb9e5(',
        'static void unpack_rgb9e5(', 'static std::pair<size_t,size_t> write_native_interval(',
        'static uint16_t pack_u8u3u3u2(', 'static void unpack_u8u3u3u2(', 'static bool full_surface_write(',
        'static SurfaceWritePatch surface_write_patch(',
        'static bool u8u3u3u2_surface_memory(',
    ]
    signatures += ['static int decode_signed5(', 'static uint16_t pack_s5s5u6(', 'static void unpack_s5s5u6(',
        'static bool s5s5u6_surface_memory(', 'static void unpack_u8s8s8u8(', 'static void pack_u8s8s8u8(',
        'static bool u8s8s8u8_surface_memory(', 'static uint32_t changed_word_mask(',
        'static bool packed_float_surface_memory(', 'static bool rgb9e5_expanded_surface_memory(', 'bool read_surface_memory(']
    source = source.replace('// CPU_REFERENCE_FUNCTIONS', function(path, 'struct SurfaceWritePatch') + ';\n' + function('vita3k/renderer/include/renderer/metal4/textures.h', 'inline uint32_t packed_10_native_word(') + '\n' + '\n'.join(function(path, x) for x in signatures))
    source = source.replace('// CONFIG_HOOK', function(path, 'static std::optional<std::array<uint32_t,17>> packed_surface_cast_config('))
    source = source.replace('// BUFFER_HOOK', function(path, 'static id<MTLBuffer> packed_cast_buffer('))
    source = source.replace('// CAST_HOOK', function(path, 'id<MTLTexture> SurfaceCaster::surface_format_cast('))
    state = 'vita3k/renderer/src/metal4/state.mm'
    source = source.replace('// REPACKED_COLOR_HOOK', function(state, 'static std::optional<SceGxmColorFormat> repacked_u2_color('))
    source = source.replace('// BIND_CAST_HOOK', function(state, '            if (inline_cast_images[index])'))
    source = source.replace('// BIND_GAMMA_HOOK', function(state, '            if (inline_gamma_images[index])'))
    msl = re.search(r'R"msl\((.*?)\)msl"', (ROOT / 'vita3k/renderer/include/renderer/metal4/packed_surface_cast.h').read_text(), re.S).group(1)
    core = msl.split('// Config:', 1)[1].split('// CORE_END', 1)[0]
    core = core[core.index('uint pc_load16'):]
    source = source.replace('// GPU_CORE_HOOK', core)
    source = source.replace('// TESTS_HOOK', Path(__file__).with_name('packed_surface_cast_cases.inl').read_text())
    output = directory / 'packed_surface_cast.mm'; output.write_text(source)
    binary = directory / 'packed_surface_cast'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g', '-Wno-nullability-completeness',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        *['-I'+str(ROOT/'vita3k'/m/'include') for m in ['renderer','gxm','util']],
        str(output), '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})
    shader = directory / 'packed_surface_cast.metal'; shader.write_text(msl)
    subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN','Metal'), '-sdk', 'macosx', 'metal',
        '-std=metal4.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-c', str(shader), '-o', str(shader.with_suffix('.air'))], check=True)
    print('PASS packed cast offline: actual integer-only packing kernel compiled as Metal 4 to AIR; no GPU execution', flush=True)


def queued_cast_sampling_test(directory):
    state = 'vita3k/renderer/src/metal4/state.mm'
    source = Path(__file__).with_name('queued_cast_sampling.mm').read_text()
    source = source.replace('// GXM_HOOK', function('vita3k/gxm/src/textures.cpp', 'SceGxmTextureBaseFormat get_base_format(') + '\n' + function('vita3k/gxm/src/textures.cpp', 'SceGxmTextureFormat get_format('))
    source = source.replace('// GAMMA_HOOK', function('vita3k/renderer/src/metal4/textures.mm', 'id<MTLTexture> SurfaceCaster::rgba8_surface_sampling('))
    source = source.replace('// REPACKED_HOOK', function(state, 'static std::optional<SceGxmColorFormat> repacked_u2_color('))
    source = source.replace('// QUEUED_HOOK', function(state, 'static std::pair<id<MTLTexture>, id<MTLTexture>> enqueue_surface_cast_sampling('))
    for hook, signature in [('// INACTIVE_CALL_HOOK','            if (!cropped)'),
        ('// CROP_CALL_HOOK','            if (inline_crops[index] && gpu_cast && !raw_cast)'),
        ('// FEEDBACK_CALL_HOOK','                    if (address==ctx.impl->guest_color.data.address() && !same_format')]:
        source = source.replace(hook, function(state, signature))
    state_source = (ROOT/state).read_text()
    source = source.replace('// GENERAL_POLICY_HOOK', re.search(r'const bool gpu_cast=!same_format[^;]+;', state_source).group())
    source = source.replace('// FEEDBACK_POLICY_HOOK', re.search(r'const bool gpu_cast=!word_alias[^;]+;', state_source).group())
    output = directory / 'queued_cast_sampling.mm'; output.write_text(source)
    binary = directory / 'queued_cast_sampling'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        '-I'+str(ROOT/'vita3k/gxm/include'), '-I'+str(ROOT/'vita3k/util/include'), str(output), '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})


def cube_fallback_submission_test(directory, implementation=None, uploaded_only=False):
    state = 'vita3k/renderer/src/metal4/state.mm'
    text = implementation if implementation is not None else (ROOT / state).read_text()
    begin = text.index('                const uintptr_t key=reinterpret_cast<uintptr_t>((__bridge void *)source);')
    end = text.index('                continue;', begin)
    source = Path(__file__).with_name('cube_fallback_submission.mm').read_text()
    textures = 'vita3k/renderer/src/metal4/textures.mm'
    source = source.replace('// RAW_SUPPORT_HOOK', function(textures, 'bool raw_texture_snapshot_supported('))
    source = source.replace('// RAW_VIEW_HOOK', function(textures, 'id<MTLTexture> raw_texture_snapshot_view('))
    source = source.replace('// FALLBACK_HOOK', text[begin:end])
    source = source.replace('// UPLOADED_HOOK', function(state, '            if (rendered.empty())', text))
    output = directory / 'cube_fallback_submission.mm'; output.write_text(source)
    binary = directory / 'cube_fallback_submission'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(output),
        '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    environment = {**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'}
    if uploaded_only: environment['VITA3K_UPLOADED_RAW_ONLY'] = '1'
    subprocess.run([str(binary)], check=True, env=environment)


def multisample_snapshot_test(directory, implementation=None):
    textures = 'vita3k/renderer/src/metal4/textures.mm'
    text = implementation if implementation is not None else (ROOT / textures).read_text()
    source = Path(__file__).with_name('multisample_snapshot.mm').read_text()
    expand = function(textures, 'void SurfaceCaster::expand_multisample(', text)
    source = source.replace('// WAIT_PARAMETER', ',bool = true' if 'bool wait_for_completion)' in expand else '')
    source = source.replace('// EXPAND_HOOK', expand)
    source = source.replace('// SNAPSHOT_HOOK', function(textures, 'id<MTLTexture> SurfaceCaster::depth_snapshot(', text))
    source = source.replace('// BITS_HOOK', function(textures, 'static MTLPixelFormat sample_bits_format(', text))
    output = directory / 'multisample_snapshot.mm'; output.write_text(source)
    binary = directory / 'multisample_snapshot'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(output),
        '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})


def pipeline_warmup_test(directory, implementation=None):
    state = 'vita3k/renderer/src/metal4/state.mm'
    source = Path(__file__).with_name('pipeline_warmup.mm').read_text()
    source = source.replace('// WARMUP_HOOK', function(state,
        'void MetalState::precompile_shader(', implementation)
        .replace('MTLRenderPipelineDescriptor', 'WarmupDescriptor'))
    output = directory / 'pipeline_warmup.mm'; output.write_text(source)
    binary = directory / 'pipeline_warmup'
    sanitizer = os.environ.get('VITA3K_WARMUP_SANITIZER', 'address,undefined')
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize='+sanitizer, '-fno-omit-frame-pointer',
        '-I'+str(ROOT/'vita3k/renderer/include'), str(output),
        '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1', 'TSAN_OPTIONS':'halt_on_error=1'})


def msaa_seed_submission_test(directory, implementation=None):
    state = 'vita3k/renderer/src/metal4/state.mm'
    text = implementation if implementation is not None else (ROOT / state).read_text()
    source = Path(__file__).with_name('msaa_seed_submission.mm').read_text()
    textures = 'vita3k/renderer/src/metal4/textures.mm'
    source = source.replace('// BITS_HOOK', function(textures, 'static MTLPixelFormat sample_bits_format('))
    source = source.replace('// SEED_HOOK', function(textures, 'void SurfaceCaster::seed_multisample(')
                            .replace('MTLRenderPipelineDescriptor', 'SeedPipelineDescriptor'))
    source = source.replace('// DRAW_HOOK', function(state,
        '        if (!ctx.impl->encoder && ctx.impl->samples>1 && ctx.impl->guest_color.data)', text))
    # Old code has no delayed bind block; its actual draw block still exposes
    # the private submit and fails the same ordering regression below.
    bind = function(state, '        if (seed_color_samples)', text) if 'if (seed_color_samples)' in text else ''
    source = source.replace('// BIND_HOOK', bind)
    output = directory / 'msaa_seed_submission.mm'; output.write_text(source)
    binary = directory / 'msaa_seed_submission'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(output),
        '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})


def finish_submission_test(directory):
    state = 'vita3k/renderer/src/metal4/state.mm'
    source = Path(__file__).with_name('finish_submission.mm').read_text()
    source = source.replace('// EXPAND_HOOK', function(state, 'static void expand_color_samples('))
    finish = function(state, 'bool MetalState::finish(')
    prefix = finish[:finish.index('        // Face routing can switch')]
    begin = finish.index('    if (publish_color || wait_without_publication)')
    end = finish.index('    // All guest-buffer GPU users', begin)
    # Only submission/retirement is isolated; publication and guest-memory
    # semantics are not modeled as a full emulator integration test.
    source = source.replace('// FINISH_SUBMISSION_HOOK', prefix+'    }\n'+finish[begin:end]+'    return published_color;\n}\n')
    scene = function(state, 'bool MetalState::end_scene(')
    begin = scene.index('    const bool defer_scene =')
    end = scene.index('    if (impl->trace_finish_timing', begin)
    source = source.replace('// SCENE_POLICY_HOOK', scene[begin:end])
    output = directory / 'finish_submission.mm'; output.write_text(source)
    binary = directory / 'finish_submission'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(output),
        '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})


def linear_surface_patch_test(directory):
    source = Path(__file__).with_name('linear_surface_patch.mm').read_text()
    source = source.replace('// FULL_WRITE_HOOK', function(
        'vita3k/renderer/src/metal4/textures.mm', 'static bool full_surface_write('))
    source = source.replace('// LINEAR_WRITER_HOOK', function(
        'vita3k/renderer/src/metal4/textures.mm', '    if (direct_linear)'))
    path = directory / 'linear_surface_patch.mm'
    path.write_text(source)
    binary = directory / 'linear_surface_patch'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                    '-I' + str(ROOT / 'vita3k/renderer/include'), str(path),
                    '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def surface_write_patch_test(directory):
    path = 'vita3k/renderer/src/metal4/textures.mm'
    baseline = subprocess.check_output(['git', 'show', '6955426851d02f8304f98e4f6f01cc514a9e6b93:' + path], cwd=ROOT, text=True)
    source = Path(__file__).with_name('surface_write_patch.mm').read_text()
    source = source.replace('// GXM_BASE_HOOK', function('vita3k/gxm/src/color.cpp', 'SceGxmColorBaseFormat get_base_format('))
    source = source.replace('// MORTON_HOOK', '\n'.join(function('vita3k/renderer/src/texture/format.cpp', x)
        for x in ('static uint32_t Part1By1(', 'uint32_t encode_morton(')))
    common = ['Map surface_memory_mapping(', 'SurfaceComponents surface_components(', 'static bool raw_storage_matches(',
        'size_t surface_memory_size(', 'static uint16_t swap_565_red_blue(', 'static uint16_t swap_5551_red_blue(',
        'static uint32_t swap_e5_red_blue(', 'static uint32_t swap_10_red_blue(', 'static uint32_t packed_10_guest_word(',
        'static uint16_t permute_4444(', 'static Map inverse_mapping(', 'static size_t packed_float_guest_pixel(',
        'static uint16_t packed_float10_from_half(', 'static uint32_t packed_float_word(', 'static void unpack_packed_float(',
        'static uint32_t pack_rgb9e5(', 'static void unpack_rgb9e5(', 'static std::pair<size_t,size_t> write_native_interval(',
        'static uint16_t pack_u8u3u3u2(', 'static void unpack_u8u3u3u2(', 'static bool full_surface_write(',
        'static int decode_signed5(', 'static uint16_t pack_s5s5u6(', 'static void unpack_s5s5u6(',
        'static void unpack_u8s8s8u8(', 'static void pack_u8s8s8u8(', 'static uint32_t changed_word_mask(']
    # The layout/packer primitives shared by both versions must be byte-equal
    # to the pinned baseline. Only complete writer bodies differ.
    for signature in common:
        assert function(path, signature) == function(path, signature, baseline), signature
    code = function('vita3k/renderer/include/renderer/metal4/textures.h', 'inline uint32_t packed_10_native_word(')
    code += '\n' + '\n'.join(function(path, x) for x in common)
    code += '\n' + function(path, 'struct SurfaceWritePatch') + ';\n' + function(path, 'static SurfaceWritePatch surface_write_patch(')
    source = source.replace('// COMMON_HOOK', code)
    writers = ['static bool u8u3u3u2_surface_memory(', 'static bool s5s5u6_surface_memory(',
        'static bool u8s8s8u8_surface_memory(', 'static bool packed_float_surface_memory(',
        'static bool rgb9e5_expanded_surface_memory(', 'static bool write_surface_memory_mapped(']
    source = source.replace('// CURRENT_HOOK', '\n'.join(function(path, x) for x in writers))
    source = source.replace('// ORIGINAL_HOOK', '\n'.join(function(path, x, baseline) for x in writers))
    generated = directory / 'surface_write_patch.mm'; generated.write_text(source)
    binary = directory / 'surface_write_patch'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        *['-I' + str(ROOT / 'vita3k' / m / 'include') for m in ('renderer','gxm','util')],
        str(generated), '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1'})


def snapshot_channels_test(directory):
    implementation = function('vita3k/renderer/src/metal4/textures.mm',
                              'id<MTLTexture> SurfaceCaster::texture_snapshot(')
    source = Path(__file__).with_name('snapshot_channels.mm').read_text().replace('// SNAPSHOT_HOOK', implementation)
    source = source.replace('// CAPTURE_HOOK', function('vita3k/renderer/src/metal4/textures.mm',
                                                      'id<MTLTexture> SurfaceCaster::sampling_snapshot('))
    path = directory / 'snapshot_channels.mm'
    path.write_text(source)
    binary = directory / 'snapshot_channels'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                    '-I' + str(ROOT / 'vita3k/renderer/include'), str(path),
                    '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})
    shader = re.search(r'@R"\((.*?)\)", nil, &error\);', implementation, re.S).group(1)
    path = directory / 'snapshot_channels.metal'
    path.write_text(shader)
    subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                    '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-c', str(path),
                    '-o', str(path.with_suffix('.air'))], check=True)
    print('PASS snapshot offline: original and packed FP32 copy kernels compiled to AIR; no GPU execution', flush=True)


def vertex_layout_test(directory, reference=None):
    source = Path(__file__).with_name('native_vertex_layout.mm').read_text()
    if reference:
        source = source.replace('// REFERENCE_HOOK', reference)
    source += '\n#undef LOG_ERROR\n#define LOG_ERROR(...)\nnamespace gxm {\n'
    source += function('vita3k/gxm/src/attributes.cpp', 'uint32_t attribute_format_size(') + '\n'
    source += function('vita3k/gxm/src/stream.cpp', 'bool is_stream_instancing(') + '\n}\n'
    path = directory / 'native_vertex_layout.mm'
    path.write_text(source)
    binary = directory / 'native_vertex_layout'
    flags = ['-DSPDLOG_FMT_EXTERNAL'] + (['-DREFERENCE_LAYOUT'] if reference else [])
    includes = ['-I' + str(ROOT / 'vita3k' / module / 'include') for module in ('renderer', 'shader', 'gxm', 'util')]
    includes += ['-I' + str(ROOT / 'external' / path) for path in ('boost', 'fmt/include', 'spdlog/include')]
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer', *includes, *flags,
                    str(path), '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def variant_keys_test(directory, reference=None):
    source = Path(__file__).with_name('variant_keys.cpp').read_text()
    hints = function('vita3k/shader/include/shader/metal4/spirv_recompiler.h', 'struct Hints {')
    source = source.replace('// HINTS_HOOK', 'namespace shader::metal3 {\n' + hints + ';\n}')
    if reference:
        source = source.replace('// REFERENCE_HOOK', reference)
    flags = ('-DREFERENCE_KEYS',) if reference else ()
    run('variant_keys', source, directory,
        ('-I' + str(ROOT / 'vita3k/renderer/include'), *flags))


def texture_lookup_test(directory, reference=None):
    source = Path(__file__).with_name('texture_lookup.cpp').read_text()
    color_bits = function('vita3k/gxm/src/color.cpp', 'size_t bits_per_pixel(')
    texture_bits = function('vita3k/gxm/src/textures.cpp', 'uint32_t bits_per_pixel(')
    # Guard the spatial bound against future upstream formats exceeding 64 bits.
    format_contract = 'namespace format_contract {\n' + color_bits + '\n' + texture_bits + '\n}\nvoid format_bounds() {\n'
    for body in (color_bits, texture_bits):
        formats = re.findall(r'case (SCE_GXM_\w+):', body)
        assert formats
        format_contract += 'for (auto format : {' + ','.join(formats) + '}) assert(format_contract::bits_per_pixel(format) <= 64);\n'
    source = source.replace('// FORMAT_BOUNDS_HOOK', format_contract + '}\n')
    source = source.replace('// COLOR_LOOKUP_HOOK', function(
        'vita3k/renderer/include/renderer/metal4/textures.h',
        'template <typename SurfaceMap, typename Eligible>\nauto find_color_subrectangle('))
    state = (ROOT / 'vita3k/renderer/src/metal4/state.mm').read_text()
    start = state.index('        const auto find_texture_surface =')
    end = state.index('        const auto rg32_linear_alias =', start)
    source = source.replace('// FINDERS_HOOK', state[start:end])
    if reference:
        first = source.index('    const auto reference =')
        last = source.index('    for (unsigned pass', first)
        source = source[:first] + reference + source[last:]
    run('texture_lookup', source, directory,
        ('-I' + str(ROOT / 'vita3k/renderer/include'),))


def uniform_bounds_test(directory):
    source = Path(__file__).with_name('uniform_bounds.cpp').read_text()
    source = source.replace('// HELPER_HOOK', function(
        'vita3k/modules/SceGxm/metal4/SceGxm_metal.inl',
        'static uint32_t metal_mapped_uniform_size_metal3('))
    source = source.replace('// REFERENCE_HOOK', r'''
static uint32_t old_mapped_uniform_size(const GxmState &gxm, Address address, uint32_t requested_size) {
    const uint64_t requested_end = uint64_t(address) + requested_size;
    const auto next = gxm.memory_mapped_regions.upper_bound(address);
    uint64_t mapped_end = address;

    // A later overlapping mapping may end before the binding while an older
    // mapping still contains it. Find the full mapped prefix from this address.
    for (auto it = next; it != gxm.memory_mapped_regions.begin();) {
        --it;
        mapped_end = std::max(mapped_end, uint64_t(it->first) + it->second.size);
        if (mapped_end >= requested_end)
            return requested_size;
    }
    if (mapped_end == address)
        return requested_size; // No GXM mapping at the binding address.

    // Adjacent or overlapping mappings extend the readable prefix.
    for (auto it = next; it != gxm.memory_mapped_regions.end() && it->first <= mapped_end; ++it) {
        mapped_end = std::max(mapped_end, uint64_t(it->first) + it->second.size);
        if (mapped_end >= requested_end)
            return requested_size;
    }
    return static_cast<uint32_t>(mapped_end - address);
}
''')
    source = source.replace('// EXPORTS_HOOK', '\n'.join(
        function('vita3k/modules/SceGxm/metal4/SceGxm_metal.inl', signature)
        for signature in ('int export_sceGxmMapMemory_metal3(', 'int export_sceGxmUnmapMemory_metal3(')))
    run('uniform_bounds', source, directory)


def link_shader_test(source, binary, extra=()):
    # Reuse the stable Xcode build's private shader library. This integration
    # check is not sanitizer coverage of its prebuilt production object files.
    build = Path(os.environ.get('VITA3K_BUILD', '/Users/jbw/Documents/Vita3K-Metal-Build/build'))
    response = next((build / 'build/shader-metal4.build/Release/Objects-normal/arm64').glob('*common-args.resp'))
    flags = [arg for arg in shlex.split(response.read_text())
             if arg not in ('-O3', '-DNDEBUG', '-flto=thin', '-fpascal-strings')]
    libs = ['vita3k/shader/Release/libshader-metal4.a', 'external/glslang/SPIRV/Release/libSPIRV.a',
            'external/glslang/glslang/Release/libglslang.a', 'external/SPIRV-Cross/Release/libspirv-cross-msl.a',
            'external/SPIRV-Cross/Release/libspirv-cross-glsl.a', 'external/SPIRV-Cross/Release/libspirv-cross-core.a',
            'vita3k/gxm/Release/libgxm.a', 'vita3k/util/Release/libutil.a',
            'external/fmt/Release/libfmt.a', 'external/spdlog/Release/libspdlog.a']
    subprocess.run(['xcrun', 'clang++', *flags, '-O1', '-g', '-Wl,-dead_strip', *extra,
                    str(source),
                    *[str(build / lib) for lib in libs], '-L/opt/homebrew/lib', '-lboost_filesystem',
                    '-framework', 'Foundation', '-framework', 'Security', '-o', str(binary)], check=True)


def texture_helper_test(directory):
    binary = directory / 'texture_helpers'
    source = (ROOT / 'vita3k/shader/src/metal4/tests/texture_helpers.cpp').read_text()
    source = source.replace('// COEFFICIENT_HOOK', function(
        'vita3k/shader/src/metal4/translator/texture_metal.cpp', 'static spv::Id get_uv_coeffs('))
    path = directory / 'texture_helpers.cpp'
    path.write_text(source)
    link_shader_test(path, binary)
    outputs = directory / 'texture-helpers'
    subprocess.run([str(binary), str(outputs)], check=True)
    pairs = sorted(outputs.glob('*-inline.spv'))
    assert len(pairs) == 40

    def instructions(path):
        data = path.read_bytes()
        words = struct.unpack('<' + 'I' * (len(data) // 4), data)
        declarations, at = [], 5
        while at < len(words) and words[at] & 65535 != 54:  # OpFunction
            count = words[at] >> 16
            assert count > 0
            declarations.append(words[at:at + count])
            at += count
        # Function return types may be declared earlier when shared. Validation
        # already checks legal declaration order; retain every instruction and
        # compare all function/control-flow instructions in their exact order.
        return words[:5], sorted(declarations), words[at:]

    def validate_pair(original):
        shared = original.with_name(original.name.replace('-inline.', '-shared.'))
        for path in (original, shared):
            subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', '--scalar-block-layout', str(path)], check=True)
            subprocess.run(['spirv-opt', '--scalar-block-layout', '-O', '--strip-debug', '--canonicalize-ids', str(path),
                            '-o', str(path.with_suffix('.normalized.spv'))], check=True)
            subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                            '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-fno-fast-math',
                            '-c', str(path.with_suffix('.metal')), '-o', str(path.with_suffix('.air'))], check=True)
        assert instructions(original.with_suffix('.normalized.spv')) == instructions(shared.with_suffix('.normalized.spv')), original.name

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate_pair, pairs))
    print('PASS texture helper equivalence: 40 inline/shared SPIR-V pairs validated; identical normalized instructions apart from global declaration order; 80 MSL shaders compiled offline; no GPU execution', flush=True)


def clip_geometry_test(directory):
    # Original routing comes from the unchanged goal baseline commit. Only
    # namespaces/layout names differ so both versions link in one CPU test.
    baseline = 'namespace legacy_shader { constexpr unsigned CAPTURE_CLIP_SLOT=15, CAPTURE_CLIP_COUNT=8, CAPTURE_OUTPUT_SLOT_COUNT=23; }\n'
    for path in ['vita3k/renderer/include/renderer/metal4/polygon_clip.h',
                 'vita3k/renderer/src/metal4/polygon_clip.cpp']:
        original = subprocess.check_output(['git', 'show', '6955426851d02f8304f98e4f6f01cc514a9e6b93:' + path], cwd=ROOT, text=True)
        original = re.sub(r'^#(?:include|pragma)[^\n]*\n', '', original, flags=re.M)
        baseline += original.replace('renderer::metal', 'baseline').replace('shader::metal::', 'legacy_shader::') + '\n'
    (directory / 'baseline_polygon.inl').write_text(baseline)
    source = (ROOT / 'vita3k/renderer/src/metal4/tests/clip_planes.cpp').read_text()
    run('clip_geometry', source, directory, extra=(
        '-I' + str(ROOT / 'vita3k/renderer/include'), '-I' + str(ROOT / 'vita3k/shader/include'),
        str(ROOT / 'vita3k/renderer/src/metal4/polygon_clip.cpp')))


def clip_replay_test(directory):
    source = Path(__file__).with_name('clip_replay.mm').read_text()
    source = source.replace('// REPLAY_HOOK', function('vita3k/renderer/src/metal4/state.mm',
        'for (const auto &primitive : routed_clip_primitives)'))
    path = directory / 'clip_replay.mm'
    path.write_text(source)
    binary = directory / 'clip_replay'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                    '-I' + str(ROOT / 'vita3k/renderer/include'), '-I' + str(ROOT / 'vita3k/shader/include'),
                    str(path), '-framework', 'Foundation', '-framework', 'Metal', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def vertex_color_clamp_test(directory):
    binary = directory / 'vertex_color_clamp'
    link_shader_test(ROOT / 'vita3k/shader/src/metal4/tests/vertex_color_clamp.cpp', binary)
    outputs = directory / 'vertex-colors'
    outputs.mkdir()
    subprocess.run([str(binary), str(outputs)], check=True)
    spirv = sorted(outputs.glob('*.spv'))
    sources = sorted(outputs.glob('*.metal'))
    assert len(spirv) == len(sources) == 24
    for path in spirv:
        subprocess.run(['spirv-val', '--target-env', 'spv1.0', '--scalar-block-layout', str(path)], check=True)
    for path in sources:
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0',
                        '-fmodules-cache-path='+str(directory/'metal-modules'), '-fno-fast-math',
                        '-c', str(path), '-o', str(path.with_suffix('.air'))], check=True)
    print('PASS vertex color offline: 24 SPIR-V modules validated and 24 MSL outputs compiled to AIR; no GPU execution', flush=True)


def clip_shader_test(directory):
    binary = directory / 'clip_shaders'
    link_shader_test(ROOT / 'vita3k/shader/src/metal4/tests/clip_shaders.cpp', binary)
    outputs = directory / 'clip-shaders'
    outputs.mkdir()
    subprocess.run([str(binary), str(outputs)], check=True)
    spirv = sorted(outputs.glob('*.spv'))
    sources = sorted(outputs.glob('*.metal'))
    assert len(spirv) == 1536 and len(sources) == 55
    def validate_spirv(path):
        subprocess.run(['spirv-val', '--target-env', 'spv1.0', '--scalar-block-layout', str(path)], check=True)
    def compile_msl(path):
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-fno-fast-math',
                        '-c', str(path), '-o', str(path.with_suffix('.air'))], check=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate_spirv, spirv))
        list(executor.map(compile_msl, sources))
    print('PASS clip offline: 1536 SPIR-V modules validated; 54 guest/capture MSL variants plus replay compiled to AIR; no GPU execution', flush=True)


def fp_constants_test(directory, reproduce=False):
    source = (ROOT/'vita3k/shader/src/metal4/tests/fp_constants.cpp').read_text()
    path = 'vita3k/shader/src/metal4/usse_utilities_metal.cpp'
    source = source.replace('// MODIFIERS_HOOK', function(path, 'static spv::Id apply_modifiers('))
    source = source.replace('// CONSTANT_HOOK', function(path, '    if (op.bank == RegisterBank::FPCONSTANT)'))
    output = directory/'fp_constants.cpp'; output.write_text(source)
    binary = directory/'fp_constants'
    link_shader_test(output,binary,('-fsanitize=address,undefined,float-cast-overflow','-fno-omit-frame-pointer'))
    if reproduce:
        result = subprocess.run([str(binary),'repro'],text=True,capture_output=True,
            env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1'})
        assert result.returncode and 'outside the range of representable values' in result.stderr,result.stderr
        print(result.stderr.splitlines()[0],flush=True)
        return
    module=directory/'fp_constants.spv'
    subprocess.run([str(binary),str(module)],check=True,env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1'})
    subprocess.run(['spirv-val','--target-env','vulkan1.2',str(module)],check=True)
    print('PASS FP constant SPIR-V: generated scalar/vector and modifier types validated; production branch sanitizer-instrumented, no GPU',flush=True)



def rotate_left_test(directory):
    source = (ROOT/'vita3k/shader/src/metal4/tests/rotate_left.cpp').read_text()
    source = source.replace('// ROTATE_HOOK', function(
        'vita3k/shader/src/metal4/translator/ialu_metal.cpp', 'static spv::Id rotate_left32('))
    path = directory/'rotate_left.cpp'; path.write_text(source)
    binary = directory/'rotate_left'
    link_shader_test(path, binary, ('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))
    outputs = directory/'rotations'; outputs.mkdir()
    subprocess.run([str(binary), str(outputs)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})
    modules = sorted(outputs.glob('*.spv')); shaders = sorted(outputs.glob('*.metal'))
    assert len(modules) == 513 and len(shaders) == 128
    def validate(path):
        subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', '--scalar-block-layout', str(path)], check=True)
    def compile_shader(path):
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-c', str(path),
                        '-o', str(path.with_suffix('.air'))], check=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate, modules))
        list(executor.map(compile_shader, shaders))
    print('PASS ROL offline: 513 SPIR-V modules validated, 128 MSL shaders compiled to AIR; no device/GPU/app', flush=True)



def i8mad_test(directory):
    source = (ROOT/'vita3k/shader/src/metal4/tests/i8mad.cpp').read_text()
    source = source.replace('// I8MAD_HOOK', function(
        'vita3k/shader/src/metal4/translator/ialu_metal.cpp', 'static spv::Id i8mad_result('))
    path = directory/'i8mad.cpp'; path.write_text(source)
    binary = directory/'i8mad'
    link_shader_test(path, binary, ('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))
    outputs = directory/'i8mad-shaders'; outputs.mkdir()
    subprocess.run([str(binary), str(outputs)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})
    modules = sorted(outputs.glob('*.spv')); shaders = sorted(outputs.glob('*.metal'))
    assert len(modules) == 256 and len(shaders) == 128
    def validate(path):
        subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', '--scalar-block-layout', str(path)], check=True)
    def compile_shader(path):
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-c', str(path),
                        '-o', str(path.with_suffix('.air'))], check=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate, modules))
        list(executor.map(compile_shader, shaders))
    print('PASS I8MAD offline: 256 helper/public SPIR-V modules validated, 128 MSL shaders compiled to AIR; no device/GPU/app', flush=True)



def i32mad_formats_test(directory):
    binary = directory/'i32mad_formats'
    link_shader_test(ROOT/'vita3k/shader/src/metal4/tests/i32mad_formats.cpp', binary,
                     ('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))
    outputs = directory/'i32mad-formats'; outputs.mkdir()
    subprocess.run([str(binary), str(outputs)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})
    modules = sorted(outputs.glob('*.spv')); shaders = sorted(outputs.glob('*.metal'))
    assert len(modules) == 768 and len(shaders) == 192
    def validate(path):
        subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', '--scalar-block-layout', str(path)], check=True)
    def compile_shader(path):
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-c', str(path),
                        '-o', str(path.with_suffix('.air'))], check=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate, modules))
        list(executor.map(compile_shader, shaders))
    print('PASS integer MAD offline: 768 IMAE/IMA16 SPIR-V modules validated, 192 MSL shaders compiled to AIR; no device/GPU/app', flush=True)


def vtst_dot_masks_test(directory):
    binary = directory/'vtst_dot_masks'
    source = (ROOT/'vita3k/shader/src/metal4/tests/vtst_dot_masks.cpp').read_text()
    source = source.replace('// TEST_PREDICATE_HOOK', function(
        'vita3k/shader/src/metal4/translator/branch_cond_metal.cpp', 'static spv::Id test_predicate('))
    source = source.replace('// TEST_INVALIDATION_HOOK', function(
        'vita3k/shader/src/metal4/usse_program_analyzer_metal.cpp', 'static bool invalidates_native_test_predicate('))
    generated = directory/'vtst_dot_masks.cpp'; generated.write_text(source)
    link_shader_test(generated, binary,
                     ('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))
    outputs = directory/'vtst-dot-masks'; outputs.mkdir()
    subprocess.run([str(binary), str(outputs)], check=True,
                   env={**os.environ, 'UBSAN_OPTIONS':'halt_on_error=1'})
    modules = sorted(outputs.glob('*.spv')); shaders = sorted(outputs.glob('*.metal'))
    assert len(modules) == 2942 and len(shaders) == 899
    pairs = sorted(outputs.glob('wordmask-*-repeated.spv'))
    assert len(pairs) == 72
    for repeated in pairs:
        reference = repeated.with_name(repeated.name.replace('-repeated.spv', '-unrolled.spv'))
        canonical = []
        for path in (repeated, reference):
            target = path.with_suffix('.canonical')
            subprocess.run(['spirv-opt', '--strip-debug', '--compact-ids', str(path), '-o', str(target)], check=True)
            canonical.append(target.read_bytes())
        assert canonical[0] == canonical[1], ('32-bit TESTMASK repeat mismatch', repeated.name)
    print('PASS VTST word-mask repeats: 72 complete repeated/unrolled public SPIR-V pairs instruction-identical after debug stripping and ID compaction; dependent overlap and scalar predicates included, no GPU', flush=True)
    def validate(path):
        subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', '--scalar-block-layout', str(path)], check=True)
    def compile_shader(path):
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'), '-c', str(path),
                        '-o', str(path.with_suffix('.air'))], check=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate, modules))
        list(executor.map(compile_shader, shaders))
    print('PASS VTST offline: 2942 SPIR-V modules validated, 899 MSL shaders compiled to AIR; no device/GPU/app', flush=True)


def shader_test(directory):
    binary = directory / 'shader_test'
    link_shader_test(ROOT / 'vita3k/shader/src/metal4/tests/shader_regressions.cpp', binary)
    outputs = directory / 'centroid-shaders'
    outputs.mkdir()
    subprocess.run([str(binary), str(outputs)], check=True)
    shaders = sorted(outputs.glob('*.metal'))
    assert len(shaders) == 88

    def validate_shader(source):
        # Validation and offline AIR compilation only: no MTLDevice, pipeline
        # creation, app launch or GPU dispatch is involved.
        subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', '--scalar-block-layout',
                        str(source.with_suffix('.spv'))], check=True)
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0', '-fmodules-cache-path='+str(directory/'metal-modules'),
                        '-c', str(source), '-o', str(source.with_suffix('.air'))], check=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate_shader, shaders))
    print('PASS centroid offline: 88 SPIR-V modules with/without native sampling metadata validated and 88 generated MSL shaders compiled to AIR; no GPU execution', flush=True)


def surface_allocation_test(directory):
    scene = function('vita3k/renderer/src/metal4/state.mm', 'void MetalState::set_context(')
    condition = scene[scene.index('const bool new_color ='):]
    condition = condition[:condition.index(';')+1]
    body = function('', 'if (new_color) {', source=scene)
    allocation = body[body.index('const auto usage ='):body.index('entry.cpu_snapshot.clear();')]
    actual = condition + '\nif (new_color) {\n' + allocation + '\n}'
    source = Path(__file__).with_name('surface_allocation.cpp').read_text().replace('// ALLOCATION_HOOK', actual)
    run('surface_allocation', source, directory)


def presentation_lifetime_test(directory):
    source = Path(__file__).with_name('presentation_lifetime.mm').read_text()
    retirement = function('vita3k/renderer/include/renderer/metal4/presentation_lifetime.h',
                          'inline void retire_presentation_layer(')
    for native, double in [('CAMetalLayer', 'FakeLayer'), ('NSView', 'FakeView'),
                           ('NSWindow', 'FakeWindow'), ('CATransaction', 'FakeTransaction')]:
        retirement = retirement.replace(native, double)
    source = source.replace('// RETIRE_HOOK', retirement)
    cleanup = function('vita3k/renderer/src/metal4/state.mm', 'void MetalState::cleanup(')
    assert cleanup.index('retire_presentation_layer') > cleanup.index('retire_commands_for_shutdown')
    assert cleanup.index('retire_presentation_layer') > cleanup.index('finish_for_shutdown')
    path = directory / 'presentation_lifetime.mm'; path.write_text(source)
    binary = directory / 'presentation_lifetime'
    subprocess.run(['xcrun', 'clang++', '-std=c++23', '-fobjc-arc', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(path),
                    '-framework', 'Foundation', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=20,
                   env={**os.environ, 'UBSAN_OPTIONS': 'halt_on_error=1'})


def library_reuse_test(directory):
    source = (ROOT / 'vita3k/renderer/src/metal4/tests/library_reuse.cpp').read_text()
    device = (ROOT / 'vita3k/renderer/src/metal4/device.mm').read_text()
    identity = re.search(r'    const std::string identity = [^\n]+;', device).group(0)
    source = source.replace('// IDENTITY_HOOK', identity.strip())
    run('library_reuse', source, directory, extra=('-I'+str(ROOT/'vita3k/renderer/include'),))


def early_mask_rejection_test(directory):
    from mask_rejection import check_module
    binary = directory / 'mask_rejection'
    link_shader_test(ROOT / 'vita3k/shader/src/metal4/tests/mask_rejection.cpp', binary,
                     ('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))
    output = directory / 'mask-shaders'
    output.mkdir()
    subprocess.run([str(binary), str(output)], check=True)
    modules = sorted(output.glob('*.spv'))
    assert len(modules) == 96
    cases = sum(check_module(path) for path in modules)
    print(f'PASS early mask semantics: {cases} emitted-SPIR-V predicate/coverage evaluations; rejecting CFG edge precedes all guest calls, framebuffer reads and interlock; single/2x/4x and ordered RGBA thresholds, no GPU/app', flush=True)
    # Extract the existing disabled-fragment shader verbatim to prove its
    # per-sample mask interface still compiles without running a guest program.
    source = '#include <shader/metal4/msl_recompiler.h>\n#include <filesystem>\n#include <fstream>\n'
    source += function('vita3k/renderer/src/metal4/state.mm', 'shader::metal::Program depth_only_program(')
    source += r'''
int main(int argc,char **argv) {for(unsigned samples:{1u,2u,4u}) {
    auto p=depth_only_program(samples);
    if(p.writes_guest_memory || (p.source.find("[[sample_id]]")!=std::string::npos)!=(samples>1)) return 1;
    std::ofstream(std::filesystem::path(argv[1])/("depth-only-"+std::to_string(samples)+".metal"))<<p.source;
}}'''
    generated = directory / 'depth_only.cpp'
    generated.write_text(source)
    depth_binary = directory / 'depth_only'
    link_shader_test(generated, depth_binary)
    subprocess.run([str(depth_binary), str(output)], check=True)
    def validate(module):
        subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', '--scalar-block-layout', str(module)], check=True)
    def compile_shader(path):
        subprocess.run(['xcrun', '--toolchain', os.environ.get('VITA3K_METAL_TOOLCHAIN', 'Metal'),
                        '-sdk', 'macosx', 'metal', '-std=metal3.0',
                        '-fmodules-cache-path='+str(directory/'metal-modules'), '-c', str(path),
                        '-o', str(path.with_suffix('.air'))], check=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate, modules))
        list(executor.map(compile_shader, sorted(output.glob('*.metal'))))
    print('PASS early mask offline: 96 SPIR-V modules validated and 99 MSL shaders compiled to AIR, including disabled-fragment mask shaders; no Metal device/GPU/app', flush=True)


def empty_mask_visibility_test(directory):
    actual = function('vita3k/renderer/src/metal4/state.mm', 'const auto empty_visibility_draw =') + ';'
    source = COMMON + r'''
using Address=uint32_t;
struct Memory { std::array<uint32_t,4> words; bool valid; };
template<class T> struct Ptr {
    Address address;
    explicit Ptr(Address value):address(value){}
    T *get(Memory &mem) const {assert(address>=64 && address<80);return reinterpret_cast<T*>(&mem.words[(address-64)/4]);}
};
bool is_valid_addr_range(Memory &mem,Address begin,Address end) {return mem.valid && begin>=64 && end<=80;}
struct State {
    Address visibility_address=64;uint32_t visibility_stride;
    bool visibility_enabled,back_visibility_enabled,visibility_increment,back_visibility_increment;
    uint32_t visibility_index,back_visibility_index;
};
int main() {
    size_t cases=0;
    for(bool mask:{false,true})for(unsigned enabled=0;enabled<4;++enabled)
    for(unsigned increment=0;increment<4;++increment)for(unsigned preserve=0;preserve<4;++preserve)
    for(unsigned stride:{0u,4u,16u})for(unsigned index:{0u,1u,7u})for(bool valid:{false,true}) {
        Memory mem{{11,22,33,44},valid};auto expected=mem.words;
        State state{};state.visibility_stride=stride;state.visibility_enabled=enabled&1;state.back_visibility_enabled=enabled&2;
        state.visibility_increment=increment&1;state.back_visibility_increment=increment&2;
        state.visibility_index=index;state.back_visibility_index=index+1;
        struct {State *impl;}ctx{&state};struct {bool is_maskupdate;}record{mask};
''' + actual + r'''
        empty_visibility_draw(preserve&1,preserve&2);
        if(!mask && valid && stride)for(unsigned face=0;face<2;++face)
            if((enabled&(1u<<face)) && !(increment&(1u<<face)) && !(preserve&(1u<<face))) {
                const auto selected=index+face;expected[selected<stride/4?selected:0]=0;
            }
        assert(mem.words==expected);++cases;
    }
    assert(cases==2304);
    std::cout<<"PASS empty mask visibility: 2304 actual production-lambda cases; masks preserve every guest query word, ordinary empty draws retain SET/INCREMENT/preserve/invalid-range/index-fallback behavior; allocator boundary controlled, ASAN/UBSAN, no GPU/app\n";
}
'''
    run('empty_mask_visibility', source, directory)


def upstream_lod_test(directory):
    module = 'vita3k/modules/SceGxm/metal4/SceGxm_metal.inl'
    header = 'vita3k/renderer/include/renderer/metal4/textures.h'
    body = '\n'.join(function(module, signature) for signature in (
        'uint32_t export_sceGxmTextureGetLodMin_metal3(',
        'int export_sceGxmTextureSetLodMin_metal3('))
    body += '\n' + '\n'.join(function(header, signature) for signature in (
        'inline uint32_t metal_lod_min(', 'inline uint32_t sampler_metadata_flags('))
    source = COMMON + r'''
struct EmuEnvState {};
#define TRACY_FUNC(...)
#define RET_ERROR(x) (x)
''' + body + r'''
int main() {
    EmuEnvState env;
    size_t cases=0;
    for(unsigned fill:{0u,0x55u,0xaau,0xffu})for(unsigned lod=0;lod<16;++lod) {
        SceGxmTexture texture;std::memset(&texture,fill,sizeof(texture));texture.type=SCE_GXM_TEXTURE_LINEAR>>29;
        auto untouched=texture;untouched.lod_min0=untouched.lod_min1=0;
        assert(export_sceGxmTextureSetLodMin_metal3(env,0,"test",&texture,lod)==0);
        assert(texture.lod_min0==lod/4 && texture.lod_min1==lod%4);
        assert(export_sceGxmTextureGetLodMin_metal3(env,0,"test",&texture)==lod);
        assert(texture.true_lod_min()==lod && metal_lod_min(texture)==lod);
        assert(((sampler_metadata_flags(texture)>>9)&15)==lod);
        auto other=texture;other.lod_min0=other.lod_min1=0;
        assert(std::memcmp(&untouched,&other,sizeof(texture))==0);++cases;
    }
    SceGxmTexture strided{};strided.type=SCE_GXM_TEXTURE_LINEAR_STRIDED>>29;
    strided.lod_min0=3;strided.lod_min1=3;const auto original=strided;
    assert(export_sceGxmTextureGetLodMin_metal3(env,0,"test",&strided)==0);
    assert(uint32_t(export_sceGxmTextureSetLodMin_metal3(env,0,"test",&strided,7))==SCE_GXM_ERROR_UNSUPPORTED);
    assert(std::memcmp(&strided,&original,sizeof(strided))==0);
    assert(((sampler_metadata_flags(strided)>>9)&15)==0);
    assert(export_sceGxmTextureGetLodMin_metal3(env,0,"test",nullptr)==SCE_GXM_ERROR_INVALID_POINTER);
    assert(uint32_t(export_sceGxmTextureSetLodMin_metal3(env,0,"test",nullptr,0))==SCE_GXM_ERROR_INVALID_POINTER);
    std::cout<<"PASS official LOD integration: 64 actual private getter/setter/sampler/metadata cases, all 16 minimum LODs, upper/lower bit layout and unrelated descriptor bytes; strided/null exclusions preserved; ASAN/UBSAN, no GPU/app\n";
}
'''
    run('official_lod', source, directory)


def color_surface_null_data_test(directory):
    module = 'vita3k/modules/SceGxm/metal4/SceGxm_metal.inl'
    shared = 'vita3k/modules/SceGxm/SceGxm.cpp'
    source = COMMON + r'''
#include <map>
#include <renderer/metal4/color_surface.h>
#include <util/align.h>
struct EmuEnvState {};
#define TRACY_FUNC(...)
#define LOG_WARN(...)
#define RET_ERROR(x) static_cast<int>(x)
#define EXPORT(ret, name, ...) ret name(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, __VA_ARGS__)
#define CALL_EXPORT(name, ...) name(emuenv, thread_id, export_name, __VA_ARGS__)
namespace gxm {
''' + function('vita3k/gxm/src/color.cpp', 'SceGxmColorBaseFormat get_base_format(') + '\n'
    source += function('vita3k/gxm/src/color.cpp', 'size_t bits_per_pixel(') + '\n'
    source += function('vita3k/renderer/src/metal4/texture_format_metal.cpp',
                       'bool convert_color_format_to_texture_format_metal3(') + '\n}\n'
    source += function(shared, 'static int init_texture_base(') + '\n'
    for name in ('Linear', 'Swizzled', 'Tiled'):
        source += function(shared, 'EXPORT(int, sceGxmTextureInit' + name + ',') + '\n'
    source += function(module, 'static void set_linear_strided_texture_stride_metal3(') + '\n'
    source += function(module, 'int export_sceGxmTextureInitLinearStrided_metal3(') + '\n'
    source += '#define sceGxmTextureInitLinearStrided export_sceGxmTextureInitLinearStrided_metal3\n'
    source += function(module, 'int export_sceGxmColorSurfaceInit_metal3(') + '\n'
    source += Path(__file__).with_name('color_surface_init.cpp').read_text()
    run('color_surface_null_data', source, directory,
        ('-I' + str(ROOT / 'vita3k/renderer/include'),
         '-I' + str(ROOT / 'vita3k/mem/include')))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='vita3k-metal-cpu-') as temporary:
        directory = Path(temporary)
        surface_allocation_test(directory)
        library_reuse_test(directory)
        presentation_lifetime_test(directory)
        early_mask_rejection_test(directory)
        upstream_lod_test(directory)
        color_surface_null_data_test(directory)
        empty_mask_visibility_test(directory)
        run('compile_jobs', Path(__file__).with_name('compile_jobs.cpp').read_text(), directory,
            ('-I' + str(ROOT / 'vita3k/renderer/include'),))
        run('shader_index', Path(__file__).with_name('shader_index.cpp').read_text(), directory,
            ('-I' + str(ROOT / 'vita3k/renderer/include'),), (str(directory / 'index-data'),))
        run('uniform', uniform_source(), directory)
        uniform_bounds_test(directory)
        run('morton', ranges_source(), directory)
        run('bc7', bc7_source(), directory)
        bc7_reference_test(directory)
        run('bc_upload', bcn_upload_source(), directory)
        dds_import_test(directory)
        cache_write_slots_test(directory)
        cache_read_concurrency_test(directory)
        guest_publication_test(directory)
        readback_buffers_test(directory)
        color_publication_test(directory)
        guest_memory_publication_test(directory)
        async_color_publication_test(directory)
        run('overlay_mips', mip_source(), directory)
        run('depth_bounds', depth_bounds_source(), directory,
            ('-I' + str(ROOT / 'vita3k/renderer/include'),))
        run('surface_changes', Path(__file__).with_name('surface_changes.cpp').read_text(), directory,
            ('-I' + str(ROOT / 'vita3k/renderer/include'),))
        run('command_dependencies', Path(__file__).with_name('command_dependencies.cpp').read_text(), directory,
            ('-I' + str(ROOT / 'vita3k/renderer/include'),))
        buffer_test(directory)
        capture_pass_test(directory)
        vertex_snapshot_test(directory)
        caster_test(directory)
        presentation_test(directory)
        surface_cache_budget_test(directory)
        fsr_presentation_test(directory)
        packed_surface_cast_test(directory)
        queued_cast_sampling_test(directory)
        msaa_seed_submission_test(directory)
        multisample_snapshot_test(directory)
        cube_fallback_submission_test(directory)
        pipeline_warmup_test(directory)
        finish_submission_test(directory)
        linear_surface_patch_test(directory)
        surface_write_patch_test(directory)
        snapshot_channels_test(directory)
        vertex_layout_test(directory)
        variant_keys_test(directory)
        texture_lookup_test(directory)
        texture_helper_test(directory)
        clip_geometry_test(directory)
        clip_replay_test(directory)
        vertex_color_clamp_test(directory)
        clip_shader_test(directory)
        fp_constants_test(directory)
        rotate_left_test(directory)
        i8mad_test(directory)
        i32mad_formats_test(directory)
        vtst_dot_masks_test(directory)
        shader_test(directory)
