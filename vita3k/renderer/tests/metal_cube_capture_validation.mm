// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/state.h>
#include <config/state.h>
#include <mem/state.h>
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <shader/uniform_block.h>
#include <shader/metal_texture.h>
#include <shader/msl_recompiler.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

static void check(bool value, const std::string &message) {
    if (!value)
        throw std::runtime_error(message);
}
static std::vector<uint8_t> read(const std::filesystem::path &path, size_t maximum = 64 * 1024 * 1024) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    check(bool(input), "Cannot read " + path.string());
    auto size = input.tellg();
    check(size > 0 && uint64_t(size) <= maximum, "Invalid file size");
    std::vector<uint8_t> bytes(size);
    input.seekg(0);
    check(bool(input.read(reinterpret_cast<char *>(bytes.data()), size)), "Short read");
    return bytes;
}
using Pixel = std::array<float, 4>;
struct Image {
    unsigned width = 0, height = 0, mips = 0, faces = 0, anisotropy = 0, count = 0;
    size_t declared_bytes = 0;
    bool complete = false;
    std::map<std::pair<unsigned, unsigned>, std::vector<Pixel>> pixels;
    id<MTLTexture> texture;
    SceGxmTexture sampler{};
};
static void check_subresource_views(renderer::metal::Device &device) {
    renderer::metal::SurfaceCaster caster(device);
    unsigned compared = 0;
    for (bool cube : {false, true})
        for (bool gamma : {false, true}) {
            const auto format = gamma ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
            auto desc = cube ? [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:format size:4 mipmapped:YES]
                             : [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                  width:4
                                                                                 height:4
                                                                              mipmapped:YES];
            desc.storageMode = MTLStorageModeShared;
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
            auto source = [device.native_device() newTextureWithDescriptor:desc];
            check(source != nil, "Subresource source allocation failed");
            auto code = [](unsigned face, unsigned mip, unsigned x, unsigned y) {
                return std::array<uint8_t, 4>{uint8_t(32 + face * 23 + x * 7), uint8_t(67 + mip * 31 + y * 5),
                                              uint8_t(103 + x * 11 + y * 3), 29};
            };
            for (unsigned face = 0; face < (cube ? 6u : 1u); ++face)
                for (unsigned mip = 0; mip < 3; ++mip) {
                    unsigned size = 4 >> mip;
                    std::vector<std::array<uint8_t, 4>> data(size * size);
                    for (unsigned y = 0; y < size; ++y)
                        for (unsigned x = 0; x < size; ++x)
                            data[y * size + x] = code(face, mip, x, y);
                    [source replaceRegion:MTLRegionMake2D(0, 0, size, size)
                              mipmapLevel:mip
                                    slice:face
                                withBytes:data.data()
                              bytesPerRow:size * 4
                            bytesPerImage:size * size * 4];
                }
            auto view = [source newTextureViewWithPixelFormat:format
                                                  textureType:source.textureType
                                                       levels:NSMakeRange(0, 3)
                                                       slices:NSMakeRange(0, cube ? 6 : 1)
                                                      swizzle:MTLTextureSwizzleChannelsMake(
                                                                  MTLTextureSwizzleBlue, MTLTextureSwizzleRed,
                                                                  MTLTextureSwizzleGreen, MTLTextureSwizzleOne)];
            check(view != nil, "Cannot create swizzled source");
            check(caster.sampling_snapshot(view, 3, 0) == nil && caster.sampling_snapshot(view, 0, cube ? 6 : 1) == nil,
                  "Invalid subresource accepted");
            for (unsigned face = 0; face < (cube ? 6u : 1u); ++face)
                for (unsigned mip = 0; mip < 3; ++mip) {
                    auto snapshot = caster.sampling_snapshot(view, mip, face);
                    unsigned size = 4 >> mip;
                    check(snapshot != nil && snapshot.width == size && snapshot.height == size,
                          "Wrong snapshot extent");
                    std::vector<Pixel> actual(size * size);
                    [snapshot getBytes:actual.data()
                           bytesPerRow:size * 16
                            fromRegion:MTLRegionMake2D(0, 0, size, size)
                           mipmapLevel:0];
                    const unsigned channels[3] = {2, 0, 1};
                    for (unsigned y = 0; y < size; ++y)
                        for (unsigned x = 0; x < size; ++x)
                            for (unsigned c = 0; c < 4; ++c) {
                                auto bytes = code(face, mip, x, y);
                                double expected = c == 3 ? 1 : bytes[channels[c]] / 255.0;
                                if (gamma && c < 3)
                                    expected = expected <= .04045 ? expected / 12.92
                                                                  : std::pow((expected + .055) / 1.055, 2.4);
                                check(std::abs(actual[y * size + x][c] - expected) < .0006,
                                      "Snapshot lost channel swizzle, gamma, face or mip");
                                ++compared;
                            }
                }
        }
    std::cout << "PASS subresource views components=" << compared
              << " 2D/cube,3mips,swizzle,linear/sRGB,invalid bounds\n";
}

// Replays the original Unit 13 reflection FS from a controlled GXM capture.
// Every face/mip is reconstructed; missing data is rejected before GPU work.

static void check_mip_pixel_centers(renderer::metal::Device &device) {
    renderer::metal::SurfaceCaster caster(device);
    NSError *error = nil;
    auto library = [device.native_device() newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;
kernel void sample_centers(texture2d<float> image [[texture(0)]], device float4 *output [[buffer(0)]],
    constant uint4 &params [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if(any(p>=params.xy)) return;
    constexpr sampler nearest(coord::normalized,address::clamp_to_edge,filter::nearest,mip_filter::nearest);
    output[p.y*params.x+p.x]=image.sample(nearest,(float2(p)+0.5f)/float2(params.xy),level(params.z));
})" options:nil error:&error];
    check(library != nil, "Cannot compile mip center sampler");
    auto pipeline =
        [device.native_device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"sample_centers"]
                                                              error:&error];
    check(pipeline != nil, "Cannot create mip center sampler");
    uint64_t compared = 0;
    unsigned chains = 0;
    for (const auto shape : {std::array<unsigned, 2>{15, 9}, {31, 17}, {17, 31}, {16, 8}, {1, 9}, {9, 1}})
        for (unsigned scale : {1u, 2u, 3u}) {
            const unsigned levels = std::bit_width(std::min(shape[0], shape[1]));
            auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                           width:shape[0]
                                                                          height:shape[1]
                                                                       mipmapped:NO];
            desc.mipmapLevelCount = levels;
            desc.storageMode = MTLStorageModeShared;
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
            auto uploaded = [device.native_device() newTextureWithDescriptor:desc];
            check(uploaded != nil, "Cannot allocate mip center input");
            std::vector<renderer::metal::CubeSurface> rendered;
            std::vector<std::vector<Pixel>> expected(levels);
            std::vector<std::array<unsigned, 2>> sizes(levels);
            for (unsigned mip = 0; mip < levels; ++mip) {
                const unsigned w = shape[0] >> mip, h = shape[1] >> mip;
                std::vector<Pixel> ram(w * h);
                for (unsigned y = 0; y < h; ++y)
                    for (unsigned x = 0; x < w; ++x)
                        ram[y * w + x] = {float(x), float(y), float(mip), -1.f};
                [uploaded replaceRegion:MTLRegionMake2D(0, 0, w, h)
                            mipmapLevel:mip
                              withBytes:ram.data()
                            bytesPerRow:w * sizeof(Pixel)];
                if (mip & 1) {
                    expected[mip] = ram;
                    sizes[mip] = {w, h};
                    continue;
                }
                const unsigned nw = w * scale, nh = h * scale;
                sizes[mip] = {nw, nh};
                expected[mip].resize(nw * nh);
                for (unsigned y = 0; y < nh; ++y)
                    for (unsigned x = 0; x < nw; ++x)
                        expected[mip][y * nw + x] = {float(x), float(y), float(mip), 1.f};
                auto sd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                             width:nw
                                                                            height:nh
                                                                         mipmapped:NO];
                sd.storageMode = MTLStorageModeShared;
                sd.usage = MTLTextureUsageShaderRead;
                auto surface = [device.native_device() newTextureWithDescriptor:sd];
                check(surface != nil, "Cannot allocate native mip center input");
                [surface replaceRegion:MTLRegionMake2D(0, 0, nw, nh)
                           mipmapLevel:0
                             withBytes:expected[mip].data()
                           bytesPerRow:nw * sizeof(Pixel)];
                rendered.push_back({surface, 0, mip});
            }
            auto image = caster.texture_snapshot(uploaded, rendered, scale);
            check(image != nil && image.mipmapLevelCount == levels, "Wrong assembled mip extent");
            for (unsigned mip = 0; mip < levels; ++mip) {
                const auto [w, h] = sizes[mip];
                auto result = [device.native_device() newBufferWithLength:w * h * sizeof(Pixel)
                                                                  options:MTLResourceStorageModeShared];
                check(result != nil, "Cannot allocate sampled mip output");
                uint32_t params[4] = {w, h, mip, 0};
                auto commands = [device.command_queue() commandBuffer];
                auto encoder = [commands computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setTexture:image atIndex:0];
                [encoder setBuffer:result offset:0 atIndex:0];
                [encoder setBytes:params length:sizeof(params) atIndex:1];
                [encoder dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
                [encoder endEncoding];
                std::string message;
                check(device.submit_and_wait(commands, message), message);
                const auto *pixels = static_cast<const Pixel *>(result.contents);
                for (unsigned p = 0; p < w * h; ++p)
                    for (unsigned c = 0; c < 4; ++c) {
                        check(pixels[p][c] == expected[mip][p][c],
                              "Lost source pixel center: " + std::to_string(shape[0]) + "x" + std::to_string(shape[1]) +
                                  " scale=" + std::to_string(scale) + " mip=" + std::to_string(mip) +
                                  " pixel=" + std::to_string(p) + " component=" + std::to_string(c));
                        ++compared;
                    }
            }
            ++chains;
        }
    std::cout << "PASS mip pixel centers chains=" << chains << " components=" << compared
              << " NPOT/rectangular,1x/2x/3x,mixed RAM/rendered levels,nearest GPU sampling\n";
}

static void check_image_cache_identity() {
    MemState mem;
    check(init(mem, false), "Cannot initialize image-cache memory");
    renderer::metal::MetalState state;
    check(state.init(), "Cannot initialize image-cache Metal device");
    Config config;
    state.late_init(config, "metal-image-cache-validation", mem);
    Ptr<uint8_t> data(alloc(mem, 65536, "Image-cache fixture"));
    check(bool(data), "Cannot allocate image-cache memory");
    std::fill(data.get(mem), data.get(mem) + 65536, uint8_t(127));
    unsigned reused = 0, invalidated = 0;
    for (auto type :
         {SCE_GXM_TEXTURE_LINEAR, SCE_GXM_TEXTURE_SWIZZLED, SCE_GXM_TEXTURE_CUBE, SCE_GXM_TEXTURE_LINEAR_STRIDED}) {
        SceGxmTexture texture{};
        texture.type = type >> 29;
        texture.width = texture.height = 15;
        texture.mip_count = 1;
        texture.lod_bias = 31;
        texture.base_format = SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 >> 24;
        texture.data_addr = data.address() >> 2;
        if (type == SCE_GXM_TEXTURE_SWIZZLED || type == SCE_GXM_TEXTURE_CUBE)
            texture.width_base2 = texture.height_base2 = 4;
        const bool strided = type == SCE_GXM_TEXTURE_LINEAR_STRIDED;
        if (strided) {
            const unsigned stride = 19;
            texture.mip_filter = stride & 1;
            texture.min_filter = (stride >> 1) & 3;
            texture.mip_count = (stride >> 3) & 15;
            texture.lod_bias = stride >> 7;
        }
        state.texture_cache.cache_and_bind_image(texture, mem);
        auto original = renderer::metal::current_texture(state.texture_cache);
        check(original != nil, "Missing initial image-cache texture");
        for (unsigned u : {0u, 1u, 2u})
            for (unsigned v : {0u, 1u, 2u})
                for (unsigned filter = 0; filter < 4; ++filter)
                    for (unsigned mag : {0u, 1u})
                        for (unsigned mip_filter : {0u, 1u})
                            for (unsigned bias : {0u, 31u, 63u})
                                for (unsigned lod : {0u, 1u, 4u, 15u}) {
                                    auto variant = texture;
                                    variant.uaddr_mode = u;
                                    variant.vaddr_mode = v;
                                    variant.mag_filter = mag;
                                    variant.lod_min0 = lod & 3;
                                    variant.lod_min1 = lod >> 2;
                                    if (!strided) {
                                        variant.min_filter = filter;
                                        variant.mip_filter = mip_filter;
                                        variant.lod_bias = bias;
                                    }
                                    state.texture_cache.cache_and_bind_image(variant, mem);
                                    check(renderer::metal::current_texture(state.texture_cache) == original,
                                          "Sampler-only change reuploaded pixels");
                                    ++reused;
                                }
        // Retain each prior texture so allocation address reuse cannot hide a stale result.
        std::vector<id<MTLTexture>> generations{original};
        const auto expect_changed = [&](const SceGxmTexture &variant) {
            state.texture_cache.cache_and_bind_image(variant, mem);
            auto current = renderer::metal::current_texture(state.texture_cache);
            check(current != nil && current != generations.back(), "Pixel/layout change reused stale upload");
            generations.push_back(current);
            ++invalidated;
        };
        const auto check_guest_pixel = [&](unsigned mip, unsigned offset, unsigned face = 0) {
            std::array<uint8_t, 4> pixel{};
            [generations.back() getBytes:pixel.data()
                             bytesPerRow:4
                           bytesPerImage:4
                              fromRegion:MTLRegionMake2D(0, 0, 1, 1)
                             mipmapLevel:mip
                                   slice:face];
            check(std::equal(pixel.begin(), pixel.end(), data.get(mem) + offset),
                  "RAM update missing from uploaded pixels");
        };
        data.get(mem)[0] ^= 1;
        expect_changed(texture);
        check_guest_pixel(0, 0);
        if (!strided) {
            data.get(mem)[1024] ^= 1;
            expect_changed(texture);
            check_guest_pixel(1, 1024);
        }
        if (type == SCE_GXM_TEXTURE_CUBE) {
            const auto face_bytes=renderer::metal::cube_texture_storage_size(texture)/6;
            for(unsigned face=1;face<6;++face) for(unsigned mip=0;mip<2;++mip) {
                const auto offset=unsigned(face*face_bytes+(mip ? 1024 : 0));
                data.get(mem)[offset]^=1;
                expect_changed(texture);
                check_guest_pixel(mip,offset,face);
            }
        }
        auto format = texture;
        format.base_format = SCE_GXM_TEXTURE_BASE_FORMAT_F32 >> 24;
        expect_changed(format);
        check(generations.back().pixelFormat == MTLPixelFormatR32Float, "Base format lost in image key");
        auto gamma = texture;
        gamma.gamma_mode = 1;
        expect_changed(gamma);
        check(generations.back().pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB, "Gamma view lost in image key");
        auto swizzle = gamma;
        swizzle.swizzle_format = 1;
        expect_changed(swizzle);
        if (strided) {
            auto wider = swizzle;
            wider.mip_count ^= 1;
            expect_changed(wider);
        }
    }
    std::cout << "PASS image cache reused=" << reused << " invalidated=" << invalidated
              << " sampler controls separated;pixel/mip/gamma/swizzle/stride changes retained\n";
}

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 2, "usage: metal-cube-capture-validation Unit13-cube-test-capture-directory");
            const std::filesystem::path root = argv[1];
            std::map<unsigned, Image> images;
            std::ifstream meta(root / "textures.txt");
            check(bool(meta), "Missing texture metadata");
            unsigned version = 0;
            std::string line;
            while (std::getline(meta, line)) {
                std::istringstream f(line);
                std::string key, stage;
                unsigned slot;
                f >> key;
                if (key == "version") {
                    f >> version;
                    check(version == 2, "Requires capture version 2");
                    continue;
                }
                if (key == "fragment_uniform") {
                    unsigned block, size;
                    f >> block >> size;
                    check(block == 2 && size == 16, "Unexpected uniform fixture");
                    continue;
                }
                if (key == "sampled_texture")
                    continue; // legacy base-level alias
                check(bool(f >> stage >> slot) && stage == "fragment" && slot < 2, "Invalid fixture texture binding");
                if (key == "sampled_image") {
                    check(!images.contains(slot), "Duplicate image");
                    std::string type;
                    unsigned native_format;
                    auto &i = images[slot];
                    f >> type >> i.width >> i.height >> i.mips >> native_format;
                    check(type == "cube" || type == "2d", "Invalid image type");
                    i.faces = type == "cube" ? 6 : 1;
                    check(i.width && i.height && i.width <= 16384 && i.height <= 16384 && i.mips && i.mips <= 15,
                          "Invalid image extent");
                    check(i.faces == 1 || i.width == i.height, "Non-square cube");
                    check((1u << (i.mips - 1)) <= std::max(i.width, i.height), "Too many mip levels");
                    size_t maximum_bytes = 0;
                    for (unsigned mip = 0; mip < i.mips; ++mip)
                        maximum_bytes +=
                            size_t(std::max(1u, i.width >> mip)) * std::max(1u, i.height >> mip) * 16 * i.faces;
                    check(maximum_bytes <= 64 * 1024 * 1024, "Image exceeds capture budget");
                } else {
                    check(images.contains(slot), "Image header missing");
                    auto &i = images.at(slot);
                    if (key == "native_sampler") {
                        f >> i.anisotropy;
                        check(i.anisotropy >= 1 && i.anisotropy <= 16, "Invalid anisotropy");
                    } else if (key == "sampled_subresource") {
                        unsigned face, mip, w, h;
                        std::string name;
                        f >> face >> mip >> w >> h >> name;
                        check(face < i.faces && mip < i.mips && w == std::max(1u, i.width >> mip) &&
                                  h == std::max(1u, i.height >> mip),
                              "Invalid subresource extent");
                        check(std::filesystem::path(name).filename() == name && name != "." && name != "..",
                              "Invalid capture filename");
                        check(!i.pixels.contains({face, mip}), "Duplicate subresource");
                        auto bytes = read(root / name);
                        check(bytes.size() == size_t(w) * h * 16, "Subresource length mismatch");
                        auto &pixels = i.pixels[{face, mip}];
                        pixels.resize(size_t(w) * h);
                        std::memcpy(pixels.data(), bytes.data(), bytes.size());
                    } else if (key == "sampled_image_complete") {
                        check(!i.complete, "Duplicate completion");
                        f >> i.count >> i.declared_bytes;
                        i.complete = true;
                    } else
                        throw std::runtime_error("Incomplete/unsupported image metadata: " + key);
                }
                check(!f.fail(), "Malformed metadata");
            }
            check(version == 2 && images.size() == 2 && images.at(0).faces == 1 && images.at(1).faces == 6,
                  "Incomplete Unit 13 image set");
            for (auto &[slot, i] : images) {
                size_t total = 0;
                for (unsigned face = 0; face < i.faces; ++face)
                    for (unsigned mip = 0; mip < i.mips; ++mip) {
                        check(i.pixels.contains({face, mip}), "Missing cube face or mip");
                        total += i.pixels.at({face, mip}).size() * 16;
                    }
                check(i.complete && i.count == i.faces * i.mips && i.pixels.size() == i.count &&
                          i.declared_bytes == total && total <= 64 * 1024 * 1024,
                      "Incomplete image or byte count");
                auto descriptor = read(root / ("fragment-texture-" + std::to_string(slot) + ".gxm"));
                check(descriptor.size() == sizeof(i.sampler), "Sampler ABI mismatch");
                std::memcpy(&i.sampler, descriptor.data(), descriptor.size());
            }
            auto uniform_bytes = read(root / "fragment-uniform-2.bin");
            const uint32_t expected_uniform[4] = {0x3c00, 0, 0, 0x3c00};
            check(uniform_bytes.size() == 16 && std::memcmp(uniform_bytes.data(), expected_uniform, 16) == 0,
                  "Not the controlled reflection material fixture");
            check(images.at(0).width == 1 && images.at(0).height == 1 &&
                      images.at(0).pixels.at({0, 0})[0] == Pixel{1, 1, 1, 1},
                  "Not the controlled base-color fixture");
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            auto native = device->native_device();
            check_subresource_views(*device);
            check_mip_pixel_centers(*device);
            check_image_cache_identity();
            for (auto &[slot, i] : images) {
                auto d = i.faces == 6
                             ? [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                     size:i.width
                                                                                mipmapped:NO]
                             : [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:i.width
                                                                                 height:i.height
                                                                              mipmapped:NO];
                d.mipmapLevelCount = i.mips;
                d.storageMode = MTLStorageModeShared;
                d.usage = MTLTextureUsageShaderRead;
                i.texture = [native newTextureWithDescriptor:d];
                check(i.texture != nil, "Cannot rebuild captured texture");
                for (auto &[sub, pixels] : i.pixels) {
                    const auto [face, mip] = sub;
                    unsigned w = std::max(1u, i.width >> mip), h = std::max(1u, i.height >> mip);
                    [i.texture replaceRegion:MTLRegionMake2D(0, 0, w, h)
                                 mipmapLevel:mip
                                       slice:face
                                   withBytes:pixels.data()
                                 bytesPerRow:w * 16
                               bytesPerImage:w * h * 16];
                }
            }
            auto source = read(root / "fragment.metal");
            shader::metal::Program program{.source = std::string(source.begin(), source.end()),
                                           .entry_point = "main_fs",
                                           .stage = shader::metal::Stage::Fragment};
            auto fs = device->compile(program, false, error);
            check(bool(fs), error);
            shader::metal::Program vertex{.source = R"(#include <metal_stdlib>
using namespace metal;struct O{float4 p [[position]];float4 uv [[user(locn4)]];float4 n [[user(locn5)]];float4 v [[user(locn6)]];};
vertex O v(uint i [[vertex_id]],constant float4 &view [[buffer(0)]]){float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
O o;o.p=float4(p[i],0,1);o.uv=float4(.5,.5,0,0);o.n=float4(0,0,1,0);o.v=view;return o;})",
                                          .entry_point = "v",
                                          .stage = shader::metal::Stage::Vertex};
            auto vs = device->compile(vertex, false, error);
            check(bool(vs), error);
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = vs->function;
            pd.fragmentFunction = fs->function;
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatRG32Float;
            auto pipeline = device->create_pipeline(pd, error);
            check(pipeline != nil, error);
            auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float
                                                                         width:8
                                                                        height:8
                                                                     mipmapped:NO];
            td.storageMode = MTLStorageModeShared;
            td.usage = MTLTextureUsageRenderTarget;
            auto target = [native newTextureWithDescriptor:td];
            td.pixelFormat = MTLPixelFormatRGBA8Unorm;
            td.usage = MTLTextureUsageShaderRead;
            auto mask = [native newTextureWithDescriptor:td];
            std::array<uint8_t, 256> white;
            white.fill(255);
            [mask replaceRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0 withBytes:white.data() bytesPerRow:32];
            auto uniform = [native newBufferWithBytes:uniform_bytes.data()
                                               length:16
                                              options:MTLResourceStorageModeShared];
            shader::RenderFragUniformBlockExtended info{};
            info.base_block.res_multiplier = 1;
            info.set_buffer_count(3);
            info.set_buffer_address(2, uniform.gpuAddress);
            info.set_texture_count(2);
            info.set_viewport_ratio(0, {1, 1});
            info.set_viewport_ratio(1, {1, 1});
            std::vector<uint8_t> info_bytes((info.get_size() + 15) & ~size_t(15));
            info.copy_to(info_bytes.data());
            shader::metal::TextureMipInfos texture_info{};
            if (std::filesystem::exists(root / "fragment-texture-info.bin")) {
                const auto bytes = read(root / "fragment-texture-info.bin", sizeof(texture_info));
                check(bytes.size() == sizeof(texture_info), "Invalid native mip metadata size");
                std::memcpy(texture_info.data(), bytes.data(), bytes.size());
            }
            unsigned draws = 0, compared = 0;
            auto &cube = images.at(1);
            auto &base = images.at(0);
            auto base_sampler = renderer::metal::make_sampler(*device, base.sampler, base.anisotropy);
            for (unsigned mip = 0; mip < cube.mips; ++mip) {
                unsigned size = std::max(1u, cube.width >> mip);
                cube.sampler.lod_min0 = mip & 3;
                cube.sampler.lod_min1 = mip >> 2;
                auto sampler = renderer::metal::make_sampler(*device, cube.sampler, cube.anisotropy);
                for (unsigned face = 0; face < 6; ++face)
                    for (unsigned x : {0u, std::min(1u, size - 1), size / 2, size - 1})
                        for (unsigned y : {0u, size / 2, size - 1}) {
                            float s = 2 * (x + .5f) / size - 1, t = 2 * (y + .5f) / size - 1;
                            std::array<std::array<float, 3>, 6> directions = {
                                {{1, -t, -s}, {-1, -t, s}, {s, 1, t}, {s, -1, -t}, {s, -t, 1}, {-s, -t, -1}}};
                            auto d = directions[face];
                            std::array<float, 4> view{-d[0], -d[1], d[2], 1};
                            auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                            pass.colorAttachments[0].texture = target;
                            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                            auto commands = [device->command_queue() commandBuffer];
                            auto enc = [commands renderCommandEncoderWithDescriptor:pass];
                            [enc setRenderPipelineState:pipeline];
                            [enc setVertexBytes:view.data() length:16 atIndex:0];
                            [enc setFragmentBytes:info_bytes.data() length:info_bytes.size() atIndex:0];
                            [enc setFragmentBytes:texture_info.data()
                                           length:sizeof(texture_info)
                                          atIndex:shader::metal::TEXTURE_INFO_BUFFER];
                            [enc useResource:uniform usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
                            [enc setFragmentTexture:base.texture atIndex:0];
                            [enc setFragmentTexture:cube.texture atIndex:1];
                            [enc setFragmentTexture:mask atIndex:17];
                            [enc setFragmentSamplerState:base_sampler atIndex:0];
                            [enc setFragmentSamplerState:sampler atIndex:1];
                            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                            [enc endEncoding];
                            check(device->submit_and_wait(commands, error), error);
                            std::array<uint8_t, 512> actual;
                            [target getBytes:actual.data()
                                 bytesPerRow:64
                                  fromRegion:MTLRegionMake2D(0, 0, 8, 8)
                                 mipmapLevel:0];
                            const auto expected = cube.pixels.at({face, mip})[y * size + x];
                            for (unsigned p = 0; p < 64; ++p)
                                for (unsigned c = 0; c < 8; ++c) {
                                    int value = c < 3    ? int(std::round(float((__fp16)expected[c]) * 255))
                                                : c == 3 ? 255
                                                : c == 6 ? 127
                                                         : 0;
                                    check(std::abs(int(actual[p * 8 + c]) - value) <= (c < 3 ? 1 : 0),
                                          "Captured cube replay mismatch face=" + std::to_string(face) +
                                              " mip=" + std::to_string(mip) + " x=" + std::to_string(x) +
                                              " y=" + std::to_string(y) + " byte=" + std::to_string(c));
                                    ++compared;
                                }
                            ++draws;
                        }
            }
            std::cout << "PASS captured Unit13 cube replay draws=" << draws << " material_bytes=" << compared
                      << " faces=6 mips=" << cube.mips << '\n';
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
