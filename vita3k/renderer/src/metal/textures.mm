// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/textures.h>
#include <renderer/metal/device.h>
#include <renderer/functions.h>
#include <renderer/gxm_types.h>
#include <gxm/functions.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <bit>
#include <cstring>
#include <cmath>
#include <vector>

namespace renderer::metal {
namespace {
using Map = std::array<MTLTextureSwizzle, 4>;
constexpr auto R = MTLTextureSwizzleRed, G = MTLTextureSwizzleGreen;
constexpr auto B = MTLTextureSwizzleBlue, A = MTLTextureSwizzleAlpha;
constexpr auto Z = MTLTextureSwizzleZero, O = MTLTextureSwizzleOne;
constexpr Map identity{R,G,B,A};
constexpr Map four[] = {{R,G,B,A}, {B,G,R,A}, {A,B,G,R}, {G,B,A,R},
    {R,G,B,O}, {B,G,R,O}, {A,B,G,O}, {G,B,A,O}};
Map texture_mapping(SceGxmTextureFormat format) {
    const auto base = gxm::get_base_format(format);
    const auto mode = (uint32_t(format) & SCE_GXM_TEXTURE_SWIZZLE_MASK) >> 12;
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_X8U24) return {R,Z,Z,O};
    // The CPU decoders already return RGB for YUV and RGBA for U8U3U3U2.
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_YUV420P2 || base == SCE_GXM_TEXTURE_BASE_FORMAT_YUV420P3
        || base == SCE_GXM_TEXTURE_BASE_FORMAT_YUV422 || base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U3U3U2)
        return identity;
    switch (gxm::get_num_components(base)) {
    case 1: {
        constexpr Map one[] = {{R,Z,Z,O}, {R,Z,Z,Z}, {R,O,O,O}, {R,R,R,R},
            {R,R,R,Z}, {R,R,R,O}, {Z,Z,Z,R}, {O,O,O,R}};
        return one[mode];
    }
    case 2: {
        constexpr Map two[] = {{R,G,Z,O}, {R,G,Z,Z}, {R,R,R,G}, {G,G,G,R}, {R,G,R,G}, {G,R,Z,Z}};
        if (mode < std::size(two)) return two[mode];
        break;
    }
    case 3:
        if (mode < 2) return mode ? Map{B,G,R,O} : Map{R,G,B,O};
        break;
    case 4: return four[mode];
    }
    throw std::runtime_error("Metal: unsupported texture channel mapping " + std::to_string(format));
}
Map surface_memory_mapping(SceGxmColorFormat format) {
    const auto mode = (uint32_t(format) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20;
    const auto base = gxm::get_base_format(format);
    switch (base) {
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8:
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8:
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16:
    case SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10: {
        // Invert the mapping from guest memory channels to logical RGBA.
        Map result = identity;
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                if (four[mode][i] == identity[j]) result[j] = identity[i];
        return result;
    }
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8:
        if (mode < 2) return mode ? Map{B,G,R,O} : Map{R,G,B,O};
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8:
    case SCE_GXM_COLOR_BASE_FORMAT_F32F32:
        if (mode < 2) return mode ? Map{G,R,Z,O} : Map{R,G,Z,O};
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_F11F11F10:
        if (mode < 2) return mode ? Map{B,G,R,O} : Map{R,G,B,O};
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U8:
        // Both R and A targets use R8 storage. For U8_A the fragment output
        // and blend state route logical alpha to physical red before storing.
        if (mode < 2) return identity;
        break;
    default:
        // Currently supported one-component targets store logical R.
        if (mode == 0) return identity;
        break;
    }
    throw std::runtime_error("Metal: unsupported surface channel mapping " + std::to_string(format));
}
}
namespace {
struct SurfaceComponents {
    uint32_t count = 0, bytes = 0;
    MTLPixelFormat native = MTLPixelFormatInvalid;
};
SurfaceComponents surface_components(SceGxmColorFormat format) {
    switch (gxm::get_base_format(format)) {
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8: return {4, 1, MTLPixelFormatRGBA8Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8: return {3, 1, MTLPixelFormatRGBA8Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8: return {4, 1, MTLPixelFormatRGBA8Snorm};
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16: return {4, 2, MTLPixelFormatRGBA16Float};
    case SCE_GXM_COLOR_BASE_FORMAT_F32F32: return {2, 4, MTLPixelFormatRG32Float};
    case SCE_GXM_COLOR_BASE_FORMAT_F32: return {1, 4, MTLPixelFormatR32Float};
    case SCE_GXM_COLOR_BASE_FORMAT_F16: return {1, 2, MTLPixelFormatR16Float};
    case SCE_GXM_COLOR_BASE_FORMAT_U8: return {1, 1, MTLPixelFormatR8Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_U16: return {1, 2, MTLPixelFormatR16Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8: return {2, 1, MTLPixelFormatRG8Unorm};
    default: return {};
    }
}
SurfaceComponents texture_components(SceGxmTextureBaseFormat format) {
    switch (format) {
#define COMPONENTS(gxm, count, bytes, mtl) case SCE_GXM_TEXTURE_BASE_FORMAT_##gxm: return {count, bytes, MTLPixelFormat##mtl}
    COMPONENTS(U8, 1, 1, R8Unorm); COMPONENTS(S8, 1, 1, R8Snorm);
    COMPONENTS(U8U8, 2, 1, RG8Unorm); COMPONENTS(S8S8, 2, 1, RG8Snorm);
    COMPONENTS(U8U8U8U8, 4, 1, RGBA8Unorm); COMPONENTS(S8S8S8S8, 4, 1, RGBA8Snorm);
    COMPONENTS(U16, 1, 2, R16Unorm); COMPONENTS(S16, 1, 2, R16Snorm); COMPONENTS(F16, 1, 2, R16Float);
    COMPONENTS(U16U16, 2, 2, RG16Unorm); COMPONENTS(S16S16, 2, 2, RG16Snorm); COMPONENTS(F16F16, 2, 2, RG16Float);
    COMPONENTS(U16U16U16U16, 4, 2, RGBA16Unorm); COMPONENTS(S16S16S16S16, 4, 2, RGBA16Snorm);
    COMPONENTS(F16F16F16F16, 4, 2, RGBA16Float); COMPONENTS(F32, 1, 4, R32Float);
    COMPONENTS(F32F32, 2, 4, RG32Float);
#undef COMPONENTS
    default: return {};
    }
}

}
std::optional<DepthMemoryLayout> depth_memory_layout(const SceGxmDepthStencilSurface &surface, uint32_t width, uint32_t height,
    SceGxmMultisampleMode mode) {
    if(surface.disabled() || !width || !height || mode>SCE_GXM_MULTISAMPLE_4X) return std::nullopt;
    const bool tiled=surface.get_type()==SCE_GXM_DEPTH_STENCIL_SURFACE_TILED;
    if(!tiled && surface.get_type()!=SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR) return std::nullopt;
    const uint64_t w=uint64_t(width)*(mode==SCE_GXM_MULTISAMPLE_4X?2:1);
    const uint64_t h=uint64_t(height)*(mode==SCE_GXM_MULTISAMPLE_NONE?1:2);
    if(w>surface.get_stride() || h>UINT32_MAX) return std::nullopt;
    uint32_t bytes=0;bool separate=false,packed=false;
    switch(surface.get_format()) {
    case SCE_GXM_DEPTH_STENCIL_FORMAT_D16:bytes=2;break;
    case SCE_GXM_DEPTH_STENCIL_FORMAT_DF32:bytes=4;break;
    case SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8:bytes=4;separate=true;break;
    case SCE_GXM_DEPTH_STENCIL_FORMAT_S8:separate=true;break;
    case SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24:bytes=4;packed=true;break;
    // DF32M includes a guest mask bit. Its memory encoding must be established
    // before depth storage may overwrite that bit.
    default:return std::nullopt;
    }
    if(packed && surface.stencil_data && surface.stencil_data!=surface.depth_data) return std::nullopt;
    const uint64_t count=uint64_t(surface.get_stride())*(tiled?((h+31)&~uint64_t(31)):h);
    const uint64_t depth_size=surface.depth_data?count*bytes:0;
    const uint64_t stencil_size=surface.stencil_data?(separate?count:packed?count*4:0):0;
    if(uint64_t(surface.depth_data.address())+depth_size>UINT32_MAX
        || uint64_t(surface.stencil_data.address())+stencil_size>UINT32_MAX) return std::nullopt;
    return DepthMemoryLayout{uint32_t(w),uint32_t(h),surface.get_stride(),bytes,size_t(depth_size),size_t(stencil_size),tiled,packed};
}
static size_t depth_sample_offset(const DepthMemoryLayout &layout,uint32_t x,uint32_t y) {
    return layout.tiled?((size_t(y/32)*(layout.stride/32)+x/32)*1024)+(y%32)*32+x%32:size_t(y)*layout.stride+x;
}
std::optional<SurfaceRect> depth_subrectangle(const SceGxmDepthStencilSurface &surface, uint32_t width, uint32_t height,
    SceGxmMultisampleMode multisample, const SceGxmTexture &texture) {
    if (!surface.depth_data || !width || !height || surface.disabled()
        || multisample > SCE_GXM_MULTISAMPLE_4X) return std::nullopt;
    const auto base = gxm::get_base_format(gxm::get_format(texture));
    const auto format = surface.get_format();
    const bool compatible = (format == SCE_GXM_DEPTH_STENCIL_FORMAT_D16 && base == SCE_GXM_TEXTURE_BASE_FORMAT_U16)
        || ((format == SCE_GXM_DEPTH_STENCIL_FORMAT_DF32 || format == SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8)
            && base == SCE_GXM_TEXTURE_BASE_FORMAT_F32)
        || (format == SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24 && base == SCE_GXM_TEXTURE_BASE_FORMAT_X8U24);
    if (!compatible) return std::nullopt;
    const auto type = texture.texture_type();
    if (type != SCE_GXM_TEXTURE_LINEAR_STRIDED && texture.true_mip_count() > 1) return std::nullopt;
    const bool linear = surface.get_type() == SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR;
    if (linear ? (type != SCE_GXM_TEXTURE_LINEAR && type != SCE_GXM_TEXTURE_LINEAR_STRIDED)
               : (surface.get_type() != SCE_GXM_DEPTH_STENCIL_SURFACE_TILED || type != SCE_GXM_TEXTURE_TILED)) return std::nullopt;
    const uint64_t memory_width = uint64_t(width) * (multisample == SCE_GXM_MULTISAMPLE_4X ? 2 : 1);
    const uint64_t memory_height = uint64_t(height) * (multisample != SCE_GXM_MULTISAMPLE_NONE ? 2 : 1);
    if (memory_width > surface.get_stride()) return std::nullopt;
    const uint32_t view_width=gxm::get_width(texture), view_height=gxm::get_height(texture);
    const uint32_t bytes = base == SCE_GXM_TEXTURE_BASE_FORMAT_U16 ? 2 : 4;
    const uint64_t pitch = type == SCE_GXM_TEXTURE_LINEAR_STRIDED ? gxm::get_stride_in_bytes(texture)
        : ((uint64_t(view_width) + (linear ? 7 : 31)) & ~uint64_t(linear ? 7 : 31)) * bytes;
    if (pitch != uint64_t(surface.get_stride()) * bytes || !view_width || !view_height) return std::nullopt;
    const uint64_t address=uint64_t(texture.data_addr)<<2;
    if (address<surface.depth_data.address()) return std::nullopt;
    const uint64_t offset=address-surface.depth_data.address();
    if (offset%bytes) return std::nullopt;
    const uint64_t sample=offset/bytes;
    uint64_t x,y;
    if (linear) {
        x=sample%surface.get_stride(); y=sample/surface.get_stride();
    } else {
        // A tiled view starts at a complete 32x32 tile. An interior-tile
        // pointer cannot preserve the view's own tile addressing across rows.
        if (sample%1024) return std::nullopt;
        const uint64_t tile=sample/1024, tiles_per_row=surface.get_stride()/32;
        x=(tile%tiles_per_row)*32; y=(tile/tiles_per_row)*32;
    }
    if (x+view_width>memory_width || y+view_height>memory_height) return std::nullopt;
    return SurfaceRect{uint32_t(x),uint32_t(y),view_width,view_height};
}
bool depth_texture_matches(const SceGxmDepthStencilSurface &surface, uint32_t width, uint32_t height,
    SceGxmMultisampleMode multisample, const SceGxmTexture &texture) {
    const auto rect=depth_subrectangle(surface,width,height,multisample,texture);
    return rect && !rect->x && !rect->y
        && rect->width==uint64_t(width)*(multisample==SCE_GXM_MULTISAMPLE_4X ? 2 : 1)
        && rect->height==uint64_t(height)*(multisample!=SCE_GXM_MULTISAMPLE_NONE ? 2 : 1);
}
std::optional<SurfaceRect> surface_subrectangle(const SceGxmColorSurface &surface, const SceGxmTexture &texture) {
    const auto type=texture.texture_type();
    if (!surface.data || !surface.width || !surface.height || surface.strideInPixels<surface.width
        || (type!=SCE_GXM_TEXTURE_LINEAR_STRIDED && texture.true_mip_count()>1)) return std::nullopt;
    SceGxmColorBaseFormat color;
    const auto base=gxm::get_base_format(gxm::get_format(texture));
    const bool same_format=renderer::texture::convert_base_texture_format_to_base_color_format(base,color)
        && color==gxm::get_base_format(surface.colorFormat);
    if (!same_format && !surface_format_cast_supported(surface.colorFormat,base)) return std::nullopt;
    const unsigned bits=gxm::bits_per_pixel(base);
    if (!bits || bits%8) return std::nullopt;
    const uint64_t bytes=bits/8, stride=uint64_t(surface.strideInPixels)*bytes;
    const auto width=gxm::get_width(texture),height=gxm::get_height(texture);
    const uint64_t address=uint64_t(texture.data_addr)<<2;
    // A compact Morton block can be a rectangular subimage, provided every
    // coordinate bit has the same address weight in both image layouts.
    if (surface.surfaceType!=SCE_GXM_COLOR_SURFACE_LINEAR) {
        const bool swizzled=surface.surfaceType==SCE_GXM_COLOR_SURFACE_SWIZZLED
            && (type==SCE_GXM_TEXTURE_SWIZZLED || type==SCE_GXM_TEXTURE_SWIZZLED_ARBITRARY)
            && std::has_single_bit(uint32_t(width)) && std::has_single_bit(uint32_t(height))
            && std::has_single_bit(surface.width) && std::has_single_bit(surface.height)
            && surface.strideInPixels==surface.width;
        if (swizzled) {
            if (surface.width>16384 || surface.height>16384 || width>surface.width || height>surface.height
                || address<surface.data.address()) return std::nullopt;
            const uint32_t side=std::min(surface.width,surface.height);
            const uint64_t offset=address-surface.data.address(), count=uint64_t(width)*height;
            if (offset%bytes || offset/bytes+count>uint64_t(surface.width)*surface.height)
                return std::nullopt;
            for (uint32_t bit=1;bit<width;bit<<=1)
                if (texture::encode_morton(bit,0,width,height)!=texture::encode_morton(bit,0,surface.width,surface.height))
                    return std::nullopt;
            for (uint32_t bit=1;bit<height;bit<<=1)
                if (texture::encode_morton(0,bit,width,height)!=texture::encode_morton(0,bit,surface.width,surface.height))
                    return std::nullopt;
            const uint32_t pixel=offset/bytes;
            const uint32_t k=std::bit_width(side)-1, upper=(pixel>>(2*k))<<k;
            const uint32_t x=(texture::decode_morton2_x(pixel)&(side-1))|(surface.width>=surface.height?upper:0);
            const uint32_t y=(texture::decode_morton2_y(pixel)&(side-1))|(surface.width<surface.height?upper:0);
            if (uint64_t(x)+width>surface.width || uint64_t(y)+height>surface.height) return std::nullopt;
            // Aligned blocks have disjoint origin/coordinate bits. For other
            // origins, verify carries along each axis; the two Morton axes
            // occupy disjoint bits, so these checks also cover every pixel.
            if (pixel%count) {
                for (uint32_t dx=1;dx<width;++dx)
                    if (texture::encode_morton(x+dx,y,surface.width,surface.height)!=pixel+texture::encode_morton(dx,0,width,height))
                        return std::nullopt;
                for (uint32_t dy=1;dy<height;++dy)
                    if (texture::encode_morton(x,y+dy,surface.width,surface.height)!=pixel+texture::encode_morton(0,dy,width,height))
                        return std::nullopt;
            }
            return SurfaceRect{x,y,width,height};
        }
        const bool tiled=surface.surfaceType==SCE_GXM_COLOR_SURFACE_TILED && type==SCE_GXM_TEXTURE_TILED
            && surface.strideInPixels==((width+31)&~31u);
        if (tiled && address==surface.data.address() && width==surface.width && height==surface.height)
            return SurfaceRect{0,0,width,height};
        return std::nullopt;
    }
    if (type!=SCE_GXM_TEXTURE_LINEAR && type!=SCE_GXM_TEXTURE_LINEAR_STRIDED) return std::nullopt;
    const uint64_t texture_stride=type==SCE_GXM_TEXTURE_LINEAR_STRIDED?gxm::get_stride_in_bytes(texture):((uint64_t(width)+7)&~uint64_t(7))*bytes;
    if (!width || !height || texture_stride!=stride || address<surface.data.address()) return std::nullopt;
    const uint64_t offset=address-surface.data.address();
    if (offset%bytes) return std::nullopt;
    const uint64_t x=(offset%stride)/bytes,y=offset/stride;
    if (x+width>surface.width || y+height>surface.height) return std::nullopt;
    return SurfaceRect{uint32_t(x),uint32_t(y),width,height};
}
size_t surface_memory_size(const SceGxmColorSurface &surface) {
    const auto components = surface_components(surface.colorFormat);
    if (!components.count || !surface.width || !surface.height || surface.width > 16384 || surface.height > 16384
        || surface.strideInPixels < surface.width)
        return 0;
    size_t rows = surface.height;
    switch (surface.surfaceType) {
    case SCE_GXM_COLOR_SURFACE_LINEAR: break;
    case SCE_GXM_COLOR_SURFACE_TILED:
        if (surface.strideInPixels % 32) return 0;
        rows = (rows + 31) & ~size_t(31);
        break;
    case SCE_GXM_COLOR_SURFACE_SWIZZLED:
        if (surface.strideInPixels != surface.width || !std::has_single_bit(uint32_t(surface.width))
            || !std::has_single_bit(uint32_t(surface.height))) return 0;
        break;
    default: return 0;
    }
    return rows * surface.strideInPixels * components.count * components.bytes;
}
static bool raw_storage_matches(MTLPixelFormat actual, MTLPixelFormat expected) {
    return actual == expected || (expected == MTLPixelFormatRGBA8Unorm && actual == MTLPixelFormatRGBA8Unorm_sRGB);
}
bool read_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<uint8_t> destination) {
    const auto size = surface_memory_size(surface);
    const auto components = surface_components(surface.colorFormat);
    if (!size || size > destination.size() || !texture || !raw_storage_matches(texture.pixelFormat, components.native)
        || texture.textureType != MTLTextureType2D || texture.storageMode != MTLStorageModeShared
        || texture.width < surface.width || texture.height < surface.height) return false;
    Map mapping;
    try { mapping = surface_memory_mapping(surface.colorFormat); }
    catch (const std::runtime_error &) { return false; }
    std::array<uint32_t, 4> channel{};
    for (uint32_t c = 0; c < components.count; ++c) {
        const auto found = std::find(identity.begin(), identity.begin() + components.count, mapping[c]);
        if (found == identity.begin() + components.count) return false;
        channel[c] = uint32_t(found - identity.begin());
    }
    const size_t pixel_bytes = components.count * components.bytes;
    const size_t native_pixel_bytes = components.count == 3 ? 4 : pixel_bytes;
    const size_t native_stride = texture.width * native_pixel_bytes;
    std::vector<uint8_t> raw(native_stride * texture.height);
    [texture getBytes:raw.data() bytesPerRow:native_stride
        fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height) mipmapLevel:0];
    for (uint32_t y = 0; y < surface.height; ++y) for (uint32_t x = 0; x < surface.width; ++x) {
        size_t offset;
        if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_TILED)
            offset = ((size_t(y / 32) * (surface.strideInPixels / 32) + x / 32) * 1024) + (y % 32) * 32 + x % 32;
        else if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_SWIZZLED)
            offset = texture::encode_morton(x, y, surface.width, surface.height);
        else offset = size_t(y) * surface.strideInPixels + x;
        // Point-select the top-left native sample of each guest pixel, matching
        // the existing GL readback convention; never average packed data words.
        const auto *input = raw.data() + (size_t(y) * texture.height / surface.height) * native_stride
            + (size_t(x) * texture.width / surface.width) * native_pixel_bytes;
        auto *output = destination.data() + offset * pixel_bytes;
        for (uint32_t c = 0; c < components.count; ++c)
            std::memcpy(output + c * components.bytes, input + channel[c] * components.bytes, components.bytes);
    }
    return true;
}
bool write_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges) {
    const size_t size = surface_memory_size(surface);
    const auto components = surface_components(surface.colorFormat);
    if (!size || size > source.size() || !texture || !raw_storage_matches(texture.pixelFormat, components.native)
        || texture.textureType != MTLTextureType2D || texture.storageMode != MTLStorageModeShared
        || texture.width < surface.width || texture.height < surface.height) return false;
    size_t previous_end = 0;
    for (const auto &range : ranges) {
        if (!range.size || range.offset < previous_end || range.offset > size || range.size > size - range.offset)
            return false;
        previous_end = range.offset + range.size;
    }
    Map mapping;
    try { mapping = surface_memory_mapping(surface.colorFormat); }
    catch (const std::runtime_error &) { return false; }
    std::array<uint32_t,4> channel{};
    for (uint32_t c = 0; c < components.count; ++c) {
        const auto found = std::find(identity.begin(), identity.begin() + components.count, mapping[c]);
        if (found == identity.begin() + components.count) return false;
        channel[c] = uint32_t(found - identity.begin());
    }
    if (ranges.empty()) return true;
    const size_t pixel_bytes = components.count * components.bytes;
    const size_t native_pixel_bytes = components.count == 3 ? 4 : pixel_bytes;
    const size_t native_stride = texture.width * native_pixel_bytes;
    std::vector<uint8_t> raw(native_stride * texture.height);
    [texture getBytes:raw.data() bytesPerRow:native_stride
        fromRegion:MTLRegionMake2D(0,0,texture.width,texture.height) mipmapLevel:0];
    bool changed = false;
    for (uint32_t y = 0; y < surface.height; ++y) for (uint32_t x = 0; x < surface.width; ++x) {
        size_t pixel;
        if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_TILED)
            pixel = ((size_t(y / 32) * (surface.strideInPixels / 32) + x / 32) * 1024) + (y % 32) * 32 + x % 32;
        else if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_SWIZZLED)
            pixel = texture::encode_morton(x,y,surface.width,surface.height);
        else pixel = size_t(y) * surface.strideInPixels + x;
        const size_t offset = pixel * pixel_bytes;
        auto range = std::lower_bound(ranges.begin(), ranges.end(), offset,
            [](const auto &range, size_t offset) { return range.offset + range.size <= offset; });
        if (range == ranges.end() || range->offset >= offset + pixel_bytes) continue;
        std::array<bool,16> written{};
        for (; range != ranges.end() && range->offset < offset + pixel_bytes; ++range)
            for (size_t byte = std::max(offset,range->offset); byte < std::min(offset+pixel_bytes,range->offset+range->size); ++byte)
                written[byte-offset] = true;
        // Preserve every native pixel/component outside the written guest bytes.
        // These boundaries are the inverse of read_surface_memory's selection.
        for (size_t ny = size_t(y)*texture.height/surface.height; ny < size_t(y+1)*texture.height/surface.height; ++ny)
            for (size_t nx = size_t(x)*texture.width/surface.width; nx < size_t(x+1)*texture.width/surface.width; ++nx)
                for (size_t byte = 0; byte < pixel_bytes; ++byte) if (written[byte]) {
                    const size_t native_byte = channel[byte/components.bytes]*components.bytes + byte%components.bytes;
                    raw[ny*native_stride+nx*native_pixel_bytes+native_byte] = source[offset+byte];
                }
        changed = true;
    }
    if (changed) [texture replaceRegion:MTLRegionMake2D(0,0,texture.width,texture.height)
        mipmapLevel:0 withBytes:raw.data() bytesPerRow:native_stride];
    return true;
}
id<MTLTexture> rgba8_gamma_view(id<MTLTexture> texture, bool srgb) {
    if (!texture || (texture.pixelFormat != MTLPixelFormatRGBA8Unorm && texture.pixelFormat != MTLPixelFormatRGBA8Unorm_sRGB))
        throw std::runtime_error("Metal: gamma view requires RGBA8 storage");
    const auto format = srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
    if (texture.pixelFormat == format) return texture;
    auto view = [texture newTextureViewWithPixelFormat:format];
    if (!view) throw std::runtime_error("Metal: cannot create gamma texture view");
    return view;
}
id<MTLTexture> sampling_view(id<MTLTexture> texture, SceGxmTextureFormat format,
    const SceGxmColorFormat *rendered_format) {
    Map mapping = texture_mapping(format);
    if (rendered_format) {
        const auto memory = surface_memory_mapping(*rendered_format);
        for (auto &channel : mapping)
            for (size_t i = 0; i < 4; ++i)
                if (channel == identity[i]) { channel = memory[i]; break; }
    }
    if (mapping == identity) return texture;
    const NSUInteger slices = texture.textureType == MTLTextureTypeCube ? 6 : texture.arrayLength;
    id<MTLTexture> result = [texture newTextureViewWithPixelFormat:texture.pixelFormat
        textureType:texture.textureType levels:NSMakeRange(0, texture.mipmapLevelCount)
        slices:NSMakeRange(0, slices)
        swizzle:MTLTextureSwizzleChannelsMake(mapping[0], mapping[1], mapping[2], mapping[3])];
    if (!result) throw std::runtime_error("Metal: cannot create texture channel view");
    return result;
}

static MTLSamplerMinMagFilter sampler_filter(uint32_t value) {
    return value == SCE_GXM_TEXTURE_FILTER_LINEAR || value == SCE_GXM_TEXTURE_FILTER_MIPMAP_LINEAR
        ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
}
uint32_t effective_sampler_anisotropy(const SceGxmTexture &texture, uint32_t requested) {
    const uint32_t minimum = texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED
        ? texture.mag_filter : texture.min_filter;
    // Point-sampled surfaces may hold packed data. Keep renderer metadata and
    // the hardware sampler consistent when the global setting is higher.
    if (sampler_filter(minimum) == MTLSamplerMinMagFilterNearest
        && sampler_filter(texture.mag_filter) == MTLSamplerMinMagFilterNearest)
        return 1;
    return std::clamp(requested, 1u, 16u);
}
id<MTLSamplerState> make_sampler(Device &device, const SceGxmTexture &texture, uint32_t anisotropy) {
    auto desc = [MTLSamplerDescriptor new];
    // Strided descriptors reuse min_filter for the row stride. Their actual
    // minification filter is the same as their magnification filter.
    desc.minFilter = sampler_filter(texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED ? texture.mag_filter : texture.min_filter);
    desc.magFilter = sampler_filter(texture.mag_filter);
    desc.mipFilter = texture.true_mip_count() > 1 ? (texture.mip_filter ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest) : MTLSamplerMipFilterNotMipmapped;
    desc.lodMinClamp = texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED ? 0.f : float(texture.lod_min0 | (texture.lod_min1 << 2));
    desc.sAddressMode = texture.uaddr_mode == SCE_GXM_TEXTURE_ADDR_REPEAT ? MTLSamplerAddressModeRepeat : texture.uaddr_mode == SCE_GXM_TEXTURE_ADDR_MIRROR ? MTLSamplerAddressModeMirrorRepeat : MTLSamplerAddressModeClampToEdge;
    desc.tAddressMode = texture.vaddr_mode == SCE_GXM_TEXTURE_ADDR_REPEAT ? MTLSamplerAddressModeRepeat : texture.vaddr_mode == SCE_GXM_TEXTURE_ADDR_MIRROR ? MTLSamplerAddressModeMirrorRepeat : MTLSamplerAddressModeClampToEdge;
    desc.maxAnisotropy = effective_sampler_anisotropy(texture, anisotropy);
    auto result = [device.native_device() newSamplerStateWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot create texture sampler");
    return result;
}

SceGxmTexture texture_image_descriptor(const SceGxmTexture &texture) {
    auto image=texture;
    image.uaddr_mode=0;image.vaddr_mode=0;image.mag_filter=0;
    image.lod_min0=0;image.lod_min1=0;
    if (texture.texture_type()!=SCE_GXM_TEXTURE_LINEAR_STRIDED) {
        image.min_filter=0;image.mip_filter=0;image.lod_bias=0;
    }
    return image;
}

SceGxmTexture cube_texture_descriptor(const SceGxmTexture &texture) {
    auto result = texture;
    switch (texture.texture_type()) {
    case SCE_GXM_TEXTURE_SWIZZLED: result.type = SCE_GXM_TEXTURE_CUBE >> 29; break;
    case SCE_GXM_TEXTURE_SWIZZLED_ARBITRARY: result.type = SCE_GXM_TEXTURE_CUBE_ARBITRARY >> 29; break;
    case SCE_GXM_TEXTURE_CUBE: case SCE_GXM_TEXTURE_CUBE_ARBITRARY: break;
    default: throw std::runtime_error("Metal: cube shader requires swizzled face storage");
    }
    if (!gxm::get_width(result) || gxm::get_width(result) != gxm::get_height(result))
        throw std::runtime_error("Metal: cube faces must be square");
    return result;
}
size_t cube_texture_storage_size(const SceGxmTexture &texture) {
    const auto cube = cube_texture_descriptor(texture);
    const auto format = gxm::get_base_format(gxm::get_format(cube));
    const uint32_t bits = gxm::bits_per_pixel(format);
    const auto [bw,bh] = gxm::get_block_size(format);
    if (!bits || !bw || !bh) throw std::runtime_error("Metal: unknown cube storage format");
    const size_t width = gxm::get_width(cube), height = gxm::get_height(cube);
    size_t w = std::bit_ceil(width), h = std::bit_ceil(height), face = 0;
    do {
        face += ((w+bw-1)/bw)*((h+bh-1)/bh)*bw*bh*bits/8;
        if (cube.mip_count == 15) break;
        w /= 2; h /= 2;
    } while (w && h);
    size_t alignment = 4;
    if (cube.mip_count != 15 && ((width>=32 && height>=32 && (bits<=8 || gxm::is_block_compressed_format(format)))
        || (width>=16 && height>=16 && (bits==16 || bits==32)) || (width>=8 && height>=8 && bits==64)))
        alignment = 2048;
    return ((face+alignment-1)/alignment)*alignment*6;
}
size_t texture_storage_size(const SceGxmTexture &texture) {
    const auto type = texture.texture_type();
    if (type == SCE_GXM_TEXTURE_CUBE || type == SCE_GXM_TEXTURE_CUBE_ARBITRARY)
        return cube_texture_storage_size(texture);
    const auto width = gxm::get_width(texture), height = gxm::get_height(texture);
    const uint32_t mips = renderer::texture::get_upload_mip(texture.true_mip_count(),width,height);
    if (type == SCE_GXM_TEXTURE_LINEAR_STRIDED || mips <= 1)
        return gxm::texture_size_first_mip(texture);
    const auto format = gxm::get_base_format(gxm::get_format(texture));
    const auto [bw,bh] = gxm::get_block_size(format);
    const size_t aw = std::max(bw,type==SCE_GXM_TEXTURE_LINEAR?8u:type==SCE_GXM_TEXTURE_TILED?32u:1u);
    const size_t ah = std::max(bh,type==SCE_GXM_TEXTURE_TILED?32u:1u);
    const uint32_t bits = gxm::bits_per_pixel(format);
    if (!bits || !aw || !ah) throw std::runtime_error("Metal: invalid texture storage format");
    size_t size = 0, w = std::bit_ceil(size_t(width)), h = std::bit_ceil(size_t(height));
    for (uint32_t mip=0;mip<mips;++mip) {
        size += ((w+aw-1)/aw)*aw*((h+ah-1)/ah)*ah*bits/8;
        w/=2; h/=2;
    }
    return size;
}

bool surface_format_cast_supported(SceGxmColorFormat color, SceGxmTextureBaseFormat texture) {
    const auto source=surface_components(color), target=texture_components(texture);
    if (!source.count || !target.count || source.count*source.bytes != target.count*target.bytes) return false;
    try {
        const auto mapping=surface_memory_mapping(color);
        for (uint32_t c=0;c<source.count;++c)
            if (std::find(identity.begin(),identity.begin()+source.count,mapping[c])==identity.begin()+source.count) return false;
    } catch (const std::runtime_error &) { return false; }
    return true;
}

float packed_alias_x_offset(float scale, uint32_t native_alias_width) {
    if (scale < 1 || !native_alias_width) throw std::runtime_error("Metal: invalid packed alias viewport");
    return (scale-1)/(2.0f*float(native_alias_width));
}

// Reinterpret storage bits for copies. In particular, do not decode/re-encode
// sRGB, normalize signed bytes, or canonicalize floating NaN payloads.
static MTLPixelFormat sample_bits_format(MTLPixelFormat format) {
    switch(format) {
    case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm: return MTLPixelFormatR8Uint;
    case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Snorm: return MTLPixelFormatRG8Uint;
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB: case MTLPixelFormatRGBA8Snorm: return MTLPixelFormatRGBA8Uint;
    case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Float: return MTLPixelFormatR16Uint;
    case MTLPixelFormatRG16Unorm: case MTLPixelFormatRG16Snorm: case MTLPixelFormatRG16Float: return MTLPixelFormatRG16Uint;
    case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm: case MTLPixelFormatRGBA16Float: return MTLPixelFormatRGBA16Uint;
    case MTLPixelFormatR32Float: return MTLPixelFormatR32Uint;
    case MTLPixelFormatRG32Float: return MTLPixelFormatRG32Uint;
    case MTLPixelFormatRGBA32Float: return MTLPixelFormatRGBA32Uint;
    default: return MTLPixelFormatInvalid;
    }
}
static id<MTLComputePipelineState> cached_compute(Device &device, id<MTLLibrary> library, NSString *name) {
    std::string error;
    auto pipeline = device.create_compute_pipeline([library newFunctionWithName:name], error);
    if (!pipeline) throw std::runtime_error(error);
    return pipeline;
}
SurfaceCaster::SurfaceCaster(Device &device) : device(device) {
    NSString *source = @R"(#include <metal_stdlib>
using namespace metal;
kernel void unpack_words(texture2d<uint, access::read> input [[texture(0)]],
    texture2d<uint, access::write> output [[texture(1)]], constant uint4 &config [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= output.get_width() || p.y >= output.get_height()) return;
    // Reinterpret each native RG32 pixel's adjacent words. Stretching each
    // guest word separately mixes color and normal under interpolated UVs.
    const uint row_words = output.get_width();
    const uint address = p.y*row_words + p.x + config.z;
    const uint x = (address%row_words)/2;
    const uint y = address/row_words;
    // The shifted alias can extend past the final rendered word. Do not read
    // beyond the GPU surface; the unsupported trailing word is deterministic.
    const uint word = y < input.get_height() ? input.read(uint2(x,y))[(address%2) ^ config.y] : 0;
    output.write(uint4(word&255, (word>>8)&255, (word>>16)&255, word>>24),p);
}
kernel void repack_components(texture2d<uint,access::read> input [[texture(0)]],
    texture2d<uint,access::write> output [[texture(1)]], constant uint4 &config [[buffer(0)]],
    constant uint4 &mapping [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint4 source=input.read(p);
    uint4 result(0);
    for (uint c=0;c<config.z;++c) for (uint b=0;b<config.w;++b) {
        const uint byte=c*config.w+b;
        const uint word=source[mapping[byte/config.y]];
        result[c] |= ((word >> ((byte%config.y)*8)) & 255u) << (b*8);
    }
    output.write(result,p);
}
kernel void copy_depth(texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x < output.get_width() && p.y < output.get_height()) output.write(input.read(p),p);
}
kernel void decode_rg_gamma(texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= output.get_width() || p.y >= output.get_height()) return;
    float4 value = input.read(p);
    value.rg = select(pow((value.rg + 0.055f) / 1.055f, float2(2.4f)),
        value.rg / 12.92f, value.rg <= 0.04045f);
    output.write(value, p);
}
uint3 sample_address(uint2 p, uint scale, uint samples) {
    const uint2 extent(samples/2,2), guest=p/scale;
    const uint sample=(guest.y%2)*extent.x+guest.x%extent.x;
    return uint3((guest/extent)*scale+p%scale,sample);
}
kernel void expand_samples(texture2d_ms<float,access::read> input [[texture(0)]],
    texture2d<float,access::write> output [[texture(1)]], constant uint &scale [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint3 q=sample_address(p,scale,input.get_num_samples());
    output.write(input.read(q.xy,q.z),p);
}
kernel void expand_integer_samples(texture2d_ms<uint,access::read> input [[texture(0)]],
    texture2d<uint,access::write> output [[texture(1)]], constant uint &scale [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint3 q=sample_address(p,scale,input.get_num_samples());
    output.write(input.read(q.xy,q.z),p);
}
kernel void expand_depth(depth2d_ms<float,access::read> input [[texture(0)]],
    texture2d<float,access::write> output [[texture(1)]], constant uint &scale [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint3 q=sample_address(p,scale,input.get_num_samples());
    output.write(float4(input.read(q.xy,q.z)),p);
}
struct DepthReadback { float depth; uint stencil; };
kernel void store_depth_samples(depth2d<float,access::read> input [[texture(0)]],
    texture2d<uint,access::read> stencil [[texture(1)]], device DepthReadback *output [[buffer(0)]],
    constant uint4 &config [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if(p.x>=config.x || p.y>=config.y) return;
    const uint2 q=p*config.z;
    output[p.y*config.x+p.x]={config.w&1 ? input.read(q) : 0.f,config.w&2 ? stencil.read(q).x : 0u};
}
kernel void store_depth_samples_ms(depth2d_ms<float,access::read> input [[texture(0)]],
    texture2d_ms<uint,access::read> stencil [[texture(1)]], device DepthReadback *output [[buffer(0)]],
    constant uint4 &config [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if(p.x>=config.x || p.y>=config.y) return;
    const uint3 q=sample_address(p*config.z,config.z,input.get_num_samples());
    output[p.y*config.x+p.x]={config.w&1 ? input.read(q.xy,q.z) : 0.f,config.w&2 ? stencil.read(q.xy,q.z).x : 0u};
}
struct DepthSeed { float depth [[depth(any)]]; uint stencil [[stencil]]; };
fragment DepthSeed seed_depth(float4 position [[position]],uint sample [[sample_id]],
    texture2d<float,access::read> depths [[texture(0)]],texture2d<uint,access::read> stencils [[texture(1)]],
    constant uint2 &config [[buffer(0)]]) {
    uint2 p=uint2(position.xy)/config.x;
    if(config.y>1) { uint2 extent(config.y/2,2);p=p*extent+uint2(sample%extent.x,sample/extent.x); }
    return {depths.read(p).x,stencils.read(p).x};
}
vertex float4 seed_vs(uint id [[vertex_id]]) {
    const float2 p[]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    return float4(p[id],0,1);
}
fragment float4 seed_fs(float4 position [[position]], uint sample [[sample_id]],
    texture2d<float,access::read> source [[texture(0)]], constant uint4 &config [[buffer(0)]]) {
    uint2 p=uint2(position.xy);
    if(config.z) {
        const uint2 extent(config.y/2,2), offset(sample%extent.x,sample/extent.x);
        p=((p/config.x)*extent+offset)*config.x+p%config.x;
    }
    return source.read(p);
}
fragment uint4 seed_fs_uint(float4 position [[position]], uint sample [[sample_id]],
    texture2d<uint,access::read> source [[texture(0)]], constant uint4 &config [[buffer(0)]]) {
    uint2 p=uint2(position.xy);
    if(config.z) {
        const uint2 extent(config.y/2,2), offset(sample%extent.x,sample/extent.x);
        p=((p/config.x)*extent+offset)*config.x+p%config.x;
    }
    return source.read(p);
})";
    NSError *error = nil;
    auto library = [device.native_device() newLibraryWithSource:source options:nil error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal: surface cast shader failed");
    pipeline = cached_compute(device, library, @"unpack_words");
    if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal: surface cast pipeline failed");
    depth_pipeline = cached_compute(device, library, @"copy_depth");
    if (!depth_pipeline) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal: depth copy pipeline failed");
    depth_store_pipeline=cached_compute(device, library, @"store_depth_samples");
    depth_store_ms_pipeline=cached_compute(device, library, @"store_depth_samples_ms");
    if(!depth_store_pipeline || !depth_store_ms_pipeline) throw std::runtime_error("Metal: depth readback pipeline failed");
    multisample_library=library;
    multisample_pipeline=cached_compute(device, library, @"expand_samples");
    multisample_depth_pipeline=cached_compute(device, library, @"expand_depth");
    multisample_integer_pipeline=cached_compute(device, library, @"expand_integer_samples");
    if (!multisample_pipeline || !multisample_depth_pipeline || !multisample_integer_pipeline) throw std::runtime_error("Metal: multisample copy pipeline failed");
}
void SurfaceCaster::expand_multisample(id<MTLTexture> source, id<MTLTexture> destination, uint32_t scale) {
    if (!scale || !source || source.textureType!=MTLTextureType2DMultisample || (source.sampleCount!=2 && source.sampleCount!=4)
        || source.width%scale || source.height%scale || !destination || destination.textureType!=MTLTextureType2D
        || destination.width!=source.width*(source.sampleCount/2) || destination.height!=source.height*2)
        throw std::runtime_error("Metal: invalid sample expansion");
    const auto bits=sample_bits_format(source.pixelFormat);
    const bool integer=bits!=MTLPixelFormatInvalid && bits==sample_bits_format(destination.pixelFormat);
    if (integer) {
        source=[source newTextureViewWithPixelFormat:bits]; destination=[destination newTextureViewWithPixelFormat:bits];
        if (!source || !destination) throw std::runtime_error("Metal: cannot view MSAA storage bits");
    }
    auto commands=[device.command_queue() commandBuffer]; auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:integer ? multisample_integer_pipeline
        : source.pixelFormat==MTLPixelFormatDepth32Float_Stencil8 ? multisample_depth_pipeline : multisample_pipeline];
    [encoder setTexture:source atIndex:0]; [encoder setTexture:destination atIndex:1];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(destination.width,destination.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding]; std::string error;
    if(!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
}
void SurfaceCaster::seed_multisample(id<MTLTexture> source, id<MTLTexture> destination, uint32_t scale, bool expanded) {
    if (!scale || !source || source.textureType!=MTLTextureType2D || !destination || destination.textureType!=MTLTextureType2DMultisample
        || (destination.sampleCount!=2 && destination.sampleCount!=4) || destination.width%scale || destination.height%scale
        || source.width!=destination.width*(expanded ? destination.sampleCount/2 : 1) || source.height!=destination.height*(expanded ? 2 : 1))
        throw std::runtime_error("Metal: invalid sample seed");
    const auto bits=sample_bits_format(source.pixelFormat);
    const bool integer=bits!=MTLPixelFormatInvalid && bits==sample_bits_format(destination.pixelFormat);
    if (integer) {
        source=[source newTextureViewWithPixelFormat:bits]; destination=[destination newTextureViewWithPixelFormat:bits];
        if (!source || !destination) throw std::runtime_error("Metal: cannot view sample seed storage bits");
    }
    auto &pipeline=seed_pipelines[{uint32_t(destination.pixelFormat),uint32_t(destination.sampleCount)}];
    std::string error;
    if (!pipeline) {
        auto desc=[MTLRenderPipelineDescriptor new];
        desc.vertexFunction=[multisample_library newFunctionWithName:@"seed_vs"];
        desc.fragmentFunction=[multisample_library newFunctionWithName:integer ? @"seed_fs_uint" : @"seed_fs"];
        desc.colorAttachments[0].pixelFormat=destination.pixelFormat;
        desc.rasterSampleCount=destination.sampleCount;
        pipeline=device.create_pipeline(desc,error);
        if (!pipeline) throw std::runtime_error(error);
    }
    auto commands=[device.command_queue() commandBuffer]; auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture=destination;
    pass.colorAttachments[0].loadAction=MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline]; [encoder setFragmentTexture:source atIndex:0];
    const uint32_t config[]={scale,uint32_t(destination.sampleCount),uint32_t(expanded),0};
    [encoder setFragmentBytes:config length:sizeof(config) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [encoder endEncoding];
    if(!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
}
bool SurfaceCaster::patch_multisample(id<MTLTexture> texture, const SceGxmColorSurface &surface, uint32_t scale,
    std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges) {
    const size_t size=surface_memory_size(surface);
    const auto components=surface_components(surface.colorFormat);
    if (!size || source.size()<size || !scale || !texture || texture.textureType!=MTLTextureType2DMultisample
        || (texture.sampleCount!=2 && texture.sampleCount!=4) || !raw_storage_matches(texture.pixelFormat,components.native)
        || texture.width*(surface.downscale ? 1 : texture.sampleCount/2)!=uint64_t(surface.width)*scale
        || texture.height*(surface.downscale ? 1 : 2)!=uint64_t(surface.height)*scale) return false;
    size_t previous_end=0;
    for (const auto &range:ranges) {
        if (!range.size || range.offset<previous_end || range.offset>size || range.size>size-range.offset) return false;
        previous_end=range.offset+range.size;
    }
    if (ranges.empty()) return true;
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat
        width:texture.width*(texture.sampleCount/2) height:texture.height*2 mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
    auto packed=[device.native_device() newTextureWithDescriptor:desc];
    if (!packed) throw std::runtime_error("Metal: cannot allocate sample patch image");
    expand_multisample(texture,packed,scale);
    // For resolved color the original guest descriptor maps each changed byte
    // to every sample of that pixel. Expanded color addresses samples separately.
    // The existing byte writer preserves all other components and native pixels.
    if (!write_surface_memory(packed,surface,source,ranges)) return false;
    seed_multisample(packed,texture,scale,true);
    return true;
}
id<MTLTexture> SurfaceCaster::cube_snapshot(id<MTLTexture> uploaded, std::span<const CubeSurface> surfaces, uint32_t scale) {
    if (!uploaded || uploaded.textureType != MTLTextureTypeCube)
        throw std::runtime_error("Metal: invalid rendered cube source");
    return texture_snapshot(uploaded,surfaces,scale);
}
id<MTLTexture> SurfaceCaster::texture_snapshot(id<MTLTexture> uploaded, std::span<const CubeSurface> surfaces, uint32_t scale) {
    if (!uploaded || (uploaded.textureType != MTLTextureTypeCube && uploaded.textureType != MTLTextureType2D) || !scale)
        throw std::runtime_error("Metal: invalid rendered texture source");
    const bool cube=uploaded.textureType==MTLTextureTypeCube;
    if (!cube_pipeline) {
        NSError *error = nil;
        auto library = [device.native_device() newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;
kernel void copy_cube_face(texture2d<float,access::read> source [[texture(0)]],
    texture2d<float,access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    uint2 size=uint2(output.get_width(),output.get_height());
    if(any(p>=size)) return;
    // Map pixel centers. Corner mapping widens the first texel when an NPOT
    // mip has more destination pixels than the rendered guest level * scale.
    uint2 coord=((2*p+1)*uint2(source.get_width(),source.get_height()))/(2*size);
    output.write(source.read(coord),p);
})" options:nil error:&error];
        if (!library) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal: cube copy compilation failed");
        cube_pipeline = cached_compute(device, library, @"copy_cube_face");
        if (!cube_pipeline) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal: cube copy pipeline failed");
    }
    auto desc = cube
        ? [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float size:uploaded.width*scale mipmapped:NO]
        : [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:uploaded.width*scale height:uploaded.height*scale mipmapped:NO];
    desc.mipmapLevelCount=uploaded.mipmapLevelCount;
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
    auto result=[device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: rendered cube allocation failed");
    auto commands=[device.command_queue() commandBuffer];
    auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:cube_pipeline];
    for (uint32_t face=0;face<(cube?6u:1u);++face) for (uint32_t mip=0;mip<uploaded.mipmapLevelCount;++mip) {
        id<MTLTexture> source=nil;
        for (const auto &surface:surfaces) if (surface.face==face && surface.mip==mip) source=surface.texture;
        if (!source) source=[uploaded newTextureViewWithPixelFormat:uploaded.pixelFormat textureType:MTLTextureType2D
            levels:NSMakeRange(mip,1) slices:NSMakeRange(face,1)];
        auto output=[result newTextureViewWithPixelFormat:result.pixelFormat textureType:MTLTextureType2D
            levels:NSMakeRange(mip,1) slices:NSMakeRange(face,1)];
        if (!source || !output) throw std::runtime_error("Metal: cannot create cube face view");
        [encoder setTexture:source atIndex:0];[encoder setTexture:output atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(output.width,output.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    }
    [encoder endEncoding];
    std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}

id<MTLTexture> SurfaceCaster::color_snapshot(id<MTLTexture> source) {
    if (!source || source.textureType != MTLTextureType2D || source.sampleCount != 1)
        throw std::runtime_error("Metal: unsupported color feedback source");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
        width:source.width height:source.height mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate color feedback snapshot");
    auto commands = [device.command_queue() commandBuffer];
    auto encoder = [commands blitCommandEncoder];
    [encoder copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
        sourceSize:MTLSizeMake(source.width,source.height,1) toTexture:result destinationSlice:0
        destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [encoder endEncoding];
    std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}
id<MTLTexture> SurfaceCaster::snapshot_subrectangle(id<MTLTexture> source, uint32_t width, uint32_t height, SurfaceRect rect) {
    if (!source || source.textureType!=MTLTextureType2D || source.sampleCount!=1
        || !width || !height || source.width<width || source.height<height
        || !rect.width || !rect.height || uint64_t(rect.x)+rect.width>width || uint64_t(rect.y)+rect.height>height)
        throw std::runtime_error("Metal: invalid snapshot subrectangle");
    const size_t x=uint64_t(rect.x)*source.width/width,y=uint64_t(rect.y)*source.height/height;
    const size_t w=(uint64_t(rect.x)+rect.width)*source.width/width-x;
    const size_t h=(uint64_t(rect.y)+rect.height)*source.height/height-y;
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat width:w height:h mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
    auto result=[device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate snapshot subrectangle");
    auto commands=[device.command_queue() commandBuffer]; auto encoder=[commands blitCommandEncoder];
    [encoder copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x,y,0)
        sourceSize:MTLSizeMake(w,h,1) toTexture:result destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [encoder endEncoding]; std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}
id<MTLTexture> SurfaceCaster::color_subrectangle(id<MTLTexture> source, const SceGxmColorSurface &surface, SurfaceRect rect) {
    return snapshot_subrectangle(source,surface.width,surface.height,rect);
}
id<MTLTexture> SurfaceCaster::rgba8_from_rg32(id<MTLTexture> source, uint32_t scale, bool swap_words,
    uint32_t word_offset, bool signed_normalized) {
    if (!scale || source.pixelFormat != MTLPixelFormatRG32Float || source.width % scale || source.height % scale || word_offset > 1)
        throw std::runtime_error("Metal: invalid RG32 surface cast dimensions/format");
    auto words = [source newTextureViewWithPixelFormat:MTLPixelFormatRG32Uint];
    if (!words) throw std::runtime_error("Metal: cannot view packed RG32 color bits");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:signed_normalized ? MTLPixelFormatRGBA8Snorm : MTLPixelFormatRGBA8Unorm width:source.width*2 height:source.height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    desc.storageMode = MTLStorageModeShared;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate reinterpreted color surface");
    auto bytes = [result newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Uint];
    if (!bytes) throw std::runtime_error("Metal: cannot write raw reinterpreted color bytes");
    auto commands = [device.command_queue() commandBuffer];
    auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:words atIndex:0];
    [encoder setTexture:bytes atIndex:1];
    const uint32_t config[] = {scale, uint32_t(swap_words), word_offset, 0};
    [encoder setBytes:config length:sizeof(config) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}
id<MTLTexture> SurfaceCaster::surface_format_cast(id<MTLTexture> source, SceGxmColorFormat color, SceGxmTextureBaseFormat texture) {
    const auto input=surface_components(color), output=texture_components(texture);
    if (!source || source.textureType!=MTLTextureType2D || source.sampleCount!=1
        || !surface_format_cast_supported(color,texture) || !raw_storage_matches(source.pixelFormat,input.native)) return nil;
    const auto mapping=surface_memory_mapping(color);
    // Ordinary equal-sized Metal formats can share storage. An identity guest
    // order needs only a typed view, without allocating or dispatching a copy.
    if (std::equal(mapping.begin(),mapping.begin()+input.count,identity.begin())) {
        if (source.pixelFormat==output.native) return source;
        auto view=[source newTextureViewWithPixelFormat:output.native];
        if (view) return view;
    }
    if (!component_cast_pipeline) {
        component_cast_pipeline=cached_compute(device, multisample_library, @"repack_components");
    }
    std::array<uint32_t,4> channels{};
    for (uint32_t c=0;c<input.count;++c)
        channels[c]=uint32_t(std::find(identity.begin(),identity.end(),mapping[c])-identity.begin());
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:output.native width:source.width height:source.height mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
    auto result=[device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: component cast allocation failed");
    auto raw_input=[source newTextureViewWithPixelFormat:sample_bits_format(source.pixelFormat)];
    auto raw_output=[result newTextureViewWithPixelFormat:sample_bits_format(output.native)];
    if (!raw_input || !raw_output) throw std::runtime_error("Metal: component cast integer view failed");
    auto commands=[device.command_queue() commandBuffer]; auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:component_cast_pipeline];
    [encoder setTexture:raw_input atIndex:0];[encoder setTexture:raw_output atIndex:1];
    const uint32_t config[]={input.count,input.bytes,output.count,output.bytes};
    [encoder setBytes:config length:sizeof(config) atIndex:0];
    [encoder setBytes:channels.data() length:sizeof(channels) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}

id<MTLTexture> SurfaceCaster::rgba8_memory_snapshot(id<MTLTexture> source, SceGxmColorFormat format) {
    return rgba8_surface_sampling(source, format, 0);
}
id<MTLTexture> SurfaceCaster::rgba8_surface_sampling(id<MTLTexture> source, SceGxmColorFormat format, uint32_t gamma) {
    if (gamma != 0 && gamma != 1 && gamma != 3)
        throw std::runtime_error("Metal: invalid RGBA8 surface texture gamma mode");
    if (gamma == 3 && !rg_gamma_pipeline) {
        rg_gamma_pipeline = cached_compute(device, multisample_library, @"decode_rg_gamma");
    }
    auto memory = sampling_view(rgba8_gamma_view(source,false), SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR, &format);
    // Float storage keeps untouched B/A values and avoids a second UNORM or
    // half conversion before filtering the partially decoded RG channels.
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:gamma == 3 ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA8Unorm
        width:source.width height:source.height mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate encoded memory-channel image");
    auto commands = [device.command_queue() commandBuffer]; auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:gamma == 3 ? rg_gamma_pipeline : depth_pipeline];
    [encoder setTexture:memory atIndex:0]; [encoder setTexture:result atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding]; std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return gamma == 1 ? rgba8_gamma_view(result, true) : result;
}
id<MTLTexture> SurfaceCaster::sampling_snapshot(id<MTLTexture> source, uint32_t mip, uint32_t face) {
    if (!source || source.sampleCount != 1 || mip >= source.mipmapLevelCount
        || (source.textureType != MTLTextureType2D && source.textureType != MTLTextureTypeCube)
        || face >= (source.textureType == MTLTextureTypeCube ? 6u : 1u)) return nil;
    switch (source.pixelFormat) {
    case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm:
    case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Snorm: case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Float:
    case MTLPixelFormatRG16Unorm: case MTLPixelFormatRG16Snorm: case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm: case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatR32Float: case MTLPixelFormatRG32Float: case MTLPixelFormatRGBA32Float:
    case MTLPixelFormatRG11B10Float: break;
    default: return nil;
    }
    if (source.textureType == MTLTextureTypeCube || mip != 0) {
        source = [source newTextureViewWithPixelFormat:source.pixelFormat textureType:MTLTextureType2D
            levels:NSMakeRange(mip,1) slices:NSMakeRange(face,1) swizzle:source.swizzle];
        if (!source) return nil;
    }
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
        width:source.width height:source.height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModeShared;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate sampled texture diagnostic");
    auto commands = [device.command_queue() commandBuffer]; auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:depth_pipeline];
    [encoder setTexture:source atIndex:0]; [encoder setTexture:result atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding]; std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}
static bool depth_memory_valid(id<MTLTexture> texture,const DepthMemoryLayout &layout,uint32_t scale,size_t depth,size_t stencil) {
    if(!texture || !scale || texture.pixelFormat!=MTLPixelFormatDepth32Float_Stencil8
        || (texture.sampleCount!=1 && texture.sampleCount!=2 && texture.sampleCount!=4)
        || depth<layout.depth_size || stencil<layout.stencil_size) return false;
    const uint32_t sx=texture.sampleCount==4?2:1,sy=texture.sampleCount>1?2:1;
    return texture.width*sx==uint64_t(layout.width)*scale && texture.height*sy==uint64_t(layout.height)*scale;
}
bool SurfaceCaster::load_depth_memory(id<MTLTexture> texture,const SceGxmDepthStencilSurface &surface,const DepthMemoryLayout &layout,uint32_t scale,
    std::span<const uint8_t> depth,std::span<const uint8_t> stencil) {
    if(!depth_memory_valid(texture,layout,scale,depth.size(),stencil.size())) return false;
    std::vector<float> depths(size_t(layout.width)*layout.height,surface.background_depth);
    std::vector<uint8_t> stencils(depths.size(),surface.stencil);
    for(uint32_t y=0;y<layout.height;++y) for(uint32_t x=0;x<layout.width;++x) {
        const size_t address=depth_sample_offset(layout,x,y),i=size_t(y)*layout.width+x;
        if(layout.depth_size) {
            if(layout.depth_bytes==2) { uint16_t v;std::memcpy(&v,depth.data()+address*2,2);depths[i]=float(v)/65535.f; }
            else if(layout.packed) {uint32_t v;std::memcpy(&v,depth.data()+address*4,4);depths[i]=float(v&0xffffff)/16777215.f;stencils[i]=v>>24;}
            else std::memcpy(&depths[i],depth.data()+address*4,4);
        }
        if(layout.stencil_size) stencils[i]=stencil[address*(layout.packed?4:1)+(layout.packed?3:0)];
    }
    auto make=[&](MTLPixelFormat format,const void *data,uint32_t bytes) {
        auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:layout.width height:layout.height mipmapped:NO];
        desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead;
        auto image=[device.native_device() newTextureWithDescriptor:desc];
        if(!image) throw std::runtime_error("Metal: depth load staging allocation failed");
        [image replaceRegion:MTLRegionMake2D(0,0,layout.width,layout.height) mipmapLevel:0 withBytes:data bytesPerRow:layout.width*bytes];return image;
    };
    auto d=make(MTLPixelFormatR32Float,depths.data(),4),s=make(MTLPixelFormatR8Uint,stencils.data(),1);
    auto &pipeline=depth_seed_pipelines[uint32_t(texture.sampleCount)];std::string error;
    if(!pipeline) {
        auto desc=[MTLRenderPipelineDescriptor new];desc.vertexFunction=[multisample_library newFunctionWithName:@"seed_vs"];
        desc.fragmentFunction=[multisample_library newFunctionWithName:@"seed_depth"];
        desc.depthAttachmentPixelFormat=desc.stencilAttachmentPixelFormat=MTLPixelFormatDepth32Float_Stencil8;
        desc.rasterSampleCount=texture.sampleCount;pipeline=device.create_pipeline(desc,error);
        if(!pipeline) throw std::runtime_error(error);
    }
    auto commands=[device.command_queue() commandBuffer];auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
    pass.depthAttachment.texture=pass.stencilAttachment.texture=texture;
    pass.depthAttachment.loadAction=pass.stencilAttachment.loadAction=MTLLoadActionDontCare;
    pass.depthAttachment.storeAction=pass.stencilAttachment.storeAction=MTLStoreActionStore;
    auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
    auto state=[MTLDepthStencilDescriptor new];state.depthCompareFunction=MTLCompareFunctionAlways;state.depthWriteEnabled=YES;
    auto stencil_state=[MTLStencilDescriptor new];stencil_state.stencilCompareFunction=MTLCompareFunctionAlways;
    stencil_state.depthStencilPassOperation=MTLStencilOperationReplace;state.frontFaceStencil=state.backFaceStencil=stencil_state;
    [encoder setDepthStencilState:[device.native_device() newDepthStencilStateWithDescriptor:state]];
    [encoder setRenderPipelineState:pipeline];[encoder setFragmentTexture:d atIndex:0];[encoder setFragmentTexture:s atIndex:1];
    const uint32_t config[]={scale,uint32_t(texture.sampleCount)};[encoder setFragmentBytes:config length:sizeof(config) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[encoder endEncoding];
    if(!device.submit_and_wait(commands,error)) throw std::runtime_error(error);return true;
}
bool SurfaceCaster::store_depth_memory(id<MTLTexture> texture,const SceGxmDepthStencilSurface &surface,const DepthMemoryLayout &layout,uint32_t scale,
    std::span<uint8_t> depth,std::span<uint8_t> stencil) {
    if(!depth_memory_valid(texture,layout,scale,depth.size(),stencil.size())) return false;
    if(!layout.depth_size && !layout.stencil_size) return true;
    // Read exactly the native sample previously selected by the CPU from the
    // expanded images. A linear shared buffer avoids two full-resolution texture
    // readbacks, their tiling conversion, and a second command-buffer wait.
    struct DepthReadback { float depth; uint32_t stencil; };
    static_assert(sizeof(DepthReadback)==8);
    const size_t bytes=size_t(layout.width)*layout.height*sizeof(DepthReadback);
    if(bytes>device.native_device().maxBufferLength) return false;
    id<MTLBuffer> buffer=depth_store_buffer;
    if(!buffer || buffer.length<bytes) {
        buffer=[device.native_device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if(!buffer) throw std::runtime_error("Metal: depth readback allocation failed");
        if(bytes<=64*1024*1024) depth_store_buffer=buffer;
    }
    auto view=[texture newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    if(!view) throw std::runtime_error("Metal: cannot view stencil for guest storage");
    const uint32_t flags=uint32_t(layout.depth_size!=0)
        | (uint32_t(layout.stencil_size || (layout.packed && layout.depth_size))<<1);
    const uint32_t config[]={layout.width,layout.height,scale,flags};
    auto commands=[device.command_queue() commandBuffer];auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:texture.sampleCount>1?depth_store_ms_pipeline:depth_store_pipeline];
    [encoder setTexture:texture atIndex:0];[encoder setTexture:view atIndex:1];
    [encoder setBuffer:buffer offset:0 atIndex:0];[encoder setBytes:config length:sizeof(config) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(layout.width,layout.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];std::string error;
    if(!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    // This buffer is reused only by subsequent synchronous calls, after the CPU
    // has consumed it. Guest rounding, tiling, stencil overrides and padding
    // handling below intentionally remain identical to the original readback.
    const auto *values=static_cast<const DepthReadback *>(buffer.contents);
    for(uint32_t y=0;y<layout.height;++y) for(uint32_t x=0;x<layout.width;++x) {
        const size_t address=depth_sample_offset(layout,x,y),i=size_t(y)*layout.width+x;
        if(layout.depth_size) {
            if(layout.depth_bytes==2) {const uint16_t v=uint16_t(std::lround(std::clamp(double(values[i].depth),0.,1.)*65535.));std::memcpy(depth.data()+address*2,&v,2);}
            else if(layout.packed) {
                const uint32_t v=uint32_t(std::llround(std::clamp(double(values[i].depth),0.,1.)*16777215.))|(uint32_t(values[i].stencil)<<24);
                std::memcpy(depth.data()+address*4,&v,4);
            } else std::memcpy(depth.data()+address*4,&values[i].depth,4);
        }
        if(layout.stencil_size) stencil[address*(layout.packed?4:1)+(layout.packed?3:0)]=values[i].stencil;
    }
    return true;
}
id<MTLTexture> SurfaceCaster::depth_snapshot(id<MTLTexture> source, bool normalized16, uint32_t scale) {
    const bool ms=source && source.textureType==MTLTextureType2DMultisample && (source.sampleCount==2 || source.sampleCount==4);
    if (!source || (!ms && (source.textureType != MTLTextureType2D || source.sampleCount != 1))
        || source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)
        throw std::runtime_error("Metal: invalid depth snapshot source");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:normalized16 ? MTLPixelFormatR16Unorm : MTLPixelFormatR32Float width:source.width*(ms ? source.sampleCount/2 : 1) height:source.height*(ms ? 2 : 1) mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    desc.storageMode = MTLStorageModeShared;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate depth snapshot");
    if (ms) { expand_multisample(source,result,scale); return result; }
    auto commands = [device.command_queue() commandBuffer];
    auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:depth_pipeline];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:result atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}
}
