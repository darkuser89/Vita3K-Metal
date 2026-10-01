// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/textures.h>
#include <renderer/metal/device.h>
#include <renderer/functions.h>
#include <renderer/gxm_types.h>
#include <gxm/functions.h>
#include <util/float_to_half.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <bit>
#include <cstring>
#include <cmath>
#include <vector>

namespace renderer::metal {
namespace {
id<MTLCommandBuffer> surface_command_buffer(Device &device, NSString *label) {
    id<MTLCommandBuffer> commands = [device.command_queue() commandBuffer];
    commands.label = label;
    return commands;
}
using Map = std::array<MTLTextureSwizzle, 4>;
constexpr auto R = MTLTextureSwizzleRed, G = MTLTextureSwizzleGreen;
constexpr auto B = MTLTextureSwizzleBlue, A = MTLTextureSwizzleAlpha;
constexpr auto Z = MTLTextureSwizzleZero, O = MTLTextureSwizzleOne;
constexpr Map identity{R,G,B,A};
constexpr Map four[] = {{R,G,B,A}, {B,G,R,A}, {A,B,G,R}, {G,B,A,R},
    {R,G,B,O}, {B,G,R,O}, {A,B,G,O}, {G,B,A,O}};
constexpr Map packed_four[] = {{A,B,G,R}, {G,B,A,R}, {R,G,B,A}, {B,G,R,A}};
Map texture_mapping(SceGxmTextureFormat format) {
    const auto base = gxm::get_base_format(format);
    const auto mode = (uint32_t(format) & SCE_GXM_TEXTURE_SWIZZLE_MASK) >> 12;
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_X8U24) return {R,Z,Z,O};
    // Packed 565 stores its red bits at the high end of the native word.
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_U5U6U5 && mode < 2)
        return mode ? Map{R,G,B,O} : Map{B,G,R,O};
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_U4U4U4U4 && mode < 8) {
        Map mapping = packed_four[mode & 3];
        if (mode >= 4) mapping[3] = O;
        return mapping;
    }
    // U2U10U10U10 uses the same eight four-channel image-view mappings as
    // Vulkan. In particular, the X2 variants select constant-one alpha.
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5) {
        // U1U5U5U5 stores alpha in bit fifteen; U5U5U5U1 stores it in
        // bit zero. Native storage follows that placement for each mode.
        if (mode < 8) return ((mode & 1) == ((mode >> 1) & 1))
            ? Map{B,G,R,mode>=4?O:A} : Map{R,G,B,mode>=4?O:A};
    }
    // These CPU decoders already return logical RGB(A) channels.
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_YUV420P2 || base == SCE_GXM_TEXTURE_BASE_FORMAT_YUV420P3
        || base == SCE_GXM_TEXTURE_BASE_FORMAT_YUV422 || base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U3U3U2
        || base == SCE_GXM_TEXTURE_BASE_FORMAT_S5S5U6)
        return identity;
    // X8S8S8U8 is decoded to logical RGB, but its alpha is always one.
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8)
        return {R,G,B,O};
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
    case SCE_GXM_COLOR_BASE_FORMAT_U5U6U5:
        if (mode < 2) return mode ? Map{R,G,B,O} : Map{B,G,R,O};
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U4U4U4U4: {
        if (mode >= std::size(packed_four)) break;
        Map result=identity;
        for (size_t i=0;i<4;++i)
            for (size_t j=0;j<4;++j)
                if (packed_four[mode][i]==identity[j]) result[j]=identity[i];
        return result;
    }
    case SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9:
        if (mode < 2) return mode ? Map{B,G,R,O} : Map{R,G,B,O};
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U1U5U5U5:
        if (mode < 4) return ((mode & 1) == ((mode >> 1) & 1)) ? Map{B,G,R,A} : identity;
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8:
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8:
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16:
    case SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10:
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
    case SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2:
        if (mode == 0) return identity;
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8:
        // Expanded storage is already in logical RGBA order for all four
        // guest byte arrangements.
        return identity;
    case SCE_GXM_COLOR_BASE_FORMAT_S5S5U6:
        if (mode<2) return identity;
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8:
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8:
        // The third and fourth modes store R/A and A/R in the two bytes.
        if (mode < 4) return mode & 1 ? Map{G,R,Z,O} : Map{R,G,Z,O};
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U16U16:
    case SCE_GXM_COLOR_BASE_FORMAT_S16S16:
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16:
    case SCE_GXM_COLOR_BASE_FORMAT_F32F32:
        if (mode < 2) return mode ? Map{G,R,Z,O} : Map{R,G,Z,O};
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_F11F11F10:
        if (mode < 2) return mode ? Map{B,G,R,O} : Map{R,G,B,O};
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U8:
    case SCE_GXM_COLOR_BASE_FORMAT_S8:
        // Both R and A targets use single-channel storage. Fragment output
        // and blend state route logical alpha to physical red before storing.
        if (mode < 2) return identity;
        break;
    case SCE_GXM_COLOR_BASE_FORMAT_U16:
    case SCE_GXM_COLOR_BASE_FORMAT_S16:
    case SCE_GXM_COLOR_BASE_FORMAT_F16:
        // Both R and G targets store their sole guest channel in native R.
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
    case SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2:
        return (uint32_t(format) & SCE_GXM_COLOR_SWIZZLE_MASK) == 0
            ? SurfaceComponents{1, 2, MTLPixelFormatRGBA8Unorm} : SurfaceComponents{};
    case SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8: return {1, 4, MTLPixelFormatRGBA16Float};
    case SCE_GXM_COLOR_BASE_FORMAT_S5S5U6:
        return ((uint32_t(format)&SCE_GXM_COLOR_SWIZZLE_MASK)>>20)<2
            ? SurfaceComponents{1,2,MTLPixelFormatRGBA16Float} : SurfaceComponents{};
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8: return {4, 1, MTLPixelFormatRGBA8Snorm};
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16: return {4, 2, MTLPixelFormatRGBA16Float};
    case SCE_GXM_COLOR_BASE_FORMAT_F32F32: return {2, 4, MTLPixelFormatRG32Float};
    case SCE_GXM_COLOR_BASE_FORMAT_F32: return {1, 4, MTLPixelFormatR32Float};
    case SCE_GXM_COLOR_BASE_FORMAT_F16: return {1, 2, MTLPixelFormatR16Float};
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16: return {2, 2, MTLPixelFormatRG16Float};
    case SCE_GXM_COLOR_BASE_FORMAT_U8: return {1, 1, MTLPixelFormatR8Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_S8: return {1, 1, MTLPixelFormatR8Snorm};
    case SCE_GXM_COLOR_BASE_FORMAT_U16: return {1, 2, MTLPixelFormatR16Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_S16: return {1, 2, MTLPixelFormatR16Snorm};
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8: return {2, 1, MTLPixelFormatRG8Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8: return {2, 1, MTLPixelFormatRG8Snorm};
    case SCE_GXM_COLOR_BASE_FORMAT_U16U16: return {2, 2, MTLPixelFormatRG16Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_S16S16: return {2, 2, MTLPixelFormatRG16Snorm};
    case SCE_GXM_COLOR_BASE_FORMAT_F11F11F10: return {1, 4, MTLPixelFormatRG11B10Float};
    case SCE_GXM_COLOR_BASE_FORMAT_U5U6U5: return {1, 2, MTLPixelFormatB5G6R5Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_U4U4U4U4: return {1, 2, MTLPixelFormatABGR4Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9: return {1, 4, MTLPixelFormatRGB9E5Float};
    case SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10: return {1, 4, MTLPixelFormatBGR10A2Unorm};
    case SCE_GXM_COLOR_BASE_FORMAT_U1U5U5U5: {
        const uint32_t mode=(uint32_t(format)&SCE_GXM_COLOR_SWIZZLE_MASK)>>20;
        if (mode >= 4) return {};
        return {1, 2, mode<2 ? MTLPixelFormatBGR5A1Unorm : MTLPixelFormatA1BGR5Unorm};
    }
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
    COMPONENTS(F32F32, 2, 4, RG32Float); COMPONENTS(U32U32, 2, 4, RG32Uint);
    COMPONENTS(F11F11F10, 1, 4, RG11B10Float);
    COMPONENTS(U32, 1, 4, R32Uint); COMPONENTS(S32, 1, 4, R32Sint);
    COMPONENTS(U5U6U5, 1, 2, B5G6R5Unorm);
    COMPONENTS(U4U4U4U4, 1, 2, ABGR4Unorm);
    COMPONENTS(SE5M9M9M9, 1, 4, RGB9E5Float);
    COMPONENTS(U2U10U10U10, 1, 4, BGR10A2Unorm);
    COMPONENTS(U1U5U5U5, 1, 2, BGR5A1Unorm);
    // The guest word is four bytes, while the decoded sample is RGBA16F.
    COMPONENTS(X8S8S8U8, 1, 4, RGBA16Float);
#undef COMPONENTS
    default: return {};
    }
}

}
static bool has_depth_mask_bit(const SceGxmDepthStencilSurface &surface) {
    return surface.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M
        || surface.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8;
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
    case SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M:bytes=4;break;
    case SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8:bytes=4;separate=true;break;
    case SCE_GXM_DEPTH_STENCIL_FORMAT_S8:separate=true;break;
    case SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24:bytes=4;packed=true;break;
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
    if (!width || !height || surface.disabled()
        || multisample > SCE_GXM_MULTISAMPLE_4X) return std::nullopt;
    const auto base = gxm::get_base_format(gxm::get_format(texture));
    const auto format = surface.get_format();
    const bool stencil = base == SCE_GXM_TEXTURE_BASE_FORMAT_U8 || base == SCE_GXM_TEXTURE_BASE_FORMAT_S8;
    const bool compatible = (stencil && (format == SCE_GXM_DEPTH_STENCIL_FORMAT_S8
            || format == SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8 || format == SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8))
        || (format == SCE_GXM_DEPTH_STENCIL_FORMAT_D16 && base == SCE_GXM_TEXTURE_BASE_FORMAT_U16)
        || ((format == SCE_GXM_DEPTH_STENCIL_FORMAT_DF32 || format == SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8
                || format == SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M || format == SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8)
            && (base == SCE_GXM_TEXTURE_BASE_FORMAT_F32 || base == SCE_GXM_TEXTURE_BASE_FORMAT_F32M))
        || (format == SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24
            && (base == SCE_GXM_TEXTURE_BASE_FORMAT_X8U24
                || (base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 && multisample == SCE_GXM_MULTISAMPLE_NONE)));
    const Address storage = stencil ? surface.stencil_data.address() : surface.depth_data.address();
    if (!compatible || !storage) return std::nullopt;
    const auto type = texture.texture_type();
    if (type != SCE_GXM_TEXTURE_LINEAR_STRIDED && texture.true_mip_count() > 1) return std::nullopt;
    const bool linear = surface.get_type() == SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR;
    if (linear ? (type != SCE_GXM_TEXTURE_LINEAR && type != SCE_GXM_TEXTURE_LINEAR_STRIDED)
               : (surface.get_type() != SCE_GXM_DEPTH_STENCIL_SURFACE_TILED || type != SCE_GXM_TEXTURE_TILED)) return std::nullopt;
    const uint64_t memory_width = uint64_t(width) * (multisample == SCE_GXM_MULTISAMPLE_4X ? 2 : 1);
    const uint64_t memory_height = uint64_t(height) * (multisample != SCE_GXM_MULTISAMPLE_NONE ? 2 : 1);
    if (memory_width > surface.get_stride()) return std::nullopt;
    const uint32_t view_width=gxm::get_width(texture), view_height=gxm::get_height(texture);
    const uint32_t bytes = stencil ? 1 : base == SCE_GXM_TEXTURE_BASE_FORMAT_U16 ? 2 : 4;
    const uint64_t pitch = type == SCE_GXM_TEXTURE_LINEAR_STRIDED ? gxm::get_stride_in_bytes(texture)
        : ((uint64_t(view_width) + (linear ? 7 : 31)) & ~uint64_t(linear ? 7 : 31)) * bytes;
    if (pitch != uint64_t(surface.get_stride()) * bytes || !view_width || !view_height) return std::nullopt;
    const uint64_t address=uint64_t(texture.data_addr)<<2;
    if (address<storage) return std::nullopt;
    const uint64_t offset=address-storage;
    if (base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 && !linear && offset) return std::nullopt;
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
std::optional<std::pair<float,float>> surface_texture_viewport(const SceGxmColorSurface &surface, const SceGxmTexture &texture) {
    const auto type=texture.texture_type();
    if (!surface.data || !surface.width || !surface.height || surface.strideInPixels<surface.width
        || surface.surfaceType!=SCE_GXM_COLOR_SURFACE_LINEAR
        || (type!=SCE_GXM_TEXTURE_LINEAR && type!=SCE_GXM_TEXTURE_LINEAR_STRIDED)
        || (type!=SCE_GXM_TEXTURE_LINEAR_STRIDED && texture.true_mip_count()>1)
        || (uint64_t(texture.data_addr)<<2)!=surface.data.address()) return std::nullopt;
    SceGxmTextureFormat mapped{};
    const auto format=gxm::get_format(texture);
    if (!gxm::convert_color_format_to_texture_format(surface.colorFormat,mapped)
        || gxm::get_base_format(mapped)!=gxm::get_base_format(format)
        || (gxm::get_base_format(mapped)==SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8 && mapped!=format)) return std::nullopt;
    const uint32_t bits=gxm::bits_per_pixel(gxm::get_base_format(format));
    if (!bits || bits%8) return std::nullopt;
    const uint32_t width=gxm::get_width(texture),height=gxm::get_height(texture);
    const uint64_t stride=type==SCE_GXM_TEXTURE_LINEAR_STRIDED?gxm::get_stride_in_bytes(texture)
        : uint64_t((width+7)&~7u)*(bits/8);
    if (stride!=uint64_t(surface.strideInPixels)*(bits/8)
        || width<surface.width || height<surface.height
        || (width==surface.width && height==surface.height)) return std::nullopt;
    return std::pair{float(width)/surface.width,float(height)/surface.height};
}

bool surface_texture_layout_overlap(const SceGxmColorSurface &surface, const SceGxmTexture &texture) {
    if (!surface.data || !surface.width || !surface.height || surface.strideInPixels < surface.width)
        return false;
    const auto type = texture.texture_type();
    uint64_t pixels = gxm::get_width(texture);
    uint64_t pitch = 0;
    SceGxmColorSurfaceType layout;
    switch (type) {
    case SCE_GXM_TEXTURE_LINEAR_STRIDED:
        layout = SCE_GXM_COLOR_SURFACE_LINEAR;
        pitch = gxm::get_stride_in_bytes(texture);
        break;
    case SCE_GXM_TEXTURE_LINEAR:
        layout = SCE_GXM_COLOR_SURFACE_LINEAR;
        pixels = (pixels + 7) & ~uint64_t(7);
        break;
    case SCE_GXM_TEXTURE_TILED:
        layout = SCE_GXM_COLOR_SURFACE_TILED;
        pixels = (pixels + 31) & ~uint64_t(31);
        break;
    case SCE_GXM_TEXTURE_SWIZZLED_ARBITRARY:
        layout = SCE_GXM_COLOR_SURFACE_SWIZZLED;
        pixels = std::bit_ceil(pixels);
        break;
    case SCE_GXM_TEXTURE_SWIZZLED:
        layout = SCE_GXM_COLOR_SURFACE_SWIZZLED;
        break;
    default:
        return false; // Cube faces are selected by the cube reader.
    }
    const uint32_t texture_bits = gxm::bits_per_pixel(gxm::get_base_format(gxm::get_format(texture)));
    const uint32_t surface_bits = gxm::bits_per_pixel(gxm::get_base_format(surface.colorFormat));
    if (!texture_bits || !surface_bits || layout != surface.surfaceType) return false;
    if (type != SCE_GXM_TEXTURE_LINEAR_STRIDED) pitch = pixels * texture_bits / 8;
    const uint64_t surface_pitch = uint64_t(surface.strideInPixels) * surface_bits / 8;
    const uint64_t address = uint64_t(texture.data_addr) << 2;
    // The final row of a linear surface ends at its visible width. Its trailing
    // stride padding cannot own a texture view starting there. Tiled and
    // swizzled surfaces retain their complete allocated row extent.
    const uint64_t owned_bytes = layout == SCE_GXM_COLOR_SURFACE_LINEAR
        ? surface_pitch * (surface.height - 1) + uint64_t(surface.width) * surface_bits / 8
        : surface_pitch * surface.height;
    return pitch && pitch == surface_pitch && address >= surface.data.address()
        && address - surface.data.address() < owned_bytes;
}

std::optional<SurfaceRect> surface_subrectangle(const SceGxmColorSurface &surface, const SceGxmTexture &texture) {
    const auto type=texture.texture_type();
    if (!surface.data || !surface.width || !surface.height || surface.strideInPixels<surface.width
        || (type!=SCE_GXM_TEXTURE_LINEAR_STRIDED && texture.true_mip_count()>1)) return std::nullopt;
    SceGxmColorBaseFormat color;
    const auto base=gxm::get_base_format(gxm::get_format(texture));
    const bool same_format=renderer::texture::convert_base_texture_format_to_base_color_format(base,color)
        && color==gxm::get_base_format(surface.colorFormat);
    if (!same_format && !surface_format_cast_supported(surface.colorFormat,base,texture.swizzle_format)) return std::nullopt;
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
    const bool packed_float = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10;
    if ((!components.count && !packed_float) || !surface.width || !surface.height || surface.width > 16384 || surface.height > 16384
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
    return rows * surface.strideInPixels * (packed_float ? 4 : components.count * components.bytes);
}
bool surface_memory_ranges(const SceGxmColorSurface &surface, SurfaceRect rectangle,
    std::vector<SurfaceMemoryRange> &ranges) {
    ranges.clear();
    const size_t size = surface_memory_size(surface);
    const uint32_t bits = gxm::bits_per_pixel(gxm::get_base_format(surface.colorFormat));
    if (!size || !bits || bits % 8) return false;
    const size_t bytes = bits / 8;
    const uint32_t x0 = std::min<uint32_t>(rectangle.x, surface.width);
    const uint32_t y0 = std::min<uint32_t>(rectangle.y, surface.height);
    const uint32_t x1 = std::min<uint64_t>(uint64_t(rectangle.x) + rectangle.width, surface.width);
    const uint32_t y1 = std::min<uint64_t>(uint64_t(rectangle.y) + rectangle.height, surface.height);
    if (x1 <= x0 || y1 <= y0) return true;
    const auto append = [&](size_t pixel, size_t count) {
        const size_t offset = pixel * bytes, length = count * bytes;
        if (!ranges.empty() && ranges.back().offset + ranges.back().size == offset)
            ranges.back().size += length;
        else ranges.push_back({offset, length});
    };
    if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_LINEAR) {
        for (uint32_t y = y0; y < y1; ++y)
            append(size_t(y) * surface.strideInPixels + x0, x1 - x0);
    } else if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_TILED) {
        // Tile-major traversal makes every emitted byte range ascending,
        // including a rectangle that starts or ends inside a 32x32 tile.
        for (uint32_t ty = y0 / 32; ty <= (y1 - 1) / 32; ++ty) {
            for (uint32_t tx = x0 / 32; tx <= (x1 - 1) / 32; ++tx) {
                const uint32_t left = std::max(x0, tx * 32), right = std::min(x1, (tx + 1) * 32);
                const uint32_t top = std::max(y0, ty * 32), bottom = std::min(y1, (ty + 1) * 32);
                const size_t tile = (size_t(ty) * (surface.strideInPixels / 32) + tx) * 1024;
                for (uint32_t y = top; y < bottom; ++y)
                    append(tile + (y - ty * 32) * 32 + left - tx * 32, right - left);
            }
        }
    } else if (!x0 && !y0 && x1 == surface.width && y1 == surface.height) {
        append(0, size_t(surface.width) * surface.height);
    } else {
        // surface_memory_size validates power-of-two Morton dimensions and
        // a compact stride. Decode in storage order to coalesce adjacent words.
        const uint32_t side = std::min(surface.width, surface.height);
        const uint32_t k = std::bit_width(side) - 1;
        for (uint32_t pixel = 0; pixel < uint32_t(surface.width) * surface.height; ++pixel) {
            const uint32_t upper = (pixel >> (2 * k)) << k;
            const uint32_t x = (texture::decode_morton2_x(pixel) & (side - 1)) | (surface.width >= surface.height ? upper : 0);
            const uint32_t y = (texture::decode_morton2_y(pixel) & (side - 1)) | (surface.width < surface.height ? upper : 0);
            if (x >= x0 && x < x1 && y >= y0 && y < y1) append(pixel, 1);
        }
    }
    for (const auto &range : ranges) {
        if (range.offset > size || range.size > size - range.offset) {
            ranges.clear();
            return false;
        }
    }
    return true;
}
static bool raw_storage_matches(MTLPixelFormat actual, MTLPixelFormat expected) {
    return actual == expected || (expected == MTLPixelFormatRGBA8Unorm && actual == MTLPixelFormatRGBA8Unorm_sRGB);
}
static uint16_t swap_565_red_blue(uint16_t value) {
    return uint16_t((value & 0x07e0) | ((value & 0xf800) >> 11) | ((value & 0x001f) << 11));
}
static uint16_t swap_5551_red_blue(uint16_t value, bool high_alpha) {
    return high_alpha
        ? uint16_t((value&0x83e0u)|((value&0x7c00u)>>10)|((value&0x001fu)<<10))
        : uint16_t((value&0x07c1u)|((value&0xf800u)>>10)|((value&0x003eu)<<10));
}
static uint32_t swap_e5_red_blue(uint32_t value) {
    return (value&0xf803fe00u)|((value&0x000001ffu)<<18)|((value&0x07fc0000u)>>18);
}
static uint32_t swap_10_red_blue(uint32_t value) {
    return (value&0xc00ffc00u)|((value&0x000003ffu)<<20)|((value&0x3ff00000u)>>20);
}
static uint32_t packed_10_guest_word(uint32_t native, uint32_t mode) {
    if (mode == 0) return native;
    if (mode == 1) return swap_10_red_blue(native);
    const uint32_t blue = native & 0x3ff;
    const uint32_t green = (native >> 10) & 0x3ff;
    const uint32_t red = (native >> 20) & 0x3ff;
    return (native >> 30) | ((mode == 2 ? blue : red) << 2)
        | (green << 12) | ((mode == 2 ? red : blue) << 22);
}
static uint16_t permute_4444(uint16_t value, const Map &mapping) {
    uint16_t result=0;
    for (size_t c=0;c<4;++c) {
        const auto found=std::find(identity.begin(),identity.end(),mapping[c]);
        const size_t source=size_t(found-identity.begin());
        result|=uint16_t(((value>>(12-4*source))&15)<<(12-4*c));
    }
    return result;
}
static Map inverse_mapping(const Map &mapping) {
    Map result=identity;
    for (size_t i=0;i<4;++i)
        for (size_t j=0;j<4;++j)
            if (mapping[i]==identity[j]) result[j]=identity[i];
    return result;
}
static size_t packed_float_guest_pixel(const SceGxmColorSurface &surface, uint32_t x, uint32_t y) {
    if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_TILED)
        return ((size_t(y / 32) * (surface.strideInPixels / 32) + x / 32) * 1024) + (y % 32) * 32 + x % 32;
    if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_SWIZZLED)
        return texture::encode_morton(x, y, surface.width, surface.height);
    return size_t(y) * surface.strideInPixels + x;
}
static uint16_t packed_float10_from_half(uint16_t bits) {
    // The existing GXM texture decoder expands each unsigned 10-bit float
    // by shifting its five exponent and five mantissa bits into binary16.
    if (bits & 0x8000) return 0;
    uint16_t result = std::min<uint16_t>((uint32_t(bits) + 16) >> 5, 0x3ff);
    if ((bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) && !(result & 31))
        result |= 1; // Keep a small NaN payload from rounding to infinity.
    return result;
}
static uint32_t packed_float_word(const uint16_t *channels, uint32_t mode) {
    __fp16 alpha_half;
    std::memcpy(&alpha_half, channels + 3, sizeof(alpha_half));
    const float alpha = static_cast<float>(alpha_half);
    const uint32_t alpha2 = std::isnan(alpha) ? 0
        : uint32_t(std::clamp(std::lround(std::clamp(alpha, 0.0f, 1.0f) * 3.0f), 0l, 3l));
    const bool blue_first = mode == 1 || mode == 2;
    const uint32_t first = packed_float10_from_half(channels[blue_first ? 2 : 0]);
    const uint32_t second = packed_float10_from_half(channels[1]);
    const uint32_t third = packed_float10_from_half(channels[blue_first ? 0 : 2]);
    if (mode & 2)
        return alpha2 | (first << 2) | (second << 12) | (third << 22);
    return first | (second << 10) | (third << 20) | (alpha2 << 30);
}
static void unpack_packed_float(uint32_t word, uint32_t mode, uint16_t *channels) {
    constexpr uint16_t alpha_half[] = {0, 0x3555, 0x3955, 0x3c00};
    const bool low_alpha = mode & 2;
    const uint16_t first = uint16_t(((word >> (low_alpha ? 2 : 0)) & 0x3ff) << 5);
    const uint16_t second = uint16_t(((word >> (low_alpha ? 12 : 10)) & 0x3ff) << 5);
    const uint16_t third = uint16_t(((word >> (low_alpha ? 22 : 20)) & 0x3ff) << 5);
    const bool blue_first = mode == 1 || mode == 2;
    channels[0] = blue_first ? third : first;
    channels[1] = second;
    channels[2] = blue_first ? first : third;
    channels[3] = alpha_half[low_alpha ? word & 3 : word >> 30];
}
static uint32_t pack_rgb9e5(const uint16_t *channels) {
    float rgb[3];
    for (uint32_t i=0;i<3;++i) {
        __fp16 half;
        std::memcpy(&half,channels+i,sizeof(half));
        const float value=static_cast<float>(half);
        rgb[i]=std::isnan(value) ? 0.0f : std::clamp(value,0.0f,65408.0f);
    }
    const float maximum=std::max({rgb[0],rgb[1],rgb[2]});
    if (maximum==0.0f) return 0;
    int exponent=std::max(-16,std::ilogb(maximum))+16;
    float scale=std::ldexp(1.0f,exponent-24);
    if (maximum/scale+0.5f>=512.0f && exponent<31) {
        ++exponent;
        scale*=2.0f;
    }
    const auto mantissa=[scale](float value) {
        return std::min(511u,uint32_t(value/scale+0.5f));
    };
    return mantissa(rgb[0]) | (mantissa(rgb[1])<<9)
        | (mantissa(rgb[2])<<18) | (uint32_t(exponent)<<27);
}
static void unpack_rgb9e5(uint32_t word, uint16_t *channels) {
    const int exponent=int(word>>27)-24;
    for (uint32_t i=0;i<3;++i) {
        const __fp16 half=std::ldexp(float((word>>(i*9))&511),exponent);
        std::memcpy(channels+i,&half,sizeof(half));
    }
    channels[3]=0x3c00; // The guest format has implicit alpha one.
}
static std::pair<size_t,size_t> write_native_interval(size_t guest, size_t guest_size,
    size_t native_size, size_t sample_extent) {
    const size_t groups=native_size/sample_extent;
    const size_t first=guest*groups/guest_size;
    const size_t last=groups<guest_size ? first+1 : (guest+1)*groups/guest_size;
    return {first*sample_extent,last*sample_extent};
}
static uint16_t pack_u8u3u3u2(const uint8_t *rgba) {
    const uint16_t r = uint16_t((uint32_t(rgba[0]) * 7 + 127) / 255);
    const uint16_t g = uint16_t((uint32_t(rgba[1]) * 7 + 127) / 255);
    const uint16_t b = uint16_t((uint32_t(rgba[2]) * 3 + 127) / 255);
    return uint16_t((uint16_t(rgba[3]) << 8) | (r << 5) | (g << 2) | b);
}
static void unpack_u8u3u3u2(uint16_t word, uint8_t *rgba) {
    const uint8_t r = uint8_t((word >> 5) & 7);
    const uint8_t g = uint8_t((word >> 2) & 7);
    const uint8_t b = uint8_t(word & 3);
    rgba[0] = uint8_t((r << 5) | (r << 2) | (r >> 1));
    rgba[1] = uint8_t((g << 5) | (g << 2) | (g >> 1));
    rgba[2] = uint8_t(b * 85);
    rgba[3] = uint8_t(word >> 8);
}
static bool u8u3u3u2_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<uint8_t> destination,
    std::span<const SurfaceMemoryRange> ranges, bool upload, uint32_t samples_x=1, uint32_t samples_y=1) {
    const size_t size = surface_memory_size(surface);
    if (!size || (upload ? source.size() : destination.size()) < size || !texture
        || !raw_storage_matches(texture.pixelFormat, MTLPixelFormatRGBA8Unorm)
        || texture.textureType != MTLTextureType2D || texture.storageMode != MTLStorageModeShared
        || !texture.width || !texture.height || !samples_x || !samples_y
        || texture.width % samples_x || texture.height % samples_y) return false;
    size_t previous_end = 0;
    for (const auto &range : ranges) {
        if (!range.size || range.offset < previous_end || range.offset > size || range.size > size - range.offset)
            return false;
        previous_end = range.offset + range.size;
    }
    if (upload && ranges.empty()) return true;
    const size_t stride = texture.width * 4;
    std::vector<uint8_t> raw(stride * texture.height);
    [texture getBytes:raw.data() bytesPerRow:stride
        fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height) mipmapLevel:0];
    bool changed = false;
    for (uint32_t y = 0; y < surface.height; ++y) for (uint32_t x = 0; x < surface.width; ++x) {
        const size_t offset = packed_float_guest_pixel(surface, x, y) * 2;
        if (upload) {
            auto range = std::lower_bound(ranges.begin(), ranges.end(), offset,
                [](const auto &range, size_t byte) { return range.offset + range.size <= byte; });
            if (range == ranges.end() || range->offset >= offset + 2) continue;
            const auto [top,bottom] = write_native_interval(y,surface.height,texture.height,samples_y);
            const auto [left,right] = write_native_interval(x,surface.width,texture.width,samples_x);
            for (size_t ny = top; ny < bottom; ++ny) for (size_t nx = left; nx < right; ++nx) {
                auto *rgba = raw.data() + ny * stride + nx * 4;
                uint16_t word = pack_u8u3u3u2(rgba);
                auto *bytes = reinterpret_cast<uint8_t *>(&word);
                for (auto part = range; part != ranges.end() && part->offset < offset + 2; ++part)
                    for (size_t byte = std::max(offset,part->offset);
                        byte < std::min(offset + 2,part->offset + part->size); ++byte)
                        bytes[byte-offset] = source[byte];
                unpack_u8u3u3u2(word,rgba);
            }
            changed = true;
        } else {
            const size_t ny = size_t(y) * texture.height / surface.height;
            const size_t nx = size_t(x) * texture.width / surface.width;
            const uint16_t word = pack_u8u3u3u2(raw.data() + ny * stride + nx * 4);
            std::memcpy(destination.data() + offset, &word, 2);
        }
    }
    if (changed) [texture replaceRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
        mipmapLevel:0 withBytes:raw.data() bytesPerRow:stride];
    return true;
}
static int decode_signed5(uint16_t bits) {
    return bits<16 ? int(bits) : int(bits)-32;
}
static uint16_t pack_s5s5u6(const uint16_t *rgba, uint32_t mode) {
    const auto quantize=[](uint16_t half,int scale,int low,int high) {
        const float value=util::decode_flt16(half);
        if (std::isnan(value)) return 0;
        if (!std::isfinite(value)) return value>0 ? high : low;
        return int(std::clamp(std::lround(value*scale),long(low),long(high)));
    };
    const uint16_t red=uint16_t(quantize(rgba[0],15,-16,15))&31;
    const uint16_t green=uint16_t(quantize(rgba[1],15,-16,15))&31;
    const uint16_t blue=uint16_t(quantize(rgba[2],63,0,63));
    return mode==0 ? uint16_t((blue<<10)|(green<<5)|red)
        : uint16_t((red<<11)|(green<<6)|blue);
}
static void unpack_s5s5u6(uint16_t word,uint32_t mode,uint16_t *rgba) {
    const uint16_t red=mode==0 ? word&31 : (word>>11)&31;
    const uint16_t green=(word>>(mode==0 ? 5 : 6))&31;
    const uint16_t blue=mode==0 ? (word>>10)&63 : word&63;
    rgba[0]=util::encode_flt16(float(decode_signed5(red))/15.f);
    rgba[1]=util::encode_flt16(float(decode_signed5(green))/15.f);
    rgba[2]=util::encode_flt16(float(blue)/63.f);
    rgba[3]=0x3c00;
}
static bool s5s5u6_surface_memory(id<MTLTexture> texture,const SceGxmColorSurface &surface,
    std::span<const uint8_t> source,std::span<uint8_t> destination,
    std::span<const SurfaceMemoryRange> ranges,bool upload,uint32_t samples_x=1,uint32_t samples_y=1) {
    const size_t size=surface_memory_size(surface);
    const uint32_t mode=(uint32_t(surface.colorFormat)&SCE_GXM_COLOR_SWIZZLE_MASK)>>20;
    if (!size || mode>=2 || (upload ? source.size() : destination.size())<size || !texture
        || texture.pixelFormat!=MTLPixelFormatRGBA16Float
        || texture.textureType!=MTLTextureType2D || texture.storageMode!=MTLStorageModeShared
        || !texture.width || !texture.height || !samples_x || !samples_y
        || texture.width%samples_x || texture.height%samples_y) return false;
    size_t previous_end=0;
    for (const auto &range:ranges) {
        if (!range.size || range.offset<previous_end || range.offset>size || range.size>size-range.offset)
            return false;
        previous_end=range.offset+range.size;
    }
    if (upload && ranges.empty()) return true;
    const size_t stride=texture.width*8;
    std::vector<uint8_t> raw(stride*texture.height);
    [texture getBytes:raw.data() bytesPerRow:stride
        fromRegion:MTLRegionMake2D(0,0,texture.width,texture.height) mipmapLevel:0];
    bool changed=false;
    for (uint32_t y=0;y<surface.height;++y) for (uint32_t x=0;x<surface.width;++x) {
        const size_t offset=packed_float_guest_pixel(surface,x,y)*2;
        if (upload) {
            auto range=std::lower_bound(ranges.begin(),ranges.end(),offset,
                [](const auto &range,size_t byte) { return range.offset+range.size<=byte; });
            if (range==ranges.end() || range->offset>=offset+2) continue;
            const auto [top,bottom]=write_native_interval(y,surface.height,texture.height,samples_y);
            const auto [left,right]=write_native_interval(x,surface.width,texture.width,samples_x);
            for (size_t ny=top;ny<bottom;++ny) for (size_t nx=left;nx<right;++nx) {
                auto *native=raw.data()+ny*stride+nx*8;
                uint16_t rgba[4];std::memcpy(rgba,native,8);
                uint16_t word=pack_s5s5u6(rgba,mode);
                auto *bytes=reinterpret_cast<uint8_t *>(&word);
                for (auto part=range;part!=ranges.end() && part->offset<offset+2;++part)
                    for (size_t byte=std::max(offset,part->offset);
                        byte<std::min(offset+2,part->offset+part->size);++byte)
                        bytes[byte-offset]=source[byte];
                unpack_s5s5u6(word,mode,rgba);
                std::memcpy(native,rgba,8);
            }
            changed=true;
        } else {
            const size_t ny=size_t(y)*texture.height/surface.height;
            const size_t nx=size_t(x)*texture.width/surface.width;
            uint16_t rgba[4];
            std::memcpy(rgba,raw.data()+ny*stride+nx*8,8);
            const uint16_t word=pack_s5s5u6(rgba,mode);
            std::memcpy(destination.data()+offset,&word,2);
        }
    }
    if (changed) [texture replaceRegion:MTLRegionMake2D(0,0,texture.width,texture.height)
        mipmapLevel:0 withBytes:raw.data() bytesPerRow:stride];
    return true;
}
static constexpr std::array<std::array<uint8_t,4>,4> mixed_signed_guest_bytes{{
    {{0,1,2,3}}, // ABGR: R, G, B, A in little-endian memory
    {{2,1,0,3}}, // ARGB: B, G, R, A
    {{3,2,1,0}}, // RGBA: A, B, G, R
    {{1,2,3,0}}  // BGRA: A, R, G, B
}};
static void unpack_u8s8s8u8(const uint8_t *guest, uint32_t mode, uint16_t *rgba) {
    const auto &order = mixed_signed_guest_bytes[mode];
    for (size_t channel=0;channel<4;++channel) {
        const int byte=guest[order[channel]];
        const int value=channel==1 || channel==2 ? (byte<128 ? byte : byte-256) : byte;
        rgba[channel]=util::encode_flt16(float(value)/float(channel==1 || channel==2 ? 127 : 255));
    }
}
static void pack_u8s8s8u8(const uint16_t *rgba, uint32_t mode, uint8_t *guest) {
    const auto &order = mixed_signed_guest_bytes[mode];
    for (size_t channel=0;channel<4;++channel) {
        const bool signed_channel=channel==1 || channel==2;
        const float value=util::decode_flt16(rgba[channel]);
        const long rounded=std::isnan(value) ? 0 : !std::isfinite(value)
            ? value>0 ? 255 : -128
            : std::lround(value*(signed_channel ? 127.f : 255.f));
        guest[order[channel]]=uint8_t(signed_channel ? int8_t(std::clamp(rounded,-128l,127l))
            : std::clamp(rounded,0l,255l));
    }
}
static bool u8s8s8u8_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<uint8_t> destination,
    std::span<const SurfaceMemoryRange> ranges, bool upload, uint32_t samples_x=1, uint32_t samples_y=1) {
    const size_t size=surface_memory_size(surface);
    const uint32_t mode=(uint32_t(surface.colorFormat)&SCE_GXM_COLOR_SWIZZLE_MASK)>>20;
    if (!size || mode>=mixed_signed_guest_bytes.size()
        || (upload ? source.size() : destination.size())<size || !texture
        || texture.pixelFormat!=MTLPixelFormatRGBA16Float
        || texture.textureType!=MTLTextureType2D || texture.storageMode!=MTLStorageModeShared
        || !texture.width || !texture.height || !samples_x || !samples_y
        || texture.width%samples_x || texture.height%samples_y) return false;
    size_t previous_end=0;
    for (const auto &range:ranges) {
        if (!range.size || range.offset<previous_end || range.offset>size || range.size>size-range.offset)
            return false;
        previous_end=range.offset+range.size;
    }
    if (upload && ranges.empty()) return true;
    const size_t stride=texture.width*8;
    std::vector<uint8_t> raw(stride*texture.height);
    [texture getBytes:raw.data() bytesPerRow:stride
        fromRegion:MTLRegionMake2D(0,0,texture.width,texture.height) mipmapLevel:0];
    bool changed=false;
    for (uint32_t y=0;y<surface.height;++y) for (uint32_t x=0;x<surface.width;++x) {
        const size_t offset=packed_float_guest_pixel(surface,x,y)*4;
        if (upload) {
            auto range=std::lower_bound(ranges.begin(),ranges.end(),offset,
                [](const auto &range,size_t byte) { return range.offset+range.size<=byte; });
            if (range==ranges.end() || range->offset>=offset+4) continue;
            const auto [top,bottom]=write_native_interval(y,surface.height,texture.height,samples_y);
            const auto [left,right]=write_native_interval(x,surface.width,texture.width,samples_x);
            for (size_t ny=top;ny<bottom;++ny) for (size_t nx=left;nx<right;++nx) {
                auto *native=raw.data()+ny*stride+nx*8;
                uint16_t rgba[4];
                std::memcpy(rgba,native,8);
                uint8_t bytes[4];
                pack_u8s8s8u8(rgba,mode,bytes);
                for (auto part=range;part!=ranges.end() && part->offset<offset+4;++part)
                    for (size_t byte=std::max(offset,part->offset);
                        byte<std::min(offset+4,part->offset+part->size);++byte)
                        bytes[byte-offset]=source[byte];
                unpack_u8s8s8u8(bytes,mode,rgba);
                std::memcpy(native,rgba,8);
            }
            changed=true;
        } else {
            const size_t ny=size_t(y)*texture.height/surface.height;
            const size_t nx=size_t(x)*texture.width/surface.width;
            uint16_t rgba[4];
            std::memcpy(rgba,raw.data()+ny*stride+nx*8,8);
            pack_u8s8s8u8(rgba,mode,destination.data()+offset);
        }
    }
    if (changed) [texture replaceRegion:MTLRegionMake2D(0,0,texture.width,texture.height)
        mipmapLevel:0 withBytes:raw.data() bytesPerRow:stride];
    return true;
}
static bool packed_float_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<uint8_t> destination,
    std::span<const SurfaceMemoryRange> ranges, bool upload, uint32_t samples_x=1, uint32_t samples_y=1) {
    const size_t size = surface_memory_size(surface);
    const uint32_t mode = (uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20;
    if (!size || (upload ? source.size() : destination.size()) < size || !texture || mode >= 4
        || texture.pixelFormat != MTLPixelFormatRGBA16Float
        || texture.textureType != MTLTextureType2D || texture.storageMode != MTLStorageModeShared
        || !texture.width || !texture.height || texture.width%samples_x || texture.height%samples_y) return false;
    size_t previous_end = 0;
    for (const auto &range : ranges) {
        if (!range.size || range.offset < previous_end || range.offset > size || range.size > size - range.offset)
            return false;
        previous_end = range.offset + range.size;
    }
    if (upload && ranges.empty()) return true;
    const size_t stride = texture.width * 8;
    std::vector<uint8_t> raw(stride * texture.height);
    [texture getBytes:raw.data() bytesPerRow:stride
        fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height) mipmapLevel:0];
    bool changed = false;
    for (uint32_t y = 0; y < surface.height; ++y) for (uint32_t x = 0; x < surface.width; ++x) {
        const size_t offset = packed_float_guest_pixel(surface, x, y) * 4;
        if (upload) {
            const auto range = std::lower_bound(ranges.begin(), ranges.end(), offset,
                [](const auto &range, size_t byte) { return range.offset + range.size <= byte; });
            if (range == ranges.end() || range->offset >= offset + 4) continue;
            uint32_t word;
            std::memcpy(&word, source.data() + offset, 4);
            uint16_t channels[4];
            unpack_packed_float(word, mode, channels);
            const auto [top,bottom]=write_native_interval(y,surface.height,texture.height,samples_y);
            const auto [left,right]=write_native_interval(x,surface.width,texture.width,samples_x);
            for (size_t ny=top;ny<bottom;++ny)
                for (size_t nx=left;nx<right;++nx)
                    std::memcpy(raw.data() + ny * stride + nx * 8, channels, 8);
            changed = true;
        } else {
            const size_t ny = size_t(y) * texture.height / surface.height;
            const size_t nx = size_t(x) * texture.width / surface.width;
            uint16_t channels[4];
            std::memcpy(channels, raw.data() + ny * stride + nx * 8, 8);
            const uint32_t word = packed_float_word(channels, mode);
            std::memcpy(destination.data() + offset, &word, 4);
        }
    }
    if (changed) [texture replaceRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
        mipmapLevel:0 withBytes:raw.data() bytesPerRow:stride];
    return true;
}
static bool rgb9e5_expanded_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<uint8_t> destination,
    std::span<const SurfaceMemoryRange> ranges, bool upload, uint32_t samples_x=1, uint32_t samples_y=1) {
    const size_t size=surface_memory_size(surface);
    const uint32_t mode=(uint32_t(surface.colorFormat)&SCE_GXM_COLOR_SWIZZLE_MASK)>>20;
    if (!size || (upload ? source.size() : destination.size())<size || !texture || mode>=2
        || texture.pixelFormat!=MTLPixelFormatRGBA16Float
        || texture.textureType!=MTLTextureType2D || texture.storageMode!=MTLStorageModeShared
        || !texture.width || !texture.height || texture.width%samples_x || texture.height%samples_y) return false;
    size_t previous_end=0;
    for (const auto &range:ranges) {
        if (!range.size || range.offset<previous_end || range.offset>size || range.size>size-range.offset)
            return false;
        previous_end=range.offset+range.size;
    }
    if (upload && ranges.empty()) return true;
    const size_t stride=texture.width*8;
    std::vector<uint8_t> raw(stride*texture.height);
    [texture getBytes:raw.data() bytesPerRow:stride
        fromRegion:MTLRegionMake2D(0,0,texture.width,texture.height) mipmapLevel:0];
    bool changed=false;
    for (uint32_t y=0;y<surface.height;++y) for (uint32_t x=0;x<surface.width;++x) {
        const size_t offset=packed_float_guest_pixel(surface,x,y)*4;
        if (upload) {
            const auto range=std::lower_bound(ranges.begin(),ranges.end(),offset,
                [](const auto &item,size_t byte) { return item.offset+item.size<=byte; });
            if (range==ranges.end() || range->offset>=offset+4) continue;
            uint32_t word;
            std::memcpy(&word,source.data()+offset,4);
            if (mode==1) word=swap_e5_red_blue(word);
            uint16_t channels[4];
            unpack_rgb9e5(word,channels);
            const auto [top,bottom]=write_native_interval(y,surface.height,texture.height,samples_y);
            const auto [left,right]=write_native_interval(x,surface.width,texture.width,samples_x);
            for (size_t ny=top;ny<bottom;++ny)
                for (size_t nx=left;nx<right;++nx)
                    std::memcpy(raw.data()+ny*stride+nx*8,channels,8);
            changed=true;
        } else {
            const size_t ny=size_t(y)*texture.height/surface.height;
            const size_t nx=size_t(x)*texture.width/surface.width;
            uint16_t channels[4];
            std::memcpy(channels,raw.data()+ny*stride+nx*8,8);
            uint32_t word=pack_rgb9e5(channels);
            if (mode==1) word=swap_e5_red_blue(word);
            std::memcpy(destination.data()+offset,&word,4);
        }
    }
    if (changed) [texture replaceRegion:MTLRegionMake2D(0,0,texture.width,texture.height)
        mipmapLevel:0 withBytes:raw.data() bytesPerRow:stride];
    return true;
}
bool read_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<uint8_t> destination) {
    if (gxm::get_base_format(surface.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_S5S5U6)
        return s5s5u6_surface_memory(texture,surface,{},destination,{},false);
    if (gxm::get_base_format(surface.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8)
        return u8s8s8u8_surface_memory(texture,surface,{},destination,{},false);
    if (gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2)
        return u8u3u3u2_surface_memory(texture, surface, {}, destination, {}, false);
    if (gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10)
        return packed_float_surface_memory(texture, surface, {}, destination, {}, false);
    if (gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
        && texture && texture.pixelFormat == MTLPixelFormatRGBA16Float)
        return rgb9e5_expanded_surface_memory(texture,surface,{},destination,{},false);
    const auto size = surface_memory_size(surface);
    const auto components = surface_components(surface.colorFormat);
    if (!size || size > destination.size() || !texture || !raw_storage_matches(texture.pixelFormat, components.native)
        || texture.textureType != MTLTextureType2D || texture.storageMode != MTLStorageModeShared
        || !texture.width || !texture.height) return false;
    const bool packed565 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U5U6U5;
    const bool packed4444 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U4U4U4U4;
    const bool packed_u1 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U1U5U5U5;
    const bool packed_e5 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9;
    const bool packed_u2 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10;
    const bool packed_f11 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_F11F11F10;
    if (packed565 && ((uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20) >= 2) return false;
    if (packed_u1 && ((uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20) >= 4) return false;
    if ((packed_e5 || packed_f11) && ((uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20) >= 2) return false;
    const bool reverse565 = packed565 && (uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) == SCE_GXM_COLOR_SWIZZLE3_BGR;
    const uint32_t u1_mode=(uint32_t(surface.colorFormat)&SCE_GXM_COLOR_SWIZZLE_MASK)>>20;
    const bool reverse_u1=packed_u1 && (u1_mode==0 || u1_mode==3);
    const bool reverse_e5 = packed_e5 && (uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) == SCE_GXM_COLOR_SWIZZLE3_RGB;
    const uint32_t u2_mode = (uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20;
    Map mapping, packed_mapping=identity;
    try {
        if (packed4444) packed_mapping=surface_memory_mapping(surface.colorFormat);
        // Vulkan synchronizes packed F11F11F10 storage as raw 32-bit words;
        // its logical RGB/BGR view mapping does not reorder guest bytes.
        mapping = packed565 || packed4444 || packed_u1 || packed_e5 || packed_u2 || packed_f11
            ? identity : surface_memory_mapping(surface.colorFormat);
    }
    catch (const std::runtime_error &) { return false; }
    std::array<uint32_t, 4> channel{};
    for (uint32_t c = 0; c < components.count; ++c) {
        const auto found = std::find(identity.begin(), identity.begin() + components.count, mapping[c]);
        if (found == identity.begin() + components.count) return false;
        channel[c] = uint32_t(found - identity.begin());
    }
    const size_t pixel_bytes = components.count * components.bytes;
    // A linear guest surface with the same native extent and byte order needs
    // no staging image or per-pixel conversion. The row pitch still preserves
    // guest padding, and packed/float payloads remain bit-exact.
    const bool native_byte_order = packed4444 ? packed_mapping == identity
        : packed565 || packed_u1 || packed_e5 || packed_u2 || packed_f11 ? true
        : std::equal(mapping.begin(), mapping.begin() + components.count, identity.begin());
    if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_LINEAR
        && texture.width == surface.width && texture.height == surface.height
        && components.count != 3 && native_byte_order
        && !reverse565 && !reverse_u1 && !reverse_e5 && (!packed_u2 || u2_mode == 0)) {
        [texture getBytes:destination.data() bytesPerRow:size_t(surface.strideInPixels) * pixel_bytes
            fromRegion:MTLRegionMake2D(0, 0, surface.width, surface.height) mipmapLevel:0];
        return true;
    }
    const size_t native_pixel_bytes = components.count == 3 ? 4 : pixel_bytes;
    const size_t native_width = texture.width, native_height = texture.height;
    const size_t native_stride = native_width * native_pixel_bytes;
    std::vector<uint8_t> raw(native_stride * native_height);
    [texture getBytes:raw.data() bytesPerRow:native_stride
        fromRegion:MTLRegionMake2D(0, 0, native_width, native_height) mipmapLevel:0];
    // Linear byte-channel surfaces need only a permutation. Keep the four
    // byte copies explicit so this common readback avoids dynamic memcpy
    // calls and Objective-C texture queries for every component and pixel.
    if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_LINEAR && components.count == 4 && components.bytes == 1
        && native_width == surface.width && native_height == surface.height) {
        for (uint32_t y = 0; y < surface.height; ++y) {
            const auto *input = raw.data() + y * native_stride;
            auto *output = destination.data() + size_t(y) * surface.strideInPixels * 4;
            for (uint32_t x = 0; x < surface.width; ++x, input += 4, output += 4) {
                output[0] = input[channel[0]];
                output[1] = input[channel[1]];
                output[2] = input[channel[2]];
                output[3] = input[channel[3]];
            }
        }
        return true;
    }
    for (uint32_t y = 0; y < surface.height; ++y) for (uint32_t x = 0; x < surface.width; ++x) {
        size_t offset;
        if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_TILED)
            offset = ((size_t(y / 32) * (surface.strideInPixels / 32) + x / 32) * 1024) + (y % 32) * 32 + x % 32;
        else if (surface.surfaceType == SCE_GXM_COLOR_SURFACE_SWIZZLED)
            offset = texture::encode_morton(x, y, surface.width, surface.height);
        else offset = size_t(y) * surface.strideInPixels + x;
        // Point-select the top-left native sample of each guest pixel, matching
        // the existing GL readback convention; never average packed data words.
        const auto *input = raw.data() + (size_t(y) * native_height / surface.height) * native_stride
            + (size_t(x) * native_width / surface.width) * native_pixel_bytes;
        auto *output = destination.data() + offset * pixel_bytes;
        if (packed565) {
            uint16_t word;
            std::memcpy(&word,input,2);
            if (reverse565) word=swap_565_red_blue(word);
            std::memcpy(output,&word,2);
        } else if (packed_u1) {
            uint16_t word;
            std::memcpy(&word,input,2);
            if (reverse_u1) word=swap_5551_red_blue(word,u1_mode<2);
            std::memcpy(output,&word,2);
        } else if (packed4444) {
            uint16_t word;
            std::memcpy(&word,input,2);
            word=permute_4444(word,packed_mapping);
            std::memcpy(output,&word,2);
        } else if (packed_e5 || packed_u2) {
            uint32_t word;
            std::memcpy(&word,input,4);
            if (reverse_e5) word=swap_e5_red_blue(word);
            if (packed_u2) word=packed_10_guest_word(word,u2_mode);
            std::memcpy(output,&word,4);
        } else for (uint32_t c = 0; c < components.count; ++c)
            std::memcpy(output + c * components.bytes, input + channel[c] * components.bytes, components.bytes);
    }
    return true;
}
static bool write_surface_memory_mapped(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges,
    uint32_t samples_x, uint32_t samples_y) {
    if (gxm::get_base_format(surface.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_S5S5U6)
        return s5s5u6_surface_memory(texture,surface,source,{},ranges,true,samples_x,samples_y);
    if (gxm::get_base_format(surface.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8)
        return u8s8s8u8_surface_memory(texture,surface,source,{},ranges,true,samples_x,samples_y);
    if (gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2)
        return u8u3u3u2_surface_memory(texture, surface, source, {}, ranges, true, samples_x, samples_y);
    if (gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10)
        return packed_float_surface_memory(texture, surface, source, {}, ranges, true,samples_x,samples_y);
    if (gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
        && texture && texture.pixelFormat == MTLPixelFormatRGBA16Float)
        return rgb9e5_expanded_surface_memory(texture,surface,source,{},ranges,true,samples_x,samples_y);
    const size_t size = surface_memory_size(surface);
    const auto components = surface_components(surface.colorFormat);
    if (!size || size > source.size() || !texture || !raw_storage_matches(texture.pixelFormat, components.native)
        || texture.textureType != MTLTextureType2D || texture.storageMode != MTLStorageModeShared
        || !texture.width || !texture.height || texture.width%samples_x || texture.height%samples_y) return false;
    size_t previous_end = 0;
    for (const auto &range : ranges) {
        if (!range.size || range.offset < previous_end || range.offset > size || range.size > size - range.offset)
            return false;
        previous_end = range.offset + range.size;
    }
    const bool packed565 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U5U6U5;
    const bool packed4444 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U4U4U4U4;
    const bool packed_u1 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U1U5U5U5;
    const bool packed_e5 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9;
    const bool packed_u2 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10;
    const bool packed_f11 = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_F11F11F10;
    if (packed565 && ((uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20) >= 2) return false;
    if (packed_u1 && ((uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20) >= 4) return false;
    if ((packed_e5 || packed_f11) && ((uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20) >= 2) return false;
    const bool reverse565 = packed565 && (uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) == SCE_GXM_COLOR_SWIZZLE3_BGR;
    const uint32_t u1_mode=(uint32_t(surface.colorFormat)&SCE_GXM_COLOR_SWIZZLE_MASK)>>20;
    const bool reverse_u1=packed_u1 && (u1_mode==0 || u1_mode==3);
    const bool reverse_e5 = packed_e5 && (uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) == SCE_GXM_COLOR_SWIZZLE3_RGB;
    const uint32_t u2_mode = (uint32_t(surface.colorFormat) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20;
    Map mapping, packed_mapping=identity;
    try {
        if (packed4444) packed_mapping=surface_memory_mapping(surface.colorFormat);
        mapping = packed565 || packed4444 || packed_u1 || packed_e5 || packed_u2 || packed_f11
            ? identity : surface_memory_mapping(surface.colorFormat);
    }
    catch (const std::runtime_error &) { return false; }
    const Map inverse_packed_mapping=packed4444 ? inverse_mapping(packed_mapping) : identity;
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
        const auto [top,bottom]=write_native_interval(y,surface.height,texture.height,samples_y);
        const auto [left,right]=write_native_interval(x,surface.width,texture.width,samples_x);
        for (size_t ny=top;ny<bottom;++ny)
            for (size_t nx=left;nx<right;++nx) {
                auto *native = raw.data()+ny*native_stride+nx*native_pixel_bytes;
                if (packed565) {
                    uint16_t word;
                    std::memcpy(&word,native,2);
                    if (reverse565) word=swap_565_red_blue(word);
                    auto *bytes=reinterpret_cast<uint8_t *>(&word);
                    for (size_t byte=0;byte<2;++byte) if (written[byte]) bytes[byte]=source[offset+byte];
                    if (reverse565) word=swap_565_red_blue(word);
                    std::memcpy(native,&word,2);
                } else if (packed_u1) {
                    uint16_t word;
                    std::memcpy(&word,native,2);
                    if (reverse_u1) word=swap_5551_red_blue(word,u1_mode<2);
                    auto *bytes=reinterpret_cast<uint8_t *>(&word);
                    for (size_t byte=0;byte<2;++byte) if (written[byte]) bytes[byte]=source[offset+byte];
                    if (reverse_u1) word=swap_5551_red_blue(word,u1_mode<2);
                    std::memcpy(native,&word,2);
                } else if (packed4444) {
                    uint16_t word;
                    std::memcpy(&word,native,2);
                    word=permute_4444(word,packed_mapping);
                    auto *bytes=reinterpret_cast<uint8_t *>(&word);
                    for (size_t byte=0;byte<2;++byte) if (written[byte]) bytes[byte]=source[offset+byte];
                    word=permute_4444(word,inverse_packed_mapping);
                    std::memcpy(native,&word,2);
                } else if (packed_e5 || packed_u2) {
                    uint32_t word;
                    std::memcpy(&word,native,4);
                    if (reverse_e5) word=swap_e5_red_blue(word);
                    if (packed_u2) word=packed_10_guest_word(word,u2_mode);
                    auto *bytes=reinterpret_cast<uint8_t *>(&word);
                    for (size_t byte=0;byte<4;++byte) if (written[byte]) bytes[byte]=source[offset+byte];
                    if (reverse_e5) word=swap_e5_red_blue(word);
                    if (packed_u2) word=packed_10_native_word(word,u2_mode);
                    std::memcpy(native,&word,4);
                } else for (size_t byte = 0; byte < pixel_bytes; ++byte) if (written[byte]) {
                    const size_t native_byte = channel[byte/components.bytes]*components.bytes + byte%components.bytes;
                    native[native_byte] = source[offset+byte];
                }
            }
        changed = true;
    }
    if (changed) [texture replaceRegion:MTLRegionMake2D(0,0,texture.width,texture.height)
        mipmapLevel:0 withBytes:raw.data() bytesPerRow:native_stride];
    return true;
}
bool write_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges) {
    return write_surface_memory_mapped(texture,surface,source,ranges,1,1);
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
    const SceGxmColorFormat *rendered_format, const ImportedTextureView *imported) {
    // GXP samplers are float-typed. Keep integer storage for guest bit-exact
    // uploads, but interpret those bits as float before applying One swizzles.
    // Otherwise an integer one becomes 0x00000001 in the float shader.
    MTLPixelFormat sampled_format = texture.pixelFormat;
    if (sampled_format == MTLPixelFormatR32Sint || sampled_format == MTLPixelFormatR32Uint)
        sampled_format = MTLPixelFormatR32Float;
    else if (sampled_format == MTLPixelFormatRG32Uint)
        sampled_format = MTLPixelFormatRG32Float;
    Map mapping;
    if (imported && imported->active) {
        // Imported DDS/PNG data uses the file's channel order, rather than the
        // original guest texture's RGB(A) swizzle. Match Vulkan's image view.
        mapping = imported->components >= 3 ? identity : texture_mapping(format);
        if (imported->components == 3) mapping[3] = O;
        bool swap_rb = imported->swap_rb;
        if (imported->base_format == SCE_GXM_TEXTURE_BASE_FORMAT_U5U6U5
            || imported->base_format == SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5)
            swap_rb = !swap_rb; // Native 565 and high-alpha 5551 are already BGR.
        if (imported->base_format == SCE_GXM_TEXTURE_BASE_FORMAT_U4U4U4U4) {
            std::swap(mapping[0], mapping[3]);
            std::swap(mapping[1], mapping[2]);
        }
        if (swap_rb) std::swap(mapping[0], mapping[2]);
    } else mapping = texture_mapping(format);
    if (rendered_format) {
        const auto memory = surface_memory_mapping(*rendered_format);
        for (auto &channel : mapping)
            for (size_t i = 0; i < 4; ++i)
                if (channel == identity[i]) { channel = memory[i]; break; }
    }
    if (mapping == identity && sampled_format == texture.pixelFormat) return texture;
    const NSUInteger slices = texture.textureType == MTLTextureTypeCube ? 6 : texture.arrayLength;
    id<MTLTexture> result = [texture newTextureViewWithPixelFormat:sampled_format
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
    // Single-level images are often render surfaces or packed data. Do not
    // widen their sampling footprint through the global anisotropy setting.
    if (texture.true_mip_count() <= 1)
        return 1;
    const uint32_t minimum = texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED
        ? texture.mag_filter : texture.min_filter;
    // Point-sampled surfaces may hold packed data. Keep renderer metadata and
    // the hardware sampler consistent when the global setting is higher.
    if (sampler_filter(minimum) == MTLSamplerMinMagFilterNearest
        && sampler_filter(texture.mag_filter) == MTLSamplerMinMagFilterNearest)
        return 1;
    return std::clamp(requested, 1u, 16u);
}
static MTLSamplerAddressMode sampler_address_mode(uint32_t mode) {
    switch (mode) {
    case SCE_GXM_TEXTURE_ADDR_REPEAT:
    case SCE_GXM_TEXTURE_ADDR_REPEAT_IGNORE_BORDER:
        return MTLSamplerAddressModeRepeat;
    case SCE_GXM_TEXTURE_ADDR_MIRROR:
        return MTLSamplerAddressModeMirrorRepeat;
    case SCE_GXM_TEXTURE_ADDR_CLAMP:
        return MTLSamplerAddressModeClampToEdge;
    case SCE_GXM_TEXTURE_ADDR_MIRROR_CLAMP:
        return MTLSamplerAddressModeMirrorClampToEdge;
    case SCE_GXM_TEXTURE_ADDR_CLAMP_FULL_BORDER:
    case SCE_GXM_TEXTURE_ADDR_CLAMP_IGNORE_BORDER:
    case SCE_GXM_TEXTURE_ADDR_CLAMP_HALF_BORDER:
        return MTLSamplerAddressModeClampToBorderColor;
    default:
        return MTLSamplerAddressModeClampToEdge;
    }
}
id<MTLSamplerState> make_sampler(Device &device, const SceGxmTexture &texture, uint32_t anisotropy) {
    auto desc = [MTLSamplerDescriptor new];
    // Strided descriptors reuse min_filter for the row stride. Their actual
    // minification filter is the same as their magnification filter.
    desc.minFilter = sampler_filter(texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED ? texture.mag_filter : texture.min_filter);
    desc.magFilter = sampler_filter(texture.mag_filter);
    desc.mipFilter = texture.true_mip_count() > 1 ? (texture.mip_filter ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest) : MTLSamplerMipFilterNotMipmapped;
    desc.lodMinClamp = texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED ? 0.f : float(metal_lod_min(texture));
    desc.sAddressMode = sampler_address_mode(texture.uaddr_mode);
    desc.tAddressMode = sampler_address_mode(texture.vaddr_mode);
    // Vulkan's unspecified VkSamplerCreateInfo::borderColor defaults to
    // VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK.
    desc.borderColor = MTLSamplerBorderColorTransparentBlack;
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
    default: throw std::runtime_error("Metal: cube shader requires swizzled face storage (type="
        + std::to_string(uint32_t(texture.texture_type())) + ", address="
        + std::to_string(uint32_t(texture.data_addr) << 2) + ", size="
        + std::to_string(gxm::get_width(texture)) + "x" + std::to_string(gxm::get_height(texture)) + ")");
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

static bool packed_color_cast_source(SceGxmColorBaseFormat base) {
    return base == SCE_GXM_COLOR_BASE_FORMAT_U5U6U5
        || base == SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2
        || base == SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8
        || base == SCE_GXM_COLOR_BASE_FORMAT_S5S5U6
        || base == SCE_GXM_COLOR_BASE_FORMAT_U4U4U4U4
        || base == SCE_GXM_COLOR_BASE_FORMAT_F11F11F10
        || base == SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
        || base == SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10
        || base == SCE_GXM_COLOR_BASE_FORMAT_U1U5U5U5;
}
static bool packed_texture_cast_target(SceGxmTextureBaseFormat texture) {
    return texture == SCE_GXM_TEXTURE_BASE_FORMAT_U5U6U5
        || texture == SCE_GXM_TEXTURE_BASE_FORMAT_U4U4U4U4
        || texture == SCE_GXM_TEXTURE_BASE_FORMAT_F11F11F10
        || texture == SCE_GXM_TEXTURE_BASE_FORMAT_SE5M9M9M9
        || texture == SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10
        || texture == SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5
        || texture == SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8;
}
bool surface_format_cast_supported(SceGxmColorFormat color, SceGxmTextureBaseFormat texture,
    uint32_t texture_swizzle) {
    const bool packed_source=packed_color_cast_source(gxm::get_base_format(color));
    // Cross-casts to 5:5:5:1 need the full texture mode to choose the alpha-bit layout.
    if (texture == SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5 && texture_swizzle >= 8) return false;
    if (texture == SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8 && texture_swizzle >= 2) return false;
    const auto source=surface_components(color), target=texture_components(texture);
    if (!source.count || !target.count || source.count*source.bytes != target.count*target.bytes) return false;
    try {
        const auto mapping=surface_memory_mapping(color);
        if (packed_source) return true;
        for (uint32_t c=0;c<source.count;++c)
            if (std::find(identity.begin(),identity.begin()+source.count,mapping[c])==identity.begin()+source.count) return false;
    } catch (const std::runtime_error &) { return false; }
    return true;
}
bool surface_format_cast_enqueueable(SceGxmColorFormat color, SceGxmTextureBaseFormat texture,
    uint32_t texture_swizzle) {
    return !packed_color_cast_source(gxm::get_base_format(color))
        && !packed_texture_cast_target(texture)
        && surface_format_cast_supported(color,texture,texture_swizzle);
}
bool surface_raw_cast_required(SceGxmColorFormat color, SceGxmTextureBaseFormat texture) {
    const auto source=surface_components(color), target=texture_components(texture);
    const auto base=gxm::get_base_format(color);
    return source.count*source.bytes==8 && target.count*target.bytes==8
        && (base==SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16 || base==SCE_GXM_COLOR_BASE_FORMAT_F32F32)
        && (source.native!=target.native || base==SCE_GXM_COLOR_BASE_FORMAT_F32F32);
}

bool surface_texture_needs_native_resolution(SceGxmColorFormat color, const SceGxmTexture &texture,
    float scale, bool use_texture_viewport) {
    if (!(scale > 0) || !std::isfinite(scale) || scale == std::floor(scale)
        || gxm::get_width(texture) > 256 || gxm::get_height(texture) > 256
        || texture.min_filter != SCE_GXM_TEXTURE_FILTER_POINT
        || texture.mag_filter != SCE_GXM_TEXTURE_FILTER_POINT)
        return false;
    const auto base = gxm::get_base_format(gxm::get_format(texture));
    SceGxmColorBaseFormat requested;
    if (!texture::convert_base_texture_format_to_base_color_format(base, requested)
        || gxm::bits_per_pixel(gxm::get_base_format(color)) != gxm::bits_per_pixel(base)
        || surface_raw_cast_required(color, base)) return false;
    // Plus returns the same-base-format texture viewport before considering
    // native reconstruction. Typeless and raw-word carriers keep their grids.
    return !use_texture_viewport || requested != gxm::get_base_format(color);
}

float packed_alias_x_offset(float scale, uint32_t native_alias_width) {
    if (scale <= 0 || !native_alias_width) throw std::runtime_error("Metal: invalid packed alias viewport");
    return (scale<1 ? 1.f : scale-1)/(2.0f*float(native_alias_width));
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
    case MTLPixelFormatRG32Float: case MTLPixelFormatRG32Uint: return MTLPixelFormatRG32Uint;
    case MTLPixelFormatRGBA32Float: return MTLPixelFormatRGBA32Uint;
    // Packed attachments must keep their storage words through MSAA seed,
    // expansion and clipped-sample restoration. Float reads canonicalize
    // NaN payloads in RG11B10Float before the color clip even draws.
    case MTLPixelFormatB5G6R5Unorm: case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm: case MTLPixelFormatA1BGR5Unorm:
        return MTLPixelFormatR16Uint;
    case MTLPixelFormatRG11B10Float: case MTLPixelFormatRGB9E5Float:
    case MTLPixelFormatBGR10A2Unorm:
        return MTLPixelFormatR32Uint;
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
    // A byte-offset view identifies one word for the whole image. Duplicate
    // it at non-unit scale; shifting interleaved columns changes the word as
    // the sampling position moves inside a rendered pixel.
    const bool separate_word = (config.y & 2u) != 0;
    const uint address = p.y*row_words + p.x + (separate_word ? 0u : config.z);
    const uint guest_x = (address%row_words)/2;
    const uint guest_y = address/row_words;
    const uint x = config.x ? guest_x*input.get_width()/config.x : guest_x;
    const uint y = config.w ? guest_y*input.get_height()/config.w : guest_y;
    // The shifted alias can extend past the final rendered word. Do not read
    // beyond the GPU surface; the unsupported trailing word is deterministic.
    const uint component = (separate_word ? config.z : address%2) ^ (config.y & 1u);
    const uint word = y < input.get_height() ? input.read(uint2(x,y))[component] : 0;
    output.write(uint4(word&255, (word>>8)&255, (word>>16)&255, word>>24),p);
}
kernel void unpack_halfwords(texture2d<uint, access::read> input [[texture(0)]],
    texture2d<uint, access::write> output [[texture(1)]], constant uint4 &config [[buffer(0)]],
    constant uint4 &memory_channels [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= output.get_width() || p.y >= output.get_height()) return;
    const uint row_words = output.get_width();
    const bool separate_word = config.y != 0;
    const uint address = p.y*row_words + p.x + (separate_word ? 0u : config.z);
    const uint guest_x = (address%row_words)/2;
    const uint guest_y = address/row_words;
    const uint x = config.x ? guest_x*input.get_width()/config.x : guest_x;
    const uint y = config.w ? guest_y*input.get_height()/config.w : guest_y;
    const uint half = (separate_word ? config.z : address%2)*2;
    uint word = 0;
    if (x < input.get_width() && y < input.get_height()) {
        const uint4 channels = input.read(uint2(x,y));
        word = (channels[memory_channels[half]] & 65535u)
            | ((channels[memory_channels[half+1]] & 65535u) << 16);
    }
    output.write(uint4(word&255u, (word>>8)&255u, (word>>16)&255u, word>>24),p);
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
kernel void scale_snapshot_words(texture2d<uint, access::read> input [[texture(0)]],
    texture2d<uint, access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= output.get_width() || p.y >= output.get_height()) return;
    const uint2 source_size(input.get_width(), input.get_height());
    const uint2 target_size(output.get_width(), output.get_height());
    const uint2 source_pixel = min(((p * 2u + 1u) * source_size) / (target_size * 2u), source_size - 1u);
    output.write(input.read(source_pixel), p);
}
float stencil_value(uint value, uint signed_mode) {
    value &= 255u;
    if (!signed_mode) return float(value)/255.f;
    int signed_value=int(value);
    if (signed_value>=128) signed_value-=256;
    return max(-1.f,float(signed_value)/127.f);
}
kernel void copy_stencil(texture2d<uint, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]], constant uint &signed_mode [[buffer(0)]],
    uint2 p [[thread_position_in_grid]]) {
    if (p.x < output.get_width() && p.y < output.get_height())
        output.write(float4(stencil_value(input.read(p).x,signed_mode)),p);
}
kernel void decode_rg_gamma(texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= output.get_width() || p.y >= output.get_height()) return;
    float4 value = input.read(p);
    value.rg = select(pow((value.rg + 0.055f) / 1.055f, float2(2.4f)),
        value.rg / 12.92f, value.rg <= 0.04045f);
    output.write(value, p);
}
uint2 sample_axis(uint p, uint output_size, uint native_size, uint guest_size, uint sample_extent) {
    if (output_size<guest_size) return uint2(min(p/sample_extent,native_size-1),p%sample_extent);
    const uint guest=min(((p+1)*guest_size-1)/output_size,guest_size-1);
    const uint groups=guest_size/sample_extent, group=guest/sample_extent;
    const uint first=group*native_size/groups, last=(group+1)*native_size/groups;
    const uint start=guest*output_size/guest_size, end=(guest+1)*output_size/guest_size;
    const uint native=first+min((p-start)*(last-first)/max(end-start,1u),last-first-1);
    return uint2(native,guest%sample_extent);
}
uint3 sample_address(uint2 p, uint2 output_size, uint2 native_size, uint2 guest_size, uint samples) {
    const uint2 x=sample_axis(p.x,output_size.x,native_size.x,guest_size.x,samples/2);
    const uint2 y=sample_axis(p.y,output_size.y,native_size.y,guest_size.y,2);
    return uint3(x.x,y.x,y.y*(samples/2)+x.y);
}
uint2 seed_axis(uint native, uint native_size, uint source_size, uint guest_size, uint sample_extent, uint sample) {
    const uint groups=guest_size/sample_extent;
    const uint group=min(native_size<groups ? native*groups/native_size
        : ((native+1)*groups-1)/native_size,groups-1);
    const uint guest=group*sample_extent+sample;
    if (source_size<guest_size)
        return uint2(min(native*sample_extent+sample,source_size-1),guest);
    const uint first=group*native_size/groups, last=(group+1)*native_size/groups;
    const uint start=guest*source_size/guest_size, end=(guest+1)*source_size/guest_size;
    // Select the first source pixel that expands back to this native pixel.
    return uint2(start+min(((native-first)*(end-start)+last-first-1)/max(last-first,1u),end-start-1),guest);
}
uint2 guest_sample_position(uint2 p, uint2 guest_size, uint2 native_size, uint samples) {
    const uint2 extent(samples/2,2), groups=guest_size/extent;
    return (p/extent)*native_size/groups;
}
uint3 guest_sample_address(uint2 p, uint2 guest_size, uint2 native_size, uint samples) {
    const uint2 extent(samples/2,2);
    const uint2 q=guest_sample_position(p,guest_size,native_size,samples);
    return uint3(q,(p.y%2)*extent.x+p.x%extent.x);
}
kernel void expand_samples(texture2d_ms<float,access::read> input [[texture(0)]],
    texture2d<float,access::write> output [[texture(1)]], constant uint2 &guest_size [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint3 q=sample_address(p,uint2(output.get_width(),output.get_height()),
        uint2(input.get_width(),input.get_height()),guest_size,input.get_num_samples());
    output.write(input.read(q.xy,q.z),p);
}
kernel void expand_integer_samples(texture2d_ms<uint,access::read> input [[texture(0)]],
    texture2d<uint,access::write> output [[texture(1)]], constant uint2 &guest_size [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint3 q=sample_address(p,uint2(output.get_width(),output.get_height()),
        uint2(input.get_width(),input.get_height()),guest_size,input.get_num_samples());
    output.write(input.read(q.xy,q.z),p);
}
kernel void resolve_raw_f16_samples(texture2d_ms<uint,access::read> input [[texture(0)]],
    texture2d<uint,access::read> resolved [[texture(1)]],
    texture2d<uint,access::write> output [[texture(2)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint4 first=input.read(p,0);
    uint4 differences=uint4(0);
    for (uint sample=1;sample<input.get_num_samples();++sample)
        differences|=input.read(p,sample)^first;
    // Pixel-frequency GXM shading writes the same raw word to every covered
    // sample. Preserve each channel independently so mixed coverage in another
    // channel cannot canonicalize an unchanged F16 NaN payload.
    output.write(select(resolved.read(p),first,differences==uint4(0)),p);
}
kernel void expand_depth(depth2d_ms<float,access::read> input [[texture(0)]],
    texture2d<float,access::write> output [[texture(1)]], constant uint2 &guest_size [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint3 q=sample_address(p,uint2(output.get_width(),output.get_height()),
        uint2(input.get_width(),input.get_height()),guest_size,input.get_num_samples());
    output.write(float4(input.read(q.xy,q.z)),p);
}
kernel void expand_stencil(texture2d_ms<uint,access::read> input [[texture(0)]],
    texture2d<float,access::write> output [[texture(1)]], constant uint3 &config [[buffer(0)]],
    uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    const uint3 q=sample_address(p,uint2(output.get_width(),output.get_height()),
        uint2(input.get_width(),input.get_height()),config.xy,input.get_num_samples());
    output.write(float4(stencil_value(input.read(q.xy,q.z).x,config.z)),p);
}
struct DepthReadback { float depth; uint stencil; };
kernel void store_depth_samples(depth2d<float,access::read> input [[texture(0)]],
    texture2d<uint,access::read> stencil [[texture(1)]], device DepthReadback *output [[buffer(0)]],
    constant uint4 &config [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if(p.x>=config.x || p.y>=config.y) return;
    // Select the first native pixel inside this guest pixel's interval.
    const uint2 native_size=uint2(input.get_width(),input.get_height());
    // A downscaled native pixel represents several guest pixels. Select the
    // same native pixel for all of them; upscaling retains the first pixel
    // inside each guest interval.
    const uint2 q=uint2(native_size.x<config.x ? p.x*native_size.x/config.x
            : (p.x*native_size.x+config.x-1)/config.x,
        native_size.y<config.y ? p.y*native_size.y/config.y
            : (p.y*native_size.y+config.y-1)/config.y);
    output[p.y*config.x+p.x]={config.w&1 ? input.read(q) : 0.f,config.w&2 ? stencil.read(q).x : 0u};
}
kernel void store_depth_samples_ms(depth2d_ms<float,access::read> input [[texture(0)]],
    texture2d_ms<uint,access::read> stencil [[texture(1)]], device DepthReadback *output [[buffer(0)]],
    constant uint4 &config [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if(p.x>=config.x || p.y>=config.y) return;
    const uint3 q=guest_sample_address(p,config.xy,uint2(input.get_width(),input.get_height()),input.get_num_samples());
    output[p.y*config.x+p.x]={config.w&1 ? input.read(q.xy,q.z) : 0.f,config.w&2 ? stencil.read(q.xy,q.z).x : 0u};
}
// The CPU path rounds a double-precision product by 2^24-1. A float product
// can round differently near half steps, so retain the exact float mantissa.
uint packed_depth_unorm(float depth,uint maximum) {
    if (!(depth>0.f)) return 0u;
    if (depth>=1.f) return maximum;
    const uint bits=as_type<uint>(depth);
    const uint exponent=(bits>>23)&255u;
    if (!exponent) return 0u;
    const uint shift=150u-exponent;
    if (shift>=64u) return 0u;
    const ulong product=ulong((bits&0x7fffffu)|0x800000u)*ulong(maximum);
    return uint((product+(1ul<<(shift-1u)))>>shift);
}
uint packed_depth24(float depth) { return packed_depth_unorm(depth,0xffffffu); }
kernel void copy_packed_depth(depth2d<float,access::read> input [[texture(0)]],
    texture2d<uint,access::read> stencil [[texture(1)]],
    texture2d<float,access::write> output [[texture(2)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x>=output.get_width() || p.y>=output.get_height()) return;
    // Match Metal's guest S8D24 store, including its exact quantization.
    // Texture channel swizzling and gamma decoding happen after packing.
    const uint word=packed_depth24(input.read(p)) | ((stencil.read(p).x&255u)<<24);
    output.write(float4(uint4(word,word>>8,word>>16,word>>24)&255u)/255.f,p);
}
kernel void store_packed_depth_samples(depth2d<float,access::read> input [[texture(0)]],
    texture2d<uint,access::read> stencil [[texture(1)]], device uint *output [[buffer(0)]],
    constant uint4 &config [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if(p.x>=config.x || p.y>=config.y) return;
    const uint2 native_size=uint2(input.get_width(),input.get_height());
    const uint2 q=uint2(native_size.x<config.x ? p.x*native_size.x/config.x
            : (p.x*native_size.x+config.x-1)/config.x,
        native_size.y<config.y ? p.y*native_size.y/config.y
            : (p.y*native_size.y+config.y-1)/config.y);
    const uint offset=config.z
        ? ((p.y/32u)*(config.z/32u)+p.x/32u)*1024u+(p.y%32u)*32u+p.x%32u
        : p.y*config.x+p.x;
    output[offset]=(config.w&4u)
        ? packed_depth24(input.read(q))|(stencil.read(q).x<<24)
        : as_type<uint>(input.read(q));
}
kernel void store_packed_depth_samples_ms(depth2d_ms<float,access::read> input [[texture(0)]],
    texture2d_ms<uint,access::read> stencil [[texture(1)]], device uint *output [[buffer(0)]],
    constant uint4 &config [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    if(p.x>=config.x || p.y>=config.y) return;
    const uint3 q=guest_sample_address(p,config.xy,uint2(input.get_width(),input.get_height()),input.get_num_samples());
    const uint offset=config.z
        ? ((p.y/32u)*(config.z/32u)+p.x/32u)*1024u+(p.y%32u)*32u+p.x%32u
        : p.y*config.x+p.x;
    output[offset]=(config.w&4u)
        ? packed_depth24(input.read(q.xy,q.z))|(stencil.read(q.xy,q.z).x<<24)
        : as_type<uint>(input.read(q.xy,q.z));
}
kernel void store_mask_samples(texture2d<float,access::read> input [[texture(0)]],
    device uint *output [[buffer(0)]], constant uint4 &config [[buffer(1)]],
    uint2 p [[thread_position_in_grid]]) {
    if(p.x>=config.x || p.y>=config.y) return;
    const uint2 native_size=uint2(input.get_width(),input.get_height());
    const uint2 q=uint2(native_size.x<config.x ? p.x*native_size.x/config.x
            : (p.x*native_size.x+config.x-1)/config.x,
        native_size.y<config.y ? p.y*native_size.y/config.y
            : (p.y*native_size.y+config.y-1)/config.y);
    output[p.y*config.x+p.x]=input.read(q).x>=0.5f;
}
kernel void store_mask_samples_ms(texture2d_ms<float,access::read> input [[texture(0)]],
    device uint *output [[buffer(0)]], constant uint4 &config [[buffer(1)]],
    uint2 p [[thread_position_in_grid]]) {
    if(p.x>=config.x || p.y>=config.y) return;
    const uint3 q=guest_sample_address(p,config.xy,uint2(input.get_width(),input.get_height()),input.get_num_samples());
    output[p.y*config.x+p.x]=input.read(q.xy,q.z).x>=0.5f;
}
struct DepthSeed { float depth [[depth(any)]]; uint stencil [[stencil]]; };
fragment DepthSeed resample_depth_grid(float4 position [[position]],
    depth2d_ms<float,access::read> source [[texture(0)]],
    texture2d_ms<uint,access::read> stencil [[texture(1)]], constant uint4 &grid [[buffer(0)]]) {
    // The Plus blit uses [0,D*s-(s-1)) -> [0,D), s=1 or 2.
    // Nearest selects expanded texel i*s. Resolution scaling still has to
    // map that texel to its native pixel and sample, via sample_address.
    const uint samples=source.get_num_samples();
    const uint2 p=uint2(position.xy)*uint2(samples/2,2);
    const uint3 q=sample_address(p,grid.xy,uint2(source.get_width(),source.get_height()),grid.zw,samples);
    return {source.read(q.xy,q.z),stencil.read(q.xy,q.z).x};
}
fragment float4 resample_depth_mask_grid(float4 position [[position]],
    texture2d_ms<float,access::read> source [[texture(0)]], constant uint4 &grid [[buffer(0)]]) {
    const uint samples=source.get_num_samples();
    const uint2 p=uint2(position.xy)*uint2(samples/2,2);
    const uint3 q=sample_address(p,grid.xy,uint2(source.get_width(),source.get_height()),grid.zw,samples);
    return source.read(q.xy,q.z);
}
fragment DepthSeed clear_depth_region_values(constant uint2 &values [[buffer(0)]]) {
    return {as_type<float>(values.x),values.y};
}
fragment float4 clear_mask_region_value(constant uint2 &values [[buffer(0)]]) {
    return float4(as_type<float>(values.x));
}
struct DepthCopy { float depth [[depth(any)]]; };
struct StencilCopy { uint stencil [[stencil]]; };
uint2 depth_patch_coordinate(uint2 native,uint sample,uint4 config,uint2 guest_size) {
    if (config.x==1) return native*guest_size/config.yz;
    const uint2 extent(config.x/2,2);
    return uint2(seed_axis(native.x,config.y,guest_size.x,guest_size.x,extent.x,sample%extent.x).y,
        seed_axis(native.y,config.z,guest_size.y,guest_size.y,extent.y,sample/extent.x).y);
}
float patched_depth(float previous,uint4 patch,uint kind) {
    // Integer guest bytes replace only their own bits. Preserve the native
    // value's remaining bytes before decoding the touched depth word again.
    const uint valid=kind==0 ? 0xffffu : kind==1 ? 0xffffffu : kind==3 ? 0x7fffffffu : 0xffffffffu;
    const uint changed=patch.y&valid;
    if (!changed) discard_fragment();
    const uint old=kind==0 ? packed_depth_unorm(previous,0xffffu)
        : kind==1 ? packed_depth24(previous) : as_type<uint>(previous);
    const uint value=((old&~changed)|(patch.x&changed))&valid;
    return kind==0 ? float(value)/65535.f : kind==1 ? float(value)/16777215.f : as_type<float>(value);
}
fragment DepthCopy patch_depth(float4 position [[position]],uint sample [[sample_id]],
    depth2d<float,access::read> previous [[texture(0)]],texture2d<uint,access::read> patches [[texture(1)]],
    constant uint4 &config [[buffer(0)]]) {
    const uint2 native=uint2(position.xy);
    const uint2 p=depth_patch_coordinate(native,sample,config,uint2(patches.get_width(),patches.get_height()));
    return {patched_depth(previous.read(native),patches.read(p),config.w)};
}
fragment DepthCopy patch_depth_ms(float4 position [[position]],uint sample [[sample_id]],
    depth2d_ms<float,access::read> previous [[texture(0)]],texture2d<uint,access::read> patches [[texture(1)]],
    constant uint4 &config [[buffer(0)]]) {
    const uint2 native=uint2(position.xy);
    const uint2 p=depth_patch_coordinate(native,sample,config,uint2(patches.get_width(),patches.get_height()));
    return {patched_depth(previous.read(native,sample),patches.read(p),config.w)};
}
fragment StencilCopy patch_stencil(float4 position [[position]],uint sample [[sample_id]],
    texture2d<uint,access::read> patches [[texture(1)]],constant uint4 &config [[buffer(0)]]) {
    const uint2 p=depth_patch_coordinate(uint2(position.xy),sample,config,uint2(patches.get_width(),patches.get_height()));
    const uint4 patch=patches.read(p);
    if (!patch.w) discard_fragment();
    return {patch.z};
}
fragment float4 patch_depth_mask(float4 position [[position]],uint sample [[sample_id]],
    texture2d<uint,access::read> patches [[texture(1)]],constant uint4 &config [[buffer(0)]]) {
    const uint2 p=depth_patch_coordinate(uint2(position.xy),sample,config,uint2(patches.get_width(),patches.get_height()));
    const uint4 patch=patches.read(p);
    if (!(patch.y&0x80000000u)) discard_fragment();
    return float4(float(patch.x>>31));
}
fragment DepthCopy copy_depth_region(float4 position [[position]],
    depth2d<float,access::read> source [[texture(0)]], constant uint4 &origins [[buffer(0)]]) {
    return {source.read(uint2(position.xy)-origins.zw+origins.xy)};
}
fragment DepthCopy copy_depth_region_ms(float4 position [[position]], uint sample [[sample_id]],
    depth2d_ms<float,access::read> source [[texture(0)]], constant uint4 &origins [[buffer(0)]]) {
    return {source.read(uint2(position.xy)-origins.zw+origins.xy,sample)};
}
fragment StencilCopy copy_stencil_region(float4 position [[position]],
    texture2d<uint,access::read> source [[texture(0)]], constant uint4 &origins [[buffer(0)]]) {
    return {source.read(uint2(position.xy)-origins.zw+origins.xy).x};
}
fragment StencilCopy copy_stencil_region_ms(float4 position [[position]], uint sample [[sample_id]],
    texture2d_ms<uint,access::read> source [[texture(0)]], constant uint4 &origins [[buffer(0)]]) {
    return {source.read(uint2(position.xy)-origins.zw+origins.xy,sample).x};
}
fragment DepthSeed seed_depth(float4 position [[position]],uint sample [[sample_id]],
    texture2d<float,access::read> depths [[texture(0)]],texture2d<uint,access::read> stencils [[texture(1)]],
    constant uint4 &config [[buffer(0)]]) {
    uint2 p=uint2(position.xy)*uint2(depths.get_width(),depths.get_height())/config.yz;
    if(config.x>1) {
        const uint2 extent(config.x/2,2),native=uint2(position.xy);
        p=uint2(seed_axis(native.x,config.y,depths.get_width(),depths.get_width(),extent.x,sample%extent.x).y,
            seed_axis(native.y,config.z,depths.get_height(),depths.get_height(),extent.y,sample/extent.x).y);
    }
    return {depths.read(p).x,stencils.read(p).x};
}
fragment float4 seed_mask(float4 position [[position]],uint sample [[sample_id]],
    texture2d<float,access::read> bits [[texture(0)]],constant uint4 &config [[buffer(0)]]) {
    uint2 p=uint2(position.xy)*uint2(bits.get_width(),bits.get_height())/config.yz;
    if(config.x>1) {
        const uint2 extent(config.x/2,2),native=uint2(position.xy);
        p=uint2(seed_axis(native.x,config.y,bits.get_width(),bits.get_width(),extent.x,sample%extent.x).y,
            seed_axis(native.y,config.z,bits.get_height(),bits.get_height(),extent.y,sample/extent.x).y);
    }
    return float4(bits.read(p).x);
}
vertex float4 seed_vs(uint id [[vertex_id]]) {
    const float2 p[]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    return float4(p[id],0,1);
}
fragment float4 seed_fs(float4 position [[position]], uint sample [[sample_id]],
    texture2d<float,access::read> source [[texture(0)]], constant uint4 &config [[buffer(0)]],
    constant uint2 &mode [[buffer(1)]]) {
    uint2 p=uint2(position.xy);
    if(mode.y) {
        const uint2 extent(mode.x/2,2),native=p;
        p=uint2(seed_axis(native.x,config.z,source.get_width(),config.x,extent.x,sample%extent.x).x,
            seed_axis(native.y,config.w,source.get_height(),config.y,extent.y,sample/extent.x).x);
    }
    return source.read(p);
}
fragment uint4 seed_fs_uint(float4 position [[position]], uint sample [[sample_id]],
    texture2d<uint,access::read> source [[texture(0)]], constant uint4 &config [[buffer(0)]],
    constant uint2 &mode [[buffer(1)]]) {
    uint2 p=uint2(position.xy);
    if(mode.y) {
        const uint2 extent(mode.x/2,2),native=p;
        p=uint2(seed_axis(native.x,config.z,source.get_width(),config.x,extent.x,sample%extent.x).x,
            seed_axis(native.y,config.w,source.get_height(),config.y,extent.y,sample/extent.x).x);
    }
    return source.read(p);
}
fragment uint4 restore_clip_uint(float4 position [[position]], uint sample [[sample_id]],
    texture2d_ms<uint,access::read> source [[texture(0)]],
    constant uint4 &clip [[buffer(0)]], constant uint2 &guest_size [[buffer(1)]]) {
    const uint2 native=uint2(position.xy);
    const uint sx=source.get_num_samples()/2;
    const uint2 guest=uint2(
        seed_axis(native.x,source.get_width(),source.get_width(),guest_size.x,sx,sample%sx).y,
        seed_axis(native.y,source.get_height(),source.get_height(),guest_size.y,2,sample/sx).y);
    if (all(guest>=clip.xy) && all(guest<=clip.zw)) discard_fragment();
    return source.read(native,sample);
}
fragment float4 restore_clip_float(float4 position [[position]], uint sample [[sample_id]],
    texture2d_ms<float,access::read> source [[texture(0)]],
    constant uint4 &clip [[buffer(0)]], constant uint2 &guest_size [[buffer(1)]]) {
    const uint2 native=uint2(position.xy);
    const uint sx=source.get_num_samples()/2;
    const uint2 guest=uint2(
        seed_axis(native.x,source.get_width(),source.get_width(),guest_size.x,sx,sample%sx).y,
        seed_axis(native.y,source.get_height(),source.get_height(),guest_size.y,2,sample/sx).y);
    if (all(guest>=clip.xy) && all(guest<=clip.zw)) discard_fragment();
    return source.read(native,sample);
}
struct PublicationBlit { float4 source; float4 destination; };
vertex float4 publication_vs(uint index [[vertex_id]]) {
    const float2 vertices[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
    return float4(vertices[index],0,1);
}
float2 publication_coordinate(float2 position, constant PublicationBlit &region) {
    return region.source.xy + (position - region.destination.xy)
        * region.source.zw / region.destination.zw;
}
fragment float4 publication_linear(float4 position [[position]],
    texture2d<float> source [[texture(0)]], constant PublicationBlit &region [[buffer(0)]]) {
    constexpr sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 size = float2(source.get_width(),source.get_height());
    return source.sample(linear_sampler,publication_coordinate(position.xy,region)/size,level(0));
}
fragment float4 publication_clipped(float4 position [[position]],
    texture2d<float> source [[texture(0)]], constant PublicationBlit &region [[buffer(0)]],
    constant float4 &guest_clip [[buffer(1)]], constant float4 &source_clip [[buffer(2)]]) {
    constexpr sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 coordinate = publication_coordinate(position.xy,region);
    // The render target was restored outside the clip before this resample.
    // Keep the bilinear footprint on the same side of each restored edge.
    if (position.x < guest_clip.x) coordinate.x = min(coordinate.x,source_clip.x-0.5f);
    else if (position.x >= guest_clip.z) coordinate.x = max(coordinate.x,source_clip.z+0.5f);
    else coordinate.x = clamp(coordinate.x,source_clip.x+0.5f,source_clip.z-0.5f);
    if (position.y < guest_clip.y) coordinate.y = min(coordinate.y,source_clip.y-0.5f);
    else if (position.y >= guest_clip.w) coordinate.y = max(coordinate.y,source_clip.w+0.5f);
    else coordinate.y = clamp(coordinate.y,source_clip.y+0.5f,source_clip.w-0.5f);
    return source.sample(linear_sampler,coordinate/float2(source.get_width(),source.get_height()),level(0));
}
fragment uint4 publication_words(float4 position [[position]],
    texture2d<uint,access::read> source [[texture(0)]], constant PublicationBlit &region [[buffer(0)]]) {
    const uint2 maximum = uint2(source.get_width()-1,source.get_height()-1);
    const uint2 pixel = min(uint2(max(floor(publication_coordinate(position.xy,region)),float2(0))),maximum);
    return source.read(pixel);
})";
    NSError *error = nil;
    auto library = [device.native_device() newLibraryWithSource:source options:nil error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal: surface cast shader failed");
    pipeline = cached_compute(device, library, @"unpack_words");
    if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal: surface cast pipeline failed");
    halfword_unpack_pipeline = cached_compute(device, library, @"unpack_halfwords");
    depth_pipeline = cached_compute(device, library, @"copy_depth");
    if (!depth_pipeline) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal: depth copy pipeline failed");
    scaled_snapshot_pipeline = cached_compute(device, library, @"scale_snapshot_words");
    if (!scaled_snapshot_pipeline) throw std::runtime_error("Metal: depth-view scaling pipeline failed");
    packed_depth_snapshot_pipeline = cached_compute(device, library, @"copy_packed_depth");
    if (!packed_depth_snapshot_pipeline) throw std::runtime_error("Metal: packed depth snapshot pipeline failed");
    stencil_pipeline = cached_compute(device, library, @"copy_stencil");
    stencil_ms_pipeline = cached_compute(device, library, @"expand_stencil");
    if (!stencil_pipeline || !stencil_ms_pipeline) throw std::runtime_error("Metal: stencil snapshot pipeline failed");
    depth_store_pipeline=cached_compute(device, library, @"store_depth_samples");
    depth_store_ms_pipeline=cached_compute(device, library, @"store_depth_samples_ms");
    if(!depth_store_pipeline || !depth_store_ms_pipeline) throw std::runtime_error("Metal: depth readback pipeline failed");
    packed_depth_store_pipeline=cached_compute(device, library, @"store_packed_depth_samples");
    packed_depth_store_ms_pipeline=cached_compute(device, library, @"store_packed_depth_samples_ms");
    if(!packed_depth_store_pipeline || !packed_depth_store_ms_pipeline) throw std::runtime_error("Metal: packed depth readback pipeline failed");
    mask_store_pipeline=cached_compute(device, library, @"store_mask_samples");
    mask_store_ms_pipeline=cached_compute(device, library, @"store_mask_samples_ms");
    if(!mask_store_pipeline || !mask_store_ms_pipeline) throw std::runtime_error("Metal: mask readback pipeline failed");
    multisample_library=library;
    multisample_pipeline=cached_compute(device, library, @"expand_samples");
    multisample_depth_pipeline=cached_compute(device, library, @"expand_depth");
    multisample_integer_pipeline=cached_compute(device, library, @"expand_integer_samples");
    raw_multisample_resolve_pipeline=cached_compute(device, library, @"resolve_raw_f16_samples");
    if (!multisample_pipeline || !multisample_depth_pipeline || !multisample_integer_pipeline
        || !raw_multisample_resolve_pipeline) throw std::runtime_error("Metal: multisample copy pipeline failed");
}
void SurfaceCaster::resolve_raw_multisample(id<MTLTexture> source, id<MTLTexture> resolved,
    id<MTLTexture> destination, id<MTLCommandBuffer> commands) {
    if (!source || !resolved || !destination || !commands
        || source.textureType!=MTLTextureType2DMultisample
        || (source.sampleCount!=2 && source.sampleCount!=4)
        || resolved.textureType!=MTLTextureType2D || destination.textureType!=MTLTextureType2D
        || source.pixelFormat!=MTLPixelFormatRGBA16Float
        || resolved.pixelFormat!=MTLPixelFormatRGBA16Float
        || destination.pixelFormat!=MTLPixelFormatRGBA16Float
        || source.width!=resolved.width || source.height!=resolved.height
        || source.width!=destination.width || source.height!=destination.height
        || commands.status!=MTLCommandBufferStatusNotEnqueued)
        throw std::runtime_error("Metal: invalid raw F16 sample resolve");
    source=[source newTextureViewWithPixelFormat:MTLPixelFormatRGBA16Uint];
    resolved=[resolved newTextureViewWithPixelFormat:MTLPixelFormatRGBA16Uint];
    destination=[destination newTextureViewWithPixelFormat:MTLPixelFormatRGBA16Uint];
    if (!source || !resolved || !destination)
        throw std::runtime_error("Metal: cannot view raw F16 sample bits");
    auto encoder=[commands computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal: cannot encode raw F16 sample resolve");
    [encoder setComputePipelineState:raw_multisample_resolve_pipeline];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:resolved atIndex:1];
    [encoder setTexture:destination atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(destination.width,destination.height,1)
        threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
}
void SurfaceCaster::expand_multisample(id<MTLTexture> source, id<MTLTexture> destination, float scale,
    uint32_t guest_width, uint32_t guest_height, id<MTLCommandBuffer> pending_commands) {
    if (!scale || !source || source.textureType!=MTLTextureType2DMultisample || (source.sampleCount!=2 && source.sampleCount!=4)
        || !destination || destination.textureType!=MTLTextureType2D
        || (pending_commands && pending_commands.status!=MTLCommandBufferStatusNotEnqueued))
        throw std::runtime_error("Metal: invalid sample expansion");
    const uint32_t sx=uint32_t(source.sampleCount/2);
    if (!guest_width) guest_width=uint32_t(std::ceil(double(destination.width)/scale));
    if (!guest_height) guest_height=uint32_t(std::ceil(double(destination.height)/scale));
    if (!guest_width || !guest_height || guest_width%sx || guest_height%2
        || destination.width>source.width*sx || destination.height>source.height*2
        || !source.width || !source.height)
        throw std::runtime_error("Metal: invalid expanded sample guest extent");
    const auto bits=sample_bits_format(source.pixelFormat);
    const bool integer=bits!=MTLPixelFormatInvalid && bits==sample_bits_format(destination.pixelFormat);
    if (integer) {
        source=[source newTextureViewWithPixelFormat:bits]; destination=[destination newTextureViewWithPixelFormat:bits];
        if (!source || !destination) throw std::runtime_error("Metal: cannot view MSAA storage bits");
    }
    auto commands=pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface expand MSAA");
    auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:integer ? multisample_integer_pipeline
        : source.pixelFormat==MTLPixelFormatDepth32Float_Stencil8 ? multisample_depth_pipeline : multisample_pipeline];
    [encoder setTexture:source atIndex:0]; [encoder setTexture:destination atIndex:1];
    const uint32_t guest_size[]={guest_width,guest_height};
    [encoder setBytes:guest_size length:sizeof(guest_size) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(destination.width,destination.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (!pending_commands) {
        std::string error;
        if(!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
}
void SurfaceCaster::seed_multisample(id<MTLTexture> source, id<MTLTexture> destination, float scale, bool expanded,
    uint32_t guest_width, uint32_t guest_height, id<MTLCommandBuffer> pending_commands) {
    if (!scale || !source || source.textureType!=MTLTextureType2D || !destination || destination.textureType!=MTLTextureType2DMultisample
        || (destination.sampleCount!=2 && destination.sampleCount!=4)
        || (!expanded && (source.width!=destination.width || source.height!=destination.height))
        || (pending_commands && pending_commands.status!=MTLCommandBufferStatusNotEnqueued))
        throw std::runtime_error("Metal: invalid sample seed");
    const uint32_t sx=uint32_t(destination.sampleCount/2);
    if (!guest_width) guest_width=uint32_t(std::ceil(double(source.width)/scale));
    if (!guest_height) guest_height=uint32_t(std::ceil(double(source.height)/scale));
    if (expanded && (!guest_width || !guest_height || guest_width%sx || guest_height%2
        || source.width>destination.width*sx || source.height>destination.height*2
        || !destination.width || !destination.height))
        throw std::runtime_error("Metal: invalid sample seed guest extent");
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
    auto commands=pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface seed MSAA");
    auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture=destination;
    pass.colorAttachments[0].loadAction=MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) throw std::runtime_error("Metal: cannot encode sample seed");
    [encoder setRenderPipelineState:pipeline]; [encoder setFragmentTexture:source atIndex:0];
    const uint32_t config[]={guest_width,guest_height,uint32_t(destination.width),uint32_t(destination.height)};
    const uint32_t mode[]={uint32_t(destination.sampleCount),uint32_t(expanded)};
    [encoder setFragmentBytes:config length:sizeof(config) atIndex:0];
    [encoder setFragmentBytes:mode length:sizeof(mode) atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [encoder endEncoding];
    if (!pending_commands && !device.submit_and_wait(commands,error)) throw std::runtime_error(error);
}
void SurfaceCaster::restore_clipped_multisample(id<MTLTexture> source, id<MTLTexture> destination,
    const SceGxmColorSurface &surface, id<MTLCommandBuffer> commands,
    const MTLSamplePosition *sample_positions) {
    if (!source || !destination || !commands
        || source.textureType != MTLTextureType2DMultisample
        || destination.textureType != MTLTextureType2DMultisample
        || source.sampleCount != destination.sampleCount
        || (source.sampleCount != 2 && source.sampleCount != 4)
        || source.width != destination.width || source.height != destination.height
        || source.pixelFormat != destination.pixelFormat
        || commands.status != MTLCommandBufferStatusNotEnqueued)
        throw std::runtime_error("Metal: invalid samplewise color clip restoration");
    const MTLPixelFormat bits = sample_bits_format(source.pixelFormat);
    if (bits != MTLPixelFormatInvalid) {
        source = [source newTextureViewWithPixelFormat:bits];
        destination = [destination newTextureViewWithPixelFormat:bits];
        if (!source || !destination)
            throw std::runtime_error("Metal: cannot view clipped multisample color bits");
    }
    auto &pipeline = clip_pipelines[{uint32_t(destination.pixelFormat), uint32_t(destination.sampleCount)}];
    if (!pipeline) {
        auto descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [multisample_library newFunctionWithName:@"seed_vs"];
        descriptor.fragmentFunction = [multisample_library newFunctionWithName:
            bits == MTLPixelFormatInvalid ? @"restore_clip_float" : @"restore_clip_uint"];
        descriptor.colorAttachments[0].pixelFormat = destination.pixelFormat;
        descriptor.rasterSampleCount = destination.sampleCount;
        std::string error;
        pipeline = device.create_pipeline(descriptor, error);
        if (!pipeline) throw std::runtime_error(error);
    }
    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = destination;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (sample_positions) [pass setSamplePositions:sample_positions count:destination.sampleCount];
    auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) throw std::runtime_error("Metal: cannot encode samplewise color clip restoration");
    const uint32_t clip[] = {surface.clip_x_min, surface.clip_y_min,
        surface.clip_x_max, surface.clip_y_max};
    const uint32_t guest_size[] = {surface.width, surface.height};
    [encoder setRenderPipelineState:pipeline];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder setFragmentBytes:clip length:sizeof(clip) atIndex:0];
    [encoder setFragmentBytes:guest_size length:sizeof(guest_size) atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
}
bool SurfaceCaster::patch_multisample(id<MTLTexture> texture, const SceGxmColorSurface &surface, float scale,
    std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges) {
    const size_t size=surface_memory_size(surface);
    const auto components=surface_components(surface.colorFormat);
    const auto native = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10
        || (gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
            && texture && texture.pixelFormat == MTLPixelFormatRGBA16Float)
        ? MTLPixelFormatRGBA16Float : components.native;
    if (!size || source.size()<size || !scale || !texture || texture.textureType!=MTLTextureType2DMultisample
        || (texture.sampleCount!=2 && texture.sampleCount!=4)
        || !raw_storage_matches(texture.pixelFormat,native)) return false;
    const uint32_t samples=uint32_t(texture.sampleCount);
    const uint32_t sx=samples/2;
    const uint32_t packed_width=surface.downscale ? uint32_t(texture.width*sx) : uint32_t(surface.width*scale);
    const uint32_t packed_height=surface.downscale ? uint32_t(texture.height*2) : uint32_t(surface.height*scale);
    if ((surface.downscale ? texture.width!=uint32_t(surface.width*scale)
            : texture.width!=(packed_width+sx-1)/sx)
        || (surface.downscale ? texture.height!=uint32_t(surface.height*scale)
            : texture.height!=(packed_height+1)/2)) return false;
    size_t previous_end=0;
    for (const auto &range:ranges) {
        if (!range.size || range.offset<previous_end || range.offset>size || range.size>size-range.offset) return false;
        previous_end=range.offset+range.size;
    }
    if (ranges.empty()) return true;
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat
        width:packed_width height:packed_height mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
    auto packed=[device.native_device() newTextureWithDescriptor:desc];
    if (!packed) throw std::runtime_error("Metal: cannot allocate sample patch image");
    const uint32_t guest_width=surface.width*(surface.downscale ? sx : 1);
    const uint32_t guest_height=surface.height*(surface.downscale ? 2 : 1);
    expand_multisample(texture,packed,scale,guest_width,guest_height);
    // For resolved color the original guest descriptor maps each changed byte
    // to every sample of that pixel. Expanded color addresses samples separately.
    // The existing byte writer preserves all other components and native pixels.
    if (!write_surface_memory_mapped(packed,surface,source,ranges,
        surface.downscale && scale<1 ? sx : 1,surface.downscale && scale<1 ? 2 : 1)) return false;
    seed_multisample(packed,texture,scale,true,guest_width,guest_height);
    return true;
}
id<MTLTexture> SurfaceCaster::rendered_cube(std::span<const id<MTLTexture>> faces,
    id<MTLCommandBuffer> commands) {
    if (faces.size()!=6 || !faces[0] || !commands || commands.status!=MTLCommandBufferStatusNotEnqueued)
        throw std::runtime_error("Metal: invalid rendered cube faces");
    const auto first=faces[0];
    for (const auto face:faces)
        if (!face || face.textureType!=MTLTextureType2D || face.sampleCount!=1
            || face.width!=first.width || face.height!=first.width || face.pixelFormat!=first.pixelFormat)
            throw std::runtime_error("Metal: incompatible rendered cube faces");
    auto desc=[MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:first.pixelFormat size:first.width mipmapped:NO];
    desc.storageMode=MTLStorageModePrivate;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
    auto result=[device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate rendered cube");
    auto blit=[commands blitCommandEncoder];
    if (!blit) throw std::runtime_error("Metal: cannot encode rendered cube copy");
    blit.label=@"Vita3K six rendered cube faces";
    for (uint32_t face=0;face<6;++face)
        [blit copyFromTexture:faces[face] sourceSlice:0 sourceLevel:0
            sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(first.width,first.height,1)
            toTexture:result destinationSlice:face destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [blit endEncoding];
    return result;
}
id<MTLTexture> SurfaceCaster::cube_snapshot(id<MTLTexture> uploaded, std::span<const CubeSurface> surfaces, float scale,
    id<MTLCommandBuffer> pending_commands) {
    if (!uploaded || uploaded.textureType != MTLTextureTypeCube)
        throw std::runtime_error("Metal: invalid rendered cube source");
    return texture_snapshot(uploaded,surfaces,scale,pending_commands);
}
id<MTLTexture> SurfaceCaster::texture_snapshot(id<MTLTexture> uploaded, std::span<const CubeSurface> surfaces, float scale,
    id<MTLCommandBuffer> pending_commands) {
    if (!uploaded || (uploaded.textureType != MTLTextureTypeCube && uploaded.textureType != MTLTextureType2D)
        || !std::isfinite(scale) || scale <= 0
        || (pending_commands && pending_commands.status != MTLCommandBufferStatusNotEnqueued))
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
    const auto scaled_width = std::max(1u,uint32_t(double(uploaded.width) * scale));
    const auto scaled_height = std::max(1u,uint32_t(double(uploaded.height) * scale));
    auto desc = cube
        ? [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float size:scaled_width mipmapped:NO]
        : [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:scaled_width height:scaled_height mipmapped:NO];
    desc.mipmapLevelCount=uploaded.mipmapLevelCount;
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
    auto result=[device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: rendered cube allocation failed");
    auto commands=pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface texture snapshot");
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
    if (!pending_commands) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
    return result;
}

id<MTLTexture> SurfaceCaster::color_snapshot(id<MTLTexture> source, id<MTLCommandBuffer> pending_commands) {
    if (!source || source.textureType != MTLTextureType2D || source.sampleCount != 1)
        throw std::runtime_error("Metal: unsupported color feedback source");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
        width:source.width height:source.height mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate color feedback snapshot");
    auto commands = pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface color snapshot");
    auto encoder = [commands blitCommandEncoder];
    [encoder copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
        sourceSize:MTLSizeMake(source.width,source.height,1) toTexture:result destinationSlice:0
        destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [encoder endEncoding];
    if (!pending_commands) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
    return result;
}
id<MTLTexture> SurfaceCaster::snapshot_subrectangle(id<MTLTexture> source, uint32_t width, uint32_t height, SurfaceRect rect) {
    auto commands=surface_command_buffer(device, @"Vita3K surface subrectangle");
    auto result=enqueue_subrectangle(source,width,height,rect,commands);
    std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}
id<MTLTexture> SurfaceCaster::scaled_snapshot(id<MTLTexture> source, uint32_t width, uint32_t height,
    id<MTLCommandBuffer> pending_commands) {
    if (!source || source.textureType != MTLTextureType2D || source.sampleCount != 1
        || !width || !height || width > source.width || height > source.height
        || (pending_commands && pending_commands.status != MTLCommandBufferStatusNotEnqueued))
        throw std::runtime_error("Metal: invalid scaled depth snapshot");
    const auto bits = sample_bits_format(source.pixelFormat);
    if (bits == MTLPixelFormatInvalid)
        throw std::runtime_error("Metal: unsupported scaled depth snapshot format");
    auto input = [source newTextureViewWithPixelFormat:bits];
    if (!input) throw std::runtime_error("Metal: cannot view depth snapshot words");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
        width:width height:height mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    auto output = [result newTextureViewWithPixelFormat:bits];
    if (!result || !output) throw std::runtime_error("Metal: cannot allocate scaled depth snapshot");
    auto commands = pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K scaled depth view");
    auto encoder = [commands computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal: cannot encode scaled depth view");
    [encoder setComputePipelineState:scaled_snapshot_pipeline];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:output atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(width,height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (!pending_commands) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
    return result;
}
id<MTLTexture> SurfaceCaster::enqueue_subrectangle(id<MTLTexture> source, uint32_t width, uint32_t height,
    SurfaceRect rect, id<MTLCommandBuffer> commands) {
    if (!source || source.textureType!=MTLTextureType2D || source.sampleCount!=1
        || !width || !height
        || !rect.width || !rect.height || uint64_t(rect.x)+rect.width>width || uint64_t(rect.y)+rect.height>height
        || !commands || commands.status!=MTLCommandBufferStatusNotEnqueued)
        throw std::runtime_error("Metal: invalid snapshot subrectangle");
    const size_t x=uint64_t(rect.x)*source.width/width,y=uint64_t(rect.y)*source.height/height;
    const size_t w=std::max<size_t>(1,(uint64_t(rect.x)+rect.width)*source.width/width-x);
    const size_t h=std::max<size_t>(1,(uint64_t(rect.y)+rect.height)*source.height/height-y);
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat width:w height:h mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
    auto result=[device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate snapshot subrectangle");
    auto encoder=[commands blitCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal: cannot begin subrectangle copy");
    [encoder copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x,y,0)
        sourceSize:MTLSizeMake(w,h,1) toTexture:result destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [encoder endEncoding];
    return result;
}
id<MTLTexture> SurfaceCaster::color_subrectangle(id<MTLTexture> source, const SceGxmColorSurface &surface, SurfaceRect rect) {
    return snapshot_subrectangle(source,surface.width,surface.height,rect);
}
id<MTLTexture> SurfaceCaster::resample_publication(id<MTLTexture> source, uint32_t width, uint32_t height,
    SurfaceRect source_rect, SurfaceRect destination_rect, bool raw_words, id<MTLCommandBuffer> pending_commands,
    const PublicationClip *clip) {
    if (!source || source.textureType != MTLTextureType2D || source.sampleCount != 1
        || !source.width || !source.height || !width || !height
        || (pending_commands && pending_commands.status != MTLCommandBufferStatusNotEnqueued)
        || !destination_rect.width || !destination_rect.height
        || uint64_t(destination_rect.x) + destination_rect.width > width
        || uint64_t(destination_rect.y) + destination_rect.height > height
        || uint64_t(source_rect.x) + source_rect.width > source.width
        || uint64_t(source_rect.y) + source_rect.height > source.height)
        throw std::runtime_error("Metal: invalid surface publication blit");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
        width:width height:height mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate surface publication image");
    id<MTLTexture> output = result;
    if (raw_words) {
        const auto bits = sample_bits_format(source.pixelFormat);
        if (bits == MTLPixelFormatInvalid)
            throw std::runtime_error("Metal: cannot preserve publication source bits");
        source = [source newTextureViewWithPixelFormat:bits];
        output = [result newTextureViewWithPixelFormat:bits];
        if (!source || !output) throw std::runtime_error("Metal: cannot view publication storage bits");
    }
    const bool clipped = clip && !raw_words;
    auto &pipeline = publication_pipelines[uint32_t(output.pixelFormat)*2 + clipped];
    std::string error;
    if (!pipeline) {
        auto descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [multisample_library newFunctionWithName:@"publication_vs"];
        descriptor.fragmentFunction = [multisample_library newFunctionWithName:raw_words ? @"publication_words"
            : clipped ? @"publication_clipped" : @"publication_linear"];
        descriptor.colorAttachments[0].pixelFormat = output.pixelFormat;
        pipeline = device.create_pipeline(descriptor,error);
        if (!pipeline) throw std::runtime_error(error);
    }
    auto commands = pending_commands ? pending_commands : surface_command_buffer(device,@"Vita3K surface publication blit");
    if (!commands) throw std::runtime_error("Metal: cannot allocate publication command buffer");
    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = output;
    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) throw std::runtime_error("Metal: cannot begin publication blit");
    const float region[] = {float(source_rect.x),float(source_rect.y),float(source_rect.width),float(source_rect.height),
        float(destination_rect.x),float(destination_rect.y),float(destination_rect.width),float(destination_rect.height)};
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:MTLViewport{double(destination_rect.x),double(destination_rect.y),
        double(destination_rect.width),double(destination_rect.height),0,1}];
    [encoder setScissorRect:MTLScissorRect{destination_rect.x,destination_rect.y,destination_rect.width,destination_rect.height}];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder setFragmentBytes:region length:sizeof(region) atIndex:0];
    if (clipped) {
        const float guest_clip[] = {float(clip->guest.x),float(clip->guest.y),
            float(clip->guest.x+clip->guest.width),float(clip->guest.y+clip->guest.height)};
        const float source_clip[] = {float(clip->source.x),float(clip->source.y),
            float(clip->source.x+clip->source.width),float(clip->source.y+clip->source.height)};
        [encoder setFragmentBytes:guest_clip length:sizeof(guest_clip) atIndex:1];
        [encoder setFragmentBytes:source_clip length:sizeof(source_clip) atIndex:2];
    }
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    if (!pending_commands && !device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}
id<MTLTexture> SurfaceCaster::rgba8_from_rg32(id<MTLTexture> source, bool swap_words,
    uint32_t word_offset, bool signed_normalized, uint32_t guest_width, uint32_t guest_height,
    id<MTLCommandBuffer> pending_commands, bool separate_word) {
    if (!source.width || !source.height || source.pixelFormat != MTLPixelFormatRG32Float || word_offset > 1
        || bool(guest_width)!=bool(guest_height)
        || (pending_commands && pending_commands.status!=MTLCommandBufferStatusNotEnqueued))
        throw std::runtime_error("Metal: invalid RG32 surface cast dimensions/format");
    auto words = [source newTextureViewWithPixelFormat:MTLPixelFormatRG32Uint];
    if (!words) throw std::runtime_error("Metal: cannot view packed RG32 color bits");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:signed_normalized ? MTLPixelFormatRGBA8Snorm : MTLPixelFormatRGBA8Unorm
        width:(guest_width ? guest_width : source.width)*2 height:guest_height ? guest_height : source.height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    desc.storageMode = MTLStorageModeShared;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate reinterpreted color surface");
    auto bytes = [result newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Uint];
    if (!bytes) throw std::runtime_error("Metal: cannot write raw reinterpreted color bytes");
    auto commands = pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface RG32 cast");
    auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:words atIndex:0];
    [encoder setTexture:bytes atIndex:1];
    const uint32_t config[] = {guest_width, uint32_t(swap_words) | (separate_word ? 2u : 0u), word_offset, guest_height};
    [encoder setBytes:config length:sizeof(config) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (!pending_commands) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
    return result;
}
id<MTLTexture> SurfaceCaster::rgba8_from_rgba16(id<MTLTexture> source, SceGxmColorFormat color,
    uint32_t word_offset, bool signed_normalized, uint32_t guest_width, uint32_t guest_height,
    id<MTLCommandBuffer> pending_commands, bool separate_word) {
    if (!source.width || !source.height || (source.pixelFormat != MTLPixelFormatRGBA16Float
        && source.pixelFormat != MTLPixelFormatRGBA16Uint) || word_offset > 1
        || gxm::get_base_format(color) != SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16
        || bool(guest_width) != bool(guest_height)
        || (pending_commands && pending_commands.status != MTLCommandBufferStatusNotEnqueued))
        throw std::runtime_error("Metal: invalid RGBA16 surface word cast");
    const auto mapping = surface_memory_mapping(color);
    uint32_t memory_channels[4];
    for (uint32_t channel = 0; channel < 4; ++channel) {
        const auto found = std::find(identity.begin(), identity.end(), mapping[channel]);
        if (found == identity.end()) throw std::runtime_error("Metal: invalid RGBA16 guest channel mapping");
        memory_channels[channel] = uint32_t(found - identity.begin());
    }
    auto words = [source newTextureViewWithPixelFormat:MTLPixelFormatRGBA16Uint];
    if (!words) throw std::runtime_error("Metal: cannot view RGBA16 surface halfwords");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
        signed_normalized ? MTLPixelFormatRGBA8Snorm : MTLPixelFormatRGBA8Unorm
        width:(guest_width ? guest_width : source.width)*2
        height:guest_height ? guest_height : source.height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    desc.storageMode = MTLStorageModeShared;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate RGBA16 word alias");
    auto bytes = [result newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Uint];
    if (!bytes) throw std::runtime_error("Metal: cannot write RGBA16 word alias bytes");
    auto commands = pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface RGBA16 word cast");
    auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:halfword_unpack_pipeline];
    [encoder setTexture:words atIndex:0];
    [encoder setTexture:bytes atIndex:1];
    const uint32_t config[] = {guest_width, uint32_t(separate_word), word_offset, guest_height};
    [encoder setBytes:config length:sizeof(config) atIndex:0];
    [encoder setBytes:memory_channels length:sizeof(memory_channels) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (!pending_commands) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
    return result;
}
id<MTLTexture> SurfaceCaster::surface_format_cast(id<MTLTexture> source, SceGxmColorFormat color,
    SceGxmTextureBaseFormat texture, uint32_t texture_swizzle, id<MTLCommandBuffer> pending_commands,
    bool raw_bits) {
    const auto input=surface_components(color);
    auto output=texture_components(texture);
    if (!source || source.textureType!=MTLTextureType2D || source.sampleCount!=1
        || !surface_format_cast_supported(color,texture,texture_swizzle)
        || !(raw_storage_matches(source.pixelFormat,input.native)
            || (gxm::get_base_format(color)==SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
                && source.pixelFormat==MTLPixelFormatRGBA16Float))) return nil;
    if (raw_bits) {
        if (input.count*input.bytes!=8 || output.count*output.bytes!=8) return nil;
        // Keep guest memory order; these are word halves, not guest texture
        // channels. The shader reconstructs the words after filtering.
        output={4,2,MTLPixelFormatRGBA16Unorm};
    }
    if (texture == SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5 && (texture_swizzle & 2))
        output.native=MTLPixelFormatA1BGR5Unorm;
    const bool packed_source=packed_color_cast_source(gxm::get_base_format(color));
    const bool packed_target=packed_texture_cast_target(texture);
    if (pending_commands && (packed_source || packed_target
        || pending_commands.status!=MTLCommandBufferStatusNotEnqueued)) return nil;
    if (packed_source || packed_target) {
        // getBytes preserves packed storage bits; read_surface_memory then puts
        // them into the guest's channel/bit order. The target format interprets
        // those same guest bytes without any normalized float round-trip.
        if (source.storageMode != MTLStorageModeShared) return nil;
        SceGxmColorSurface surface{};
        surface.colorFormat=color;
        surface.width=uint32_t(source.width);
        surface.height=uint32_t(source.height);
        surface.strideInPixels=surface.width;
        surface.surfaceType=SCE_GXM_COLOR_SURFACE_LINEAR;
        const size_t pitch=size_t(source.width)*input.count*input.bytes;
        std::vector<uint8_t> bytes(pitch*source.height);
        if (!read_surface_memory(source,surface,bytes)) return nil;
        if (texture == SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10
            && texture_swizzle < 8 && (texture_swizzle & 2)) {
            for (size_t offset=0;offset<bytes.size();offset+=4) {
                uint32_t guest;
                std::memcpy(&guest,bytes.data()+offset,4);
                const uint32_t native=packed_10_native_word(guest,texture_swizzle&3);
                std::memcpy(bytes.data()+offset,&native,4);
            }
        }
        std::vector<uint16_t> decoded;
        const void *upload_bytes=bytes.data();
        size_t upload_pitch=pitch;
        if (texture == SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8) {
            decoded.resize(size_t(source.width)*source.height*4);
            const auto format=static_cast<SceGxmTextureFormat>(uint32_t(texture) | (texture_swizzle<<12));
            renderer::texture::convert_x8s8s8u8_to_f16f16f16f16(decoded.data(),bytes.data(),
                uint32_t(source.width),uint32_t(source.height),format);
            upload_bytes=decoded.data();
            upload_pitch=size_t(source.width)*8;
        }
        auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:output.native
            width:source.width height:source.height mipmapped:NO];
        desc.storageMode=MTLStorageModeShared;
        desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
        auto result=[device.native_device() newTextureWithDescriptor:desc];
        if (!result) return nil;
        [result replaceRegion:MTLRegionMake2D(0,0,source.width,source.height)
            mipmapLevel:0 withBytes:upload_bytes bytesPerRow:upload_pitch];
        return result;
    }
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
    auto commands=pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface format cast");
    auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:component_cast_pipeline];
    [encoder setTexture:raw_input atIndex:0];[encoder setTexture:raw_output atIndex:1];
    const uint32_t config[]={input.count,input.bytes,output.count,output.bytes};
    [encoder setBytes:config length:sizeof(config) atIndex:0];
    [encoder setBytes:channels.data() length:sizeof(channels) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (!pending_commands) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
    return result;
}

bool raw_texture_snapshot_supported(id<MTLTexture> source) {
    if (!source || source.sampleCount!=1
        || (source.textureType!=MTLTextureType2D && source.textureType!=MTLTextureTypeCube)) return false;
    switch (source.pixelFormat) {
    case MTLPixelFormatRGBA16Float: case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRG32Float: case MTLPixelFormatRG32Uint: return true;
    default: return false;
    }
}
id<MTLTexture> SurfaceCaster::raw_texture_snapshot(id<MTLTexture> source, id<MTLCommandBuffer> pending_commands) {
    if (!raw_texture_snapshot_supported(source)) return nil;
    uint32_t count, bytes;
    switch (source.pixelFormat) {
    case MTLPixelFormatRGBA16Float: case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm:
        count=4; bytes=2; break;
    case MTLPixelFormatRG32Float: case MTLPixelFormatRG32Uint:
        count=2; bytes=4; break;
    default: return nil;
    }
    if (pending_commands && pending_commands.status!=MTLCommandBufferStatusNotEnqueued) return nil;
    if (source.pixelFormat==MTLPixelFormatRGBA16Unorm) return source;
    auto view=[source newTextureViewWithPixelFormat:MTLPixelFormatRGBA16Unorm];
    if (view) return view;
    // Cross-component views need not be supported. Split the same integer
    // storage words explicitly without a float read on any face or mip.
    if (!component_cast_pipeline)
        component_cast_pipeline=cached_compute(device,multisample_library,@"repack_components");
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Unorm
        width:source.width height:source.height mipmapped:NO];
    desc.textureType=source.textureType;
    desc.mipmapLevelCount=source.mipmapLevelCount;
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
    auto result=[device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: raw texture carrier allocation failed");
    auto commands=pending_commands ? pending_commands : surface_command_buffer(device,@"Vita3K raw texture carrier");
    auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:component_cast_pipeline];
    const uint32_t config[]={count,bytes,4,2}, mapping[]={0,1,2,3};
    [encoder setBytes:config length:sizeof(config) atIndex:0];
    [encoder setBytes:mapping length:sizeof(mapping) atIndex:1];
    for (uint32_t face=0;face<(source.textureType==MTLTextureTypeCube ? 6u : 1u);++face)
        for (uint32_t mip=0;mip<source.mipmapLevelCount;++mip) {
            auto input=[source newTextureViewWithPixelFormat:sample_bits_format(source.pixelFormat)
                textureType:MTLTextureType2D levels:NSMakeRange(mip,1) slices:NSMakeRange(face,1)];
            auto output=[result newTextureViewWithPixelFormat:MTLPixelFormatRGBA16Uint
                textureType:MTLTextureType2D levels:NSMakeRange(mip,1) slices:NSMakeRange(face,1)];
            if (!input || !output) throw std::runtime_error("Metal: raw texture carrier integer view failed");
            [encoder setTexture:input atIndex:0];[encoder setTexture:output atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(output.width,output.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
        }
    [encoder endEncoding];
    if (!pending_commands) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
    return result;
}

id<MTLTexture> SurfaceCaster::rgba8_memory_snapshot(id<MTLTexture> source, SceGxmColorFormat format) {
    return rgba8_surface_sampling(source, format, 0);
}
id<MTLTexture> SurfaceCaster::rgba8_surface_sampling(id<MTLTexture> source, SceGxmColorFormat format, uint32_t gamma,
    id<MTLCommandBuffer> pending_commands) {
    if (gamma != 0 && gamma != 1 && gamma != 3)
        throw std::runtime_error("Metal: invalid RGBA8 surface texture gamma mode");
    if (pending_commands && pending_commands.status != MTLCommandBufferStatusNotEnqueued)
        throw std::runtime_error("Metal: gamma conversion requires an open command buffer");
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
    auto commands = pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface RGBA8 sampling");
    auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:gamma == 3 ? rg_gamma_pipeline : depth_pipeline];
    [encoder setTexture:memory atIndex:0]; [encoder setTexture:result atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (!pending_commands) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    }
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
    case MTLPixelFormatB5G6R5Unorm: case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatRGB9E5Float: case MTLPixelFormatBGR10A2Unorm: break;
    case MTLPixelFormatA1BGR5Unorm: case MTLPixelFormatBGR5A1Unorm: break;
    case MTLPixelFormatBC1_RGBA: case MTLPixelFormatBC1_RGBA_sRGB:
    case MTLPixelFormatBC2_RGBA: case MTLPixelFormatBC2_RGBA_sRGB:
    case MTLPixelFormatBC3_RGBA: case MTLPixelFormatBC3_RGBA_sRGB:
    case MTLPixelFormatBC4_RUnorm: case MTLPixelFormatBC4_RSnorm:
    case MTLPixelFormatBC5_RGUnorm: case MTLPixelFormatBC5_RGSnorm:
    case MTLPixelFormatBC6H_RGBUfloat: case MTLPixelFormatBC6H_RGBFloat:
    case MTLPixelFormatBC7_RGBAUnorm: case MTLPixelFormatBC7_RGBAUnorm_sRGB: break;
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
    auto commands = surface_command_buffer(device, @"Vita3K surface sampling snapshot"); auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:depth_pipeline];
    [encoder setTexture:source atIndex:0]; [encoder setTexture:result atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding]; std::string error;
    if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return result;
}
static bool msaa_depth_extent_valid(id<MTLTexture> texture,const DepthMemoryLayout &layout,float scale) {
    const uint32_t sx=texture.sampleCount==4?2:1,sy=texture.sampleCount>1?2:1;
    const auto matches=[scale](uint32_t native,uint32_t guest,uint32_t samples) {
        const uint64_t base=uint64_t(double(guest/samples)*scale);
        const uint64_t expanded=uint64_t(double(guest)*scale);
        return native>=base && native<=(expanded+samples-1)/samples;
    };
    return matches(uint32_t(texture.width),layout.width,sx)
        && matches(uint32_t(texture.height),layout.height,sy);
}
static bool depth_memory_valid(id<MTLTexture> texture,const DepthMemoryLayout &layout,float scale,size_t depth,size_t stencil) {
    if(!texture || !scale || texture.pixelFormat!=MTLPixelFormatDepth32Float_Stencil8
        || (texture.sampleCount!=1 && texture.sampleCount!=2 && texture.sampleCount!=4)
        || depth<layout.depth_size || stencil<layout.stencil_size) return false;
    if(texture.sampleCount==1)
        return scale<1 ? texture.width>0 && texture.height>0
            : texture.width>=layout.width && texture.height>=layout.height;
    return msaa_depth_extent_valid(texture,layout,scale);
}
void SurfaceCaster::clear_depth_region(id<MTLTexture> texture,SurfaceRect rect,float depth,uint32_t stencil,
    DepthMemoryWrite aspects,id<MTLCommandBuffer> commands,const MTLSamplePosition *sample_positions) {
    if (!texture || !commands || commands.status>=MTLCommandBufferStatusCommitted
        || !rect.width || !rect.height || uint64_t(rect.x)+rect.width>texture.width
        || uint64_t(rect.y)+rect.height>texture.height || (!aspects.depth && !aspects.stencil && !aspects.mask)
        || (aspects.mask && (aspects.depth || aspects.stencil))
        || texture.pixelFormat!=(aspects.mask ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatDepth32Float_Stencil8))
        throw std::runtime_error("Metal: invalid depth/mask clear region");
    auto &pipeline=depth_clear_pipelines[{uint32_t(texture.sampleCount),aspects.mask}];
    std::string error;
    if (!pipeline) {
        auto desc=[MTLRenderPipelineDescriptor new];
        desc.vertexFunction=[multisample_library newFunctionWithName:@"seed_vs"];
        desc.fragmentFunction=[multisample_library newFunctionWithName:aspects.mask
            ? @"clear_mask_region_value" : @"clear_depth_region_values"];
        if (aspects.mask) desc.colorAttachments[0].pixelFormat=texture.pixelFormat;
        else desc.depthAttachmentPixelFormat=desc.stencilAttachmentPixelFormat=texture.pixelFormat;
        desc.rasterSampleCount=texture.sampleCount;
        pipeline=device.create_pipeline(desc,error);
        if (!pipeline) throw std::runtime_error(error);
    }
    auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
    if (aspects.mask) {
        pass.colorAttachments[0].texture=texture;
        pass.colorAttachments[0].loadAction=MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    } else {
        pass.depthAttachment.texture=pass.stencilAttachment.texture=texture;
        pass.depthAttachment.loadAction=pass.stencilAttachment.loadAction=MTLLoadActionLoad;
        pass.depthAttachment.storeAction=pass.stencilAttachment.storeAction=MTLStoreActionStore;
        if (sample_positions) pass.depthAttachment.storeAction=MTLStoreActionCustomSampleDepthStore;
    }
    if (sample_positions) [pass setSamplePositions:sample_positions count:texture.sampleCount];
    auto state=[MTLDepthStencilDescriptor new];
    state.depthCompareFunction=MTLCompareFunctionAlways;state.depthWriteEnabled=aspects.depth;
    if (aspects.stencil) {
        auto descriptor=[MTLStencilDescriptor new];
        descriptor.stencilCompareFunction=MTLCompareFunctionAlways;
        descriptor.depthStencilPassOperation=MTLStencilOperationReplace;descriptor.writeMask=0xff;
        state.frontFaceStencil=state.backFaceStencil=descriptor;
    }
    auto depth_state=[device.native_device() newDepthStencilStateWithDescriptor:state];
    if (!depth_state) throw std::runtime_error("Metal: cannot allocate partial depth clear state");
    auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) throw std::runtime_error("Metal: cannot encode partial depth clear");
    [encoder setRenderPipelineState:pipeline];[encoder setDepthStencilState:depth_state];
    [encoder setViewport:MTLViewport{0,0,double(texture.width),double(texture.height),0,1}];
    [encoder setScissorRect:MTLScissorRect{rect.x,rect.y,rect.width,rect.height}];
    const uint32_t values[]={std::bit_cast<uint32_t>(depth),stencil};
    [encoder setFragmentBytes:values length:sizeof(values) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
}
bool SurfaceCaster::resample_depth(id<MTLTexture> source,id<MTLTexture> destination,
    uint32_t grid_width,uint32_t grid_height,uint32_t guest_width,uint32_t guest_height,
    id<MTLCommandBuffer> commands,id<MTLTexture> source_mask,id<MTLTexture> destination_mask,
    const MTLSamplePosition *sample_positions) {
    if (!commands || commands.status>=MTLCommandBufferStatusCommitted || !source || !destination
        || source==destination || source.pixelFormat!=MTLPixelFormatDepth32Float_Stencil8
        || destination.pixelFormat!=source.pixelFormat || source.sampleCount!=destination.sampleCount
        || source.textureType!=MTLTextureType2DMultisample || destination.textureType!=MTLTextureType2DMultisample
        || (source.sampleCount!=2 && source.sampleCount!=4)
        || !grid_width || !grid_height || !guest_width || !guest_height
        || guest_width%(source.sampleCount/2) || guest_height%2
        || grid_width>source.width*(source.sampleCount/2) || grid_height>source.height*2
        || grid_width<destination.width*(source.sampleCount/2) || grid_height<destination.height*2
        || bool(source_mask)!=bool(destination_mask)) return false;
    const auto matches=[](id<MTLTexture> mask,id<MTLTexture> depth) {
        return mask.pixelFormat==MTLPixelFormatRGBA8Unorm && mask.width==depth.width
            && mask.height==depth.height && mask.sampleCount==depth.sampleCount
            && mask.textureType==MTLTextureType2DMultisample;
    };
    if (source_mask && (source_mask==destination_mask || !matches(source_mask,source)
        || !matches(destination_mask,destination))) return false;
    auto stencil=[source newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    if (!stencil) return false;
    std::string error;
    for (uint32_t aspect=0;aspect<(source_mask ? 2u : 1u);++aspect) {
        const bool mask=aspect==1;
        auto &pipeline=depth_resample_pipelines[{uint32_t(source.sampleCount),mask}];
        if (!pipeline) {
            auto desc=[MTLRenderPipelineDescriptor new];
            desc.vertexFunction=[multisample_library newFunctionWithName:@"seed_vs"];
            desc.fragmentFunction=[multisample_library newFunctionWithName:mask
                ? @"resample_depth_mask_grid" : @"resample_depth_grid"];
            desc.rasterSampleCount=source.sampleCount;
            if (mask) desc.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
            else desc.depthAttachmentPixelFormat=desc.stencilAttachmentPixelFormat=destination.pixelFormat;
            pipeline=device.create_pipeline(desc,error);
            if (!pipeline) throw std::runtime_error(error);
        }
        auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
        if (mask) {
            pass.colorAttachments[0].texture=destination_mask;
            pass.colorAttachments[0].loadAction=MTLLoadActionDontCare;
            pass.colorAttachments[0].storeAction=MTLStoreActionStore;
        } else {
            pass.depthAttachment.texture=pass.stencilAttachment.texture=destination;
            pass.depthAttachment.loadAction=pass.stencilAttachment.loadAction=MTLLoadActionDontCare;
            pass.depthAttachment.storeAction=pass.stencilAttachment.storeAction=MTLStoreActionStore;
            if (sample_positions) pass.depthAttachment.storeAction=MTLStoreActionCustomSampleDepthStore;
        }
        if (sample_positions) [pass setSamplePositions:sample_positions count:source.sampleCount];
        auto state=[MTLDepthStencilDescriptor new];
        state.depthCompareFunction=MTLCompareFunctionAlways;state.depthWriteEnabled=!mask;
        if (!mask) {
            auto descriptor=[MTLStencilDescriptor new];
            descriptor.stencilCompareFunction=MTLCompareFunctionAlways;
            descriptor.depthStencilPassOperation=MTLStencilOperationReplace;descriptor.writeMask=0xff;
            state.frontFaceStencil=state.backFaceStencil=descriptor;
        }
        auto depth_state=[device.native_device() newDepthStencilStateWithDescriptor:state];
        if (!depth_state) throw std::runtime_error("Metal: cannot allocate depth resampling state");
        auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
        if (!encoder) throw std::runtime_error("Metal: cannot encode depth resampling");
        encoder.label=mask ? @"Vita3K sample-rate mask view" : @"Vita3K sample-rate depth/stencil view";
        [encoder setRenderPipelineState:pipeline];[encoder setDepthStencilState:depth_state];
        [encoder setViewport:MTLViewport{0,0,double(destination.width),double(destination.height),0,1}];
        [encoder setFragmentTexture:mask ? source_mask : source atIndex:0];
        if (!mask) [encoder setFragmentTexture:stencil atIndex:1];
        const uint32_t grid[]={grid_width,grid_height,guest_width,guest_height};
        [encoder setFragmentBytes:grid length:sizeof(grid) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
    }
    return true;
}
bool SurfaceCaster::patch_depth_memory(id<MTLTexture> texture,const SceGxmDepthStencilSurface &surface,
    const DepthMemoryLayout &layout,float scale,std::span<const uint8_t> depth,std::span<const uint8_t> stencil,
    std::span<const SurfaceMemoryRange> depth_changes,std::span<const SurfaceMemoryRange> stencil_changes,
    id<MTLTexture> mask,id<MTLCommandBuffer> commands,DepthMemoryWrite *written) {
    if (written) *written={};
    const bool masked=has_depth_mask_bit(surface);
    if (!depth_memory_valid(texture,layout,scale,depth.size(),stencil.size())
        || !layout.width || !layout.height || (layout.tiled && layout.stride%32)
        || (commands && commands.status>=MTLCommandBufferStatusCommitted)
        || (masked && (!mask || mask.pixelFormat!=MTLPixelFormatRGBA8Unorm
            || mask.width!=texture.width || mask.height!=texture.height || mask.sampleCount!=texture.sampleCount)))
        return false;
    const auto valid=[](auto changes,size_t size) {
        return std::all_of(changes.begin(),changes.end(),[size](const auto &range) {
            return range.offset<=size && range.size<=size-range.offset;
        });
    };
    if (!valid(depth_changes,layout.depth_size) || !valid(stencil_changes,layout.stencil_size)) return false;
    if (depth_changes.empty() && stencil_changes.empty()) return true;
    const auto normalize=[](auto input) {
        std::vector<SurfaceMemoryRange> ranges(input.begin(),input.end()),result;
        std::sort(ranges.begin(),ranges.end(),[](const auto &a,const auto &b) { return a.offset<b.offset; });
        for (const auto &range:ranges) {
            if (!range.size) continue;
            if (!result.empty() && range.offset<=result.back().offset+result.back().size)
                result.back().size=std::max(result.back().offset+result.back().size,range.offset+range.size)-result.back().offset;
            else result.push_back(range);
        }
        return result;
    };
    const auto depth_ranges=normalize(depth_changes),stencil_ranges=normalize(stencil_changes);
    const auto changed=[](size_t offset,const auto &ranges) {
        const auto it=std::lower_bound(ranges.begin(),ranges.end(),offset,[](const auto &range,size_t byte) {
            return range.offset+range.size<=byte;
        });
        return it!=ranges.end() && it->offset<=offset;
    };
    const uint32_t kind=layout.depth_bytes==2 ? 0 : layout.packed ? 1 : masked ? 3 : 2;
    const uint32_t depth_bits=kind==0 ? 0xffffu : kind==1 ? 0xffffffu : kind==3 ? 0x7fffffffu : 0xffffffffu;
    // Each guest sample supplies a word and byte write mask, plus a separate
    // stencil value/write flag. Padding never becomes a visible GPU patch.
    static_assert(sizeof(std::array<uint32_t,4>)==16);
    std::vector<std::array<uint32_t,4>> patches(size_t(layout.width)*layout.height);
    bool patch_depth=false,patch_stencil=false,patch_mask=false;
    for (uint32_t y=0;y<layout.height;++y) for (uint32_t x=0;x<layout.width;++x) {
        const size_t address=depth_sample_offset(layout,x,y);
        auto &patch=patches[size_t(y)*layout.width+x];
        if (layout.depth_size) {
            std::memcpy(&patch[0],depth.data()+address*layout.depth_bytes,layout.depth_bytes);
            for (uint32_t b=0;b<layout.depth_bytes;++b)
                if (changed(address*layout.depth_bytes+b,depth_ranges)) patch[1]|=0xffu<<(8*b);
            if (layout.packed) {
                patch[2]=patch[0]>>24;
                patch[3]=bool(patch[1]&0xff000000u);
            }
        }
        if (layout.stencil_size) {
            const size_t offset=address*(layout.packed ? 4 : 1)+(layout.packed ? 3 : 0);
            patch[2]=stencil[offset];
            patch[3]|=changed(offset,stencil_ranges);
        }
        patch_depth|=bool(patch[1]&depth_bits);
        patch_stencil|=bool(patch[3]);
        patch_mask|=masked && bool(patch[1]&0x80000000u);
    }
    if (!patch_depth && !patch_stencil && !patch_mask) return true;
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint
        width:layout.width height:layout.height mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead;
    auto input=[device.native_device() newTextureWithDescriptor:desc];
    if (!input) throw std::runtime_error("Metal: depth patch staging allocation failed");
    [input replaceRegion:MTLRegionMake2D(0,0,layout.width,layout.height) mipmapLevel:0
        withBytes:patches.data() bytesPerRow:size_t(layout.width)*sizeof(patches[0])];
    const bool submit_here=commands==nil;
    if (submit_here) commands=surface_command_buffer(device,@"Vita3K depth memory patch");
    id<MTLTexture> previous=nil;
    if (patch_depth) {
        // Sampling the render attachment itself is not legal here. Retain its
        // exact native samples before any partial-byte depth writes occur.
        auto copy_desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat
            width:texture.width height:texture.height mipmapped:NO];
        copy_desc.storageMode=MTLStorageModePrivate;
        copy_desc.textureType=texture.textureType;copy_desc.sampleCount=texture.sampleCount;
        copy_desc.usage=MTLTextureUsageShaderRead;
        previous=[device.native_device() newTextureWithDescriptor:copy_desc];
        if (!previous) throw std::runtime_error("Metal: depth patch snapshot allocation failed");
        auto blit=[commands blitCommandEncoder];
        if (!blit) throw std::runtime_error("Metal: cannot snapshot depth before patch");
        [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
            sourceSize:MTLSizeMake(texture.width,texture.height,1) toTexture:previous
            destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [blit endEncoding];
    }
    std::string error;
    for (uint32_t aspect=0;aspect<3;++aspect) {
        if (!(aspect==0 ? patch_depth : aspect==1 ? patch_stencil : patch_mask)) continue;
        auto &pipeline=depth_patch_pipelines[{uint32_t(texture.sampleCount),aspect}];
        if (!pipeline) {
            auto pipeline_desc=[MTLRenderPipelineDescriptor new];
            pipeline_desc.vertexFunction=[multisample_library newFunctionWithName:@"seed_vs"];
            NSString *name=aspect==0 ? (texture.sampleCount>1 ? @"patch_depth_ms" : @"patch_depth")
                : aspect==1 ? @"patch_stencil" : @"patch_depth_mask";
            pipeline_desc.fragmentFunction=[multisample_library newFunctionWithName:name];
            if (aspect==2) pipeline_desc.colorAttachments[0].pixelFormat=mask.pixelFormat;
            else pipeline_desc.depthAttachmentPixelFormat=pipeline_desc.stencilAttachmentPixelFormat=texture.pixelFormat;
            pipeline_desc.rasterSampleCount=texture.sampleCount;
            pipeline=device.create_pipeline(pipeline_desc,error);
            if (!pipeline) throw std::runtime_error(error);
        }
        auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
        if (aspect==2) {
            pass.colorAttachments[0].texture=mask;
            pass.colorAttachments[0].loadAction=MTLLoadActionLoad;
            pass.colorAttachments[0].storeAction=MTLStoreActionStore;
        } else {
            pass.depthAttachment.texture=pass.stencilAttachment.texture=texture;
            pass.depthAttachment.loadAction=pass.stencilAttachment.loadAction=MTLLoadActionLoad;
            pass.depthAttachment.storeAction=pass.stencilAttachment.storeAction=MTLStoreActionStore;
        }
        auto state=[MTLDepthStencilDescriptor new];
        state.depthCompareFunction=MTLCompareFunctionAlways;state.depthWriteEnabled=aspect==0;
        if (aspect==1) {
            auto stencil_state=[MTLStencilDescriptor new];
            stencil_state.stencilCompareFunction=MTLCompareFunctionAlways;
            stencil_state.depthStencilPassOperation=MTLStencilOperationReplace;
            stencil_state.writeMask=0xff;
            state.frontFaceStencil=state.backFaceStencil=stencil_state;
        }
        auto depth_state=[device.native_device() newDepthStencilStateWithDescriptor:state];
        if (!depth_state) throw std::runtime_error("Metal: depth patch state allocation failed");
        auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
        if (!encoder) throw std::runtime_error("Metal: cannot encode depth memory patch");
        [encoder setRenderPipelineState:pipeline];[encoder setDepthStencilState:depth_state];
        [encoder setViewport:MTLViewport{0,0,double(texture.width),double(texture.height),0,1}];
        if (aspect==0) [encoder setFragmentTexture:previous atIndex:0];
        [encoder setFragmentTexture:input atIndex:1];
        const uint32_t config[]={uint32_t(texture.sampleCount),uint32_t(texture.width),uint32_t(texture.height),kind};
        [encoder setFragmentBytes:config length:sizeof(config) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
    }
    if (submit_here && !device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    if (written) *written={patch_depth,patch_stencil,patch_mask};
    return true;
}
bool SurfaceCaster::copy_depth_stencil_region(id<MTLTexture> source,id<MTLTexture> destination,
    SurfaceRect source_rect,SurfaceRect destination_rect,bool stencil,id<MTLCommandBuffer> commands) {
    const auto fits=[](id<MTLTexture> texture,SurfaceRect rect) {
        return rect.width && rect.height && uint64_t(rect.x)+rect.width<=texture.width
            && uint64_t(rect.y)+rect.height<=texture.height;
    };
    if (!commands || commands.status>=MTLCommandBufferStatusCommitted || !source || !destination
        || source==destination || source.pixelFormat!=MTLPixelFormatDepth32Float_Stencil8
        || destination.pixelFormat!=source.pixelFormat || source.sampleCount!=destination.sampleCount
        || (source.sampleCount!=1 && source.sampleCount!=2 && source.sampleCount!=4)
        || !fits(source,source_rect) || !fits(destination,destination_rect)
        || source_rect.width!=destination_rect.width || source_rect.height!=destination_rect.height)
        return false;
    id<MTLTexture> input=stencil ? [source newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8] : source;
    if (!input) return false;
    auto &pipeline=depth_copy_pipelines[{uint32_t(source.sampleCount),stencil}];
    std::string error;
    if (!pipeline) {
        auto desc=[MTLRenderPipelineDescriptor new];
        desc.vertexFunction=[multisample_library newFunctionWithName:@"seed_vs"];
        NSString *function=stencil
            ? (source.sampleCount>1 ? @"copy_stencil_region_ms" : @"copy_stencil_region")
            : (source.sampleCount>1 ? @"copy_depth_region_ms" : @"copy_depth_region");
        desc.fragmentFunction=[multisample_library newFunctionWithName:function];
        desc.depthAttachmentPixelFormat=desc.stencilAttachmentPixelFormat=MTLPixelFormatDepth32Float_Stencil8;
        desc.rasterSampleCount=source.sampleCount;
        pipeline=device.create_pipeline(desc,error);
        if (!pipeline) throw std::runtime_error(error);
    }
    auto state=[MTLDepthStencilDescriptor new];
    state.depthCompareFunction=MTLCompareFunctionAlways;
    state.depthWriteEnabled=!stencil;
    if (stencil) {
        auto descriptor=[MTLStencilDescriptor new];
        descriptor.stencilCompareFunction=MTLCompareFunctionAlways;
        descriptor.depthStencilPassOperation=MTLStencilOperationReplace;
        descriptor.writeMask=0xff;
        state.frontFaceStencil=state.backFaceStencil=descriptor;
    }
    auto depth_state=[device.native_device() newDepthStencilStateWithDescriptor:state];
    if (!depth_state) throw std::runtime_error("Metal: depth transfer state allocation failed");
    auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
    pass.depthAttachment.texture=pass.stencilAttachment.texture=destination;
    pass.depthAttachment.loadAction=pass.stencilAttachment.loadAction=MTLLoadActionLoad;
    pass.depthAttachment.storeAction=pass.stencilAttachment.storeAction=MTLStoreActionStore;
    auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) throw std::runtime_error("Metal: cannot encode depth/stencil aspect transfer");
    encoder.label=stencil ? @"Vita3K stencil aspect transfer" : @"Vita3K depth aspect transfer";
    [encoder setRenderPipelineState:pipeline];
    [encoder setDepthStencilState:depth_state];
    [encoder setViewport:MTLViewport{0,0,double(destination.width),double(destination.height),0,1}];
    [encoder setScissorRect:MTLScissorRect{destination_rect.x,destination_rect.y,destination_rect.width,destination_rect.height}];
    [encoder setFragmentTexture:input atIndex:0];
    const uint32_t origins[]={source_rect.x,source_rect.y,destination_rect.x,destination_rect.y};
    [encoder setFragmentBytes:origins length:sizeof(origins) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}
bool SurfaceCaster::load_depth_memory(id<MTLTexture> texture,const SceGxmDepthStencilSurface &surface,const DepthMemoryLayout &layout,float scale,
    std::span<const uint8_t> depth,std::span<const uint8_t> stencil,id<MTLCommandBuffer> commands) {
    if(!depth_memory_valid(texture,layout,scale,depth.size(),stencil.size())) return false;
    std::vector<float> depths(size_t(layout.width)*layout.height,surface.background_depth);
    std::vector<uint8_t> stencils(depths.size(),surface.stencil);
    for(uint32_t y=0;y<layout.height;++y) for(uint32_t x=0;x<layout.width;++x) {
        const size_t address=depth_sample_offset(layout,x,y),i=size_t(y)*layout.width+x;
        if(layout.depth_size) {
            if(layout.depth_bytes==2) { uint16_t v;std::memcpy(&v,depth.data()+address*2,2);depths[i]=float(v)/65535.f; }
            else if(layout.packed) {uint32_t v;std::memcpy(&v,depth.data()+address*4,4);depths[i]=float(v&0xffffff)/16777215.f;stencils[i]=v>>24;}
            else if(has_depth_mask_bit(surface)) {
                uint32_t word;
                std::memcpy(&word,depth.data()+address*4,4);
                depths[i]=std::bit_cast<float>(word&0x7fffffffu);
            } else std::memcpy(&depths[i],depth.data()+address*4,4);
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
    const bool submit_here=commands==nil;
    if(submit_here) commands=surface_command_buffer(device, @"Vita3K surface depth load");
    auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
    pass.depthAttachment.texture=pass.stencilAttachment.texture=texture;
    pass.depthAttachment.loadAction=pass.stencilAttachment.loadAction=MTLLoadActionDontCare;
    pass.depthAttachment.storeAction=pass.stencilAttachment.storeAction=MTLStoreActionStore;
    auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
    auto state=[MTLDepthStencilDescriptor new];state.depthCompareFunction=MTLCompareFunctionAlways;state.depthWriteEnabled=YES;
    auto stencil_state=[MTLStencilDescriptor new];stencil_state.stencilCompareFunction=MTLCompareFunctionAlways;
    stencil_state.depthStencilPassOperation=MTLStencilOperationReplace;state.frontFaceStencil=state.backFaceStencil=stencil_state;
    [encoder setDepthStencilState:[device.native_device() newDepthStencilStateWithDescriptor:state]];
    [encoder setRenderPipelineState:pipeline];[encoder setFragmentTexture:d atIndex:0];[encoder setFragmentTexture:s atIndex:1];
    const uint32_t config[]={uint32_t(texture.sampleCount),uint32_t(texture.width),uint32_t(texture.height),0};
    [encoder setFragmentBytes:config length:sizeof(config) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[encoder endEncoding];
    if(submit_here && !device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return true;
}
bool SurfaceCaster::load_mask_memory(id<MTLTexture> mask,const SceGxmDepthStencilSurface &surface,
    const DepthMemoryLayout &layout,float scale,std::span<const uint8_t> depth,
    id<MTLCommandBuffer> commands) {
    if(!has_depth_mask_bit(surface) || !mask || !scale || !layout.depth_size
        || depth.size()<layout.depth_size || mask.pixelFormat!=MTLPixelFormatRGBA8Unorm
        || (mask.sampleCount!=1 && mask.sampleCount!=2 && mask.sampleCount!=4)) return false;
    if(mask.sampleCount==1 ? (scale>=1 && (mask.width<layout.width || mask.height<layout.height))
        : !msaa_depth_extent_valid(mask,layout,scale)) return false;
    std::vector<uint8_t> values(size_t(layout.width)*layout.height);
    for(uint32_t y=0;y<layout.height;++y) for(uint32_t x=0;x<layout.width;++x) {
        uint32_t word;
        std::memcpy(&word,depth.data()+depth_sample_offset(layout,x,y)*4,4);
        values[size_t(y)*layout.width+x]=(word&0x80000000u)?255:0;
    }
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
        width:layout.width height:layout.height mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead;
    auto source=[device.native_device() newTextureWithDescriptor:desc];
    if(!source) throw std::runtime_error("Metal: mask load staging allocation failed");
    [source replaceRegion:MTLRegionMake2D(0,0,layout.width,layout.height) mipmapLevel:0
        withBytes:values.data() bytesPerRow:layout.width];
    auto &pipeline=mask_seed_pipelines[uint32_t(mask.sampleCount)];
    std::string error;
    if(!pipeline) {
        auto pipeline_desc=[MTLRenderPipelineDescriptor new];
        pipeline_desc.vertexFunction=[multisample_library newFunctionWithName:@"seed_vs"];
        pipeline_desc.fragmentFunction=[multisample_library newFunctionWithName:@"seed_mask"];
        pipeline_desc.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
        pipeline_desc.rasterSampleCount=mask.sampleCount;
        pipeline=device.create_pipeline(pipeline_desc,error);
        if(!pipeline) throw std::runtime_error(error);
    }
    const bool submit_here=commands==nil;
    if(submit_here) commands=surface_command_buffer(device, @"Vita3K surface mask load");
    auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture=mask;
    pass.colorAttachments[0].loadAction=MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];[encoder setFragmentTexture:source atIndex:0];
    const uint32_t config[]={uint32_t(mask.sampleCount),uint32_t(mask.width),uint32_t(mask.height),0};
    [encoder setFragmentBytes:config length:sizeof(config) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    if(submit_here && !device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    return true;
}
bool SurfaceCaster::store_depth_memory(id<MTLTexture> texture,const SceGxmDepthStencilSurface &surface,const DepthMemoryLayout &layout,float scale,
    std::span<uint8_t> depth,std::span<uint8_t> stencil,id<MTLTexture> mask) {
    auto commands=surface_command_buffer(device, @"Vita3K surface depth store");
    DepthStoreReadback readback;
    if(!enqueue_depth_store(texture,surface,layout,scale,depth.size(),stencil.size(),mask,commands,readback)) return false;
    if(!readback.depth) return true;
    std::string error;
    if(!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    finish_depth_store(readback,surface,layout,depth,stencil);
    return true;
}
bool SurfaceCaster::enqueue_depth_store(id<MTLTexture> texture,const SceGxmDepthStencilSurface &surface,
    const DepthMemoryLayout &layout,float scale,size_t depth_size,size_t stencil_size,
    id<MTLTexture> mask,id<MTLCommandBuffer> commands,DepthStoreReadback &readback) {
    if(!commands || !depth_memory_valid(texture,layout,scale,depth_size,stencil_size)) return false;
    if(!layout.depth_size && !layout.stencil_size) return true;
    const bool masked=has_depth_mask_bit(surface);
    if(masked && (!mask || mask.pixelFormat!=MTLPixelFormatRGBA8Unorm
        || mask.width!=texture.width || mask.height!=texture.height
        || mask.sampleCount!=texture.sampleCount)) return false;
    // Read exactly the native sample previously selected by the CPU from the
    // expanded images. A linear shared buffer avoids two full-resolution texture
    // readbacks, their tiling conversion, and a second command-buffer wait.
    struct DepthReadback { float depth; uint32_t stencil; };
    static_assert(sizeof(DepthReadback)==8);
    // A complete 32x32 tile has no guest padding when the stride and height
    // match the visible extent. Scatter on the GPU, then copy its already
    // guest-ordered words instead of packing every pixel on the CPU.
    const bool contiguous=layout.stride==layout.width
        && (!layout.tiled || ((layout.width%32u)==0 && (layout.height%32u)==0));
    const bool direct_depth=contiguous && layout.depth_bytes==4
        && layout.depth_size==size_t(layout.width)*layout.height*4
        && !layout.stencil_size && !masked
        && (layout.packed || surface.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32
            || surface.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8);
    const size_t bytes=size_t(layout.width)*layout.height*(direct_depth?sizeof(uint32_t):sizeof(DepthReadback));
    if(bytes>device.native_device().maxBufferLength) return false;
    id<MTLBuffer> buffer=depth_store_buffer;
    if(!buffer || buffer.length<bytes) {
        buffer=[device.native_device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if(!buffer) throw std::runtime_error("Metal: depth readback allocation failed");
        if(bytes<=64*1024*1024) depth_store_buffer=buffer;
    }
    id<MTLBuffer> mask_buffer;
    if(masked) {
        const size_t mask_bytes=size_t(layout.width)*layout.height*sizeof(uint32_t);
        if(mask_bytes>device.native_device().maxBufferLength) return false;
        mask_buffer=mask_store_buffer;
        if(!mask_buffer || mask_buffer.length<mask_bytes) {
            mask_buffer=[device.native_device() newBufferWithLength:mask_bytes options:MTLResourceStorageModeShared];
            if(!mask_buffer) throw std::runtime_error("Metal: mask readback allocation failed");
            if(mask_bytes<=64*1024*1024) mask_store_buffer=mask_buffer;
        }
    }
    auto view=[texture newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    if(!view) throw std::runtime_error("Metal: cannot view stencil for guest storage");
    const uint32_t flags=uint32_t(layout.depth_size!=0)
        | (uint32_t(layout.stencil_size || (layout.packed && layout.depth_size))<<1)
        | (uint32_t(layout.packed)<<2);
    const uint32_t config[]={layout.width,layout.height,direct_depth && layout.tiled?layout.stride:0,flags};
    auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:direct_depth
        ? (texture.sampleCount>1?packed_depth_store_ms_pipeline:packed_depth_store_pipeline)
        : (texture.sampleCount>1?depth_store_ms_pipeline:depth_store_pipeline)];
    [encoder setTexture:texture atIndex:0];[encoder setTexture:view atIndex:1];
    [encoder setBuffer:buffer offset:0 atIndex:0];[encoder setBytes:config length:sizeof(config) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(layout.width,layout.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if(masked) {
        auto mask_encoder=[commands computeCommandEncoder];
        [mask_encoder setComputePipelineState:mask.sampleCount>1?mask_store_ms_pipeline:mask_store_pipeline];
        [mask_encoder setTexture:mask atIndex:0];
        [mask_encoder setBuffer:mask_buffer offset:0 atIndex:0];
        [mask_encoder setBytes:config length:sizeof(config) atIndex:1];
        [mask_encoder dispatchThreads:MTLSizeMake(layout.width,layout.height,1)
            threadsPerThreadgroup:MTLSizeMake(8,8,1)];
        [mask_encoder endEncoding];
    }
    readback.depth=buffer;
    readback.mask=mask_buffer;
    readback.packed_direct=direct_depth;
    return true;
}
void SurfaceCaster::finish_depth_store(const DepthStoreReadback &readback,const SceGxmDepthStencilSurface &surface,
    const DepthMemoryLayout &layout,std::span<uint8_t> depth,std::span<uint8_t> stencil) {
    if(!readback.depth) return;
    if(readback.packed_direct) {
        std::memcpy(depth.data(),readback.depth.contents,size_t(layout.width)*layout.height*sizeof(uint32_t));
        return;
    }
    // The caller has completed the command buffer. Guest rounding, tiling,
    // stencil overrides and padding remain identical to the original readback.
    struct DepthReadback { float depth; uint32_t stencil; };
    const bool masked=has_depth_mask_bit(surface);
    const auto *values=static_cast<const DepthReadback *>(readback.depth.contents);
    const auto *masks=masked?static_cast<const uint32_t *>(readback.mask.contents):nullptr;
    for(uint32_t y=0;y<layout.height;++y) for(uint32_t x=0;x<layout.width;++x) {
        const size_t address=depth_sample_offset(layout,x,y),i=size_t(y)*layout.width+x;
        if(layout.depth_size) {
            if(layout.depth_bytes==2) {const uint16_t v=uint16_t(std::lround(std::clamp(double(values[i].depth),0.,1.)*65535.));std::memcpy(depth.data()+address*2,&v,2);}
            else if(layout.packed) {
                const uint32_t v=uint32_t(std::llround(std::clamp(double(values[i].depth),0.,1.)*16777215.))|(uint32_t(values[i].stencil)<<24);
                std::memcpy(depth.data()+address*4,&v,4);
            } else if(masked) {
                const uint32_t word=(std::bit_cast<uint32_t>(values[i].depth)&0x7fffffffu)
                    | (masks[i]?0x80000000u:0);
                std::memcpy(depth.data()+address*4,&word,4);
            } else std::memcpy(depth.data()+address*4,&values[i].depth,4);
        }
        if(layout.stencil_size) stencil[address*(layout.packed?4:1)+(layout.packed?3:0)]=values[i].stencil;
    }
}
id<MTLTexture> SurfaceCaster::depth_snapshot(id<MTLTexture> source, bool normalized16, float scale,
    uint32_t guest_width, uint32_t guest_height, bool wait_for_completion,
    id<MTLCommandBuffer> pending_commands) {
    const bool ms=source && source.textureType==MTLTextureType2DMultisample && (source.sampleCount==2 || source.sampleCount==4);
    if (!source || (!ms && (source.textureType != MTLTextureType2D || source.sampleCount != 1))
        || source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)
        throw std::runtime_error("Metal: invalid depth snapshot source");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:normalized16 ? MTLPixelFormatR16Unorm : MTLPixelFormatR32Float width:source.width*(ms ? source.sampleCount/2 : 1) height:source.height*(ms ? 2 : 1) mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    desc.storageMode = MTLStorageModeShared;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate depth snapshot");
    if (ms) {
        expand_multisample(source,result,scale,guest_width,guest_height,pending_commands); return result;
    }
    auto commands = pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface depth snapshot");
    auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:depth_pipeline];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:result atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (pending_commands) return result;
    if (wait_for_completion) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    } else {
        [commands addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.error) LOG_ERROR("Metal depth snapshot: {}", completed.error.localizedDescription.UTF8String);
        }];
        [commands commit];
    }
    return result;
}
id<MTLTexture> SurfaceCaster::packed_depth_snapshot(id<MTLTexture> source, bool wait_for_completion,
    id<MTLCommandBuffer> pending_commands) {
    if (!source || source.textureType != MTLTextureType2D || source.sampleCount != 1
        || source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)
        throw std::runtime_error("Metal: invalid packed depth snapshot source");
    auto stencil = [source newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    if (!stencil) throw std::runtime_error("Metal: cannot view packed depth stencil");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:source.width height:source.height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
    desc.storageMode = MTLStorageModeShared;
    auto result = [device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate packed depth snapshot");
    auto commands = pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K packed depth snapshot");
    auto encoder = [commands computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal: cannot encode packed depth snapshot");
    [encoder setComputePipelineState:packed_depth_snapshot_pipeline];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:stencil atIndex:1];
    [encoder setTexture:result atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (pending_commands) return result;
    if (wait_for_completion) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    } else {
        [commands addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.error) LOG_ERROR("Metal packed depth snapshot: {}", completed.error.localizedDescription.UTF8String);
        }];
        [commands commit];
    }
    return result;
}
id<MTLTexture> SurfaceCaster::stencil_snapshot(id<MTLTexture> source, bool signed_normalized, float scale,
    uint32_t guest_width, uint32_t guest_height, bool wait_for_completion,
    id<MTLCommandBuffer> pending_commands) {
    const bool ms=source && source.textureType==MTLTextureType2DMultisample
        && (source.sampleCount==2 || source.sampleCount==4);
    if (!source || !scale || (!ms && (source.textureType!=MTLTextureType2D || source.sampleCount!=1))
        || source.pixelFormat!=MTLPixelFormatDepth32Float_Stencil8)
        throw std::runtime_error("Metal: invalid stencil snapshot source");
    auto stencil=[source newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    if (!stencil) throw std::runtime_error("Metal: cannot view stencil for texture sampling");
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
        signed_normalized ? MTLPixelFormatR8Snorm : MTLPixelFormatR8Unorm
        width:source.width*(ms ? source.sampleCount/2 : 1)
        height:source.height*(ms ? 2 : 1) mipmapped:NO];
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
    desc.storageMode=MTLStorageModeShared;
    auto result=[device.native_device() newTextureWithDescriptor:desc];
    if (!result) throw std::runtime_error("Metal: cannot allocate stencil snapshot");
    auto commands=pending_commands ? pending_commands : surface_command_buffer(device, @"Vita3K surface stencil snapshot");
    auto encoder=[commands computeCommandEncoder];
    [encoder setComputePipelineState:ms ? stencil_ms_pipeline : stencil_pipeline];
    [encoder setTexture:stencil atIndex:0];
    [encoder setTexture:result atIndex:1];
    if (!guest_width) guest_width=uint32_t(std::ceil(double(result.width)/scale));
    if (!guest_height) guest_height=uint32_t(std::ceil(double(result.height)/scale));
    const uint32_t config[]={guest_width,guest_height,uint32_t(signed_normalized)};
    if (ms) [encoder setBytes:config length:sizeof(config) atIndex:0];
    else [encoder setBytes:&config[2] length:sizeof(uint32_t) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(result.width,result.height,1)
        threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [encoder endEncoding];
    if (pending_commands) return result;
    if (wait_for_completion) {
        std::string error;
        if (!device.submit_and_wait(commands,error)) throw std::runtime_error(error);
    } else {
        [commands addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.error) LOG_ERROR("Metal stencil snapshot: {}", completed.error.localizedDescription.UTF8String);
        }];
        [commands commit];
    }
    return result;
}
}
