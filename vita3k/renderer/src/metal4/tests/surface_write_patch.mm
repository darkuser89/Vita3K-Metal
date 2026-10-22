// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Complete current/private baseline writers at a CPU-only texture boundary.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <gxm/types.h>
#include <util/float_to_half.h>
#include <renderer/metal4/linear_surface_patch.h>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
using namespace renderer::metal;
namespace gxm {
// GXM_BASE_HOOK
}
namespace texture {
// MORTON_HOOK
}
struct SceGxmColorSurface { SceGxmColorFormat colorFormat; uint32_t width,height,strideInPixels; SceGxmColorSurfaceType surfaceType; };
struct SurfaceMemoryRange { size_t offset,size; };
using Map=std::array<MTLTextureSwizzle,4>;
constexpr auto R=MTLTextureSwizzleRed,G=MTLTextureSwizzleGreen,B=MTLTextureSwizzleBlue,A=MTLTextureSwizzleAlpha,Z=MTLTextureSwizzleZero,O=MTLTextureSwizzleOne;
constexpr Map identity{R,G,B,A};
constexpr Map four[]={{R,G,B,A},{B,G,R,A},{A,B,G,R},{G,B,A,R},{R,G,B,O},{B,G,R,O},{A,B,G,O},{G,B,A,O}};
constexpr Map packed_four[]={{A,B,G,R},{G,B,A,R},{R,G,B,A},{B,G,R,A}};
struct SurfaceComponents { uint32_t count=0,bytes=0; MTLPixelFormat native=MTLPixelFormatInvalid; };
static constexpr std::array<std::array<uint8_t,4>,4> mixed_signed_guest_bytes{{{{0,1,2,3}},{{2,1,0,3}},{{3,2,1,0}},{{1,2,3,0}}}};
// COMMON_HOOK
namespace current {
// CURRENT_HOOK
}
namespace original {
// ORIGINAL_HOOK
}
@interface WriteTexture : NSObject {
@public
    std::vector<uint8_t> pixels;
    size_t bytes, reads, writes, read_bytes, write_bytes;
    MTLRegion written;
}
@property NSUInteger width,height;
@property MTLPixelFormat pixelFormat;
@property MTLStorageMode storageMode;
- (MTLTextureType)textureType;
- (id<MTLBuffer>)buffer;
- (NSUInteger)bufferBytesPerRow;
- (NSUInteger)bufferOffset;
- (void)getBytes:(void *)data bytesPerRow:(NSUInteger)stride fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level;
- (void)replaceRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level withBytes:(const void *)data bytesPerRow:(NSUInteger)stride;
@end
@implementation WriteTexture
- (instancetype)init { if((self=[super init])) _storageMode=MTLStorageModeShared; return self; }
- (MTLTextureType)textureType { return MTLTextureType2D; }
- (id<MTLBuffer>)buffer { return nil; }
- (NSUInteger)bufferBytesPerRow { return 0; }
- (NSUInteger)bufferOffset { return 0; }
- (void)getBytes:(void *)data bytesPerRow:(NSUInteger)stride fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level {
    assert(!level && !region.origin.z && region.size.depth==1);
    assert(region.origin.x+region.size.width<=self.width && region.origin.y+region.size.height<=self.height);
    assert(stride>=region.size.width*bytes);
    for(size_t y=0;y<region.size.height;++y)
        std::memcpy(static_cast<uint8_t *>(data)+y*stride,
            pixels.data()+((region.origin.y+y)*self.width+region.origin.x)*bytes,region.size.width*bytes);
    ++reads;read_bytes+=region.size.width*region.size.height*bytes;
}
- (void)replaceRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level withBytes:(const void *)data bytesPerRow:(NSUInteger)stride {
    assert(!level && !region.origin.z && region.size.depth==1);
    assert(region.origin.x+region.size.width<=self.width && region.origin.y+region.size.height<=self.height);
    assert(stride>=region.size.width*bytes);
    for(size_t y=0;y<region.size.height;++y)
        std::memcpy(pixels.data()+((region.origin.y+y)*self.width+region.origin.x)*bytes,
            static_cast<const uint8_t *>(data)+y*stride,region.size.width*bytes);
    ++writes;write_bytes+=region.size.width*region.size.height*bytes;written=region;
}
@end
static size_t checks=0,bounds_checks=0,byte_checks=0,download_checks=0;
static LinearSurfacePatch oracle_bounds(const SceGxmColorSurface &s,size_t b,std::span<const SurfaceMemoryRange> ranges) {
    size_t left=s.width,top=s.height,right=0,bottom=0;
    for(unsigned y=0;y<s.height;++y) for(unsigned x=0;x<s.width;++x) {
        const size_t pixel=s.surfaceType==SCE_GXM_COLOR_SURFACE_TILED
            ? ((size_t(y/32)*(s.strideInPixels/32)+x/32)*1024)+(y%32)*32+x%32
            : s.surfaceType==SCE_GXM_COLOR_SURFACE_SWIZZLED ? texture::encode_morton(x,y,s.width,s.height)
            : size_t(y)*s.strideInPixels+x;
        for(auto r:ranges) if(r.offset<(pixel+1)*b && r.offset+r.size>pixel*b) {
            left=std::min(left,size_t(x));right=std::max(right,size_t(x)+1);
            top=std::min(top,size_t(y));bottom=std::max(bottom,size_t(y)+1);break;
        }
    }
    return left<right && top<bottom ? LinearSurfacePatch{left,top,right-left,bottom-top}:LinearSurfacePatch{};
}
static void check(const SceGxmColorSurface &s,MTLPixelFormat format,size_t native_bytes,
    unsigned groups_x,unsigned groups_y,unsigned sx,unsigned sy,const std::vector<SurfaceMemoryRange> &ranges) {
    @autoreleasepool {
        auto now=[WriteTexture new],old=[WriteTexture new];
        now.width=old.width=groups_x*sx;now.height=old.height=groups_y*sy;
        now.pixelFormat=old.pixelFormat=format;now->bytes=old->bytes=native_bytes;
        now->pixels.resize(now.width*now.height*native_bytes);
        for(size_t i=0;i<now->pixels.size();++i) now->pixels[i]=uint8_t(i*19+(i>>8)*73+47);
        old->pixels=now->pixels;
        std::vector<uint8_t> source(surface_memory_size(s));
        for(size_t i=0;i<source.size();++i) source[i]=uint8_t(i*37+(i>>8)*59+3);
        const auto components=surface_components(s.colorFormat);
        const size_t b=gxm::get_base_format(s.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10?4:components.count*components.bytes;
        const auto layout=s.surfaceType==SCE_GXM_COLOR_SURFACE_TILED?SurfacePatchLayout::Tiled:
            s.surfaceType==SCE_GXM_COLOR_SURFACE_SWIZZLED?SurfacePatchLayout::Morton:SurfacePatchLayout::Linear;
        const auto expected=oracle_bounds(s,b,ranges);
        const auto actual=guest_surface_patch(s.width,s.height,b,s.strideInPixels,layout,std::span<const SurfaceMemoryRange>(ranges));
        assert(actual.x==expected.x && actual.y==expected.y && actual.width==expected.width && actual.height==expected.height);
        ++bounds_checks;
        const bool a=current::write_surface_memory_mapped((id<MTLTexture>)now,s,source,ranges,sx,sy);
        const bool z=original::write_surface_memory_mapped((id<MTLTexture>)old,s,source,ranges,sx,sy);
        assert(a==z && a);assert(now->pixels==old->pixels);
        if(expected.width && expected.height) {
            // Enumerate the old writer's integer footprints independently of
            // the new endpoint-based native rectangle construction.
            size_t left=now.width,top=now.height,right=0,bottom=0;
            for(size_t y=expected.y;y<expected.y+expected.height;++y)
                for(size_t x=expected.x;x<expected.x+expected.width;++x) {
                    const size_t l=x*groups_x/s.width, t=y*groups_y/s.height;
                    const size_t r=groups_x<s.width?l+1:(x+1)*groups_x/s.width;
                    const size_t btm=groups_y<s.height?t+1:(y+1)*groups_y/s.height;
                    left=std::min(left,l*sx);right=std::max(right,r*sx);
                    top=std::min(top,t*sy);bottom=std::max(bottom,btm*sy);
                }
            assert(now->written.origin.x==left && now->written.origin.y==top);
            assert(now->written.size.width==right-left && now->written.size.height==bottom-top);
            assert(now->write_bytes==(right-left)*(bottom-top)*native_bytes);
            assert(now->read_bytes==0 || now->read_bytes==now->write_bytes);
        } else assert(!now->reads && !now->writes);
        assert(now->read_bytes<=old->read_bytes && now->write_bytes<=old->write_bytes);
        byte_checks+=now->pixels.size();++checks;
        // Expanded formats share upload/download functions. Verify that their
        // full-image readback behavior stays identical after the refactor.
        if(ranges.size()==1 && ranges[0].offset==0 && ranges[0].size==source.size()) {
            std::vector<uint8_t> n(source.size(),0xa5),o=n;bool handled=true;
            switch(gxm::get_base_format(s.colorFormat)) {
#define READ(base,fn) case SCE_GXM_COLOR_BASE_FORMAT_##base: \
                assert(current::fn((id<MTLTexture>)now,s,{},n,{},false,sx,sy)); \
                assert(original::fn((id<MTLTexture>)old,s,{},o,{},false,sx,sy));break
                READ(U8U3U3U2,u8u3u3u2_surface_memory);
                READ(S5S5U6,s5s5u6_surface_memory);
                READ(U8S8S8U8,u8s8s8u8_surface_memory);
                READ(U2F10F10F10,packed_float_surface_memory);
                case SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9:
                    if(format==MTLPixelFormatRGBA16Float) {
                        assert(current::rgb9e5_expanded_surface_memory((id<MTLTexture>)now,s,{},n,{},false,sx,sy));
                        assert(original::rgb9e5_expanded_surface_memory((id<MTLTexture>)old,s,{},o,{},false,sx,sy));
                    } else handled=false;break;
                default:handled=false;break;
#undef READ
            }
            if(handled) { assert(n==o);++download_checks; }
        }
    }
}
int main() {
    std::mt19937 rng(0x26555246);
    const std::pair<SceGxmColorBaseFormat,unsigned> formats[]={
        {SCE_GXM_COLOR_BASE_FORMAT_U8,2},{SCE_GXM_COLOR_BASE_FORMAT_S8,2},
        {SCE_GXM_COLOR_BASE_FORMAT_U8U8,4},{SCE_GXM_COLOR_BASE_FORMAT_S8S8,4},
        {SCE_GXM_COLOR_BASE_FORMAT_U16,2},{SCE_GXM_COLOR_BASE_FORMAT_S16,2},{SCE_GXM_COLOR_BASE_FORMAT_F16,2},
        {SCE_GXM_COLOR_BASE_FORMAT_U16U16,2},{SCE_GXM_COLOR_BASE_FORMAT_S16S16,2},{SCE_GXM_COLOR_BASE_FORMAT_F16F16,2},
        {SCE_GXM_COLOR_BASE_FORMAT_U8U8U8,2},{SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8,8},{SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8,8},
        {SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16,8},{SCE_GXM_COLOR_BASE_FORMAT_F32,1},{SCE_GXM_COLOR_BASE_FORMAT_F32F32,2},
        {SCE_GXM_COLOR_BASE_FORMAT_U5U6U5,2},{SCE_GXM_COLOR_BASE_FORMAT_U4U4U4U4,4},
        {SCE_GXM_COLOR_BASE_FORMAT_U1U5U5U5,4},{SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10,4},
        {SCE_GXM_COLOR_BASE_FORMAT_F11F11F10,2},{SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9,2},
        {SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2,1},{SCE_GXM_COLOR_BASE_FORMAT_S5S5U6,2},
        {SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8,4},{SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10,4}};
    for(auto [base,modes]:formats) for(unsigned mode=0;mode<modes;++mode)
    for(auto layout:{SCE_GXM_COLOR_SURFACE_LINEAR,SCE_GXM_COLOR_SURFACE_TILED,SCE_GXM_COLOR_SURFACE_SWIZZLED})
    for(auto [w,h]:std::array<std::pair<unsigned,unsigned>,4>{{{1,1},{3,5},{8,4},{33,35}}}) {
        if(layout==SCE_GXM_COLOR_SURFACE_SWIZZLED && (!std::has_single_bit(w)||!std::has_single_bit(h))) continue;
        SceGxmColorSurface s{SceGxmColorFormat(uint32_t(base)|(mode<<20)),w,h,
            layout==SCE_GXM_COLOR_SURFACE_TILED?((w+31)/32+1)*32:layout==SCE_GXM_COLOR_SURFACE_LINEAR?w+3:w,layout};
        const auto c=surface_components(s.colorFormat);
        const auto b=base==SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10?4:c.count*c.bytes;
        const size_t size=surface_memory_size(s);assert(size);
        for(bool expanded:{false,true}) {
            if(expanded && base!=SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9) continue;
            const auto f=expanded||base==SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10?MTLPixelFormatRGBA16Float:c.native;
            const size_t nb=f==MTLPixelFormatRGBA16Float?8:c.count==3||base==SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2?4:c.count*c.bytes;
            for(auto [sx,sy]:std::array<std::pair<unsigned,unsigned>,4>{{{1,1},{1,2},{2,1},{2,2}}})
            for(auto [gw,gh]:std::array<std::pair<unsigned,unsigned>,3>{{{w,h},{w+3,h+2},{std::max(1u,w/2),std::max(1u,h/2)}}}) {
                const auto verify=[&](std::vector<SurfaceMemoryRange> ranges) { check(s,f,nb,gw,gh,sx,sy,ranges); };
                verify({});verify({{0,size}});if(size>1)verify({{0,1},{1,size-1}});
                for(size_t at:{size_t(0),size_t(b-1),size_t(b),size/2,size-1})if(at<size)verify({{at,1}});
                for(unsigned trial=0;trial<4;++trial) {
                    const size_t first=rng()%size;verify({{first,1+rng()%(size-first)}});
                    std::vector<SurfaceMemoryRange> ranges;
                    for(unsigned part=0;part<8;++part) {
                        const size_t begin=size*part/8,end=size*(part+1)/8;
                        if(begin==end)continue;
                        const size_t at=begin+rng()%(end-begin);
                        ranges.push_back({at,std::min(end-at,size_t(1+rng()%17))});
                    }
                    verify(ranges);
                }
            }
        }
    }
    // Sparse 1024-square fixtures verify that allocations and transfers are
    // bounded even for expanded, packed, RGB24 and Morton-backed surfaces.
    for(auto base:{SCE_GXM_COLOR_BASE_FORMAT_U8U8U8,SCE_GXM_COLOR_BASE_FORMAT_U5U6U5,
        SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10,SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9}) {
        SceGxmColorSurface s{SceGxmColorFormat(base),1024,1024,1024,SCE_GXM_COLOR_SURFACE_SWIZZLED};
        auto c=surface_components(s.colorFormat);const auto b=base==SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10?4:c.count*c.bytes;
        const auto f=base==SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10?MTLPixelFormatRGBA16Float:c.native;
        check(s,f,f==MTLPixelFormatRGBA16Float?8:c.count==3?4:b,1024,1024,1,1,{{size_t(texture::encode_morton(721,512,1024,1024))*b,1}});
    }
    size_t large_bounds=0,invalid_bounds=0;
    for(auto layout:{SurfacePatchLayout::Linear,SurfacePatchLayout::Tiled,SurfacePatchLayout::Morton})
    for(size_t b:{1,2,3,4,8,16}) {
        for(unsigned n=0;n<1000;++n) {
            const unsigned x=rng()%16384,y=rng()%16384;
            const size_t p=layout==SurfacePatchLayout::Tiled?((size_t(y/32)*512+x/32)*1024)+(y%32)*32+x%32:
                layout==SurfacePatchLayout::Morton?texture::encode_morton(x,y,16384,16384):size_t(y)*16384+x;
            const std::array<SurfaceMemoryRange,1> r{{{p*b+rng()%b,1}}};
            auto patch=guest_surface_patch(16384,16384,b,16384,layout,std::span<const SurfaceMemoryRange>(r));
            assert(patch.x==x && patch.y==y && patch.width==1 && patch.height==1);++large_bounds;
        }
        const size_t bytes=size_t(16384)*16384*b;
        const std::array<SurfaceMemoryRange,1> full{{{0,bytes}}};
        const auto patch=guest_surface_patch(16384,16384,b,16384,layout,std::span<const SurfaceMemoryRange>(full));
        assert(!patch.x && !patch.y && patch.width==16384 && patch.height==16384);++large_bounds;
        for(auto r:{SurfaceMemoryRange{bytes,1},SurfaceMemoryRange{0,bytes+1},SurfaceMemoryRange{0,0}}) {
            bool rejected=false;try { guest_surface_patch(16384,16384,b,16384,layout,std::span<const SurfaceMemoryRange>(&r,1)); }
            catch(const std::invalid_argument &) {rejected=true;}
            assert(rejected);++invalid_bounds;
        }
        for(size_t bad_stride:{size_t(0),std::numeric_limits<size_t>::max()}) {
            bool rejected=false;try { guest_surface_patch(16384,16384,b,bad_stride,layout,std::span<const SurfaceMemoryRange>()); }
            catch(const std::invalid_argument &) {rejected=true;}
            assert(rejected);++invalid_bounds;
        }
    }
    std::cout<<"PASS surface write patch: "<<checks<<" complete production/baseline writer comparisons, "<<byte_checks
        <<" native bytes, "<<download_checks<<" expanded readbacks; linear/tiled/Morton, 26 formats, RGB24 alpha, packed/expanded, scale and sample footprints; ASAN/UBSAN, CPU texture doubles\n";
    std::cout<<"PASS surface patch bounds: "<<bounds_checks<<" exact guest/native bounds against per-pixel oracle, "<<large_bounds
        <<" 16384-square size/edge checks and "<<invalid_bounds<<" rejected layouts/ranges; padding-only edits transfer nothing; sparse 1024-square updates remain bounded\n";
}
