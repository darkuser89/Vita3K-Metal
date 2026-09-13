// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <renderer/gxm_types.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    @autoreleasepool {
        try {
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error.c_str());
            renderer::metal::SurfaceCaster caster(*device);
            {
                SceGxmDepthStencilSurface depth{};
                depth.depth_data = Ptr<void>(0x61256000);
                depth.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_D16);
                depth.set_type(SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR);
                depth.set_stride(1472);
                SceGxmTexture texture{};
                texture.type = SCE_GXM_TEXTURE_LINEAR_STRIDED >> 29;
                texture.width = 1439; texture.height = 815; texture.base_format = 9;
                texture.data_addr = depth.depth_data.address() >> 2;
                const uint32_t encoded = 2944 / 4 - 1;
                texture.mip_filter = encoded & 1; texture.min_filter = (encoded >> 1) & 3;
                texture.mip_count = (encoded >> 3) & 15; texture.lod_bias = (encoded >> 7) & 63;
                const auto matches = [&](uint32_t w, uint32_t h, SceGxmMultisampleMode mode) {
                    return renderer::metal::depth_texture_matches(depth, w, h, mode, texture);
                };
                check(matches(720,408,SCE_GXM_MULTISAMPLE_4X), "LBP D16 full allocation was not matched");
                check(matches(1440,408,SCE_GXM_MULTISAMPLE_2X), "2X depth sample rows were not matched");
                check(matches(1440,816,SCE_GXM_MULTISAMPLE_NONE), "Single-sample D16 allocation was not matched");
                check(!matches(720,408,SCE_GXM_MULTISAMPLE_NONE), "Wrong depth sample dimensions accepted");
                check(!matches(720,408,static_cast<SceGxmMultisampleMode>(3)), "Invalid multisample mode accepted");
                texture.data_addr++;
                check(!matches(720,408,SCE_GXM_MULTISAMPLE_4X), "Partial depth view accepted as full allocation");
                texture.data_addr--;
                texture.min_filter ^= 1;
                check(!matches(720,408,SCE_GXM_MULTISAMPLE_4X), "Wrong depth byte stride accepted");
                texture.min_filter ^= 1;
                depth.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_DF32);
                check(!matches(720,408,SCE_GXM_MULTISAMPLE_4X), "D16 texture reinterpreted floating depth storage");
                depth.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_D16);
                depth.set_type(SCE_GXM_DEPTH_STENCIL_SURFACE_TILED);
                check(!matches(720,408,SCE_GXM_MULTISAMPLE_4X), "Linear depth view matched tiled storage");
                depth.set_stride(1440);
                texture = {}; texture.type = SCE_GXM_TEXTURE_TILED >> 29;
                texture.width = 1439; texture.height = 815; texture.base_format = 9;
                texture.data_addr = depth.depth_data.address() >> 2;
                check(matches(720,408,SCE_GXM_MULTISAMPLE_4X), "Tiled D16 full allocation was not matched");
                texture.mip_count = 1;
                check(!matches(720,408,SCE_GXM_MULTISAMPLE_4X), "Mipmapped depth view accepted as single level");
                texture.mip_count = 0;
                depth.set_type(SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR);
                texture.type = SCE_GXM_TEXTURE_LINEAR >> 29;
                check(matches(720,408,SCE_GXM_MULTISAMPLE_4X), "Packed linear D16 allocation was not matched");
                depth.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_DF32);
                texture.base_format = SCE_GXM_TEXTURE_BASE_FORMAT_F32 >> 24;
                check(matches(720,408,SCE_GXM_MULTISAMPLE_4X), "DF32 full allocation was not matched");
                depth.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24);
                check(!matches(720,408,SCE_GXM_MULTISAMPLE_4X), "Floating view reinterpreted packed depth");
                texture.base_format = SCE_GXM_TEXTURE_BASE_FORMAT_X8U24 >> 24;
                check(matches(720,408,SCE_GXM_MULTISAMPLE_4X), "Packed depth full allocation was not matched");
                std::cout << "PASS full depth allocation matching: D16/DF32/D24, sample footprints, layouts, stride/format/mip/offset guards\n";
            }
            {
                unsigned count=0;
                const SceGxmDepthStencilFormat formats[]={SCE_GXM_DEPTH_STENCIL_FORMAT_D16,SCE_GXM_DEPTH_STENCIL_FORMAT_DF32,
                    SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8,SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24};
                for(auto format:formats) for(auto mode:{SCE_GXM_MULTISAMPLE_NONE,SCE_GXM_MULTISAMPLE_2X,SCE_GXM_MULTISAMPLE_4X}) {
                    const uint32_t sx=mode==SCE_GXM_MULTISAMPLE_4X?2:1, sy=mode==SCE_GXM_MULTISAMPLE_NONE?1:2;
                    const uint32_t w=96*sx,h=64*sy,bytes=format==SCE_GXM_DEPTH_STENCIL_FORMAT_D16?2:4;
                    SceGxmDepthStencilSurface depth{};
                    depth.depth_data=Ptr<void>(0x200000);depth.set_format(format);depth.set_stride(w);
                    for(auto type:{SCE_GXM_TEXTURE_LINEAR_STRIDED,SCE_GXM_TEXTURE_LINEAR,SCE_GXM_TEXTURE_TILED}) {
                        depth.set_type(type==SCE_GXM_TEXTURE_TILED?SCE_GXM_DEPTH_STENCIL_SURFACE_TILED:SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR);
                        for(unsigned bottom=0;bottom<2;++bottom) {
                            const uint32_t x=type==SCE_GXM_TEXTURE_LINEAR_STRIDED?2:0;
                            const uint32_t y=bottom?h-32:0, vw=type==SCE_GXM_TEXTURE_LINEAR_STRIDED?w-4:w, vh=32;
                            SceGxmTexture t{};t.type=type>>29;t.width=vw-1;t.height=vh-1;
                            t.base_format=(format==SCE_GXM_DEPTH_STENCIL_FORMAT_D16?SCE_GXM_TEXTURE_BASE_FORMAT_U16:
                                format==SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24?SCE_GXM_TEXTURE_BASE_FORMAT_X8U24:SCE_GXM_TEXTURE_BASE_FORMAT_F32)>>24;
                            const uint64_t offset=type==SCE_GXM_TEXTURE_TILED?(uint64_t(y/32)*(w/32)+x/32)*1024*bytes:(uint64_t(y)*w+x)*bytes;
                            t.data_addr=(depth.depth_data.address()+offset)>>2;
                            if(type==SCE_GXM_TEXTURE_LINEAR_STRIDED) {
                                const uint32_t pitch=w*bytes/4-1;
                                t.mip_filter=pitch&1;t.min_filter=(pitch>>1)&3;t.mip_count=(pitch>>3)&15;t.lod_bias=(pitch>>7)&63;
                            }
                            auto rect=renderer::metal::depth_subrectangle(depth,96,64,mode,t);
                            check(rect && rect->x==x && rect->y==y && rect->width==vw && rect->height==vh,"Depth crop coordinates/sample footprint mismatch");
                            check(!renderer::metal::depth_texture_matches(depth,96,64,mode,t),"Partial depth crop classified as full allocation");
                            auto bad=t;bad.height=h;
                            check(!renderer::metal::depth_subrectangle(depth,96,64,mode,bad),"Depth crop crossing bottom accepted");
                            bad=t;bad.data_addr=(depth.depth_data.address()-4)>>2;
                            check(!renderer::metal::depth_subrectangle(depth,96,64,mode,bad),"Depth crop before allocation accepted");
                            bad=t;bad.base_format=SCE_GXM_TEXTURE_BASE_FORMAT_U8>>24;
                            check(!renderer::metal::depth_subrectangle(depth,96,64,mode,bad),"Incompatible depth crop format accepted");
                            if(type==SCE_GXM_TEXTURE_TILED) {
                                bad=t;bad.data_addr++;
                                check(!renderer::metal::depth_subrectangle(depth,96,64,mode,bad),"Interior tile address accepted as tiled origin");
                            } else if(type==SCE_GXM_TEXTURE_LINEAR_STRIDED) {
                                bad=t;bad.width=w;
                                check(!renderer::metal::depth_subrectangle(depth,96,64,mode,bad),"Depth crop crossing right edge accepted");
                            }
                            ++count;
                        }
                    }
                }
                std::cout<<"PASS "<<count<<" depth subrectangle mappings:4encodings,3sample footprints,3layouts,top/bottom origins and rejection guards\n";
            }
            struct Case { SceGxmColorFormat format; MTLPixelFormat native; unsigned count, bytes; std::array<unsigned,4> channels; };
            const Case cases[] = {
                {SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR, MTLPixelFormatRGBA8Unorm_sRGB,4,1,{0,1,2,3}},
                {SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB, MTLPixelFormatRGBA8Unorm_sRGB,4,1,{2,1,0,3}},
                {SCE_GXM_COLOR_FORMAT_U8U8U8_BGR, MTLPixelFormatRGBA8Unorm,3,1,{0,1,2}},
                {SCE_GXM_COLOR_FORMAT_U8U8U8_RGB, MTLPixelFormatRGBA8Unorm,3,1,{2,1,0}},
                {SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR, MTLPixelFormatRGBA8Unorm,4,1,{0,1,2,3}},
                {SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB, MTLPixelFormatRGBA8Unorm,4,1,{2,1,0,3}},
                {SCE_GXM_COLOR_FORMAT_U8U8U8U8_RGBA, MTLPixelFormatRGBA8Unorm,4,1,{3,2,1,0}},
                {SCE_GXM_COLOR_FORMAT_U8U8U8U8_BGRA, MTLPixelFormatRGBA8Unorm,4,1,{3,0,1,2}},
                {SCE_GXM_COLOR_FORMAT_S8S8S8S8_ABGR, MTLPixelFormatRGBA8Snorm,4,1,{0,1,2,3}},
                {SCE_GXM_COLOR_FORMAT_F16F16F16F16_ABGR, MTLPixelFormatRGBA16Float,4,2,{0,1,2,3}},
                {SCE_GXM_COLOR_FORMAT_F16F16F16F16_ARGB, MTLPixelFormatRGBA16Float,4,2,{2,1,0,3}},
                {SCE_GXM_COLOR_FORMAT_F32F32_GR, MTLPixelFormatRG32Float,2,4,{0,1}},
                {SCE_GXM_COLOR_FORMAT_F32F32_RG, MTLPixelFormatRG32Float,2,4,{1,0}},
                {SCE_GXM_COLOR_FORMAT_F32_R, MTLPixelFormatR32Float,1,4,{0}},
                {SCE_GXM_COLOR_FORMAT_F16_R, MTLPixelFormatR16Float,1,2,{0}},
                {SCE_GXM_COLOR_FORMAT_U8_R, MTLPixelFormatR8Unorm,1,1,{0}},
                {SCE_GXM_COLOR_FORMAT_U16_R, MTLPixelFormatR16Unorm,1,2,{0}},
                {SCE_GXM_COLOR_FORMAT_U8U8_GR, MTLPixelFormatRG8Unorm,2,1,{0,1}},
                {SCE_GXM_COLOR_FORMAT_U8U8_RG, MTLPixelFormatRG8Unorm,2,1,{1,0}},
            };
            {
                SceGxmColorSurface surface{}; surface.data=Ptr<void>(0x100000);
                surface.width=surface.height=surface.strideInPixels=768; surface.surfaceType=SCE_GXM_COLOR_SURFACE_LINEAR;
                surface.colorFormat=SCE_GXM_COLOR_FORMAT_U16_R;
                SceGxmTexture texture{}; texture.type=SCE_GXM_TEXTURE_LINEAR_STRIDED>>29;
                texture.width=texture.height=383; texture.base_format=9;
                const unsigned encoded_stride=1536/4-1;
                texture.mip_filter=encoded_stride&1; texture.min_filter=(encoded_stride>>1)&3;
                texture.mip_count=(encoded_stride>>3)&15; texture.lod_bias=(encoded_stride>>7)&63;
                const unsigned offsets[]={0,0x300,0x90000};
                for(unsigned i=0;i<3;++i) {
                    texture.data_addr=(surface.data.address()+offsets[i])>>2;
                    const auto rect=renderer::metal::surface_subrectangle(surface,texture);
                    check(rect && rect->x==(i==1?384:0) && rect->y==(i==2?384:0) && rect->width==384 && rect->height==384,
                        "Observed LBP U16 alias mapped to wrong subrectangle");
                }
                texture.data_addr=(surface.data.address()+0x304)>>2;
                check(!renderer::metal::surface_subrectangle(surface,texture),"Rectangle crossing the right edge was accepted");
                texture.data_addr=(surface.data.address()-4)>>2;
                check(!renderer::metal::surface_subrectangle(surface,texture),"Address before surface was accepted");
                texture.data_addr=surface.data.address()>>2; texture.min_filter=0;
                check(!renderer::metal::surface_subrectangle(surface,texture),"Different byte stride was accepted");
                std::cout<<"PASS exact observed LBP U16 subrectangles at base,+0x300,+0x90000 and bounds/stride guards\n";
            }
            {
                SceGxmColorSurface surface{}; surface.data=Ptr<void>(0x100000);
                surface.width=surface.height=surface.strideInPixels=32;
                surface.colorFormat=SCE_GXM_COLOR_FORMAT_F32_R;
                SceGxmTexture texture{};texture.type=SCE_GXM_TEXTURE_LINEAR_STRIDED>>29;
                texture.width=texture.height=15;texture.mip_count=3;texture.min_filter=3;texture.mip_filter=1;
                const auto format=uint32_t(SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR);
                texture.base_format=(format>>24)&31;texture.format0=format>>31;
                texture.data_addr=(surface.data.address()+32*4*3+4*5)>>2;
                auto rect=renderer::metal::surface_subrectangle(surface,texture);
                check(rect && rect->x==5 && rect->y==3 && rect->width==16 && rect->height==16,"F32/RGBA8 crop lost byte coordinates");
                texture.width=31;
                check(!renderer::metal::surface_subrectangle(surface,texture),"Format crop crossing right edge was accepted");
                for(bool tiled:{false,true}) {
                    surface.surfaceType=tiled?SCE_GXM_COLOR_SURFACE_TILED:SCE_GXM_COLOR_SURFACE_SWIZZLED;
                    texture={};texture.type=(tiled?SCE_GXM_TEXTURE_TILED:SCE_GXM_TEXTURE_SWIZZLED)>>29;
                    if(tiled) texture.width=texture.height=31;
                    else texture.width_base2=texture.height_base2=5;
                    texture.base_format=(format>>24)&31;texture.format0=format>>31;texture.mip_count=15;
                    texture.data_addr=surface.data.address()>>2;
                    check(renderer::metal::surface_subrectangle(surface,texture).has_value(),"Complete nonlinear format alias rejected");
                    ++texture.data_addr;
                    check(!renderer::metal::surface_subrectangle(surface,texture),"Nonlinear byte offset accepted as rectangle");
                    --texture.data_addr;texture.mip_count=1;
                    check(!renderer::metal::surface_subrectangle(surface,texture),"Single surface accepted as full mip chain");
                }
                std::cout<<"PASS F32/RGBA8 linear crop,full tiled/swizzled alias and bounds/mip guards\n";
            }
            {
                // Independent address oracle: consume Y then X bits until one
                // dimension runs out, then append the remaining coordinate bits.
                const auto morton=[](uint32_t x,uint32_t y,uint32_t w,uint32_t h) {
                    uint32_t address=0,output_bit=0;
                    for(uint32_t bit=1;bit<w || bit<h;bit<<=1) {
                        if(bit<h) { if(y&bit) address|=1u<<output_bit; ++output_bit; }
                        if(bit<w) { if(x&bit) address|=1u<<output_bit; ++output_bit; }
                    }
                    return address;
                };
                unsigned checked=0,accepted=0;
                for(uint32_t sw:{1u,2u,4u,8u,16u}) for(uint32_t sh:{1u,2u,4u,8u,16u}) {
                    SceGxmColorSurface surface{};surface.data=Ptr<void>(0x100000);
                    surface.width=surface.strideInPixels=sw;surface.height=sh;
                    surface.colorFormat=SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;surface.surfaceType=SCE_GXM_COLOR_SURFACE_SWIZZLED;
                    std::vector<std::pair<uint32_t,uint32_t>> origins(sw*sh);
                    for(uint32_t y=0;y<sh;++y) for(uint32_t x=0;x<sw;++x) origins[morton(x,y,sw,sh)]={x,y};
                    for(uint32_t wb=0;(1u<<wb)<=sw;++wb) for(uint32_t hb=0;(1u<<hb)<=sh;++hb) {
                        const uint32_t w=1u<<wb,h=1u<<hb;
                        for(uint32_t offset=0;offset<sw*sh;++offset) {
                            SceGxmTexture t{};t.type=SCE_GXM_TEXTURE_SWIZZLED>>29;t.width_base2=wb;t.height_base2=hb;t.mip_count=15;t.base_format=12;
                            t.data_addr=(surface.data.address()+offset*4)>>2;
                            const auto [x0,y0]=origins[offset];
                            bool expected=x0+w<=sw && y0+h<=sh;
                            for(uint32_t y=0;expected && y<h;++y) for(uint32_t x=0;expected && x<w;++x)
                                expected=morton(x0+x,y0+y,sw,sh)==offset+morton(x,y,w,h);
                            const auto rect=renderer::metal::surface_subrectangle(surface,t);
                            if(bool(rect)!=expected) {
                                std::cerr<<"Morton case source="<<sw<<'x'<<sh<<" view="<<w<<'x'<<h<<" offset="<<offset<<" expected="<<expected<<" accepted="<<bool(rect)<<'\n';
                                throw std::runtime_error("Morton crop acceptance disagrees with independent byte-address oracle");
                            }
                            if(rect) {check(rect->x==x0 && rect->y==y0 && rect->width==w && rect->height==h,"Morton crop origin mismatch");++accepted;}
                            ++checked;
                        }
                    }
                }
                std::cout<<"PASS "<<checked<<" Morton subimage layouts/offsets against independent per-pixel address oracle; "<<accepted<<" accepted\n";
                unsigned gpu_cases=0;size_t verified_bytes=0;
                for(bool wide:{false,true}) for(unsigned half:{0u,1u}) for(unsigned scale:{1u,2u,3u}) {
                    SceGxmColorSurface surface{};surface.data=Ptr<void>(0x64814000);
                    surface.width=surface.strideInPixels=wide?256:128;surface.height=wide?128:256;
                    surface.colorFormat=SCE_GXM_COLOR_FORMAT_U8_A;surface.surfaceType=SCE_GXM_COLOR_SURFACE_SWIZZLED;
                    SceGxmTexture t{};t.type=SCE_GXM_TEXTURE_SWIZZLED>>29;t.width_base2=t.height_base2=7;t.mip_count=15;
                    t.swizzle_format=6;t.data_addr=(surface.data.address()+half*128*128)>>2;
                    const auto rect=renderer::metal::surface_subrectangle(surface,t);
                    check(rect && rect->x==(wide?half*128:0) && rect->y==(wide?0:half*128),"DOA alpha half-surface rejected/mislocated");
                    const uint32_t nw=surface.width*scale,nh=surface.height*scale;
                    auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:nw height:nh mipmapped:NO];
                    td.storageMode=MTLStorageModeShared;td.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
                    auto source=[device->native_device() newTextureWithDescriptor:td];
                    std::vector<uint8_t> pixels(nw*nh);
                    for(uint32_t y=0;y<nh;++y) for(uint32_t x=0;x<nw;++x) pixels[y*nw+x]=(x*17+y*29+(x/scale)^((y/scale)*13))&255;
                    [source replaceRegion:MTLRegionMake2D(0,0,nw,nh) mipmapLevel:0 withBytes:pixels.data() bytesPerRow:nw];
                    auto cropped=caster.color_subrectangle(source,surface,*rect);
                    auto cast=caster.surface_format_cast(cropped,surface.colorFormat,SCE_GXM_TEXTURE_BASE_FORMAT_U8);
                    check(cast && cast.width==128*scale && cast.height==128*scale,"DOA alpha crop format/size mismatch");
                    std::vector<uint8_t> actual(cast.width*cast.height);
                    [cast getBytes:actual.data() bytesPerRow:cast.width fromRegion:MTLRegionMake2D(0,0,cast.width,cast.height) mipmapLevel:0];
                    for(uint32_t y=0;y<cast.height;++y) for(uint32_t x=0;x<cast.width;++x)
                        check(actual[y*cast.width+x]==pixels[(y+rect->y*scale)*nw+x+rect->x*scale],"DOA alpha crop changed native pixel bytes");
                    verified_bytes+=actual.size();++gpu_cases;
                    ++t.data_addr;check(!renderer::metal::surface_subrectangle(surface,t),"Misaligned Morton block accepted");
                }
                std::cout<<"PASS "<<gpu_cases<<" DOA U8_A half-surface GPU crops/casts, tall/wide, both halves,1x/2x/3x: "<<verified_bytes<<" exact bytes\n";
            }
            unsigned passed = 0;
            for (const auto &test:cases) for (auto layout:{SCE_GXM_COLOR_SURFACE_LINEAR,SCE_GXM_COLOR_SURFACE_TILED,SCE_GXM_COLOR_SURFACE_SWIZZLED})
                for (double scale:{1.0,1.5,2.0,3.0}) {
                    SceGxmColorSurface surface{};
                    surface.width=layout==SCE_GXM_COLOR_SURFACE_SWIZZLED?64:37;
                    surface.height=layout==SCE_GXM_COLOR_SURFACE_SWIZZLED?32:35;
                    surface.strideInPixels=layout==SCE_GXM_COLOR_SURFACE_SWIZZLED?64:96;
                    surface.surfaceType=layout; surface.colorFormat=test.format;
                    const size_t nw=surface.width*scale,nh=surface.height*scale,bpp=test.count*test.bytes;
                    const size_t native_count=test.count==3?4:test.count,nbpp=native_count*test.bytes;
                    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:test.native width:nw height:nh mipmapped:NO];
                    desc.storageMode=MTLStorageModeShared;
                    auto texture=[device->native_device() newTextureWithDescriptor:desc];
                    check(texture!=nil,"Cannot create readback texture");
                    std::vector<uint8_t> input(nw*nh*nbpp);
                    for(size_t y=0;y<nh;++y) for(size_t x=0;x<nw;++x) for(size_t c=0;c<native_count;++c) for(size_t b=0;b<test.bytes;++b)
                        input[(y*nw+x)*nbpp+c*test.bytes+b]=uint8_t(x*31+y*73+c*47+b*109);
                    // Include noncanonical float payloads. Raw readback must not
                    // evaluate or normalize them, even when channels are swapped.
                    if (test.bytes==4) for(size_t c=0;c<test.count;++c) {
                        const uint32_t word=0x7fa01234u+uint32_t(c);
                        std::memcpy(input.data()+c*4,&word,4);
                    }
                    [texture replaceRegion:MTLRegionMake2D(0,0,nw,nh) mipmapLevel:0 withBytes:input.data() bytesPerRow:nw*nbpp];
                    id<MTLTexture> snapshot = layout==SCE_GXM_COLOR_SURFACE_LINEAR && scale==2
                        ? caster.color_snapshot(texture) : nil;
                    const renderer::metal::SurfaceRect rectangles[]={{0,0,uint32_t(surface.width/2),uint32_t(surface.height/2)},
                        {uint32_t(surface.width/2),0,uint32_t(surface.width/2),uint32_t(surface.height/2)},
                        {0,uint32_t(surface.height/2),uint32_t(surface.width/2),uint32_t(surface.height/2)},
                        {3,5,uint32_t(surface.width-7),uint32_t(surface.height-9)}};
                    for(const auto rect:rectangles) {
                        auto crop=caster.color_subrectangle(texture,surface,rect);
                        const size_t left=rect.x*nw/surface.width,top=rect.y*nh/surface.height;
                        const size_t right=(rect.x+rect.width)*nw/surface.width,bottom=(rect.y+rect.height)*nh/surface.height;
                        check(crop.width==right-left && crop.height==bottom-top,"Native subrectangle boundaries differ");
                        std::vector<uint8_t> cropped(crop.width*crop.height*nbpp);
                        [crop getBytes:cropped.data() bytesPerRow:crop.width*nbpp fromRegion:MTLRegionMake2D(0,0,crop.width,crop.height) mipmapLevel:0];
                        for(size_t y=top;y<bottom;++y) for(size_t x=left;x<right;++x) for(size_t b=0;b<nbpp;++b)
                            check(cropped[((y-top)*crop.width+x-left)*nbpp+b]==input[(y*nw+x)*nbpp+b],"Subrectangle changed a native component or selected the wrong pixel");
                    }
                    const size_t rows=layout==SCE_GXM_COLOR_SURFACE_TILED?64:surface.height;
                    const size_t size=rows*surface.strideInPixels*bpp;
                    check(renderer::metal::surface_memory_size(surface)==size,"Wrong guest memory extent");
                    std::vector<uint8_t> output(size+32,0xa7),expected(size+32,0xa7);
                    check(renderer::metal::read_surface_memory(texture,surface,{output.data()+16,size}),"Readback rejected valid layout");
                    for(size_t y=0;y<surface.height;++y) for(size_t x=0;x<surface.width;++x) {
                        size_t at=y*surface.strideInPixels+x;
                        if(layout==SCE_GXM_COLOR_SURFACE_TILED) {
                            const size_t tile_x=x>>5,tile_y=y>>5;
                            at=tile_y*3*1024+tile_x*1024+(y&31)*32+(x&31);
                        } else if(layout==SCE_GXM_COLOR_SURFACE_SWIZZLED) {
                            // Independent rectangular Morton bit interleaving.
                            at=(x/32)*1024;
                            for(unsigned bit=0;bit<5;++bit) at|=((y>>bit)&1)<<(2*bit)|((x>>bit)&1)<<(2*bit+1);
                        }
                        for(size_t c=0;c<test.count;++c) for(size_t b=0;b<test.bytes;++b)
                            expected[16+at*bpp+c*test.bytes+b]=input[((y*nh/surface.height)*nw+x*nw/surface.width)*nbpp+test.channels[c]*test.bytes+b];
                    }
                    check(output==expected,"Readback changed a component, scale sample, layout, or padding byte");
                    const auto saved=output;
                    check(!renderer::metal::read_surface_memory(texture,surface,{output.data()+16,size-1}),"Short output buffer accepted");
                    check(output==saved,"Rejected readback partially wrote output");
                    std::vector<uint8_t> guest(size);
                    for(size_t byte=0;byte<size;++byte) guest[byte]=uint8_t(byte*119+13);
                    const renderer::metal::SurfaceMemoryRange writes[]={{1,3*bpp+1},{size/2+1,bpp+1},{size-5,3}};
                    check(renderer::metal::write_surface_memory(texture,surface,guest,writes),"Partial GPU update rejected");
                    auto expected_native=input;
                    for(size_t ny=0;ny<nh;++ny) for(size_t nx=0;nx<nw;++nx) {
                        // Invert floor(guest_coordinate * native/guest extent).
                        const size_t x=((nx+1)*surface.width-1)/nw,y=((ny+1)*surface.height-1)/nh;
                        size_t at=y*surface.strideInPixels+x;
                        if(layout==SCE_GXM_COLOR_SURFACE_TILED)
                            at=(y/32)*3*1024+(x/32)*1024+(y%32)*32+x%32;
                        else if(layout==SCE_GXM_COLOR_SURFACE_SWIZZLED) {
                            at=(x/32)*1024;
                            for(unsigned bit=0;bit<5;++bit) at|=((y>>bit)&1)<<(2*bit)|((x>>bit)&1)<<(2*bit+1);
                        }
                        for(size_t c=0;c<test.count;++c) for(size_t b=0;b<test.bytes;++b) {
                            const size_t address=at*bpp+c*test.bytes+b;
                            for(const auto &range:writes) if(address>=range.offset && address-range.offset<range.size)
                                expected_native[(ny*nw+nx)*nbpp+test.channels[c]*test.bytes+b]=guest[address];
                        }
                    }
                    std::vector<uint8_t> actual(input.size());
                    [texture getBytes:actual.data() bytesPerRow:nw*nbpp fromRegion:MTLRegionMake2D(0,0,nw,nh) mipmapLevel:0];
                    check(actual==expected_native,"Partial write lost untouched native pixels/components or used the wrong guest byte");
                    const renderer::metal::SurfaceMemoryRange invalid[]={{size-1,2}};
                    check(!renderer::metal::write_surface_memory(texture,surface,guest,invalid),"Out-of-range texture write accepted");
                    [texture getBytes:actual.data() bytesPerRow:nw*nbpp fromRegion:MTLRegionMake2D(0,0,nw,nh) mipmapLevel:0];
                    check(actual==expected_native,"Invalid write changed GPU memory");
                    if (snapshot) {
                        [snapshot getBytes:actual.data() bytesPerRow:nw*nbpp fromRegion:MTLRegionMake2D(0,0,nw,nh) mipmapLevel:0];
                        check(actual==input,"Color feedback snapshot lost bits or changed after the original target was updated");
                    }
                    ++passed;
                }
            std::cout<<"PASS "<<passed<<" raw surface readbacks: channel orders, 1x/1.5x/2x/3x, padded linear, partial tiles, rectangular Morton, NaN bits and sentinels\n";
            std::cout<<"PASS "<<passed<<" partial GPU updates: unaligned guest bytes, all channel/layout/scale cases, untouched native detail and invalid-range rejection\n";
            std::cout<<"PASS "<<std::size(cases)<<" native color feedback snapshots retain every original byte after source updates\n";
            std::cout<<"PASS "<<passed*4<<" native bit-preserving subrectangle copies across formats/layouts/scales, including fractional edge positions\n";
            unsigned sample_updates=0;
            for(const auto &test:cases) for(unsigned samples:{2u,4u}) for(unsigned scale:{1u,2u,3u})
                for(bool resolved:{false,true}) for(auto layout:{SCE_GXM_COLOR_SURFACE_LINEAR,SCE_GXM_COLOR_SURFACE_TILED,SCE_GXM_COLOR_SURFACE_SWIZZLED}) {
                    @autoreleasepool {
                        const unsigned sx=samples/2,sy=2,nw=8*scale,nh=4*scale,pw=nw*sx,ph=nh*sy;
                        const size_t bpp=test.count*test.bytes,nbpp=(test.count==3?4:test.count)*test.bytes;
                        SceGxmColorSurface surface{};
                        surface.data=Ptr<void>(0x100000); surface.colorFormat=test.format; surface.surfaceType=layout;
                        surface.width=8*(resolved?1:sx); surface.height=4*(resolved?1:sy); surface.downscale=resolved;
                        surface.strideInPixels=layout==SCE_GXM_COLOR_SURFACE_SWIZZLED ? surface.width : layout==SCE_GXM_COLOR_SURFACE_TILED ? 32 : surface.width+4;
                        const size_t size=size_t(surface.strideInPixels)*(layout==SCE_GXM_COLOR_SURFACE_TILED?32:surface.height)*bpp;
                        check(renderer::metal::surface_memory_size(surface)==size,"MSAA guest memory extent mismatch");
                        std::cout<<"MSAA byte case format="<<uint32_t(test.format)<<" native="<<test.native<<" samples="<<samples
                            <<" scale="<<scale<<" resolved="<<resolved<<" layout="<<uint32_t(layout)<<'\n';
                        auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:test.native width:pw height:ph mipmapped:NO];
                        desc.storageMode=MTLStorageModeShared;
                        desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
                        auto input_texture=[device->native_device() newTextureWithDescriptor:desc];
                        auto output_texture=[device->native_device() newTextureWithDescriptor:desc];
                        check(input_texture && output_texture,"Cannot allocate sample byte images");
                        desc.width=nw;desc.height=nh;desc.textureType=MTLTextureType2DMultisample;desc.sampleCount=samples;
                        desc.storageMode=MTLStorageModePrivate;desc.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
                        auto native=[device->native_device() newTextureWithDescriptor:desc];
                        check(native!=nil,"Cannot allocate sample byte target");
                        std::vector<uint8_t> original(size_t(pw)*ph*nbpp);
                        for(size_t i=0;i<original.size();++i) original[i]=uint8_t(i*73+i/19+11);
                        if(test.bytes==4) {
                            const uint32_t bits[]={0x7fc12345u,0xff812345u,0x7f800000u,0xff800000u,0x80000000u,0x00000001u,0x3f800000u,0x7f7fffffu};
                            for(size_t i=0;i<original.size()/4;++i) std::memcpy(original.data()+i*4,&bits[i%8],4);
                        } else if(test.bytes==2) {
                            const uint16_t bits[]={0x7e55,0xfc01,0x7c00,0xfc00,0x8000,0x0001,0x3c00,0x7bff};
                            for(size_t i=0;i<original.size()/2;++i) std::memcpy(original.data()+i*2,&bits[i%8],2);
                        }
                        if(test.count==3) for(size_t i=3;i<original.size();i+=4) original[i]=255;
                        [input_texture replaceRegion:MTLRegionMake2D(0,0,pw,ph) mipmapLevel:0 withBytes:original.data() bytesPerRow:pw*nbpp];
                        caster.seed_multisample(input_texture,native,scale,true);
                        auto read_samples=[&] {
                            caster.expand_multisample(native,output_texture,scale);
                            std::vector<uint8_t> actual(original.size());
                            [output_texture getBytes:actual.data() bytesPerRow:pw*nbpp fromRegion:MTLRegionMake2D(0,0,pw,ph) mipmapLevel:0];
                            return actual;
                        };
                        check(read_samples()==original,"Sample seed/expansion changed raw NaN, signed, gamma or channel bits");
                        auto offset=[&](size_t x,size_t y) {
                            if(layout==SCE_GXM_COLOR_SURFACE_LINEAR) return y*surface.strideInPixels+x;
                            if(layout==SCE_GXM_COLOR_SURFACE_TILED) return (y/32)*(surface.strideInPixels/32)*1024+(x/32)*1024+(y%32)*32+x%32;
                            size_t result=(x/surface.height)*surface.height*surface.height;
                            for(unsigned bit=0;(1u<<bit)<surface.height;++bit) result|=((y>>bit)&1)<<(2*bit)|((x>>bit)&1)<<(2*bit+1);
                            return result;
                        };
                        std::vector<uint8_t> guest(size);
                        for(size_t i=0;i<size;++i) guest[i]=uint8_t(0xa5+i*29);
                        std::vector<renderer::metal::SurfaceMemoryRange> ranges={{1,1},
                            {offset(surface.width/2,surface.height/2)*bpp+(bpp-1)/2,1},
                            {offset(surface.width-1,surface.height-1)*bpp+bpp-1,1}};
                        if(layout!=SCE_GXM_COLOR_SURFACE_SWIZZLED) ranges.push_back({surface.width*bpp,1});
                        std::sort(ranges.begin(),ranges.end(),[](const auto &a,const auto &b){return a.offset<b.offset;});
                        check(caster.patch_multisample(native,surface,scale,guest,ranges),"Sample patch rejected a valid byte range");
                        auto expected=original;
                        for(size_t y=0;y<ph;++y) for(size_t x=0;x<pw;++x) {
                            const size_t gx=x/(scale*(resolved?sx:1)),gy=y/(scale*(resolved?sy:1));
                            const size_t address=offset(gx,gy)*bpp;
                            for(unsigned c=0;c<test.count;++c) for(unsigned byte=0;byte<test.bytes;++byte)
                                for(const auto &range:ranges) if(address+c*test.bytes+byte==range.offset)
                                    expected[(y*pw+x)*nbpp+test.channels[c]*test.bytes+byte]=guest[range.offset];
                        }
                        check(read_samples()==expected,"Sample patch changed an untouched byte/sample or used wrong guest layout");
                        const renderer::metal::SurfaceMemoryRange invalid[]={{size-1,2}};
                        check(!caster.patch_multisample(native,surface,scale,guest,invalid),"Out-of-range sample patch accepted");
                        check(!caster.patch_multisample(native,surface,scale,{guest.data(),size-1},ranges),"Short sample patch source accepted");
                        check(caster.patch_multisample(native,surface,scale,guest,{}),"Empty sample patch rejected");
                        check(read_samples()==expected,"Rejected or empty sample patch changed native bytes");
                        ++sample_updates;
                    }
                }
            std::cout<<"PASS "<<sample_updates<<" raw MSAA seed/expand/partial-patch cases:19formats,2/4samples,1x/2x/3x,resolved/expanded,3layouts,NaNpayloads,signed/gamma bits,untouched components and rejection guards\n";
            return 0;
        } catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
