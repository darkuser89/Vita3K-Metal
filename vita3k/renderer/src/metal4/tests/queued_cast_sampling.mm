// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Production sampling helper and gamma encoder with CPU boundary doubles only.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <gxm/types.h>
#include <array>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace gxm {
// GXM_HOOK
}
static std::vector<std::string> calls;
static bool fail_cast,fail_gamma_allocation,fail_pipeline;
static unsigned submits,allocations;
static id<MTLTexture> last_cast_source=nil;
static void require(bool b,const std::string &error) { if(!b)throw std::runtime_error(error); }
@interface SamplingTexture : NSObject
@property NSUInteger width,height;
@property MTLPixelFormat pixelFormat;
@end
@implementation SamplingTexture
@end
static id<MTLTexture> image(MTLPixelFormat format,NSUInteger w=7,NSUInteger h=3) {
    auto t=[SamplingTexture new];t.pixelFormat=format;t.width=w;t.height=h;return (id<MTLTexture>)t;
}
@interface SamplingEncoder : NSObject
- (void)setComputePipelineState:(id)p;
- (void)setTexture:(id<MTLTexture>)t atIndex:(NSUInteger)i;
- (void)dispatchThreads:(MTLSize)size threadsPerThreadgroup:(MTLSize)group;
- (void)endEncoding;
@end
@implementation SamplingEncoder
- (void)setComputePipelineState:(id)p { calls.push_back("pipeline:"+std::string([(NSString *)p UTF8String])); }
- (void)setTexture:(id<MTLTexture>)t atIndex:(NSUInteger)i { assert(t);calls.push_back("texture:"+std::to_string(i)+":"+std::to_string(t.pixelFormat)); }
- (void)dispatchThreads:(MTLSize)s threadsPerThreadgroup:(MTLSize)g {
    assert(s.depth==1&&g.width==8&&g.height==8&&g.depth==1);
    calls.push_back("dispatch:"+std::to_string(s.width)+":"+std::to_string(s.height));
}
- (void)endEncoding { calls.push_back("end"); }
@end
@interface V3KMetal4Commands : NSObject
@property MTLCommandBufferStatus status;
- (SamplingEncoder *)computeCommandEncoder;
@end
@implementation V3KMetal4Commands
- (instancetype)init { if((self=[super init]))_status=MTLCommandBufferStatusNotEnqueued;return self; }
- (SamplingEncoder *)computeCommandEncoder { assert(_status==MTLCommandBufferStatusNotEnqueued);calls.push_back("encoder");return [SamplingEncoder new]; }
@end
@interface SamplingDevice : NSObject
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d;
@end
@implementation SamplingDevice
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d {
    ++allocations;if(fail_gamma_allocation)return nil;
    assert(d.storageMode==MTLStorageModeShared&&d.usage==(MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView));
    calls.push_back("allocate:"+std::to_string(d.pixelFormat));return image(d.pixelFormat,d.width,d.height);
}
@end
struct Device {
    id<MTLDevice> native=(id<MTLDevice>)[SamplingDevice new];
    id<MTLDevice> native_device() {return native;}
    bool submit_and_wait(V3KMetal4Commands *c,std::string &) {++submits;c.status=MTLCommandBufferStatusCompleted;return true;}
};
static V3KMetal4Commands *surface_command_buffer(Device &,NSString *) {return [V3KMetal4Commands new];}
static id<MTLComputePipelineState> cached_compute(Device &,id<MTLLibrary>,NSString *name) {
    if(fail_pipeline)throw std::runtime_error("pipeline");return (id<MTLComputePipelineState>)name;
}
static id<MTLTexture> rgba8_gamma_view(id<MTLTexture> t,bool gamma) {
    calls.push_back("gamma_view:"+std::to_string(gamma));
    return image(t.pixelFormat==MTLPixelFormatRGBA8Unorm||t.pixelFormat==MTLPixelFormatRGBA8Unorm_sRGB
        ? gamma ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm : t.pixelFormat,t.width,t.height);
}
static id<MTLTexture> sampling_view(id<MTLTexture> t,SceGxmTextureFormat format,const SceGxmColorFormat *color) {
    assert(format==SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR&&color);
    calls.push_back("memory_order:"+std::to_string(uint32_t(*color)));return t;
}
struct SurfaceCaster {
    Device &device;
    id<MTLComputePipelineState> rg_gamma_pipeline=nil,depth_pipeline=nil;
    id<MTLLibrary> multisample_library=nil;
    explicit SurfaceCaster(Device &d):device(d){}
    id<MTLComputePipelineState> compute_pipeline(id<MTLComputePipelineState> __strong &p,NSString *name) {
        if(!p)p=cached_compute(device,multisample_library,name);return p;
    }
    id<MTLTexture> surface_format_cast(id<MTLTexture> source,SceGxmColorFormat color,SceGxmTextureBaseFormat target,uint32_t swizzle,V3KMetal4Commands *commands) {
        if(fail_cast)return nil;
        last_cast_source=source;
        if(commands)assert(commands.status==MTLCommandBufferStatusNotEnqueued);else ++submits;
        calls.push_back("cast:"+std::to_string(uint32_t(color))+":"+std::to_string(uint32_t(target))+":"+std::to_string(swizzle));
        // The real packer/encoder is exercised by packed_surface_cast.mm.
        return image(source.pixelFormat,source.width,source.height);
    }
    id<MTLTexture> rgba8_surface_sampling(id<MTLTexture>,SceGxmColorFormat,uint32_t,V3KMetal4Commands * = nil);
};
// GAMMA_HOOK
// REPACKED_HOOK
// QUEUED_HOOK
static SceGxmTexture descriptor(SceGxmTextureBaseFormat base,unsigned mode,unsigned gamma) {
    SceGxmTexture texture{};texture.base_format=(uint32_t(base)>>24)&31;texture.format0=uint32_t(base)>>31;
    texture.swizzle_format=mode;texture.gamma_mode=gamma;return texture;
}
static void encoding_tests() {
    Device device;size_t cases=0;
    for(auto base:{SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10,SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8,SCE_GXM_TEXTURE_BASE_FORMAT_U16})
    for(unsigned mode=0;mode<8;++mode)for(unsigned gamma:{0u,1u,3u})
    for(auto native:{MTLPixelFormatRGBA8Unorm,MTLPixelFormatRGBA8Unorm_sRGB,MTLPixelFormatBGR10A2Unorm,MTLPixelFormatRGBA16Float}) {
        SurfaceCaster reference{device},actual{device};auto source=image(native);auto texture=descriptor(base,mode,gamma);
        const auto color=SCE_GXM_COLOR_FORMAT_U5U6U5_BGR;
        calls.clear();submits=0;
        auto cast=reference.surface_format_cast(source,color,base,mode,nil);
        id<MTLTexture> decoded=nil;
        const bool needs_decode=native==MTLPixelFormatRGBA8Unorm_sRGB||gamma;
        if(needs_decode) {
            auto expected_color=base==SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10&&(mode&2)
                ? SceGxmColorFormat(uint32_t(SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10)|((mode&3)<<20))
                : SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
            decoded=reference.rgba8_surface_sampling(cast,expected_color,gamma);
        }
        const auto expected=calls;assert(submits==1+unsigned(needs_decode));
        calls.clear();submits=0;auto commands=[V3KMetal4Commands new];
        auto result=enqueue_surface_cast_sampling(actual,source,color,texture,commands);
        assert(result.first&&bool(result.second)==needs_decode&&calls==expected&&submits==0);
        assert(commands.status==MTLCommandBufferStatusNotEnqueued&&result.first.pixelFormat==native);
        if(needs_decode)assert(result.second.pixelFormat==decoded.pixelFormat&&result.second.width==7&&result.second.height==3);
        ++cases;
    }
    // Invalid/committed queue is rejected before encoding or allocation.
    for(auto status:{MTLCommandBufferStatusNotEnqueued,MTLCommandBufferStatusEnqueued,MTLCommandBufferStatusCommitted,MTLCommandBufferStatusScheduled,MTLCommandBufferStatusCompleted,MTLCommandBufferStatusError}) {
        auto c=[V3KMetal4Commands new];c.status=status;SurfaceCaster caster{device};calls.clear();const auto n=allocations;
        bool threw=false;try{enqueue_surface_cast_sampling(caster,image(MTLPixelFormatRGBA8Unorm),SCE_GXM_COLOR_FORMAT_U5U6U5_BGR,descriptor(SCE_GXM_TEXTURE_BASE_FORMAT_U16,0,1),status==MTLCommandBufferStatusNotEnqueued?nil:c);}catch(const std::runtime_error &){threw=true;}
        assert(threw&&calls.empty()&&allocations==n);
    }
    for(unsigned failure=0;failure<4;++failure) {
        SurfaceCaster caster{device};auto c=[V3KMetal4Commands new];auto t=descriptor(SCE_GXM_TEXTURE_BASE_FORMAT_U16,0,failure==3?2:3);
        fail_cast=failure==0;fail_gamma_allocation=failure==1;fail_pipeline=failure==2;calls.clear();submits=0;
        bool threw=false;try{enqueue_surface_cast_sampling(caster,image(MTLPixelFormatRGBA8Unorm),SCE_GXM_COLOR_FORMAT_U5U6U5_BGR,t,c);}catch(const std::runtime_error &){threw=true;}
        assert(threw&&submits==0&&c.status==MTLCommandBufferStatusNotEnqueued);
        assert(std::find(calls.begin(),calls.end(),"encoder")==calls.end());
        fail_cast=fail_gamma_allocation=fail_pipeline=false;t.gamma_mode=3;
        assert(enqueue_surface_cast_sampling(caster,image(MTLPixelFormatRGBA8Unorm),SCE_GXM_COLOR_FORMAT_U5U6U5_BGR,t,c).second);
    }
    std::cout<<"PASS queued cast sampling: "<<cases<<" actual helper/gamma encodings match synchronous format/channel/pipeline sequences with zero submissions; U2 alpha, sRGB, RG gamma, queue validation, failures and retry; CPU boundary doubles only\n";
}
struct GuestAddress { uint32_t address() const {return 4096;} };
struct Guest { GuestAddress data; SceGxmColorFormat colorFormat=SCE_GXM_COLOR_FORMAT_U5U6U5_BGR; };
struct Entry {Guest guest;id<MTLTexture> color;};
struct RenderImpl {Device *device;std::unique_ptr<SurfaceCaster> caster;std::map<uint32_t,Entry> surfaces;};
struct ContextImpl {Guest guest_color;id<MTLTexture> color;V3KMetal4Commands *commands=nil;};
static bool test_raw,test_enqueueable,test_rect;
static bool surface_raw_cast_required(SceGxmColorFormat,SceGxmTextureBaseFormat) {return test_raw;}
static bool surface_format_cast_enqueueable(SceGxmColorFormat,SceGxmTextureBaseFormat,uint32_t) {return test_enqueueable;}
static std::optional<int> cached_surface_subrectangle(const Guest &,const SceGxmTexture &) {return test_rect?std::optional(1):std::nullopt;}
static id<MTLTexture> raw_surface_color(const Entry &e) {return e.color;}
static void routing_tests() {
    Device device;size_t cases=0;
    for(unsigned route=0;route<3;++route)for(unsigned gamma:{0u,1u,3u})for(unsigned mask=0;mask<512;++mask) {
        @autoreleasepool {
            const bool same_format=mask&1,enqueueable_cast=mask&2,raw_cast=mask&4,capture_draw=mask&8;
            const bool active_target=mask&16,rectangle=mask&32,word_alias=mask&64,found=mask&128,has_raw_feedback=mask&256;
            test_raw=raw_cast;test_enqueueable=enqueueable_cast;test_rect=rectangle;
            auto texture=descriptor(SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10,2,gamma);
            Entry entry;entry.color=image(MTLPixelFormatBGR10A2Unorm);
            ContextImpl context;context.color=active_target?entry.color:image(MTLPixelFormatRGBA16Float);
            struct {ContextImpl *impl;} ctx{&context};
            RenderImpl storage{&device,std::make_unique<SurfaceCaster>(device),{}};auto impl=&storage;
            if(found)impl->surfaces[4096]=entry;
            const auto base=gxm::get_base_format(gxm::get_format(texture));
            const uint32_t address=4096;const bool cropped=route==1;
            auto image=entry.color;auto color_feedback=::image(MTLPixelFormatBGR10A2Unorm);
            id<MTLTexture> raw_color_feedback=has_raw_feedback?::image(MTLPixelFormatBGR10A2Unorm):nil;
            std::array<id<MTLTexture>,1> inline_cast_images{},inline_gamma_images{};
            std::array<bool,1> inline_crops{};bool expanded=false;
            const auto prepare_inline_commands=[&]{calls.push_back("prepare");if(!context.commands)context.commands=[V3KMetal4Commands new];};
            const auto prepare_inline_color=[&]{prepare_inline_commands();expanded=true;};
            calls.clear();submits=0;last_cast_source=nil;
            for(size_t index=0;index<1;++index) {
                if(route==2) {
                    const auto active=impl->surfaces.find(4096);
                    // FEEDBACK_POLICY_HOOK
                    const bool inline_feedback=!capture_draw&&(same_format||gpu_cast);
                    if(inline_feedback) {
                        prepare_inline_color();
                        // FEEDBACK_CALL_HOOK
                    }
                } else {
                    // GENERAL_POLICY_HOOK
                    inline_crops[index]=(same_format||gpu_cast)&&!capture_draw;
                    if(route==0) {
                        // INACTIVE_CALL_HOOK
                    } else {
                        // CROP_CALL_HOOK
                    }
                }
            }
            const bool generic=!same_format&&enqueueable_cast&&!raw_cast&&!capture_draw;
            const bool expected=generic&&(route==0?!active_target:route==1?true:!word_alias&&active_target&&found&&rectangle);
            assert(bool(inline_cast_images[0])==expected);
            assert(bool(inline_gamma_images[0])==(expected&&gamma));
            assert(submits==0);
            if(expected) {
                assert(context.commands.status==MTLCommandBufferStatusNotEnqueued);
                assert(!calls.empty()&&calls.front()=="prepare");
                assert(last_cast_source==(route==2?(raw_color_feedback?raw_color_feedback:color_feedback):entry.color));
                if(route==1&&active_target)assert(expanded);
                auto first_encoder=std::find(calls.begin(),calls.end(),"encoder");
                auto first_cast=std::find_if(calls.begin(),calls.end(),[](const std::string &c){return c.starts_with("cast:");});
                assert(first_cast!=calls.end()&&(first_encoder==calls.end()||first_cast<first_encoder));
            }
            ++cases;
        }
    }
    std::cout<<"PASS queued cast routing: "<<cases<<" actual inactive/crop/feedback call blocks and eligibility expressions; gamma and U2 metadata, active/inactive producer ordering, capture/raw/unsupported/alias/layout exclusions and raw-feedback source selection; no GPU\n";
}
int main(){@autoreleasepool{encoding_tests();routing_tests();}}
