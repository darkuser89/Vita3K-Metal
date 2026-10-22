// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Actual scene color allocation statements; CPU allocator boundary only.
#include <gxm/types.h>
#include <cassert>
#include <cstdint>
#include <iostream>

struct Texture {
    unsigned width=0, height=0, pixelFormat=0, identity=0;
    explicit operator bool() const { return identity != 0; }
};
#define nil Texture{}
constexpr unsigned MTLTextureUsageRenderTarget=1, MTLTextureUsageShaderRead=2,
    MTLTextureUsagePixelFormatView=4, MTLTextureUsageShaderWrite=8;
struct Device {};
static unsigned allocations, linear_attempts;
static bool linear_available;
static Texture make_linear_buffer_texture(Device &, unsigned format, unsigned w, unsigned h, unsigned usage) {
    assert(usage==15);++linear_attempts;
    return linear_available ? Texture{w,h,format,++allocations+100} : Texture{};
}
static Texture make_texture(Device &, unsigned format, unsigned w, unsigned h, unsigned usage) {
    assert(usage==15);return {w,h,format,++allocations+100};
}
int main() {
    size_t cases=0;
    for (bool populated:{false,true})
    for (unsigned old_w:{31u,32u,64u})for (unsigned old_h:{15u,16u,32u})
    for (unsigned old_format:{1u,2u,3u})for (unsigned format:{1u,2u,3u})
    for (unsigned samples:{1u,2u,4u})for (float res_multiplier:{.5f,1.f,2.f})
    for (auto type:{SCE_GXM_COLOR_SURFACE_LINEAR,SCE_GXM_COLOR_SURFACE_TILED})
    for (bool available:{false,true}) {
        const unsigned color_width=32,color_height=16;
        struct {Texture color;} entry{{old_w,old_h,old_format,populated?17u:0u}};
        const auto before=entry.color;
        struct {SceGxmColorSurfaceType surfaceType;} surface{type};
        Device device;struct {Device *device;} storage{&device};auto *impl=&storage;
        allocations=linear_attempts=0;linear_available=available;
        // ALLOCATION_HOOK
        const bool reuse=populated && old_w==32 && old_h==16 && old_format==format;
        assert(entry.color && entry.color.width==32 && entry.color.height==16 && entry.color.pixelFormat==format);
        assert(allocations==!reuse);
        assert((entry.color.identity==before.identity)==reuse);
        assert(linear_attempts==(!reuse && samples==1 && res_multiplier==1.f && type==SCE_GXM_COLOR_SURFACE_LINEAR));
        ++cases;
    }
    assert(cases==5832);
    std::cout<<"PASS scene color allocation: "<<cases<<" actual production allocation branches; cache hits retain owner, changed dimensions/formats replace old owner, 1x/2x/4x samples, scales, linear allocation/fallback; ASAN/UBSAN, no device/GPU/app\n";
}
