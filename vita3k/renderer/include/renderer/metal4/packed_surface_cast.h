// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
namespace renderer::metal {
// Integer-only packing: never round-trip raw native payloads through a sampled
// float. Tests compile these exact helper bodies as C++ against the CPU packers.
inline constexpr const char packed_surface_cast_msl[] = R"msl(
#include <metal_stdlib>
using namespace metal;
// Config: width,height,inputPitch,outputPitch,inputBytes,outputBytes,guestBytes,
// sourceKind,sourceMode,targetKind,targetMode,count,componentBytes,channels[4].
uint pc_load16(thread const uint *v, uint i) { return v[i] | (v[i+1] << 8); }
uint pc_load32(thread const uint *v) { return pc_load16(v,0) | (pc_load16(v,2) << 16); }
void pc_store(thread uint *v, uint word, uint count) {
    for (uint i=0;i<count;++i) v[i]=(word>>(8*i))&255;
}
uint pc_round_shift(uint value, uint shift) {
    return shift ? (value+(1u<<(shift-1)))>>shift : value;
}
int pc_quantize_half(uint bits, uint scale, int low, int high) {
    const uint exponent=(bits>>10)&31, fraction=bits&1023;
    if (exponent==31) return fraction ? 0 : (bits&32768 ? low : high);
    const uint value=(fraction+(exponent ? 1024 : 0))*scale;
    const uint e=exponent ? exponent : 1;
    const uint rounded=e>=25 ? value<<(e-25) : pc_round_shift(value,25-e);
    const int signed_value=bits&32768 ? -int(rounded) : int(rounded);
    return clamp(signed_value,low,high);
}
uint pc_positive_half(uint bits) {
    if ((bits&32768) || ((bits&0x7c00)==0x7c00 && (bits&1023))) return 0;
    return min(bits,0x7bfcu); // 65408, also the positive-infinity clamp
}
uint pc_e5_mantissa(uint bits, uint exponent) {
    const uint e=(bits>>10)&31, value=(bits&1023)+(e ? 1024 : 0);
    return pc_round_shift(value,exponent-(e ? e-1 : 0));
}
uint pc_rgb9e5(thread const uint *v) {
    const uint r=pc_positive_half(pc_load16(v,0)), g=pc_positive_half(pc_load16(v,2)), b=pc_positive_half(pc_load16(v,4));
    const uint biggest=max(r,max(g,b));
    if (!biggest) return 0;
    uint exponent=(biggest>>10) ? (biggest>>10)+1 : (biggest>=512 ? 1 : 0);
    if (pc_e5_mantissa(biggest,exponent)>=512 && exponent<31) ++exponent;
    return min(pc_e5_mantissa(r,exponent),511u)
        | (min(pc_e5_mantissa(g,exponent),511u)<<9)
        | (min(pc_e5_mantissa(b,exponent),511u)<<18) | (exponent<<27);
}
// Exact rational -> binary16, ties to even. These normalized byte values cannot
// double-round through the CPU's binary32 division to a different half value.
uint pc_normalized_half(uint numerator, uint denominator, bool negative) {
    if (!numerator) return 0;
    int exponent=0;
    while (numerator<denominator) { numerator<<=1; --exponent; }
    uint value=(numerator<<10)/denominator;
    const uint remainder=(numerator<<10)%denominator;
    if (2*remainder>denominator || (2*remainder==denominator && (value&1))) ++value;
    if (value==2048) { value=1024; ++exponent; }
    return (negative ? 32768u : 0u) | (uint(exponent+15)<<10) | (value-1024);
}
uint pc_e5_swap(uint word) { return (word&0xf803fe00u)|((word&511)<<18)|((word>>18)&511); }
uint pc_u2_swap(uint word) { return (word&0xc00ffc00u)|((word&1023)<<20)|((word>>20)&1023); }
void pc_pixel(thread const uint *native, thread uint *guest, constant uint *c) {
    uint word=pc_load32(native);
    const uint mode=c[8];
    switch(c[7]) {
    case 0: // Unpacked components, in guest memory-channel order.
        for(uint channel=0;channel<c[11];++channel)
            for(uint byte=0;byte<c[12];++byte) guest[channel*c[12]+byte]=native[c[13+channel]*c[12]+byte];
        break;
    case 1: // 565
        word&=65535;
        if (!mode) word=(word&0x07e0u)|((word&0xf800u)>>11)|((word&31)<<11);
        pc_store(guest,word,2); break;
    case 2: // 5551; preserve the CPU path's two native alpha-bit layouts.
        word&=65535;
        if (mode==0 || mode==3) word=mode<2
            ? (word&0x83e0u)|((word&0x7c00u)>>10)|((word&31)<<10)
            : (word&0x07c1u)|((word&0xf800u)>>10)|((word&0x3eu)<<10);
        pc_store(guest,word,2); break;
    case 3: { // 4444 nibble permutation, independent of byte endianness.
        uint permuted=0;
        for(uint i=0;i<4;++i) permuted|=((word>>(12-4*c[13+i]))&15)<<(12-4*i);
        pc_store(guest,permuted,2); break;
    }
    case 4: // Native RGB9E5 storage.
        pc_store(guest,mode ? pc_e5_swap(word) : word,4); break;
    case 5: // U2U10U10U10 guest alpha-first or alpha-last layouts.
        if (mode==1) word=pc_u2_swap(word);
        else if (mode>=2) {
            const uint first=word&1023, green=(word>>10)&1023, third=(word>>20)&1023;
            word=(word>>30)|((mode==2 ? first : third)<<2)|(green<<12)|((mode==2 ? third : first)<<22);
        }
        pc_store(guest,word,4); break;
    case 6: // F11F11F10 is synchronized as raw words, with no RGB/BGR shuffle.
        pc_store(guest,word,4); break;
    case 7: // U8U3U3U2 expanded in RGBA8.
        word=(native[3]<<8)|(((native[0]*7+127)/255)<<5)|(((native[1]*7+127)/255)<<2)|((native[2]*3+127)/255);
        pc_store(guest,word,2); break;
    case 8: { // S5S5U6 expanded in RGBA16F.
        const uint r=uint(pc_quantize_half(pc_load16(native,0),15,-16,15))&31;
        const uint g=uint(pc_quantize_half(pc_load16(native,2),15,-16,15))&31;
        const uint b=uint(pc_quantize_half(pc_load16(native,4),63,0,63));
        pc_store(guest,mode ? (r<<11)|(g<<6)|b : (b<<10)|(g<<5)|r,2); break;
    }
    case 9: { // Mixed signed guest bytes expanded in RGBA16F.
        const uint orders[16]={0,1,2,3, 2,1,0,3, 3,2,1,0, 1,2,3,0};
        for(uint i=0;i<4;++i) {
            const bool signed_channel=i==1 || i==2;
            guest[orders[mode*4+i]]=uint(pc_quantize_half(pc_load16(native,i*2),signed_channel ? 127 : 255,signed_channel ? -128 : 0,signed_channel ? 127 : 255))&255;
        }
        break;
    }
    case 10: // RGB9E5 expanded in RGBA16F.
        word=pc_rgb9e5(native);
        pc_store(guest,mode ? pc_e5_swap(word) : word,4); break;
    }
    if(c[9]==1) { // Rebuild Metal's alpha-high U2U10U10U10 word.
        word=pc_load32(guest);
        const uint first=(word>>2)&1023, green=(word>>12)&1023, third=(word>>22)&1023;
        word=(c[10]==2 ? first : third)|(green<<10)|((c[10]==2 ? third : first)<<20)|((word&3)<<30);
        pc_store(guest,word,4);
    } else if(c[9]==2) { // X8S8S8U8 output is expanded RGBA16F, with alpha one.
        const uint red=guest[c[10] ? 2 : 0], green=guest[1], blue=guest[c[10] ? 0 : 2];
        const int g=max(green<128 ? int(green) : int(green)-256,-127);
        const int b=max(blue<128 ? int(blue) : int(blue)-256,-127);
        const uint halfs[4]={pc_normalized_half(red,255,false),pc_normalized_half(uint(abs(g)),127,g<0),pc_normalized_half(uint(abs(b)),127,b<0),0x3c00};
        for(uint i=0;i<4;++i) { guest[2*i]=halfs[i]&255; guest[2*i+1]=halfs[i]>>8; }
    }
}
// CORE_END
kernel void repack_packed_surface(device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]], constant uint *c [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    if(p.x>=c[0] || p.y>=c[1]) return;
    uint native[8]={}, guest[8]={};
    for(uint i=0;i<c[4];++i) native[i]=input[p.y*c[2]+p.x*c[4]+i];
    pc_pixel(native,guest,c);
    for(uint i=0;i<c[5];++i) output[p.y*c[3]+p.x*c[5]+i]=uchar(guest[i]);
}
)msl";
}
