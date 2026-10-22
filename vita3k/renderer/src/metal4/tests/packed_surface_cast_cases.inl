// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
static std::vector<uint8_t> reference_bytes(CastTexture *t,SceGxmColorFormat color,SceGxmTextureBaseFormat target,unsigned swizzle) {
    const auto input=surface_components(color);
    SceGxmColorSurface surface{color,uint32_t(t.width),uint32_t(t.height),uint32_t(t.width),SCE_GXM_COLOR_SURFACE_LINEAR};
    std::vector<uint8_t> bytes(t.width*t.height*input.count*input.bytes);
    if(!read_surface_memory((id<MTLTexture>)t,surface,bytes,nullptr))return {};
    if(target==SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10&&swizzle<8&&(swizzle&2))
        for(size_t offset=0;offset<bytes.size();offset+=4) {
            uint32_t word;std::memcpy(&word,bytes.data()+offset,4);
            word=packed_10_native_word(word,swizzle&3);std::memcpy(bytes.data()+offset,&word,4);
        }
    if(target==SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8) {
        std::vector<uint16_t> expanded(t.width*t.height*4);
        renderer::texture::convert_x8s8s8u8_to_f16f16f16f16(expanded.data(),bytes.data(),uint32_t(t.width),uint32_t(t.height),SceGxmTextureFormat(uint32_t(target)|(swizzle<<12)));
        bytes.resize(expanded.size()*2);std::memcpy(bytes.data(),expanded.data(),bytes.size());
    }
    return bytes;
}
static void numeric_tests() {
    size_t cases=0;
    std::mt19937 rng(0x28b17);
    for(unsigned bits=0;bits<65536;++bits) {
        uint16_t halfs[4]{uint16_t(bits),uint16_t(bits),uint16_t(bits),uint16_t(bits)};
        uint32_t v[8]{};for(unsigned i=0;i<4;++i){v[2*i]=halfs[i]&255;v[2*i+1]=halfs[i]>>8;}
        assert(gpu::pc_rgb9e5(v)==pack_rgb9e5(halfs));++cases;
        for(unsigned mode=0;mode<2;++mode) {
            uint32_t c[17]{},out[8]{};c[7]=8;c[8]=mode;
            gpu::pc_pixel(v,out,c);
            assert(gpu::pc_load16(out,0)==pack_s5s5u6(halfs,mode));++cases;
        }
        for(unsigned mode=0;mode<4;++mode) {
            uint32_t c[17]{},out[8]{};uint8_t expected[4]{};c[7]=9;c[8]=mode;
            gpu::pc_pixel(v,out,c);pack_u8s8s8u8(halfs,mode,expected);
            for(unsigned i=0;i<4;++i) assert(out[i]==expected[i]);++cases;
        }
    }
    for(unsigned i=0;i<200000;++i) {
        uint16_t halfs[4]{uint16_t(rng()),uint16_t(rng()),uint16_t(rng()),uint16_t(rng())};
        uint32_t v[8]{};for(unsigned j=0;j<4;++j){v[2*j]=halfs[j]&255;v[2*j+1]=halfs[j]>>8;}
        assert(gpu::pc_rgb9e5(v)==pack_rgb9e5(halfs));++cases;
    }
    for(unsigned mode=0;mode<2;++mode) for(unsigned component=0;component<4;++component) for(unsigned value=0;value<256;++value) {
        uint8_t bytes[4]{17,253,128,0};bytes[component]=value;
        uint16_t expected[4]{};
        renderer::texture::convert_x8s8s8u8_to_f16f16f16f16(expected,bytes,1,1,SceGxmTextureFormat(uint32_t(SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8)|(mode<<12)));
        uint32_t v[8]{bytes[0],bytes[1],bytes[2],bytes[3]},out[8]{},c[17]{};
        c[7]=6;c[9]=2;c[10]=mode;gpu::pc_pixel(v,out,c);
        for(unsigned j=0;j<4;++j)assert(gpu::pc_load16(out,2*j)==expected[j]);++cases;
    }
    std::cout<<"PASS packed cast arithmetic: "<<cases<<" exact CPU-packer comparisons, all 65536 half encodings, 200000 mixed-exponent RGB9E5 words, signed quantization and normalized-byte decode\n";
}
static void encoding_tests() {
    const SceGxmColorBaseFormat colors[]={
        SCE_GXM_COLOR_BASE_FORMAT_U5U6U5,SCE_GXM_COLOR_BASE_FORMAT_U1U5U5U5,SCE_GXM_COLOR_BASE_FORMAT_U4U4U4U4,
        SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9,SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10,SCE_GXM_COLOR_BASE_FORMAT_F11F11F10,
        SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2,SCE_GXM_COLOR_BASE_FORMAT_S5S5U6,SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8,
        SCE_GXM_COLOR_BASE_FORMAT_U8U8,SCE_GXM_COLOR_BASE_FORMAT_S8S8,SCE_GXM_COLOR_BASE_FORMAT_U16,SCE_GXM_COLOR_BASE_FORMAT_S16,SCE_GXM_COLOR_BASE_FORMAT_F16,
        SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8,SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8,SCE_GXM_COLOR_BASE_FORMAT_U16U16,
        SCE_GXM_COLOR_BASE_FORMAT_S16S16,SCE_GXM_COLOR_BASE_FORMAT_F16F16,SCE_GXM_COLOR_BASE_FORMAT_F32};
    const SceGxmTextureBaseFormat targets[]={
        SCE_GXM_TEXTURE_BASE_FORMAT_U5U6U5,SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5,SCE_GXM_TEXTURE_BASE_FORMAT_U4U4U4U4,
        SCE_GXM_TEXTURE_BASE_FORMAT_SE5M9M9M9,SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10,SCE_GXM_TEXTURE_BASE_FORMAT_F11F11F10,SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8,
        SCE_GXM_TEXTURE_BASE_FORMAT_U8U8,SCE_GXM_TEXTURE_BASE_FORMAT_S8S8,SCE_GXM_TEXTURE_BASE_FORMAT_U16,SCE_GXM_TEXTURE_BASE_FORMAT_S16,SCE_GXM_TEXTURE_BASE_FORMAT_F16,
        SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8,SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8,SCE_GXM_TEXTURE_BASE_FORMAT_U16U16,
        SCE_GXM_TEXTURE_BASE_FORMAT_S16S16,SCE_GXM_TEXTURE_BASE_FORMAT_F16F16,SCE_GXM_TEXTURE_BASE_FORMAT_F32};
    Device device;SurfaceCaster caster{device};std::mt19937 rng(0x28eac0de);size_t cases=0,pixels=0;
    for(auto base:colors)for(unsigned mode=0;mode<8;++mode)for(auto target:targets)for(unsigned swizzle=0;swizzle<8;++swizzle) {
        auto color=SceGxmColorFormat(uint32_t(base)|(mode<<20));
        if(!(packed_color_cast_source(base)||packed_texture_cast_target(target))||!surface_format_cast_supported(color,target,swizzle))continue;
        for(unsigned variant=0;variant<(base==SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9?2:1);++variant)for(unsigned size=0;size<3;++size) {
            @autoreleasepool {
                auto source=[CastTexture new];source.pixelFormat=variant?MTLPixelFormatRGBA16Float:surface_components(color).native;
                source.width=size==0?1:size==1?7:17;source.height=size==0?1:size==1?3:5;
                // Exercise encoded sRGB storage, not sampled/decoded gamma values.
                if(source.pixelFormat==MTLPixelFormatRGBA8Unorm&&size==2)source.pixelFormat=MTLPixelFormatRGBA8Unorm_sRGB;
                source->pixels.resize(source.width*source.height*native_bytes(source.pixelFormat));
                for(auto &byte:source->pixels)byte=uint8_t(rng());
                auto expected=reference_bytes(source,color,target,swizzle);
                if(expected.empty())continue; // combinations rejected by the old CPU readback are not newly advertised
                assert(surface_format_cast_enqueueable(color,target,swizzle));
                const auto reads=source->cpu_reads,submits=submissions;
                auto commands=[V3KMetal4Commands new];
                const bool own=(cases%71)==0;
                auto result=(CastTexture *)caster.surface_format_cast((id<MTLTexture>)source,color,target,swizzle,own?nil:commands);
                assert(result&&source->cpu_reads==reads&&result->cpu_writes==0);
                assert(result.width==source.width&&result.height==source.height);
                const auto format=target==SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5&&(swizzle&2)?MTLPixelFormatA1BGR5Unorm:texture_components(target).native;
                assert(result.pixelFormat==format);
                if(own)assert(submissions==submits+1);
                else { assert(submissions==submits&&commands.encoder->work.count==3);[commands runGPU]; }
                if(result->pixels!=expected) { std::cerr<<"packed mismatch color="<<std::hex<<uint32_t(color)<<" target="<<uint32_t(target)<<" swizzle="<<swizzle<<" variant="<<variant<<std::dec<<"\n";std::abort(); }
                ++cases;pixels+=source.width*source.height;
            }
        }
    }
    assert(libraries==1&&pipelines==1&&cases>5000);
    // Queued producer data must be read on the simulated GPU, not while encoding.
    auto source=[CastTexture new];source.width=3;source.height=2;source.pixelFormat=MTLPixelFormatB5G6R5Unorm;source->pixels.resize(12,0);
    const auto color=SCE_GXM_COLOR_FORMAT_U5U6U5_BGR;const auto target=SCE_GXM_TEXTURE_BASE_FORMAT_U16;
    auto c=[V3KMetal4Commands new];[c.encoder->work addObject:^{std::fill(source->pixels.begin(),source->pixels.end(),0x5a);}];
    auto result=(CastTexture *)caster.surface_format_cast((id<MTLTexture>)source,color,target,0,c);
    assert(result&&c.encoder->work.count==4&&source->pixels[0]==0&&source->cpu_reads==0);[c runGPU];
    assert(result->pixels==reference_bytes(source,color,target,0));
    // The new GPU-only intermediate path also works with private input storage.
    source.storageMode=MTLStorageModePrivate;c=[V3KMetal4Commands new];
    auto private_result=(CastTexture *)caster.surface_format_cast((id<MTLTexture>)source,color,target,0,c);assert(private_result);[c runGPU];assert(private_result->pixels==result->pixels);source.storageMode=MTLStorageModeShared;
    for(unsigned failure=0;failure<9;++failure) {
        SurfaceCaster fresh{device};c=[V3KMetal4Commands new];const auto before=allocations;
        fail_library=failure==0;fail_pipeline=failure==1;fail_buffer=failure==2;fail_texture=failure==3;
        max_buffer=failure==4?1:UINT64_MAX;source.framebufferOnly=failure==5;source.sampleCount=failure==6?2:1;
        if(failure==7)c.status=MTLCommandBufferStatusCommitted;
        if(failure==8)source.width=16385;
        bool threw=false;id<MTLTexture> rejected=nil;
        try { rejected=fresh.surface_format_cast((id<MTLTexture>)source,color,target,0,c); } catch(const std::runtime_error &){threw=true;}
        assert(!rejected&&threw==(failure<2)&&c.encoder->work.count==0);
        if(failure>=4)assert(allocations==before);
        fail_library=fail_pipeline=fail_buffer=fail_texture=false;max_buffer=UINT64_MAX;source.framebufferOnly=NO;source.sampleCount=1;source.width=3;c.status=MTLCommandBufferStatusNotEnqueued;
        assert(fresh.surface_format_cast((id<MTLTexture>)source,color,target,0,c));[c runGPU];
    }
    fail_submit=true;bool threw=false;try {(void)caster.surface_format_cast((id<MTLTexture>)source,color,target,0,nil);}catch(const std::runtime_error &){threw=true;}fail_submit=false;assert(threw);
    std::cout<<"PASS packed cast integration: "<<cases<<" complete format/mode/layout encodings, "<<pixels<<" pixels byte-identical to actual CPU readback and decode; same-submission producer, private source, lazy reuse, allocation/compile/submit failures and retry\n";
}
static void scratch_and_binding_tests() {
    Device device; SurfaceCaster caster{device};
    const auto color=SCE_GXM_COLOR_FORMAT_U5U6U5_BGR;
    const auto target=SCE_GXM_TEXTURE_BASE_FORMAT_U16;
    __weak id<MTLBuffer> old_input=nil,old_output=nil;
    auto commands=[V3KMetal4Commands new];
    std::vector<CastTexture *> results;
    std::vector<std::vector<uint8_t>> expected;
    const auto before=submissions;
    @autoreleasepool {
        for(unsigned i=0;i<32;++i) {
            auto source=[CastTexture new];source.width=i<8?3:i<16?31:7;source.height=5;
            source.pixelFormat=MTLPixelFormatB5G6R5Unorm;
            source->pixels.resize(source.width*source.height*2,uint8_t(13*i+7));
            expected.push_back(reference_bytes(source,color,target,0));
            auto previous_input=caster.packed_cast_input,previous_output=caster.packed_cast_output;
            results.push_back((CastTexture *)caster.surface_format_cast((id<MTLTexture>)source,color,target,0,commands));
            if(i==0){old_input=caster.packed_cast_input;old_output=caster.packed_cast_output;}
            if(i&&i!=8)assert(caster.packed_cast_input==previous_input&&caster.packed_cast_output==previous_output);
            if(i==8)assert(caster.packed_cast_input!=previous_input&&caster.packed_cast_output!=previous_output);
        }
    }
    assert(old_input&&old_output&&submissions==before&&commands.encoder->work.count==96);
    [commands runGPU];
    for(size_t i=0;i<results.size();++i)assert(results[i]->pixels==expected[i]);
    assert(!old_input&&!old_output); // retired commands release replaced scratch
    id<MTLBuffer> slot=nil;
    constexpr uint64_t cap=32u*1024*1024;
    @autoreleasepool {
        auto cached=packed_cast_buffer(device,slot,cap);assert(cached==slot&&slot.length==cap);
        auto oversized=packed_cast_buffer(device,slot,cap+1);assert(oversized&&oversized!=slot&&slot==cached);
        assert(packed_cast_buffer(device,slot,1)==cached);
        fail_buffer=true;assert(!packed_cast_buffer(device,slot,cap+2)&&slot==cached);fail_buffer=false;
    }
    size_t bindings=0;
    for(auto base:{SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10,SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8,SCE_GXM_TEXTURE_BASE_FORMAT_U16})
        for(unsigned mode=0;mode<8;++mode)for(bool gamma:{false,true}) {
            auto format=SceGxmTextureFormat(uint32_t(base)|(mode<<12));
            auto metadata=binding_metadata(format,gamma);
            const bool relocated=!gamma&&base==SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10&&(mode&2);
            assert(metadata.has_value()==relocated);
            if(metadata)assert(uint32_t(*metadata)==(uint32_t(SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10)|((mode&3)<<20)));
            ++bindings;
        }
    std::cout<<"PASS packed cast lifetime/binding: 32 delayed casts survive scratch growth/reuse, replaced scratch released after retirement, 32-MiB cache limit and allocation failure preserve cache; "<<bindings<<" actual renderer binding blocks preserve relocated U2 alpha and cleared gamma metadata\n";
}
int main() { @autoreleasepool { numeric_tests();encoding_tests();scratch_and_binding_tests(); } }
