// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <gxm/types.h>
#include <renderer/metal4/variant_keys.h>
#include <cassert>
#include <bitset>
#include <iostream>
#include <random>
#include <vector>
// HINTS_HOOK
struct KeyMeta { std::string hash; };
struct KeyBinding {
    std::shared_ptr<KeyMeta> vertex_program, fragment_program;
    std::vector<SceGxmVertexAttribute> attributes;
};
// REFERENCE_HOOK
static void blend_random(SceGxmBlendInfo &b, std::mt19937 &random) {
    b.colorFunc=static_cast<SceGxmBlendFunc>(random()%6); b.alphaFunc=static_cast<SceGxmBlendFunc>(random()%6);
    b.colorSrc=static_cast<SceGxmBlendFactor>(random()%16); b.colorDst=static_cast<SceGxmBlendFactor>(random()%16);
    b.alphaSrc=static_cast<SceGxmBlendFactor>(random()%16); b.alphaDst=static_cast<SceGxmBlendFactor>(random()%16);
    b.colorMask=static_cast<SceGxmColorMask>(random()%16);
}
static void unused_texture_test() {
    using namespace renderer::metal;
    constexpr auto format = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
    for (uint32_t mask = 0; mask < 65536; ++mask) {
        const std::bitset<16> vertex(mask), fragment(mask ^ 65535);
        shader::metal3::Hints original{};
        for (size_t slot = 0; slot < 16; ++slot) {
            original.vertex_textures[slot] = SCE_GXM_TEXTURE_FORMAT_F32_R;
            original.fragment_textures[slot] = SCE_GXM_TEXTURE_FORMAT_S8S8S8S8_ABGR;
            original.metal_vertex_lod_bias[slot] = std::bit_cast<float>(0x7fc00001u + uint32_t(slot));
            original.metal_fragment_lod_bias[slot] = std::bit_cast<float>(0x80000000u | uint32_t(slot));
        }
        auto canonical = original, stale = original;
        for (size_t slot = 0; slot < 16; ++slot) {
            if (!vertex[slot]) { stale.vertex_textures[slot] = format; stale.metal_vertex_lod_bias[slot] = 12.f; }
            if (!fragment[slot]) { stale.fragment_textures[slot] = format; stale.metal_fragment_lod_bias[slot] = -32.f; }
        }
        canonicalize_shader_textures(canonical, vertex, fragment, format);
        canonicalize_shader_textures(stale, vertex, fragment, format);
        for (size_t slot = 0; slot < 16; ++slot) {
            assert(canonical.vertex_textures[slot] == (vertex[slot] ? original.vertex_textures[slot] : format));
            assert(canonical.fragment_textures[slot] == (fragment[slot] ? original.fragment_textures[slot] : format));
            assert(std::bit_cast<uint32_t>(canonical.metal_vertex_lod_bias[slot])
                == (vertex[slot] ? std::bit_cast<uint32_t>(original.metal_vertex_lod_bias[slot]) : 0));
            assert(std::bit_cast<uint32_t>(canonical.metal_fragment_lod_bias[slot])
                == (fragment[slot] ? std::bit_cast<uint32_t>(original.metal_fragment_lod_bias[slot]) : 0));
        }
        for (bool stage : {false, true})
            assert(shader_key_words(canonical, stage, false, 17, 4, false, true)
                == shader_key_words(stale, stage, false, 17, 4, false, true));
    }
    std::cout << "PASS unused texture hints: all 65536 resource masks, vertex/fragment isolation, active format and every LOD bit preserved, stale inactive state has identical keys; ASAN/UBSAN\n";
}
int main() {
    unused_texture_test();
    using namespace renderer::metal;
    for(unsigned flags=0;flags<64;++flags) {
        const bool allowed=can_omit_mask_test(flags&1,flags&2,flags&4,flags&8,flags&16,flags&32);
        assert(allowed==(flags==7));
    }
    auto v=std::make_shared<KeyBinding>(), f=std::make_shared<KeyBinding>();
    v->vertex_program=std::make_shared<KeyMeta>(std::string(64,'a'));
    f->fragment_program=std::make_shared<KeyMeta>(std::string(64,'b'));
    shader::metal3::Hints hints{};
    std::array<ShaderVariantKeyCache<KeyBinding>,2> stages;
    size_t hashes=0;
    auto hash=[&] { ++hashes; return v->vertex_program->hash; };
    auto words=shader_key_words(hints,true,false,0,1,false,false);
    const auto original=stages[0].get(v,v,words,hash);
    const auto revision=stages[0].revision();
    const auto *storage=stages[0].get(v,v,words,hash).data();
    for(size_t i=0;i<10000;++i) {
        assert(stages[0].get(v,v,shader_key_words(hints,true,false,0,1,false,false),hash)==original);
        assert(stages[0].get(v,v,words,hash).data()==storage);
    }
    assert(hashes==1 && stages[0].revision()==revision);
    // A failed miss preserves the previous key and its revision.
    auto changed=words;changed.prefix[1]^=1;
    bool failed=false;
    try {stages[0].get(v,v,changed,[]()->std::string {throw std::runtime_error("test");});}
    catch(const std::runtime_error &) {failed=true;}
    assert(failed && stages[0].revision()==revision && stages[0].get(v,v,words,hash)==original);
    // Compare only the selected stage; retain every LOD bit, including signed
    // zero and NaN payloads. Inactive blend state cannot create new variants.
    auto same=hints;same.color_format=SCE_GXM_COLOR_FORMAT_F16F16F16F16_ABGR;
    same.metal_fragment_lod_bias[0]=4; same.metal_quantized_color_blend=true;
    assert(shader_key_words(same,true,false,0,1,false,false)==words);
    for(size_t slot=0;slot<16;++slot) {
        auto h=hints;h.vertex_textures[slot]=SCE_GXM_TEXTURE_FORMAT_F32_R;
        assert(!(shader_key_words(h,true,false,0,1,false,false)==words));
        for(uint32_t bits:{0x80000000u,0x7fc00001u,0x7fc00002u,0x3f800000u}) {
            h=hints;h.metal_vertex_lod_bias[slot]=std::bit_cast<float>(bits);
            assert(!(shader_key_words(h,true,false,0,1,false,false)==words));
        }
    }
    auto fragment=shader_key_words(hints,false,false,0,1,false,false);
    auto unmasked=shader_key_words(hints,false,false,0,1,false,false,true);
    assert(unmasked.mask_test_omitted && !(unmasked==fragment));
    assert(shader_key_words(hints,true,false,0,1,false,false,true)==words);
    assert(!shader_key_words(hints,false,true,0,1,false,false,true).mask_test_omitted);
    assert(!shader_key_words(hints,false,false,0,1,true,false,true).mask_test_omitted);
    const auto ordinary_key=stages[1].get(f,v,fragment,[&]{return f->fragment_program->hash;});
    const auto unmasked_key=stages[1].get(f,v,unmasked,[&]{return f->fragment_program->hash;});
    assert(unmasked_key==ordinary_key+"-mask-one");
    assert(stages[1].get(f,v,fragment,[&]{return f->fragment_program->hash;})==ordinary_key);
    std::cout << "PASS known-one mask keys: all 64 eligibility states, separate fragment/pipeline identity, vertex/depth-only/mask-update unchanged, restoration after mask mutation\n";
    same=hints;same.metal_red_alpha_blend.colorSrc=SCE_GXM_BLEND_FACTOR_ONE;
    assert(shader_key_words(same,false,false,0,1,false,false)==fragment);
    auto depth=shader_key_words(hints,false,true,0,4,false,false);
    assert(stages[1].get({}, {},depth,[]()->std::string {assert(false);return {};}).starts_with("metal-depth-only"));
    stages[1].get(f,v,fragment,[&] {return f->fragment_program->hash;});
    std::weak_ptr<const KeyBinding> prior_fragment=f;
    auto replacement=std::make_shared<KeyBinding>(*f);
    replacement->fragment_program=std::make_shared<KeyMeta>(std::string(64,'c'));
    f.reset();assert(!prior_fragment.expired());
    assert(stages[1].get(replacement,v,fragment,[&] {return replacement->fragment_program->hash;}).starts_with(std::string(64,'c')));
    assert(prior_fragment.expired());f=replacement;
    PipelineVariantKeyCache<KeyBinding> pipelines;
    std::array<uint32_t,14> target{};
    auto pipeline=pipelines.get(v,{1,1},target,"vertex","fragment","layout");
    std::string expected="vertexfragmentlayout";for(auto x:target)append_variant_word(expected,x);
    assert(pipeline==expected);
    const auto *pipeline_storage=pipelines.get(v,{1,1},target,"vertex","fragment","layout").data();
    for(size_t i=0;i<10000;++i)assert(pipelines.get(v,{1,1},target,"vertex","fragment","layout").data()==pipeline_storage);
    for(size_t i=0;i<target.size();++i) {
        auto t=target;t[i]=uint32_t(i+1);auto e=std::string("vertexfragmentlayout");for(auto x:t)append_variant_word(e,x);
        assert(pipelines.get(v,{1,1},t,"vertex","fragment","layout")==e);
    }
    assert(pipelines.get(v,{2,1},target,"new","fragment","layout").starts_with("newfragmentlayout"));
    assert(pipelines.get(v,{2,2},target,"new","other","layout").starts_with("newotherlayout"));
    auto other=std::make_shared<KeyBinding>(*v);
    assert(pipelines.get(other,{2,2},target,"new","other","another").starts_with("newotheranother"));
    std::mt19937 random(0x4d344b59);
    uint64_t digest=14695981039346656037ull;
    size_t compared=0;
    for(size_t iteration=0;iteration<16000;++iteration) {
        if(iteration%17==0) {
            v=std::make_shared<KeyBinding>(*v);v->attributes.clear();
            const size_t count=random()%32;
            for(size_t i=0;i<count;++i)v->attributes.push_back({uint16_t(random()%16),uint16_t(random()),
                static_cast<SceGxmAttributeFormat>(random()%11),uint8_t(random()%5),uint16_t(random())});
        }
        hints.metal_missing_vertex_outputs=random(); hints.metal_far_clip=random()%2;
        hints.metal_mip_sampling=random()%2;hints.metal_float_cube_filter=random()%2;
        hints.color_format=static_cast<SceGxmColorFormat>(random());
        hints.metal_raw_color_attachment=random()%2;hints.metal_raw_color_fetch=random()%2;
        hints.metal_red_alpha_shader_blend=random()%2; hints.metal_quantized_color_blend=random()%2;
        blend_random(hints.metal_red_alpha_blend,random);blend_random(hints.metal_quantized_blend,random);
        hints.metal_output_register_size=random();hints.metal_output_register_format=static_cast<SceGxmOutputRegisterFormat>(random()%4);
        for(size_t slot=0;slot<16;++slot) {
            hints.vertex_textures[slot]=static_cast<SceGxmTextureFormat>(random());
            hints.fragment_textures[slot]=static_cast<SceGxmTextureFormat>(random());
            hints.metal_vertex_lod_bias[slot]=std::bit_cast<float>(uint32_t(random()));
            hints.metal_fragment_lod_bias[slot]=std::bit_cast<float>(uint32_t(random()));
        }
        const uint32_t features=random(),samples=1u<<(random()%3);
        const bool mask=random()%2,gamma=random()%2;
        for(size_t mode=0;mode<3;++mode) {
            const bool vertex=mode==0,depth_only=mode==2;
            const auto bits=shader_key_words(hints,vertex,depth_only,features,samples,mask,gamma);
            const auto &key=stages[vertex?0:1].get(vertex?v:f,v,bits,[&] {return vertex?v->vertex_program->hash:f->fragment_program->hash;});
#ifdef REFERENCE_KEYS
            assert(key==reference_key(*v,*f,hints,vertex,depth_only,features,samples,mask,gamma));
#endif
            for(uint8_t byte:key) {digest^=byte;digest*=1099511628211ull;}
            ++compared;
        }
    }
    std::cout << "variant corpus " << compared << " digest " << std::hex << digest << std::dec << '\n';
#ifndef REFERENCE_KEYS
    // Golden FNV-1a digest from the previous production key builder over this
    // deterministic corpus; the initial comparison checked each key directly.
    assert(digest==0xf5c4ebccb6252fb7ull);
#endif
    std::cout << "PASS variant keys: 48000 byte-identical baseline variants, 10000 shader/pipeline cache hits, every texture/LOD bit and pipeline target field, binding/revision changes and failed-miss recovery\n";
}
