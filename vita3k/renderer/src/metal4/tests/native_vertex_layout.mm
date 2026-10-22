// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/native_vertex_layout.h>
#include <cassert>
#include <iostream>
#include <random>
#include <sstream>
using namespace renderer::metal;
struct CountingInfos : shader::metal3::usse::AttributeInformationMap {
    mutable size_t lookups = 0;
    const_iterator find(int key) const { ++lookups; return shader::metal3::usse::AttributeInformationMap::find(key); }
};
struct TestBinding {
    std::vector<SceGxmVertexAttribute> attributes;
    std::vector<SceGxmVertexStream> streams;
    CountingInfos attribute_infos;
};
static void word(std::string &s, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) s.push_back(char(value >> (8 * i)));
}
static std::string serialize(const NativeVertexLayout &layout) {
    std::string s;
    word(s, layout.key.size()); s += layout.key;
    word(s, layout.streams_used);
    for (const auto &a : layout.vertex_attributes) {
        word(s,a.format); word(s,a.offset); word(s,a.buffer_index); word(s,a.source_offset); word(s,a.byte_size);
    }
    for (size_t i = 0; i < layout.vertex_stream_layouts.size(); ++i) {
        const auto &v = layout.vertex_stream_layouts[i];
        word(s,v.stride); word(s,v.step_function); word(s,v.step_rate);
        word(s,layout.stream_extents[i]); word(s,layout.repack_stream[i]);
    }
    return s;
}
// REFERENCE_HOOK
static TestBinding simple(uint16_t stride, uint16_t offset, uint8_t components = 3, uint16_t index = 0) {
    TestBinding p;
    p.streams.push_back({stride,index});
    p.attributes.push_back({0,offset,SCE_GXM_ATTRIBUTE_FORMAT_F32,components,0});
    p.attribute_infos.emplace(0, shader::metal3::usse::AttributeInformation(0,SCE_GXM_PARAMETER_TYPE_F32,components,false,false,false));
    return p;
}
int main() {
    auto a = build_native_vertex_layout(simple(12,0));
    assert(a.streams_used == 1 && !a.repack_stream[0] && a.vertex_stream_layouts[0].stride == 12);
    assert(a.vertex_attributes[0].format == MTLVertexFormatFloat3 && a.vertex_attributes[0].byte_size == 12);
    auto unaligned = build_native_vertex_layout(simple(16,1));
    assert(unaligned.repack_stream[0] && unaligned.vertex_attributes[0].source_offset == 1);
    assert(unaligned.vertex_attributes[0].offset == 0 && unaligned.vertex_stream_layouts[0].stride == 12);
    auto crossing = build_native_vertex_layout(simple(5,0,2));
    assert(crossing.repack_stream[0] && crossing.vertex_stream_layouts[0].stride == 8);
    auto broadcast = build_native_vertex_layout(simple(0,0));
    assert(broadcast.vertex_stream_layouts[0].step_function == MTLVertexStepFunctionConstant);
    assert(broadcast.vertex_stream_layouts[0].step_rate == 0 && broadcast.vertex_stream_layouts[0].stride == 12);
    auto instanced = build_native_vertex_layout(simple(8,0,2,SCE_GXM_INDEX_SOURCE_EACH_INSTANCE_32BIT));
    assert(instanced.vertex_stream_layouts[0].step_function == MTLVertexStepFunctionPerInstance);
    assert(instanced.vertex_stream_layouts[0].step_rate == 1);
    auto array = simple(18,0);
    array.attribute_infos.at(0) = shader::metal3::usse::AttributeInformation(0,SCE_GXM_PARAMETER_TYPE_S16,3,true,true,true,3);
    auto raw = build_native_vertex_layout(array);
    assert(raw.stream_extents[0] == 18 && raw.vertex_stream_layouts[0].stride == 20);
    assert(raw.vertex_attributes[0].format == MTLVertexFormatUShort4);
    assert(raw.vertex_attributes[2].format == MTLVertexFormatUShort && raw.vertex_attributes[2].source_offset == 16);
    auto signed_input = simple(4,0,1);
    signed_input.attributes[0].format = SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED;
    signed_input.attribute_infos.at(0).is_signed = true;
    assert(build_native_vertex_layout(signed_input).vertex_attributes[0].format == MTLVertexFormatInt);
    FrozenVertexLayoutCache<TestBinding> cache;
    auto frozen = std::make_shared<TestBinding>(simple(12,0));
    auto expected = serialize(cache.get(frozen));
    const auto lookups = frozen->attribute_infos.lookups;
    for (size_t i = 0; i < 10000; ++i) assert(serialize(cache.get(frozen)) == expected);
    assert(frozen->attribute_infos.lookups == lookups);
    auto invalid = std::make_shared<TestBinding>(simple(12,0)); invalid->attributes[0].streamIndex = 20;
    bool rejected = false;
    try { cache.get(invalid); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected && serialize(cache.get(frozen)) == expected);
    assert(frozen->attribute_infos.lookups == lookups);
    rejected = false;
    try { cache.get({}); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    std::weak_ptr<const TestBinding> old = frozen;
    auto replacement = std::make_shared<TestBinding>(simple(16,1)); frozen.reset();
    assert(!old.expired());
    assert(serialize(cache.get(replacement)) != expected && old.expired());
    assert(replacement->attribute_infos.lookups == 1);
    // Fixed seed corpus covers each ordinary/raw type, arrays, all locations,
    // absent inputs, overlaps, every stream index, stride and malformed shapes.
    std::mt19937 random(0x4d34564c);
    uint64_t digest = 1469598103934665603ull;
    size_t accepted = 0, refused = 0;
    for (size_t test = 0; test < 50000; ++test) {
        TestBinding binding;
        const auto streams = random() % (SCE_GXM_MAX_VERTEX_STREAMS + 2);
        for (size_t i = 0; i < streams; ++i) binding.streams.push_back({uint16_t(random()%68),uint16_t(random()%4)});
        const auto attributes = random() % 12;
        for (size_t i = 0; i < attributes; ++i) {
            SceGxmVertexAttribute input{};
            input.streamIndex=random()%(SCE_GXM_MAX_VERTEX_STREAMS+1); input.offset=random()%80;
            input.format=static_cast<SceGxmAttributeFormat>(random()%12); input.componentCount=random()%6; input.regIndex=i;
            binding.attributes.push_back(input);
            if (random()%5) binding.attribute_infos.emplace(i, shader::metal3::usse::AttributeInformation(
                random()%33,static_cast<SceGxmParameterType>(random()%12),random()%6,
                random()%2,random()%2,random()%2,random()%5));
        }
        const auto outcome = [&](auto build) {
            try { return std::string("ok")+serialize(build(binding)); }
            catch (const std::runtime_error &e) { return std::string("error:")+e.what(); }
        };
        const auto result = outcome([](const auto &p) { return build_native_vertex_layout(p); });
#ifdef REFERENCE_LAYOUT
        assert(result == outcome([](const auto &p) { return reference_layout(p); }));
#endif
        if (result.starts_with("ok")) ++accepted; else ++refused;
        for (uint8_t byte : result) { digest ^= byte; digest *= 1099511628211ull; }
    }
    std::cout << "layout corpus " << accepted << " accepted " << refused << " rejected digest " << std::hex << digest << std::dec << '\n';
#ifndef REFERENCE_LAYOUT
    // Golden digest of the pre-cache production algorithm, fieldwise/keywise
    // serialization over this same deterministic corpus (no struct padding).
    assert(accepted == 7871 && refused == 42129 && digest == 0x0860146f7bb8b0e4ull);
#endif
    std::cout << "PASS vertex layout: 50000 baseline layout/key/error cases, raw arrays, repacking, instancing and 10000 identity-cache hits without rebuilding\n";
}
