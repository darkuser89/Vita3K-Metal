// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <features/state.h>
#include <shader/metal4/spirv_recompiler.h>
#include <shader/metal4/msl_recompiler.h>
#include <shader/metal4/usse_types.h>
#include <SPIRV/GLSL.std.450.h>
#include <SPIRV/spirv.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

int main(int argc, char **argv) {
    assert(argc == 2);
    // Flag updates must preserve the enum object and remain constant-evaluable.
    static_assert([] {
        using namespace shader::metal3::usse;
        auto flags = Absolute;
        if (&(flags |= Negative) != &flags) return false;
        if (&(flags &= Absolute | GPI) != &flags) return false;
        return flags == Absolute && (Negative | GPI) == static_cast<RegisterFlags>(6);
    }());
    spdlog::set_level(spdlog::level::off);
    struct Input { SceGxmProgram program; SceGxmProgramVertexVaryings varyings; uint64_t instruction; } input{};
    input.program.magic = 0x00505847;
    input.program.size = sizeof(input);
    input.program.varyings_offset = offsetof(Input, varyings) - offsetof(SceGxmProgram, varyings_offset);
    input.program.primary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, primary_program_offset);
    input.program.primary_program_instr_count = 1;
    input.program.secondary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset);
    input.program.secondary_program_offset_end = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset_end);
    for (char bit : std::string("11111----000-----------101--------------------------------------"))
        input.instruction = (input.instruction << 1) | (bit == '1');
    FeatureState features{};
    features.enable_memory_mapping = true;
    std::vector<SceGxmVertexAttribute> attributes;
    shader::metal3::Hints hints{};
    hints.attributes = &attributes;
    size_t cases = 0;
    for (unsigned colors = 0; colors < 4; ++colors)
    for (unsigned capture = 0; capture < 3; ++capture)
    for (bool fallback : {false, true}) {
        input.varyings.vertex_outputs1 = 0x200 | ((colors & 1) ? 0x800 : 0) | ((colors & 2) ? 0x400 : 0);
        input.varyings.vertex_outputs2 = 7; // TEXCOORD0, four components
        hints.metal_capture_vertex_outputs = capture != 0;
        hints.metal_capture_vertex_outputs_compact = capture == 2;
        hints.metal_missing_vertex_outputs = fallback ? SCE_GXM_VERTEX_PROGRAM_OUTPUT_COLOR0 | SCE_GXM_VERTEX_PROGRAM_OUTPUT_COLOR1 : 0;
        const auto words = shader::metal3::convert_gxp(input.program, "vertex-color-test", features,
            shader::metal3::Target::SpirVMetal, hints, false).spirv;
        std::map<uint32_t, uint32_t> locations, constants;
        std::map<uint32_t, std::vector<uint32_t>> composites, clamps;
        std::multimap<uint32_t, uint32_t> stores;
        for (size_t at = 5; at < words.size();) {
            const auto *w = words.data() + at;
            const auto count = w[0] >> 16;
            assert(count && at + count <= words.size());
            switch (w[0] & 0xffff) {
            case spv::OpDecorate:
                if (count == 4 && w[2] == spv::DecorationLocation) locations[w[3]] = w[1];
                break;
            case spv::OpConstant: if (count == 4) constants[w[2]] = w[3]; break;
            case spv::OpConstantComposite: composites[w[2]] = {w + 3, w + count}; break;
            case spv::OpExtInst:
                if (w[4] == GLSLstd450FClamp) {
                    assert(count == 8);
                    clamps[w[2]] = {w[5], w[6], w[7]};
                }
                break;
            case spv::OpStore: stores.emplace(w[1], w[2]); break;
            }
            at += count;
        }
        std::set<uint32_t> output_clamps;
        for (unsigned color = 0; color < 2; ++color) {
            const bool written = colors & (1u << color);
            assert(locations.contains(color + 1) == (written || fallback));
            if (!locations.contains(color + 1)) continue;
            const auto range = stores.equal_range(locations.at(color + 1));
            assert(std::distance(range.first, range.second) == 1);
            const auto value = range.first->second;
            assert(clamps.contains(value) == written);
            if (!written) continue;
            output_clamps.insert(value);
            const auto &args = clamps.at(value);
            for (auto [bound, bits] : {std::pair{args[1], 0u}, std::pair{args[2], 0x3f800000u}}) {
                assert(composites.at(bound).size() == 4);
                for (auto component : composites.at(bound)) assert(constants.at(component) == bits);
            }
            size_t uses = 0;
            for (auto [target, source] : stores) uses += source == value;
            // Capture/replay receives the same already clamped RGBA value.
            assert(uses == (capture ? 2 : 1));
        }
        assert(output_clamps.size() == clamps.size());
        // Fog and texture coordinates retain their unconstrained domains.
        for (unsigned location : {3u, 4u}) {
            const auto range = stores.equal_range(locations.at(location));
            assert(range.first != range.second);
            for (auto it = range.first; it != range.second; ++it) assert(!clamps.contains(it->second));
        }
        const auto stem = std::filesystem::path(argv[1]) / ("color-" + std::to_string(cases));
        std::ofstream binary(stem.string() + ".spv", std::ios::binary);
        binary.write(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(uint32_t));
        std::ofstream(stem.string() + ".metal") << shader::metal::convert_spirv(words).source;
        ++cases;
    }
    assert(cases == 24);
    std::cout << "PASS vertex color clamp: 24 real GXP translations; COLOR0/COLOR1 RGBA [0,1] before interpolation; dense/compact capture shares the clamped value; missing outputs, fog and texture coordinates preserved; no GPU execution\n";
}
