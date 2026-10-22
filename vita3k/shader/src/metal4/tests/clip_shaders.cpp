// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <features/state.h>
#include <shader/metal4/spirv_recompiler.h>
#include <shader/metal4/msl_recompiler.h>
#include <shader/metal4/metal_capture.h>
#include <shader/metal4/msl_capture.h>
#include <SPIRV/spirv.hpp>
#include <bit>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
using namespace shader::metal3;

int main(int argc, char **argv) {
    assert(argc == 2);
    spdlog::set_level(spdlog::level::off);
    static_assert(metal::CAPTURE_OUTPUT_SLOT_COUNT == shader::metal::CAPTURE_OUTPUT_SLOT_COUNT);
    static_assert(metal::CAPTURE_CLIP_COUNT == 11 && metal::NATIVE_CLIP_COUNT == 8);
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
    Hints hints{}; hints.attributes = &attributes;
    hints.color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
    size_t cases = 0, msl_cases = 0;
    for (unsigned mask = 0; mask < 256; ++mask)
    for (unsigned mode = 0; mode < 3; ++mode)
    for (bool far : {false, true}) {
        input.varyings.vertex_outputs1 = mask;
        hints.metal_capture_vertex_outputs = mode != 0;
        hints.metal_capture_vertex_outputs_compact = mode == 2;
        hints.metal_far_clip = far;
        auto spirv = convert_gxp(input.program, "clip-test", features, Target::SpirVMetal, hints, false).spirv;
        std::map<uint32_t, uint32_t> constants, pointers, arrays, variables, bindings;
        std::map<uint32_t, std::vector<uint32_t>> chains;
        std::vector<std::pair<uint32_t,uint32_t>> stores;
        uint32_t clip = 0;
        for (size_t at = 5; at < spirv.size();) {
            const auto *w = spirv.data() + at; const auto count = w[0] >> 16;
            assert(count && at + count <= spirv.size());
            switch (w[0] & 0xffff) {
            case spv::OpConstant: if (count == 4) constants[w[2]] = w[3]; break;
            case spv::OpIMul: case spv::OpIAdd:
                if (constants.contains(w[3]) && constants.contains(w[4]))
                    constants[w[2]] = (w[0] & 0xffff) == spv::OpIMul ? constants[w[3]] * constants[w[4]] : constants[w[3]] + constants[w[4]];
                break;
            case spv::OpTypePointer: pointers[w[1]] = w[3]; break;
            case spv::OpTypeArray: arrays[w[1]] = w[3]; break;
            case spv::OpVariable: variables[w[2]] = w[1]; break;
            case spv::OpAccessChain: chains[w[2]] = {w + 3, w + count}; break;
            case spv::OpStore: stores.emplace_back(w[1], w[2]); break;
            case spv::OpDecorate:
                if (w[2] == spv::DecorationBuiltIn && w[3] == spv::BuiltInClipDistance) clip = w[1];
                if (w[2] == spv::DecorationBinding) bindings[w[1]] = w[3];
                break;
            }
            at += count;
        }
        const unsigned planes = 3 + std::popcount(mask), native = std::min(planes, 8u);
        assert(clip && constants.at(arrays.at(pointers.at(variables.at(clip)))) == native);
        uint32_t capture = 0;
        for (auto [id,binding] : bindings) if (binding == 5) capture = id;
        assert(bool(capture) == (mode != 0));
        std::array<unsigned, 8> native_stores{};
        std::array<unsigned, 26> capture_stores{};
        for (auto [target,value] : stores) if (chains.contains(target)) {
            const auto &chain = chains.at(target);
            if (chain[0] == clip) {
                const auto index = constants.at(chain.back()); assert(index < native);
                ++native_stores[index];
            }
            if (capture && chain[0] == capture && mode == 2) {
                const auto index = constants.at(chain.back()); assert(index < 26);
                ++capture_stores[index];
            }
        }
        for (unsigned plane = 0; plane < 8; ++plane) assert(native_stores[plane] == (plane < native));
        if (mode == 2) {
            assert(capture_stores[0] == 2);
            for (unsigned slot = 1; slot < 14; ++slot) assert(capture_stores[slot] == 1);
            assert(capture_stores[14] == 2);
            for (unsigned plane = 0; plane < 11; ++plane) assert(capture_stores[15 + plane] == 1u + (plane < planes));
        }
        const auto stem = std::filesystem::path(argv[1]) / ("clip-" + std::to_string(mask) + "-" + std::to_string(mode) + "-" + std::to_string(far));
        std::ofstream binary(stem.string() + ".spv", std::ios::binary);
        binary.write(reinterpret_cast<const char *>(spirv.data()), spirv.size() * sizeof(uint32_t));
        // Offline compile each plane count (0..8), both far modes and all
        // three capture variants. Every sparse mask above is SPIR-V validated.
        if (mask == (1u << std::popcount(mask)) - 1) {
            std::ofstream(stem.string() + ".metal") << shader::metal::convert_spirv(spirv).source;
            ++msl_cases;
        }
        ++cases;
    }
    std::ofstream(std::filesystem::path(argv[1]) / "clip-replay.metal") << shader::metal::point_replay_vertex_program().source;
    assert(cases == 1536 && msl_cases == 54);
    std::cout << "PASS clip shader interfaces: 1536 GXP translations, all 256 guest-plane masks, far on/off, ordinary/dense/compact capture, exact 8 native and 11 captured plane bounds/stores; no GPU execution\n";
}
