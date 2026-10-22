// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <SPIRV/GLSL.std.450.h>

static void test_centroid_queries(const std::filesystem::path &output) {
    struct Input {
        SceGxmProgram program;
        SceGxmProgramVertexVaryings varyings;
        std::array<SceGxmProgramAttributeDescriptor, 6> descriptors;
        uint64_t instruction;
    };
    size_t cases = 0;
    // Ten user locations, query-only/PA sharing, center/centroid/mixed queries,
    // projection through z/w, and descriptor order. PointCoord keeps its
    // separate builtin path; this test does not claim centroid point emulation.
    for (unsigned coord = 0; coord <= 10; ++coord)
    for (unsigned mode = 0; mode < 4; ++mode)
    for (bool metadata : {false, true}) {
        Input input{};
        auto &program = input.program;
        program.magic = 0x00505847;
        program.size = sizeof(input);
        program.program_flags = SCE_GXM_PROGRAM_FLAG_FRAGMENT;
        program.varyings_offset = offsetof(Input, varyings) - offsetof(SceGxmProgram, varyings_offset);
        program.primary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, primary_program_offset);
        program.primary_program_instr_count = 1;
        program.secondary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset);
        program.secondary_program_offset_end = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset_end);
        input.varyings.vertex_outputs1 = offsetof(Input, descriptors)
            - offsetof(Input, varyings) - offsetof(SceGxmProgramVertexVaryings, vertex_outputs1);
        input.varyings.output_param_type = SCE_GXM_PARAMETER_TYPE_F32;
        input.varyings.output_comp_count = 4;
        for (char bit : std::string("11111----000-----------101--------------------------------------"))
            input.instruction = (input.instruction << 1) | (bit == '1');
        auto add_pa = [&] {
            auto &d = input.descriptors[input.varyings.varyings_count++];
            d.attribute_info = (coord == 10 ? 0x40000000u : coord * 0x1000u) | (3u << 22) | 0xFu;
            d.size = 3u << 4;
        };
        auto add_query = [&](bool centroid, unsigned projection) {
            auto &d = input.descriptors[input.varyings.varyings_count++];
            d.attribute_info = 0xF000u | (coord == 10 ? 0x400u : coord)
                | (centroid ? 0x10u : 0u) | (projection << 8);
            d.size = 3u << 6;
            d.component_info = 3u << 4; // F32, four query result registers.
        };
        if (mode == 2) { add_pa(); add_pa(); }
        add_query(mode != 0, coord == 10 ? 1 : mode == 2 ? 2 : mode == 3 ? 3 : 1);
        if (mode >= 2) { add_query(false, 1); add_query(true, 1); }
        if (mode == 3) add_pa(); // A query may precede its shared iterator.
        program.primary_reg_count = input.varyings.varyings_count * 4;
        FeatureState features{};
        std::vector<SceGxmVertexAttribute> attributes;
        Hints hints{};
        hints.attributes = &attributes;
        hints.color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
        hints.fragment_textures[0] = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
        hints.metal_samples = mode == 2 ? 2 : mode == 3 ? 4 : 1;
        hints.metal_mip_sampling = metadata;
        auto spirv = convert_gxp(program, "centroid-test", features, Target::SpirVMetal, hints, false).spirv;
        std::map<uint32_t, uint32_t> locations, builtins;
        std::set<uint32_t> inputs, loaded_inputs, centroid_results;
        std::vector<uint32_t> centroid_inputs;
        bool capability = false, projected_centroid = false;
        for (size_t at = 5; at < spirv.size();) {
            const auto *w = spirv.data() + at;
            const uint32_t count = w[0] >> 16;
            assert(count && at + count <= spirv.size());
            switch (w[0] & 0xFFFF) {
            case spv::OpCapability: capability |= w[1] == spv::CapabilityInterpolationFunction; break;
            case spv::OpDecorate:
                assert(w[2] != spv::DecorationCentroid);
                if (w[2] == spv::DecorationLocation) locations[w[1]] = w[3];
                if (w[2] == spv::DecorationBuiltIn) builtins[w[1]] = w[3];
                break;
            case spv::OpVariable: if (w[3] == spv::StorageClassInput) inputs.insert(w[2]); break;
            case spv::OpLoad: loaded_inputs.insert(w[3]); break;
            case spv::OpExtInst:
                if (w[4] == GLSLstd450InterpolateAtCentroid) {
                    centroid_inputs.push_back(w[5]); centroid_results.insert(w[2]);
                }
                break;
            case spv::OpVectorShuffle:
                if (centroid_results.contains(w[3])) {
                    assert(w[3] == w[4] && (count == 7 || count == 8) && w[5] == 0 && w[6] == 1);
                    if (count == 8) {
                        assert(w[7] == (mode == 2 ? 3u : 2u));
                        projected_centroid = true;
                    }
                }
                break;
            }
            at += count;
        }
        const size_t expected = coord == 10 || mode == 0 ? 0 : mode == 1 ? 1 : 2;
        assert(centroid_inputs.size() == expected && capability == (expected != 0));
        unsigned matching_inputs = 0;
        uint32_t shared_input = 0;
        for (auto id : inputs) {
            if (coord < 10 ? locations.contains(id) && locations.at(id) == 4 + coord
                           : builtins.contains(id) && builtins.at(id) == spv::BuiltInPointCoord) {
                ++matching_inputs; shared_input = id;
            }
        }
        assert(matching_inputs == 1);
        for (auto id : centroid_inputs) assert(id == shared_input);
        if (mode >= 2 || mode == 0 || coord == 10) assert(loaded_inputs.contains(shared_input));
        assert(projected_centroid == (coord < 10 && mode >= 2));
        auto msl = shader::metal::convert_spirv(spirv).source;
        assert((msl.find(".interpolate_at_centroid()") != std::string::npos) == (expected != 0));
        if (expected && mode >= 2) assert(msl.find(".interpolate_at_center()") != std::string::npos);
        if (coord == 10) assert(msl.find("[[point_coord]]") != std::string::npos);
        if (!output.empty()) {
            const auto stem = output / ("centroid-" + std::to_string(coord) + "-" + std::to_string(mode)
                + (metadata ? "-metadata" : "-plain"));
            std::ofstream(stem.string() + ".metal") << msl;
            std::ofstream binary(stem.string() + ".spv", std::ios::binary);
            binary.write(reinterpret_cast<const char *>(spirv.data()), spirv.size() * sizeof(uint32_t));
        }
        ++cases;
    }
    std::cout << "PASS centroid queries: " << cases
        << " GXP-to-SPIR-V/MSL fixtures, 10 user varyings, center/centroid separation, shared/duplicate/late PA, projection and preserved PointCoord path; no GPU execution\n";
}
