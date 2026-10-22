// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <features/state.h>
#include <shader/metal4/spirv_recompiler.h>
#include <shader/metal4/msl_recompiler.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace shader::metal3;
int main(int argc, char **argv) {
    assert(argc == 2);
    spdlog::set_level(spdlog::level::off);
    const std::filesystem::path directory(argv[1]);
    size_t cases = 0;
    for (unsigned samples : {1u, 2u, 4u})
    for (bool mask : {false, true})
    for (bool update : {false, true})
    for (unsigned fetch : {0u, 1u, 2u, 3u})
    for (bool store : {false, true}) {
        struct Input {
            SceGxmProgram program;
            SceGxmProgramVertexVaryings varyings;
            uint64_t instruction;
        } input{};
        input.program.magic = 0x00505847;
        input.program.size = sizeof(input);
        input.program.program_flags = SCE_GXM_PROGRAM_FLAG_FRAGMENT
            | (fetch ? SCE_GXM_PROGRAM_FLAG_FRAGCOLOR_USED : 0);
        input.program.varyings_offset = offsetof(Input, varyings) - offsetof(SceGxmProgram, varyings_offset);
        input.program.primary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, primary_program_offset);
        input.program.primary_program_instr_count = 1;
        input.program.secondary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset);
        input.program.secondary_program_offset_end = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset_end);
        input.program.temp_reg_count1 = 128;
        input.program.default_uniform_buffer_count = store ? 4 : 0;
        input.varyings.output_param_type = SCE_GXM_PARAMETER_TYPE_F32;
        input.varyings.output_comp_count = 4;
        if (store) {
            // Actual VLDST STR: op1=2, temporary address/data registers.
            const std::string layout = "111oopppsnmycrbakkkkddeetgffihjlqquuvvvvvvvwwwwwwwxxxxxxxzzzzzzz";
            unsigned field = 0;
            for (int bit = 63; bit >= 0; --bit) {
                const bool set = layout[bit] == '1' || (layout[bit] == 'o' && ((2u >> field++) & 1));
                if (set) input.instruction |= uint64_t(1) << (63 - bit);
            }
        } else {
            for (char bit : std::string("11111----000-----------101--------------------------------------"))
                input.instruction = (input.instruction << 1) | (bit == '1');
        }
        FeatureState features{};
        features.use_mask_bit = mask;
        features.direct_fragcolor = fetch != 3;
        features.support_shader_interlock = fetch == 3;
        features.support_unknown_format = fetch != 3;
        features.enable_memory_mapping = store;
        std::vector<SceGxmVertexAttribute> attributes;
        Hints hints{};
        hints.attributes = &attributes;
        hints.color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
        hints.metal_samples = samples;
        hints.metal_raw_color_fetch = fetch == 2;
        hints.metal_raw_color_attachment = fetch == 2;
        auto module = convert_gxp(input.program, "mask-rejection", features, Target::SpirVMetal, hints, update).spirv;
        auto native = shader::metal::convert_spirv(module);
        assert(native.writes_guest_memory == (store && !update));
        assert(native.uses_framebuffer_fetch == (fetch == 1 || fetch == 2));
        const auto name = "s" + std::to_string(samples) + "-m" + std::to_string(mask)
            + "-u" + std::to_string(update) + "-f" + std::to_string(fetch) + "-w" + std::to_string(store);
        std::ofstream spv(directory / (name + ".spv"), std::ios::binary);
        spv.write(reinterpret_cast<const char *>(module.data()), module.size() * sizeof(uint32_t));
        std::ofstream(directory / (name + ".metal")) << native.source;
        ++cases;
    }
    std::cout << "PASS mask public translations: " << cases << " actual GXP-to-SPIR-V-to-MSL variants; single/MSAA, mask on/off, mask updates, framebuffer/raw fetch and real guest STR; guest-write classification preserved\n";
}
