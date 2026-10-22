// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <features/state.h>
#include <shader/metal4/usse_utilities.h>
#include <shader/metal4/usse_decoder_helpers.h>
#include <shader/metal4/usse_disasm.h>
#include <shader/metal4/spirv_recompiler.h>
#include <shader/metal4/msl_recompiler.h>
#include <bit>
#include <cassert>
#include <iostream>
#include <sstream>
#include <future>

using namespace shader::metal3;
using namespace shader::metal3::usse;
#include "centroid_regressions.inl"

int main(int argc, char **argv) {
    spdlog::set_level(spdlog::level::off);
    int evaluations = 0;
    LOG_DISASM("{}", ++evaluations);
    assert(evaluations == 0);
    std::stringstream diagnostic;
    disasm::disasm_storage = &diagnostic;
    LOG_DISASM("{}", ++evaluations);
    assert(evaluations == 1 && diagnostic.str() == "1\n");
    disasm::disasm_storage = nullptr;

    FeatureState features{};
    spv::SpvBuildLogger logger;
    spv::Builder builder(spv::Spv_1_0, 0, &logger);
    builder.setMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    builder.addCapability(spv::CapabilityShader);
    auto *main = builder.makeEntryPoint("main");
    builder.addEntryPoint(spv::ExecutionModelVertex, main, "main");
    SpirvShaderParameters parameters{};
    parameters.native_metal = true;
    utils::SpirvUtilFunctions utilities{};
    utilities.native_metal = true;
    utilities.std_builtins = builder.import("GLSL.std.450");
    size_t cases = 0;
    for (auto type : {DataType::F16, DataType::F32, DataType::UINT16,
            DataType::INT16, DataType::UINT32, DataType::INT32}) {
        for (uint32_t mask = 1; mask < 16; ++mask) {
            Operand op{};
            op.bank = RegisterBank::FPCONSTANT; op.num = 1; op.type = type;
            op.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
            const auto value = utils::load(builder, parameters, utilities, features, op, mask, 0);
            assert(value != spv::NoResult);
            auto result_type = builder.getTypeId(value);
            assert(builder.getNumTypeComponents(result_type) == std::popcount(mask));
            if (std::popcount(mask) > 1) result_type = builder.getContainedTypeId(result_type);
            if (type == DataType::UINT16 || type == DataType::UINT32) assert(builder.isUintType(result_type));
            else if (type == DataType::INT16 || type == DataType::INT32) assert(builder.isIntType(result_type));
            else assert(builder.isFloatType(result_type));
            ++cases;
        }
    }
    for (unsigned type = 0; type < 3; ++type)
    for (unsigned extended = 0; extended < 2; ++extended)
    for (unsigned swizzle = 0; swizzle < 16; ++swizzle) {
        bool rejected = false;
        try { (void)decode_vec34_swizzle(swizzle, extended, type); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected == (type == 1 && extended && swizzle == 15));
    }
    Operand empty{};
    empty.bank = RegisterBank::FPCONSTANT; empty.num = 1; empty.type = DataType::F32;
    empty.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
    bool empty_rejected = false;
    try { (void)utils::load(builder, parameters, utilities, features, empty, 0, 0); }
    catch (const std::runtime_error &) { empty_rejected = true; }
    assert(empty_rejected);

    // Minimal vertex program exercises the public GXP translation boundary.
    struct Input { SceGxmProgram program; SceGxmProgramVertexVaryings varyings; uint64_t instruction; } input{};
    input.program.magic = 0x00505847;
    input.program.size = sizeof(input);
    input.program.varyings_offset = offsetof(Input, varyings) - offsetof(SceGxmProgram, varyings_offset);
    input.program.primary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, primary_program_offset);
    input.program.primary_program_instr_count = 1;
    input.program.secondary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset);
    input.program.secondary_program_offset_end = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset_end);
    std::vector<SceGxmVertexAttribute> attributes;
    Hints hints{}; hints.attributes = &attributes;
    hints.color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
    // Illegal opcodes, unimplemented I8MAD2, and reserved VDUAL combinations.
    for (uint64_t opcode : {22, 23, 24, 25, 27, 4}) {
        input.instruction = opcode << 59;
        bool rejected = false;
        disasm::disasm_storage = &diagnostic;
        try { (void)convert_gxp(input.program, "cpu-test", features, Target::SpirVMetal, hints, false); }
        catch (const std::runtime_error &error) {
            rejected = std::string(error.what()).find("Metal USSE PC 0 opcode") != std::string::npos;
        }
        assert(rejected);
        assert(disasm::disasm_storage == &diagnostic);
    }
    input.instruction = (uint64_t(0b00111) << 59) | (uint64_t(7) << 56); // VMOV, PN
    bool pn_rejected = false;
    try { (void)convert_gxp(input.program, "cpu-test", features, Target::SpirVMetal, hints, false); }
    catch (const std::runtime_error &error) { pn_rejected = std::string(error.what()).find("unsupported USSE predicate 7") != std::string::npos; }
    assert(pn_rejected && disasm::disasm_storage == &diagnostic);
    // These variants previously logged an error but returned success with no
    // instruction result. Exercise the public translator, not a return-value mock.
    const auto encode = [](std::string_view layout, char field, unsigned value) {
        assert(layout.size() == 64);
        uint64_t result = 0;
        unsigned field_bit = 0;
        for (int i = 63; i >= 0; --i) {
            const bool bit = layout[i] == '1'
                || (layout[i] == field && ((value >> field_bit++) & 1));
            if (bit) result |= uint64_t(1) << (63 - i);
        }
        return result;
    };
    for (uint64_t instruction : {
            encode("10000ppcsnaaderbmooofllggghhhittkkjjqqqqqqquvvwwxyzzzzzzzAAAAAAA", 'g', 7),
            encode("10010ppmsnccderbowwwwaalllfff-ttkkgguuuuuuu-------hhhhhhhiiiiiii", 'l', 1),
            encode("10001ppcsnooderbmallfgghhhiiikttjjqquuuuuuuvvvvvvvwwwwwwwxxxxxxx", 'f', 1),
            encode("11100pppsn-ymrceffaaddlltbbggkhhiijjoooooooqqqqqqquuuuuuuvvvvvvv", '-', 0) }) {
        input.instruction = instruction;
        bool rejected = false;
        try { (void)convert_gxp(input.program, "cpu-invalid-variant", features, Target::SpirVMetal, hints, false); }
        catch (const std::runtime_error &error) {
            rejected = std::string(error.what()).find("Metal USSE PC 0 opcode") != std::string::npos;
        }
        assert(rejected && disasm::disasm_storage == &diagnostic);
    }
    input.instruction = 0;
    for (char bit : std::string("11111----000-----------101--------------------------------------"))
        input.instruction = (input.instruction << 1) | (bit == '1');
    assert(!convert_gxp(input.program, "cpu-nop", features, Target::SpirVMetal, hints, false).spirv.empty());
    assert(disasm::disasm_storage == &diagnostic);
    disasm::disasm_storage = nullptr;
    const auto expected = convert_gxp(input.program, "parallel-nop", features, Target::SpirVMetal, hints, false).spirv;
    const auto expected_msl = shader::metal::convert_gxp(input.program, "parallel-nop", features, hints, false).source;
    assert(!expected_msl.empty());
    auto translate = [input, features, attributes, hints, expected, expected_msl]() mutable {
        hints.attributes = &attributes;
        std::stringstream local;
        disasm::disasm_storage = &local;
        for (size_t i = 0; i < 100; ++i) {
            assert(convert_gxp(input.program, "parallel-nop", features, Target::SpirVMetal, hints, false).spirv == expected);
            assert(shader::metal::convert_gxp(input.program, "parallel-nop", features, hints, false).source == expected_msl);
            assert(disasm::disasm_storage == &local);
        }
        disasm::disasm_storage = nullptr;
    };
    auto first = std::async(std::launch::async, translate);
    auto second = std::async(std::launch::async, translate);
    first.get(); second.get();
    assert(disasm::disasm_storage == nullptr);
    std::cout << "PASS parallel translation: 200 actual private GXP-to-SPIR-V and GXP-to-MSL translations each, deterministic output and thread-local diagnostics\n";
    std::cout << "PASS shader: " << cases << " constant types, 96 swizzles, 11 public translation rejections, valid NOP and diagnostic RAII\n";
    test_centroid_queries(argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path{});
}
