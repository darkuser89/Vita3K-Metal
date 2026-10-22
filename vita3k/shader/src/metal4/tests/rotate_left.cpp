// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <features/state.h>
#include <shader/metal4/usse_translator.h>
#include <shader/metal4/msl_recompiler.h>
#include <array>
#include <bit>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
using namespace shader::metal3;
// ROTATE_HOOK

// Independent arithmetic oracle; no shift expression copied from production.
// Encoding reference: PVR_PSP2 ae4df4b723f3f1aa629b1f3b276b9cca6d663891,
// tools/intern/useasm/useasm.c EncodeBitwiseInstruction and usedisasm.c
// DecodeBitwiseInstruction/RotateLeft. This is not a Vita hardware oracle.
static uint32_t evaluate(spv::Builder &b, spv::Id id, spv::Id value_id,
    spv::Id count_id, uint32_t value, uint32_t count) {
    if (id == value_id) return value;
    if (id == count_id) return count;
    if (b.getOpCode(id) == spv::OpConstant) return b.getConstantScalar(id);
    const auto a = evaluate(b, b.getIdOperand(id, 0), value_id, count_id, value, count);
    const auto c = evaluate(b, b.getIdOperand(id, 1), value_id, count_id, value, count);
    switch (b.getOpCode(id)) {
    case spv::OpBitwiseAnd: return a & c;
    case spv::OpBitwiseOr: return a | c;
    case spv::OpISub: return a - c;
    case spv::OpShiftLeftLogical: assert(c < 32); return a << c;
    case spv::OpShiftRightLogical: assert(c < 32); return a >> c;
    default: assert(false); return 0;
    }
}
static uint64_t encode(char field, unsigned value) {
    constexpr std::string_view layout = "01ooopppsnrydecxaaaaittttthhbwkkffggjjjjjjjlllllllmmmmmmmqqqqqqq";
    static_assert(layout.size() == 64);
    uint64_t result = 0; unsigned bit = 0;
    for (int i = 63; i >= 0; --i) {
        if (layout[i] == '1' || (layout[i] == field && ((value >> bit++) & 1)))
            result |= uint64_t(1) << (63 - i);
    }
    return result;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    spdlog::set_level(spdlog::level::off);
    spv::SpvBuildLogger logger; spv::Builder b(spv::Spv_1_0, 0, &logger);
    b.setMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    b.addCapability(spv::CapabilityShader);
    auto *main = b.makeEntryPoint("main");
    b.addEntryPoint(spv::ExecutionModelVertex, main, "main");
    const auto type = b.makeUintType(32);
    const auto value_id = b.createUndefined(type), count_id = b.createUndefined(type);
    const auto result = rotate_left32(b, value_id, count_id);
    size_t numeric = 0;
    const auto check = [&](uint32_t value, uint32_t count) {
        assert(evaluate(b, result, value_id, count_id, value, count) == std::rotl(value, int(count % 32)));
        ++numeric;
    };
    for (auto value : {0u, 1u, 0xffffffffu, 0x80000000u, 0xaaaaaaaau, 0x55555555u})
        for (uint32_t count = 0; count < 256; ++count) check(value, count);
    for (unsigned bit = 0; bit < 32; ++bit)
        for (uint32_t count = 0; count < 64; ++count) check(1u << bit, count);
    uint32_t seed = 0xa034;
    auto random = [&]() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; };
    for (unsigned i = 0; i < 200000; ++i) { const auto v = random(), c = random(); check(v, c); }
    for (uint32_t count : {0x80000000u, 0xfffffffeu, 0xffffffffu}) check(0x89abcdef, count);
    b.leaveFunction(); std::vector<uint32_t> words; b.dump(words);
    const auto directory = std::filesystem::path(argv[1]);
    auto write = [&](std::string name, const std::vector<uint32_t> &code) {
        std::ofstream out(directory / (name + ".spv"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(code.data()), code.size() * 4); assert(out);
    };
    write("arithmetic", words);
    std::cout << "PASS ROL arithmetic: " << numeric << " emitted-SPIR-V evaluations against std::rotl, all low-five-bit counts, boundary/full-width random counts, no shift by 32; ASAN/UBSAN, no GPU\n";

    struct Input { SceGxmProgram program; SceGxmProgramVertexVaryings varyings; uint64_t instruction; } input{};
    input.program.magic = 0x00505847; input.program.size = sizeof(input);
    input.program.varyings_offset = offsetof(Input, varyings) - offsetof(SceGxmProgram, varyings_offset);
    input.program.primary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, primary_program_offset);
    input.program.primary_program_instr_count = 1;
    input.program.secondary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset);
    input.program.secondary_program_offset_end = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset_end);
    input.program.temp_reg_count1 = 128;
    std::vector<SceGxmVertexAttribute> attributes; Hints hints{}; hints.attributes = &attributes;
    hints.color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
    FeatureState features{};
    size_t translations = 0, rotations = 0, rejected = 0;
    for (bool immediate : {false, true}) for (unsigned invert : {0u, 1u})
    for (unsigned rotation = 0; rotation < 32; ++rotation) for (unsigned repeat : {0u, 1u, 3u, 15u}) {
        input.instruction = encode('o', 4) | encode('b', 1) | encode('t', rotation)
            | encode('i', invert) | encode('a', repeat) | encode('r', 1)
            | encode('m', 32) | encode('q', immediate ? 31 : 64)
            | encode('x', immediate) | encode('g', immediate ? 2 : 0);
        const auto code = convert_gxp(input.program, "rol-test", features, Target::SpirVMetal, hints, false).spirv;
        struct Binary { unsigned op, a, b; };
        std::map<uint32_t, Binary> ops; std::map<uint32_t, uint32_t> constants;
        for (size_t at = 5; at < code.size();) {
            const auto *w = code.data() + at; const auto n = w[0] >> 16, op = w[0] & 65535;
            assert(n && at + n <= code.size());
            if (op == spv::OpConstant && n == 4) constants[w[2]] = w[3];
            if (n == 5 && (op == spv::OpBitwiseOr || op == spv::OpShiftLeftLogical
                    || op == spv::OpShiftRightLogical || op == spv::OpBitwiseAnd || op == spv::OpISub))
                ops[w[2]] = {op, w[3], w[4]};
            at += n;
        }
        size_t found = 0;
        for (const auto &[id, op] : ops) if (op.op == spv::OpBitwiseOr && ops.contains(op.a) && ops.contains(op.b)) {
            const auto left = ops.at(op.a), right = ops.at(op.b);
            if (left.op != spv::OpShiftLeftLogical || right.op != spv::OpShiftRightLogical || left.a != right.a) continue;
            if (!ops.contains(left.b) || !ops.contains(right.b)) continue;
            const auto lc = ops.at(left.b), rc = ops.at(right.b);
            assert(lc.op == spv::OpBitwiseAnd && constants.at(lc.b) == 31);
            assert(rc.op == spv::OpBitwiseAnd && constants.at(rc.b) == 31);
            const auto sub = ops.at(rc.a); assert(sub.op == spv::OpISub && constants.at(sub.a) == 0 && sub.b == left.b);
            if (immediate) {
                uint32_t expected = std::rotl(uint32_t(31), int(rotation));
                if (invert) expected = ~expected;
                assert(constants.at(lc.a) == expected);
            }
            ++found;
        }
        assert(found == repeat + 1); rotations += found;
        const auto name = "rol-" + std::to_string(translations++);
        write(name, code);
        // One MSL per source/rotation/invert combination, including the maximum repeat.
        if (repeat == 15) std::ofstream(directory / (name + ".metal")) << shader::metal::convert_spirv(code).source;
        input.instruction |= encode('w', 1);
        bool threw = false;
        try { (void)convert_gxp(input.program, "rol-partial", features, Target::SpirVMetal, hints, false); }
        catch (const std::runtime_error &e) { threw = std::string(e.what()).find("Metal USSE PC 0 opcode") != std::string::npos; }
        assert(threw); ++rejected;
    }
    std::cout << "PASS ROL integration: " << translations << " public GXP translations, " << rotations
              << " rotation graphs, immediate/register, all source rotations/inversion, repeat 1/2/4/16; " << rejected << " partial variants still explicitly rejected\n";
}
