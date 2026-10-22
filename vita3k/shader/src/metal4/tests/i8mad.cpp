// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <features/state.h>
#include <shader/metal4/usse_translator.h>
#include <shader/metal4/msl_recompiler.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
using namespace shader::metal3;
// I8MAD_HOOK

using Value = std::array<uint32_t, 4>;
using Inputs = std::array<Value, 3>;
struct Controls {
    unsigned color;
    bool alpha, alpha1, alpha2, c0, a0, c1, a1, c2, a2, negative, saturated;
    explicit Controls(unsigned flags) : color(flags & 3), alpha(flags & 4), alpha1(flags & 8), alpha2(flags & 16),
        c0(flags & 32), a0(flags & 64), c1(flags & 128), a1(flags & 256), c2(flags & 512), a2(flags & 1024),
        negative(flags & 2048), saturated(flags & 4096) {}
};
// Independent lane arithmetic: selector mapping and clamp bounds are documented
// in PVR_PSP2 ae4df4b723f3f1aa629b1f3b276b9cca6d663891, usc2/inst.h
// (IIMA8, FPMA_PARAMS), useasm/usedisasm.c DecodeIMA8Instruction.
// This verifies those software contracts, not numerical Vita hardware output.
static Value oracle(const Inputs &in, const Controls &c) {
    Value out{};
    for (unsigned lane = 0; lane < 4; ++lane) {
        int add;
        if (lane == 3) add = in[c.alpha ? 1 : 0][3];
        else switch (c.color) {
        case 0: add = in[0][lane]; break;
        case 1: add = in[1][lane]; break;
        case 2: add = in[0][3]; break;
        default: add = in[1][3]; break;
        }
        int lhs = in[1][lane != 3 && c.alpha1 ? 3 : lane];
        int rhs = in[2][lane != 3 && c.alpha2 ? 3 : lane];
        if (lane == 3 ? c.a0 : c.c0) add = 255 - add;
        if (lane == 3 ? c.a1 : c.c1) lhs = 255 - lhs;
        if (lane == 3 ? c.a2 : c.c2) rhs = 255 - rhs;
        int result = lhs * rhs + (c.negative ? -add : add);
        if (c.saturated) result = std::clamp(result, 0, 255);
        out[lane] = uint32_t(result);
    }
    return out;
}
struct Graph {
    struct Node { unsigned op; std::vector<uint32_t> operands; };
    std::map<uint32_t, Node> nodes;
    std::map<uint32_t, unsigned> raw_ops;
    std::map<uint32_t, std::string> names;
    std::vector<std::vector<uint32_t>> calls;
    explicit Graph(const std::vector<uint32_t> &words) {
        for (size_t at = 5; at < words.size();) {
            auto *w = words.data() + at; const auto n = w[0] >> 16, op = w[0] & 65535;
            assert(n && at + n <= words.size());
            if (op == spv::OpName) names[w[1]] = std::string(reinterpret_cast<const char *>(w + 2));
            if (op == spv::OpFunctionCall) calls.emplace_back(w + 2, w + n);
            switch (op) {
            case spv::OpConstant: case spv::OpConstantComposite: case spv::OpUndef:
            case spv::OpVectorExtractDynamic: case spv::OpCompositeConstruct: case spv::OpCompositeExtract:
            case spv::OpVectorShuffle: case spv::OpBitwiseXor: case spv::OpIMul:
            case spv::OpIAdd: case spv::OpISub: case spv::OpSelect:
            case spv::OpULessThan: case spv::OpUGreaterThan:
                nodes[w[2]] = {op, {w + 3, w + n}}; break;
            }
            if (n >= 3) raw_ops[w[2]] = op;
            at += n;
        }
    }
    Value evaluate(uint32_t id, std::map<uint32_t, Value> &cache) const {
        if (cache.contains(id)) return cache.at(id);
        if (!nodes.contains(id)) { std::cerr << "Missing expression ID " << id << " opcode " << raw_ops.at(id) << "\n"; std::abort(); }
        const auto &node = nodes.at(id); const auto &a = node.operands;
        Value out{};
        if (node.op == spv::OpConstant) out.fill(a.at(0));
        else if (node.op == spv::OpConstantComposite || node.op == spv::OpCompositeConstruct) {
            assert(a.size() == 4);
            for (unsigned i = 0; i < 4; ++i) out[i] = evaluate(a[i], cache)[0];
        } else if (node.op == spv::OpVectorExtractDynamic) {
            assert(a.size() == 2); const auto lane = evaluate(a[1], cache)[0]; assert(lane < 4);
            out.fill(evaluate(a[0], cache)[lane]);
        } else if (node.op == spv::OpCompositeExtract) {
            assert(a.size() == 2 && a[1] < 4); out.fill(evaluate(a[0], cache)[a[1]]);
        } else {
            const auto lhs = evaluate(a.at(0), cache), rhs = evaluate(a.at(1), cache);
            if (node.op == spv::OpVectorShuffle) {
                assert(a.size() == 6);
                for (unsigned i = 0; i < 4; ++i) { assert(a[i+2] < 8); out[i] = a[i+2] < 4 ? lhs[a[i+2]] : rhs[a[i+2]-4]; }
            } else {
                Value third{}; if (node.op == spv::OpSelect) third = evaluate(a.at(2), cache);
                for (unsigned i = 0; i < 4; ++i) switch (node.op) {
                case spv::OpBitwiseXor: out[i] = lhs[i] ^ rhs[i]; break;
                case spv::OpIMul: out[i] = lhs[i] * rhs[i]; break;
                case spv::OpIAdd: out[i] = lhs[i] + rhs[i]; break;
                case spv::OpISub: out[i] = lhs[i] - rhs[i]; break;
                case spv::OpULessThan: out[i] = lhs[i] < rhs[i]; break;
                case spv::OpUGreaterThan: out[i] = lhs[i] > rhs[i]; break;
                case spv::OpSelect: out[i] = lhs[i] ? rhs[i] : third[i]; break;
                default: assert(false);
                }
            }
        }
        return cache[id] = out;
    }
    void check(uint32_t result, std::array<uint32_t,3> ids, const Inputs &in, const Controls &c) const {
        std::map<uint32_t, Value> values{{ids[0],in[0]}, {ids[1],in[1]}, {ids[2],in[2]}};
        const auto actual = evaluate(result, values), expected = oracle(in, c);
        // UINT8 store takes the low eight bits; saturated output is already bounded.
        for (unsigned i = 0; i < 4; ++i) {
            assert(actual[i] == expected[i]);
            assert((actual[i] & 255) == (expected[i] & 255));
        }
    }
};
static uint64_t encode(char field, unsigned value) {
    constexpr std::string_view layout = "10011ppcsneedarbmtttuolfghijkqvvwwxxyyyyyyyzzzzzzzAAAAAAABBBBBBB";
    static_assert(layout.size() == 64); uint64_t result = 0; unsigned bit = 0;
    for (int i = 63; i >= 0; --i)
        if (layout[i] == '1' || (layout[i] == field && ((value >> bit++) & 1))) result |= uint64_t(1) << (63 - i);
    return result;
}
int main(int argc, char **argv) {
    assert(argc == 2); spdlog::set_level(spdlog::level::off);
    const auto directory = std::filesystem::path(argv[1]);
    auto write = [&](std::string name, const std::vector<uint32_t> &words, bool msl) {
        std::ofstream out(directory / (name + ".spv"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(words.data()), words.size()*4); assert(out);
        if (msl) std::ofstream(directory / (name + ".metal")) << shader::metal::convert_spirv(words).source;
    };
    std::array<Inputs,32> cases{};
    for (unsigned i = 0; i < 8; ++i) {
        constexpr unsigned boundary[]{0,1,2,127,128,253,254,255};
        for (unsigned source = 0; source < 3; ++source)
            for (unsigned lane = 0; lane < 4; ++lane) cases[i][source][lane] = boundary[(i+source*3+lane)%8];
    }
    uint32_t seed = 0x8a034;
    for (unsigned i = 8; i < cases.size(); ++i) for (auto &source : cases[i]) for (auto &v : source) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; v = seed & 255;
    }
    // Dedicated distinct RGB/alpha fixture exposes csel0=src1 versus csel1=alpha.
    cases[0] = {{{1,2,3,4}, {10,20,30,40}, {1,2,3,4}}};
    assert((oracle(cases[0], Controls(9)) == Value{50,100,150,164}));
    const Inputs saturating{{{255,0,1,128},{0,255,255,1},{255,255,1,1}}};
    assert((oracle(saturating, Controls(4096)) == Value{255,255,255,129}));
    assert((oracle(saturating, Controls(4096|2048)) == Value{0,255,254,0}));
    size_t helper_cases = 0, public_cases = 0, public_results = 0, saved = 0;
    struct Input { SceGxmProgram program; SceGxmProgramVertexVaryings varyings; uint64_t instruction; } input{};
    input.program.magic = 0x00505847; input.program.size = sizeof(input);
    input.program.varyings_offset = offsetof(Input, varyings) - offsetof(SceGxmProgram, varyings_offset);
    input.program.primary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, primary_program_offset);
    input.program.primary_program_instr_count = 1;
    input.program.secondary_program_offset = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset);
    input.program.secondary_program_offset_end = offsetof(Input, instruction) - offsetof(SceGxmProgram, secondary_program_offset_end);
    input.program.temp_reg_count1 = 128;
    std::vector<SceGxmVertexAttribute> attributes; Hints hints{}; hints.attributes = &attributes;
    hints.color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR; FeatureState features{};
    for (unsigned flags = 0; flags < 8192; ++flags) {
        const Controls c(flags);
        spv::SpvBuildLogger logger; spv::Builder b(spv::Spv_1_0, 0, &logger);
        b.setMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450); b.addCapability(spv::CapabilityShader);
        auto *main = b.makeEntryPoint("main"); b.addEntryPoint(spv::ExecutionModelVertex, main, "main");
        const auto type = b.makeVectorType(b.makeUintType(32), 4);
        std::array<spv::Id,3> ids{b.createUndefined(type), b.createUndefined(type), b.createUndefined(type)};
        const auto result = i8mad_result(b, ids[0], ids[1], ids[2], c.color, c.alpha, c.alpha1, c.alpha2,
            c.c0, c.a0, c.c1, c.a1, c.c2, c.a2, c.negative, c.saturated);
        b.leaveFunction(); std::vector<uint32_t> words; b.dump(words); Graph graph(words);
        for (const auto &in : cases) { graph.check(result, ids, in, c); ++helper_cases; }
        if (flags % 64 == (flags / 64) % 64) write("helper-" + std::to_string(flags), words, false);

        constexpr unsigned repeats[]{0,1,3,7}; const auto repeat = repeats[(flags ^ (flags >> 4)) & 3];
        input.instruction = encode('e',c.color) | encode('l',c.alpha) | encode('i',c.alpha1) | encode('j',c.alpha2)
            | encode('o',c.c0) | encode('h',c.a0) | encode('c',c.c1) | encode('g',c.a1)
            | encode('m',c.c2) | encode('f',c.a2) | encode('k',c.negative) | encode('u',c.saturated)
            | encode('t',repeat) | encode('z',32) | encode('A',64) | encode('B',96);
        words = convert_gxp(input.program, "i8mad-test", features, Target::SpirVMetal, hints, false).spirv;
        const Graph public_graph(words); unsigned source = 0, results = 0;
        for (const auto &call : public_graph.calls) {
            if (!public_graph.names.contains(call.at(1))) { std::cerr << "Missing function name " << call.at(1) << " flags " << flags << "\n"; std::abort(); }
            const auto &name = public_graph.names.at(call.at(1));
            if (name.find("unpack4xU8") == 0) { assert(source < 3); ids[source++] = call.at(0); }
            if (name.find("pack4xU8") == 0) {
                assert(source == 3 && call.size() == 3); source = 0;
                for (unsigned i = 0; i < 4; ++i) {
                    public_graph.check(call.at(2), ids, cases[(i*7+results)%cases.size()], c); ++public_cases;
                }
                ++results;
            }
        }
        assert(source == 0 && results == repeat+1); public_results += results;
        if (flags % 64 == (flags / 64) % 64) {
            write("public-" + std::to_string(flags), words, true); ++saved;
        }
    }
    assert(saved == 128);
    std::cout << "PASS I8MAD arithmetic: " << helper_cases << " vector cases across all 8192 selector/complement/negate/saturation controls; 32 boundary/distinct-lane/seeded inputs, exact arithmetic and UINT8 packing versus independent lane oracle; extracted helper ASAN/UBSAN\n";
    std::cout << "PASS I8MAD public translation: 8192 GXP translations, " << public_results << " actual load-to-pack result graphs and " << public_cases << " numerical vector evaluations, repeat 1/2/4/8, source selectors independent of multiply swizzles/modifiers; no GPU/hardware oracle\n";
}
