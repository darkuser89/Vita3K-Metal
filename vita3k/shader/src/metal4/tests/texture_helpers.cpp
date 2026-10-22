// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Emit both shared and original inline production bodies. No GPU/device calls.
#include <features/state.h>
#include <shader/metal4/msl_recompiler.h>
#include <shader/metal4/usse_translator.h>
#include <SPIRV/GLSL.std.450.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace shader::metal3;
using namespace shader::metal3::usse;
// COEFFICIENT_HOOK
struct Case {
    bool vertex = false;
    int dim = 2, lod = 0, gather = -1;
    bool gather_uv = false, cube_filter = true, metadata = true, mixed = false;
    DataType destination = DataType::F32;
    unsigned components = 2;
};
static std::vector<uint32_t> generate(const Case &base, bool shared) {
    auto c = base;
    spv::SpvBuildLogger logger;
    spv::Builder b(spv::Spv_1_0, 0, &logger);
    b.setMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    b.addCapability(spv::CapabilityShader); b.addCapability(spv::CapabilityImageQuery);
    auto main = b.makeEntryPoint("main");
    auto entry = b.addEntryPoint(c.vertex ? spv::ExecutionModelVertex : spv::ExecutionModelFragment, main, "main");
    if (!c.vertex) b.addExecutionMode(main, spv::ExecutionModeOriginUpperLeft);
    const auto f = b.makeFloatType(32), f4 = b.makeVectorType(f, 4), u = b.makeUintType(32), u2 = b.makeVectorType(u, 2);
    const auto uv_input = b.createVariable(spv::NoPrecision, spv::StorageClassInput, f4, "uv");
    b.addDecoration(uv_input, spv::DecorationLocation, 0); entry->addIdOperand(uv_input);
    const auto output = b.createVariable(spv::NoPrecision, spv::StorageClassOutput, f4, "result");
    b.addDecoration(output, c.vertex ? spv::DecorationBuiltIn : spv::DecorationLocation,
        c.vertex ? spv::BuiltInPosition : 0); entry->addIdOperand(output);
    SpirvShaderParameters parameters{}; parameters.native_metal = true;
    parameters.native_cube_float_filter = c.cube_filter;
    parameters.native_texture_lod_bias[0] = .5f; parameters.native_texture_lod_bias[1] = -.25f;
    if (!c.vertex) {
        parameters.native_frag_coord = b.createVariable(spv::NoPrecision, spv::StorageClassInput, f4, "position");
        b.addDecoration(parameters.native_frag_coord, spv::DecorationBuiltIn, spv::BuiltInFragCoord);
        entry->addIdOperand(parameters.native_frag_coord);
    }
    if (c.metadata) {
        const auto sizes = b.makeArrayType(u2, b.makeUintConstant(16), 0);
        b.addDecoration(sizes, spv::DecorationArrayStride, 8);
        const auto info = b.makeStructType({b.makeVectorType(u,4), sizes, f4}, "MetalTextureMipInfo");
        for (unsigned member = 0; member < 3; ++member)
            b.addMemberDecoration(info, member, spv::DecorationOffset, member == 0 ? 0 : member == 1 ? 16 : 144);
        const auto infos = b.makeArrayType(info, b.makeUintConstant(16), 0);
        b.addDecoration(infos, spv::DecorationArrayStride, 160);
        const auto block = b.makeStructType({infos}, "MetalTextureMipBlock");
        b.addDecoration(block, spv::DecorationBlock); b.addMemberDecoration(block, 0, spv::DecorationOffset, 0);
        parameters.native_texture_info = b.createVariable(spv::NoPrecision, spv::StorageClassUniform, block, "metadata");
        b.addDecoration(parameters.native_texture_info, spv::DecorationDescriptorSet, 0);
        b.addDecoration(parameters.native_texture_info, spv::DecorationBinding, 4);
    }
    const auto image_type = b.makeImageType(f, c.dim == 3 ? spv::DimCube : spv::Dim2D,
        false, false, false, 1, spv::ImageFormatUnknown);
    const auto sampled_type = b.makeSampledImageType(image_type);
    std::array<spv::Id, 2> textures;
    for (unsigned i = 0; i < textures.size(); ++i) {
        textures[i] = b.createVariable(spv::NoPrecision, spv::StorageClassUniformConstant, sampled_type,
            i ? "texture1" : "texture0");
        b.addDecoration(textures[i], spv::DecorationDescriptorSet, c.vertex ? 2 : 3);
        b.addDecoration(textures[i], spv::DecorationBinding, i);
    }
    utils::SpirvUtilFunctions utilities{}; utilities.native_metal = true;
    utilities.std_builtins = b.import("GLSL.std.450");
    FeatureState features{}; SceGxmProgram program{};
    if (!c.vertex) program.program_flags = SCE_GXM_PROGRAM_FLAG_FRAGMENT;
    NonDependentTextureQueryCallInfos queries;
    USSERecompiler compiler(b, program, features, parameters, utilities, nullptr, queries, spv::NoResult);
    spv::Id sum = b.makeCompositeConstant(f4, std::vector<spv::Id>(4, b.makeFloatConstant(0)));
    for (unsigned call = 0; call < 6; ++call) {
        if (call == 3) compiler.visitor.reset_for_new_session();
        if (base.mixed) {
            constexpr DataType destinations[] = {DataType::F32, DataType::F16, DataType::UINT8,
                DataType::INT8, DataType::UINT16, DataType::INT16};
            constexpr int lods[] = {0, 1, 3, 4, 2, 0};
            c.destination = destinations[call]; c.lod = lods[call];
            c.components = c.lod == 4 ? 3 : call % 2 ? 4 : 2;
            c.gather = call >= 4 ? int(call - 4) : -1;
            c.gather_uv = call == 5;
        }
        auto value = b.createLoad(uv_input, spv::NoPrecision);
        value = b.createBinOp(spv::OpFAdd, f4, value,
            b.makeCompositeConstant(f4, std::vector<spv::Id>(4, b.makeFloatConstant(float(call) / 16))));
        std::vector<spv::IdImmediate> indices{{true, value}, {true, value}};
        for (unsigned component = 0; component < c.components; ++component) indices.push_back({false, component});
        const auto coord = b.createOp(spv::OpVectorShuffle, b.makeVectorType(f, c.components), indices);
        spv::Id extra1 = 0, extra2 = 0;
        if (c.lod == 1 || c.lod == 2) extra1 = b.createCompositeExtract(value, f, 3);
        if (c.lod == 3) {
            const auto gradient_type = b.makeVectorType(f, c.dim);
            extra1 = b.makeCompositeConstant(gradient_type, std::vector<spv::Id>(c.dim, b.makeFloatConstant(.1f + call * .01f)));
            extra2 = b.makeCompositeConstant(gradient_type, std::vector<spv::Id>(c.dim, b.makeFloatConstant(.2f + call * .02f)));
        }
        const unsigned slot = call % 2;
        const auto tex = b.createLoad(textures[slot], spv::NoPrecision);
        auto result = shared
            ? compiler.visitor.do_fetch_texture(tex, slot, c.dim, {coord, int(DataType::F32)}, c.destination,
                c.lod, extra1, extra2, c.gather, c.gather_uv)
            : compiler.visitor.emit_fetch_texture(tex, slot, c.dim, coord, c.destination,
                c.lod, extra1, extra2, c.gather, c.gather_uv);
        if (is_integer_data_type(c.destination)) result = b.createUnaryOp(spv::OpBitcast, f4, result);
        sum = b.createBinOp(spv::OpFAdd, f4, sum, result);
        if (call == 0 && c.dim == 2 && c.components == 2) {
            const auto coefficients = get_uv_coeffs(b, utilities.std_builtins, tex, coord,
                b.makeFloatConstant(0), parameters.native_texture_info, slot);
            sum = b.createBinOp(spv::OpFAdd, f4, sum, b.createCompositeConstruct(f4,
                {b.createCompositeExtract(coefficients, f, 0), b.createCompositeExtract(coefficients, f, 1),
                    b.makeFloatConstant(0), b.makeFloatConstant(0)}));
        }
    }
    b.createStore(sum, output); b.makeReturn(true); b.leaveFunction();
    std::vector<uint32_t> words; b.dump(words);
    unsigned functions = 0, calls = 0;
    for (size_t i = 5; i < words.size(); i += words[i] >> 16) {
        functions += (words[i] & 65535) == spv::OpFunction;
        calls += (words[i] & 65535) == spv::OpFunctionCall;
    }
    assert(functions == 1 + (shared && c.metadata ? (c.mixed ? 6 : 2) : 0));
    assert(calls == (shared && c.metadata ? 6 : 0));
    return words;
}
int main(int argc, char **argv) {
    assert(argc == 2); spdlog::set_level(spdlog::level::off);
    const std::filesystem::path output = argv[1]; std::filesystem::create_directory(output);
    std::vector<Case> cases;
    for (int dim : {2, 3}) {
        for (int lod = 0; lod <= (dim == 2 ? 4 : 3); ++lod) {
            Case c; c.dim = dim; c.lod = lod; c.components = lod == 4 ? 3 : dim; cases.push_back(c);
            if (lod == 2 || lod == 3) { c.vertex = true; cases.push_back(c); }
        }
        for (int component = 0; component < 4; ++component) for (bool uv : {false, true}) {
            Case c; c.dim = dim; c.components = dim; c.gather = component; c.gather_uv = uv; cases.push_back(c);
        }
    }
    for (auto type : {DataType::F16, DataType::C10, DataType::UINT8, DataType::INT8,
             DataType::UINT16, DataType::INT16, DataType::UINT32, DataType::INT32}) {
        Case c; c.destination = type; c.components = 4; cases.push_back(c);
    }
    Case c; c.dim = c.components = 3; c.cube_filter = false; cases.push_back(c);
    c = {}; c.metadata = false; cases.push_back(c);
    c = {}; c.mixed = true; cases.push_back(c);
    size_t inline_words = 0, shared_words = 0, inline_msl = 0, shared_msl = 0;
    for (size_t i = 0; i < cases.size(); ++i) for (bool shared : {false, true}) {
        const auto words = generate(cases[i], shared);
        (shared ? shared_words : inline_words) += words.size();
        const auto name = std::to_string(i) + (shared ? "-shared" : "-inline");
        std::ofstream file(output / (name + ".spv"), std::ios::binary);
        file.write(reinterpret_cast<const char *>(words.data()), words.size() * 4);
        const auto source = shader::metal::convert_spirv(words).source;
        (shared ? shared_msl : inline_msl) += source.size();
        std::ofstream(output / (name + ".metal")) << source;
    }
    assert(shared_words < inline_words);
    std::cout << "PASS texture helper generation: " << cases.size() << " inline/shared production pairs; "
        << inline_words << " -> " << shared_words << " SPIR-V words, " << inline_msl << " -> " << shared_msl
        << " MSL bytes; dynamic metadata retained\n";
}
