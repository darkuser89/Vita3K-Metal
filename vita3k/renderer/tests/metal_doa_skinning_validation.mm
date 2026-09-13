// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <renderer/metal/buffers.h>
#include <shader/uniform_block.h>
#include <shader/usse_program_analyzer.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

static void check(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
static std::vector<uint8_t> read(const std::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    check(bool(f), "Cannot open fixture " + p.string());
    const auto size = f.tellg();
    check(size > 0 && size < 4 * 1024 * 1024, "Invalid fixture size");
    std::vector<uint8_t> result(size);
    f.seekg(0);
    check(bool(f.read(reinterpret_cast<char *>(result.data()), size)), "Cannot read fixture");
    return result;
}
static void replace(std::string &s, const std::string &from, const std::string &to) {
    const auto at = s.find(from);
    check(at != std::string::npos && s.find(from, at + 1) == std::string::npos, "Captured shader instrumentation is ambiguous");
    s.replace(at, from.size(), to);
}
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: metal-doa-skinning-validation <captured DOA draw directory>\n";
        return 2;
    }
    @autoreleasepool {
        try {
            const std::filesystem::path draw(argv[1]);
            auto gxp_bytes = read(draw / "vertex.gxp");
            const auto &gxp = *reinterpret_cast<const SceGxmProgram *>(gxp_bytes.data());
            UniformBufferSizes sizes;
            shader::usse::get_uniform_buffer_sizes(gxp, sizes);
            check(sizes[4] == SCE_GXM_MAX_UB_IN_FLOAT_UNIT, "DOA palette is still restricted to one matrix");
            auto code = read(draw / "vertex.metal");
            std::string source(code.begin(), code.end());
            // Observe the original four-bone transform before lighting reuses its registers.
            // Keep every translated instruction and native vertex fetch unchanged.
            replace(source, "thread spvUnsafeArray<float4, 20>& outs, constant GxmRenderVertBufferBlock& renderVertInfo)", "thread spvUnsafeArray<float4, 20>& outs, constant GxmRenderVertBufferBlock& renderVertInfo, thread float4& skin_result)");
            replace(source, "    internals[2] = _1704;", "    internals[2] = _1704;\n    skin_result = float4(internals[2].xyz, 1.0);");
            replace(source, "vertex main_vs_out main_vs(", "vertex void main_vs(");
            replace(source, "renderVertInfo [[buffer(0)]])", "renderVertInfo [[buffer(0)]], device float4 *capture [[buffer(1)]], uint capture_id [[vertex_id]])");
            replace(source, "    primary_program(pa, sa, internals, r, p, outs, renderVertInfo);", "    float4 skin_result;\n    primary_program(pa, sa, internals, r, p, outs, renderVertInfo, skin_result);\n    capture[capture_id] = skin_result;");
            replace(source, "    return out;", "    return;");
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *native_error = nil;
            auto library = [device->native_device() newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&native_error];
            check(library != nil, native_error ? native_error.localizedDescription.UTF8String : "Compilation failed");
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [library newFunctionWithName:@"main_vs"];
            pd.rasterizationEnabled = NO;
            auto layout = [MTLVertexDescriptor vertexDescriptor];
            const MTLVertexFormat formats[] = { MTLVertexFormatFloat3, MTLVertexFormatHalf2, MTLVertexFormatChar3Normalized, MTLVertexFormatUChar4Normalized, MTLVertexFormatUChar4 };
            const unsigned offsets[] = { 0, 23, 12, 15, 19 };
            for (unsigned i = 0; i < 5; ++i) {
                layout.attributes[i].format = formats[i];
                layout.attributes[i].offset = offsets[i];
                layout.attributes[i].bufferIndex = 4;
            }
            layout.layouts[4].stride = 28;
            layout.layouts[4].stepFunction = MTLVertexStepFunctionPerVertex;
            layout.layouts[4].stepRate = 1;
            pd.vertexDescriptor = layout;
            auto pipeline = device->create_pipeline(pd, error);
            check(pipeline != nil, error);
            auto original = read(draw / "stream-4.bin");
            check(original.size() % 28 == 0, "Unexpected vertex stride");
            const size_t count = original.size() / 28;
            auto info = read(draw / "vertex-render-info.bin");
            check(info.size() == 112, "Unexpected native uniform address layout");
            std::map<unsigned, std::vector<uint8_t>> uniforms;
            for (unsigned block : { 0u, 6u, 7u, 8u })
                uniforms[block] = read(draw / ("uniform-" + std::to_string(block) + ".bin"));
            // The captured palette omitted bone 1. Supply explicit test matrices,
            // retaining captured vertex positions, weights and other uniform blocks.
            std::array<float, SCE_GXM_MAX_UB_IN_FLOAT_UNIT> palette{};
            for (unsigned bone = 0; bone < 128; ++bone) {
                const float rows[12] = { 1 + bone / 256.f, .125f, 0, bone / 32.f, 0, .75f + bone / 512.f, .25f, -bone / 64.f, .0625f, 0, 1 + bone / 128.f, bone / 128.f };
                std::copy_n(rows, 12, palette.data() + bone * 12);
            }
            auto output = [device->native_device() newBufferWithLength:count * 16 options:MTLResourceStorageModeShared];
            unsigned failures_before = 0, checked = 0;
            const auto captured_palette = read(draw / "uniform-4.bin");
            check(captured_palette.size() <= sizeof(palette), "Captured palette exceeds upload limit");
            const bool complete_capture = captured_palette.size() > 48;
            for (unsigned fixture = 0; fixture < (complete_capture ? 3u : 2u); ++fixture) {
                auto vertices = original;
                if (fixture == 2) {
                    std::fill(palette.begin(), palette.end(), 0);
                    std::memcpy(palette.data(), captured_palette.data(), captured_palette.size());
                    for (size_t v = 0; v < count; ++v)
                        for (unsigned k = 0; k < 4; ++k)
                            check((size_t(vertices[v * 28 + 19 + k]) + 1) * 48 <= captured_palette.size(), "Captured vertex reads beyond captured palette");
                }
                if (fixture == 1)
                    for (size_t v = 0; v < count; ++v)
                        for (unsigned k = 0; k < 4; ++k) {
                            vertices[v * 28 + 19 + k] = (v * 13 + k * 17) % 128;
                            vertices[v * 28 + 15 + k] = std::array<uint8_t, 4>{ 128, 64, 32, 31 }[k];
                        }
                auto geometry = [device->native_device() newBufferWithBytes:vertices.data() length:vertices.size() options:MTLResourceStorageModeShared];
                for (bool complete : { false, true }) {
                    renderer::metal::UploadBufferArena arena;
                    auto seed = arena.allocate(*device, 1024 * 1024);
                    std::memset(static_cast<uint8_t *>(seed.buffer.contents) + seed.offset, 0, seed.buffer.length);
                    arena.reset_after_completion();
                    std::vector<renderer::metal::GuestBufferRange> ranges;
                    std::vector<unsigned> blocks;
                    for (auto &[block, bytes] : uniforms) {
                        blocks.push_back(block);
                        ranges.push_back({ bytes.data(), bytes.size() });
                    }
                    blocks.push_back(4);
                    ranges.push_back({ reinterpret_cast<uint8_t *>(palette.data()), complete ? sizes[4] * 4u : 48u });
                    renderer::metal::GuestBufferBindings bindings(*device, ranges, getpagesize(), true, &arena);
                    auto render_info = info;
                    for (size_t i = 0; i < blocks.size(); ++i) {
                        uint64_t address = bindings.address(ranges[i]);
                        std::memcpy(render_info.data() + 40 + blocks[i] * 8, &address, 8);
                    }
                    auto commands = [device->command_queue() commandBuffer];
                    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    pass.renderTargetWidth = 1;
                    pass.renderTargetHeight = 1;
                    pass.defaultRasterSampleCount = 1;
                    auto enc = [commands renderCommandEncoderWithDescriptor:pass];
                    [enc setRenderPipelineState:pipeline];
                    [enc setVertexBuffer:geometry offset:0 atIndex:4];
                    [enc setVertexBuffer:output offset:0 atIndex:1];
                    [enc setVertexBytes:render_info.data() length:render_info.size() atIndex:0];
                    bindings.make_resident(enc);
                    [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:count];
                    [enc endEncoding];
                    check(device->submit_and_wait(commands, error), error);
                    const auto *actual = static_cast<const float *>(output.contents);
                    for (size_t v = 0; v < count; ++v) {
                        float pos[4] = { 0, 0, 0, 1 };
                        std::memcpy(pos, vertices.data() + v * 28, 12);
                        for (unsigned c = 0; c < 3; ++c) {
                            float row[4] = {};
                            for (unsigned k = 0; k < 4; ++k) {
                                unsigned bone = vertices[v * 28 + 19 + k];
                                float weight = vertices[v * 28 + 15 + k] / 255.f;
                                for (unsigned j = 0; j < 4; ++j)
                                    row[j] = k ? std::fma(weight, palette[bone * 12 + c * 4 + j], row[j]) : weight * palette[bone * 12 + c * 4 + j];
                            }
                            double expected = 0;
                            for (unsigned j = 0; j < 4; ++j)
                                expected += double(pos[j]) * row[j];
                            bool match = std::isfinite(actual[v * 4 + c]) && std::abs(actual[v * 4 + c] - expected) < 2e-5 * std::max(1.0, std::abs(expected));
                            if (complete) {
                                check(match, "Full palette transform differs from independent weighted matrix reference");
                                ++checked;
                            } else if (!match)
                                ++failures_before;
                        }
                        if (complete) {
                            check(actual[v * 4 + 3] == 1, "Invalid capture marker");
                            ++checked;
                        }
                    }
                }
            }
            check(failures_before > 0, "Truncated original palette did not reproduce corruption");
            std::cout << "PASS DOA original vertex skinning: truncated 48-byte palette mismatches " << failures_before << " components; complete palette matches " << checked << " CPU-reference channels, captured two-bone weights and four-bone indices 0..127; real captured palette checked: " << complete_capture << '\n';
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
