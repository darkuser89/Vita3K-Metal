// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <shader/uniform_block.h>
#include <stdexcept>

using Vec = std::array<float, 4>;
using SkinVertex = std::array<Vec, 6>;
static void check(bool value, const std::string &message) {
    if (!value)
        throw std::runtime_error(message);
}
static float halfround(float value) { return float((__fp16)value); }
static float dot3(const Vec &a, const Vec &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Execute the original Stealth four-bone skinning program. Instrument only its
// entry point and final output stores; all translated guest instructions stay
// unchanged. Rasterization is disabled so clipped vertices remain observable.
// The CPU oracle describes weighted matrix transforms, with the F16 roundings
// specified by the original GXP instructions. This is not a gameplay capture.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 2, "usage: metal-stealth-vertex-validation original-58880ba8.gxp");
            std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
            check(bool(file), "Cannot open GXP");
            auto size = file.tellg();
            check(size > 0 && size < 1024 * 1024, "Invalid GXP size");
            std::vector<uint32_t> words((size_t(size) + 3) / 4);
            file.seekg(0);
            check(bool(file.read(reinterpret_cast<char *>(words.data()), size)), "Cannot read GXP");
            auto program = compile_gxp_fixture(words, size, "stealth-skinning", "gxp-mapped-viewport");
            check(program.stage == shader::metal::Stage::Vertex && !program.writes_guest_memory,
                  "Expected a read-only vertex program");
            auto source = program.source;
            const std::string entry = "[[buffer(0)]])";
            auto at = source.find(entry);
            check(at != std::string::npos && source.find(entry, at + 1) == std::string::npos,
                  "Unexpected entry point ABI");
            source.replace(at, entry.size(),
                           "[[buffer(0)]], device float4 *capture [[buffer(1)]], uint capture_id [[vertex_id]])");
            at = source.find("    return out;");
            check(at != std::string::npos && source.find("    return out;", at + 1) == std::string::npos,
                  "Unexpected output ABI");
            source.replace(at, std::strlen("    return out;"), "    return;");
            source.insert(at, "    capture[capture_id*4] = out.gl_Position;\n"
                              "    capture[capture_id*4+1] = out.v_Color0;\n"
                              "    capture[capture_id*4+2] = out.v_TexCoord0;\n"
                              "    capture[capture_id*4+3] = out.v_TexCoord1;\n");
            at = source.find("vertex main_vs_out main_vs(");
            check(at != std::string::npos, "Unexpected vertex entry point");
            source.replace(at, std::strlen("vertex main_vs_out main_vs("), "vertex void main_vs(");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            check(device != nil, "No Metal device");
            auto queue = [device newCommandQueue];
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *error = nil;
            auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                options:options
                                                  error:&error];
            check(library != nil, error ? error.localizedDescription.UTF8String : "Shader compilation failed");
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [library newFunctionWithName:@"main_vs"];
            pd.rasterizationEnabled = NO;
            auto layout = [MTLVertexDescriptor new];
            for (int i = 0; i < 6; ++i) {
                layout.attributes[i].format = MTLVertexFormatFloat4;
                layout.attributes[i].offset = i * sizeof(Vec);
                layout.attributes[i].bufferIndex = 4;
            }
            layout.layouts[4].stride = sizeof(SkinVertex);
            pd.vertexDescriptor = layout;
            auto pipeline = [device newRenderPipelineStateWithDescriptor:pd error:&error];
            check(pipeline != nil, error ? error.localizedDescription.UTF8String : "Pipeline failed");
            constexpr int count = 64;
            int failures = 0, comparisons = 0;
            float maxPositionError = 0, maxNormalError = 0;
            for (int pass = 0; pass < 5; ++pass) {
                // 128-byte header, then 64-byte bone stride; the last row is
                // poison to detect accidental 3-row stride/address mistakes.
                std::array<Vec, 8 + 64 * 4> uniforms{};
                uniforms[0] = {1, .125f, 0, .25f};
                uniforms[1] = {0, .75f, .25f, -.125f};
                uniforms[2] = {0, 0, .5f, .375f};
                uniforms[3] = {.125f, 0, 0, 1};
                uniforms[4] = pass % 2 ? Vec{.75f, -.25f, .125f, 0} : Vec{1, 0, 0, 0};
                uniforms[5] = pass % 2 ? Vec{.125f, 1, -.125f, 0} : Vec{0, 1, 0, 0};
                uniforms[6] = pass % 2 ? Vec{-.25f, .125f, .75f, 0} : Vec{0, 0, 1, 0};
                for (int b = 0; b < 64; ++b) {
                    uniforms[8 + b * 4] = {1 + b / 128.f, (b % 3 - 1) / 8.f, 0, b / 64.f};
                    uniforms[9 + b * 4] = {0, .75f + (b % 7) / 16.f, (b % 5 - 2) / 16.f, -b / 128.f};
                    uniforms[10 + b * 4] = {(b % 3 - 1) / 16.f, 0, .5f + (b % 11) / 16.f, b / 256.f};
                    uniforms[11 + b * 4] = {999, -999, 777, -777};
                }
                std::array<SkinVertex, count> vertices{};
                std::array<std::array<Vec, 4>, count> expected{};
                shader::RenderVertUniformBlockExtended info{};
                info.set_buffer_count(1);
                info.base_block.viewport_flip = {pass == 2 ? -1.f : 1.f, pass == 3 ? -1.f : 1.f, 1, 1};
                info.base_block.viewport_flag = pass == 4 ? 0 : 1;
                info.base_block.screen_width = 960;
                info.base_block.screen_height = 544;
                info.base_block.z_offset = pass == 3 ? -.5f : 0;
                info.base_block.z_scale = pass == 3 ? .75f : 1;
                for (int v = 0; v < count; ++v) {
                    auto &a = vertices[v];
                    a[0] = {(v % 7 - 3) / 8.f, (v % 11 - 5) / 8.f, (v % 5 - 2) / 8.f, 1};
                    a[1] = {1 + (v % 3) / 4.f, (v % 7 - 3) / 4.f, .5f + (v % 5) / 4.f, 0};
                    a[2] = {v / 64.f, -.5f, .25f, -1};
                    a[3] = {v / 63.f, (63 - v) / 63.f, 88, 99};
                    a[4] = v < 32 ? Vec{} : Vec{.125f, .25f, .5f, .125f};
                    if (v < 32)
                        a[4][v % 4] = 1;
                    for (int k = 0; k < 4; ++k)
                        a[5][k] = float((v * 13 + k * 17 + pass * 7) % 64);
                    Vec pos{}, normal{};
                    Vec inputNormal = a[1];
                    inputNormal[1] = -inputNormal[1];
                    for (int k = 0; k < 4; ++k) {
                        const int bone = int(a[5][k]);
                        for (int c = 0; c < 3; ++c) {
                            const auto &row = uniforms[8 + bone * 4 + c];
                            pos[c] += a[4][k] * (dot3(row, a[0]) + row[3]);
                            normal[c] += a[4][k] * dot3(row, inputNormal);
                        }
                    }
                    const float length = std::sqrt(dot3(normal, normal));
                    for (int c = 0; c < 3; ++c)
                        normal[c] = halfround(normal[c] / length * (c == 1 ? -1 : 1));
                    Vec transformed{};
                    for (int c = 0; c < 3; ++c) {
                        const float first = halfround(normal[0] * halfround(uniforms[4][c]));
                        const float second = halfround(std::fma(normal[1], halfround(uniforms[5][c]), first));
                        // The final MAD writes an internal F32 register.
                        transformed[c] = std::fma(normal[2], halfround(uniforms[6][c]), second);
                    }
                    const float reciprocal = 1 / std::sqrt(halfround(dot3(transformed, transformed)));
                    auto &e = expected[v];
                    for (int c = 0; c < 4; ++c) {
                        e[0][c] = dot3(uniforms[c], pos) + uniforms[c][3];
                        e[1][c] = a[2][c] + 1;
                    }
                    if (pass == 4) {
                        e[0][0] = e[0][0] * 2 / 960 - 1;
                        e[0][1] = e[0][1] * 2 / 544 - 1;
                        e[0][2] = e[0][3];
                    } else {
                        for (int c = 0; c < 4; ++c)
                            e[0][c] *= info.base_block.viewport_flip[c];
                        e[0][2] =
                            std::max(e[0][2] / e[0][3] * info.base_block.z_scale + info.base_block.z_offset, 0.f) *
                            e[0][3];
                    }
                    e[2] = {a[3][0], a[3][1], transformed[0] * reciprocal, transformed[1] * reciprocal};
                    e[3] = {e[2][2], e[2][3], transformed[2] * reciprocal, 0};
                }
                auto uniform = [device newBufferWithBytes:uniforms.data()
                                                   length:sizeof(uniforms)
                                                  options:MTLResourceStorageModeShared];
                auto input = [device newBufferWithBytes:vertices.data()
                                                 length:sizeof(vertices)
                                                options:MTLResourceStorageModeShared];
                auto output = [device newBufferWithLength:(count * 4 + 2) * sizeof(Vec)
                                                  options:MTLResourceStorageModeShared];
                check(uniform && input && output, "Buffer allocation failed");
                std::memset(output.contents, 0xA5, output.length);
                info.set_buffer_address(0, uniform.gpuAddress);
                std::vector<uint8_t> bytes((info.get_size() + 15) & ~size_t(15));
                info.copy_to(bytes.data());
                auto commands = [queue commandBuffer];
                auto rp = [MTLRenderPassDescriptor renderPassDescriptor];
                rp.renderTargetWidth = 1;
                rp.renderTargetHeight = 1;
                rp.defaultRasterSampleCount = 1;
                auto encoder = [commands renderCommandEncoderWithDescriptor:rp];
                check(encoder != nil, "Render encoder failed");
                [encoder setRenderPipelineState:pipeline];
                [encoder setVertexBytes:bytes.data() length:bytes.size() atIndex:0];
                [encoder setVertexBuffer:output offset:sizeof(Vec) atIndex:1];
                [encoder setVertexBuffer:input offset:0 atIndex:4];
                [encoder useResource:uniform usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
                [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:count];
                [encoder endEncoding];
                [commands commit];
                [commands waitUntilCompleted];
                check(commands.status == MTLCommandBufferStatusCompleted,
                      commands.error ? commands.error.localizedDescription.UTF8String : "GPU execution failed");
                const auto actual = static_cast<const Vec *>(output.contents) + 1;
                for (int v = 0; v < count; ++v)
                    for (int slot = 0; slot < 4; ++slot)
                        for (int c = 0; c < 4; ++c) {
                            const float x = actual[v * 4 + slot][c], y = expected[v][slot][c];
                            const float delta = std::abs(x - y);
                            const bool normal = slot == 3 || (slot == 2 && c >= 2);
                            (normal ? maxNormalError : maxPositionError) =
                                std::max(normal ? maxNormalError : maxPositionError, delta);
                            ++comparisons;
                            if (!std::isfinite(x) || delta > (normal ? .001f : .00001f)) {
                                if (failures++ < 16)
                                    std::cerr << "Mismatch pass=" << pass << " vertex=" << v << " slot=" << slot
                                              << " component=" << c << " actual=" << x << " expected=" << y << '\n';
                            }
                        }
                const auto raw = static_cast<const uint8_t *>(output.contents);
                for (size_t i = 0; i < sizeof(Vec); ++i)
                    check(raw[i] == 0xA5 && raw[output.length - sizeof(Vec) + i] == 0xA5, "Output guard overwritten");
            }
            std::cout << "RESULT vertices=320 components=" << comparisons << " failed=" << failures
                      << " max_position_error=" << maxPositionError << " max_normal_error=" << maxNormalError << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
