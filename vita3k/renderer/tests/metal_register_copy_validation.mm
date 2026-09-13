// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void check(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

// Unit 13's original f4c6949a fragment program contains one full VMOV.f16
// from sa0 to pa0. Execute its unchanged translated primary function with
// every 16-bit pattern, observing registers before attachment conversion.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 2, "usage: metal-register-copy-validation original-f4c6949a.gxp");
            std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
            check(bool(input), "Cannot open original GXP");
            const auto size = input.tellg();
            check(size > 0 && size < 1024 * 1024, "Invalid GXP length");
            std::vector<uint32_t> words((size_t(size) + 3) / 4);
            input.seekg(0);
            check(bool(input.read(reinterpret_cast<char *>(words.data()), size)), "Cannot read GXP");
            auto program = compile_gxp_fixture(words, size, "unit13-register-copy", "gxp-mapped-viewport");
            check(program.stage == shader::metal::Stage::Fragment && !program.writes_guest_memory,
                  "Expected a read-only fragment program");
            const auto source = program.source + R"(
kernel void copy_probe(uint id [[thread_position_in_grid]], device uint2 *result [[buffer(0)]]) {
    const uint a = id | (((id + 0x1234u) & 65535u) << 16u);
    const uint b = (id ^ 0xa5a5u) | ((id ^ 0x5a5au) << 16u);
    spvUnsafeArray<float4, 32> pa;
    spvUnsafeArray<float4, 32> sa;
    sa[0] = as_type<float4>(uint4(a, b, 0u, 0u));
    primary_program(pa, sa);
    result[id] = as_type<uint2>(float2(pa[0].x, pa[0].y));
}
)";
            auto device = MTLCreateSystemDefaultDevice();
            check(device != nil, "No Metal device");
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *error = nil;
            auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                options:options error:&error];
            check(library != nil, error ? error.localizedDescription.UTF8String : "Compilation failed");
            auto pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"copy_probe"]
                                                                 error:&error];
            check(pipeline != nil, error ? error.localizedDescription.UTF8String : "Pipeline failed");
            constexpr size_t count = 65536;
            auto output = [device newBufferWithLength:(count * 2 + 8) * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
            check(output != nil, "Buffer allocation failed");
            std::memset(output.contents, 0xA5, output.length);
            auto commands = [[device newCommandQueue] commandBuffer];
            auto encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:output offset:16 atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            check(commands.status == MTLCommandBufferStatusCompleted,
                  commands.error ? commands.error.localizedDescription.UTF8String : "GPU failed");
            const auto *actual = static_cast<const uint32_t *>(output.contents) + 4;
            size_t failed = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t expected[] = {i | (((i + 0x1234u) & 65535u) << 16u),
                                            (i ^ 0xa5a5u) | ((i ^ 0x5a5au) << 16u)};
                for (size_t word = 0; word < 2; ++word) {
                    if (actual[i * 2 + word] == expected[word])
                        continue;
                    if (failed++ < 8)
                        std::cerr << "Mismatch input=" << i << " word=" << word << " actual=" << std::hex
                                  << actual[i * 2 + word] << " expected=" << expected[word] << std::dec << '\n';
                }
            }
            const auto *raw = static_cast<const uint8_t *>(output.contents);
            for (size_t i = 0; i < 16; ++i)
                check(raw[i] == 0xA5 && raw[output.length - 16 + i] == 0xA5, "Output guard changed");
            std::cout << "RESULT patterns=" << count << " packed_words=" << count * 2 << " failed=" << failed << '\n';
            return failed ? 1 : 0;
        } catch (const std::exception &error) {
            std::cerr << "FAIL " << error.what() << '\n';
            return 1;
        }
    }
}
