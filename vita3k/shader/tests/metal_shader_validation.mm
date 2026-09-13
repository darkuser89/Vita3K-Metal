// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later

#include <shader/msl_recompiler.h>
#include "metal_shader_fixture.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Explicitly run against a local corpus; no copyrighted fixtures are distributed.
// Compilation validates the native compiler path, not game rendering correctness.
int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: metal-shader-validation <spirv|gxp-fetch|gxp-interlock|gxp-interlock-f16|gxp-mask|gxp-mask-update|gxp-mapped|gxp-mapped-viewport|gxp-mapped-mips|gxp-msaa2|gxp-msaa4|gxp-mapped-half> <corpus> <output> [native-lod-bias]\n";
        return 2;
    }
    const std::string mode = argv[1];
    float lod_bias = 0.f;
    if (argc == 5) {
        try { lod_bias = std::stof(argv[4]); }
        catch (...) { return 2; }
        if (!std::isfinite(lod_bias) || lod_bias < -3.875f || lod_bias > 4.f) return 2;
    }
    std::cout << std::unitbuf;
    if (mode != "spirv" && mode != "gxp-fetch" && mode != "gxp-interlock" && mode != "gxp-interlock-f16"
        && mode != "gxp-mask" && mode != "gxp-mask-update" && mode != "gxp-mapped" && mode != "gxp-mapped-viewport" && mode != "gxp-mapped-mips" && mode != "gxp-msaa2" && mode != "gxp-msaa4" && mode != "gxp-mapped-half")
        return 2;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal3]) {
            std::cerr << "A Metal 3 GPU is required\n";
            return 2;
        }
        if ((mode == "gxp-interlock" || mode == "gxp-interlock-f16") && !device.rasterOrderGroupsSupported) {
            std::cerr << "Raster order groups are unavailable\n";
            return 2;
        }
        std::cout << "Device: " << device.name.UTF8String << "; mode=" << mode << '\n';
        std::filesystem::create_directories(argv[3]);
        uint32_t passed = 0, failed = 0, fetch = 0;
        for (const auto &file : std::filesystem::recursive_directory_iterator(argv[2])) {
            if (file.path().extension() != (mode == "spirv" ? ".spv" : ".gxp"))
                continue;
            @autoreleasepool {
                const auto relative_path = std::filesystem::relative(file.path(), argv[2]);
                std::cout << "BEGIN " << relative_path.string() << '\n';
                try {
                    std::ifstream input(file.path(), std::ios::binary | std::ios::ate);
                    if (!input)
                        throw std::runtime_error("Cannot open shader");
                    const std::streamsize size = input.tellg();
                    if (size < 20 || size > 64 * 1024 * 1024)
                        throw std::runtime_error("Invalid shader length");
                    // Word storage provides the alignment required by GXP and SPIR-V.
                    std::vector<uint32_t> words((size + 3) / 4);
                    input.seekg(0);
                    if (!input.read(reinterpret_cast<char *>(words.data()), size))
                        throw std::runtime_error("Cannot read shader");
                    shader::metal::Program generated;
                    if (mode == "spirv") {
                        if (size % 4)
                            throw std::runtime_error("Truncated SPIR-V word");
                        generated = shader::metal::convert_spirv(words);
                    } else {
                        generated = compile_gxp_fixture(words, size, file.path().stem().string(), mode, lod_bias);
                    }
                    // Preserve game directories so equally named shaders cannot overwrite evidence.
                    auto relative = relative_path;
                    relative.replace_extension(".metal");
                    const auto output = std::filesystem::path(argv[3]) / relative;
                    std::filesystem::create_directories(output.parent_path());
                    std::ofstream source(output);
                    source << generated.source;
                    if (!source)
                        throw std::runtime_error("Cannot write generated MSL");
                    MTLCompileOptions *options = [MTLCompileOptions new];
                    options.languageVersion = MTLLanguageVersion3_0;
                    options.fastMathEnabled = NO;
                    NSError *error = nil;
                    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:generated.source.c_str()] options:options error:&error];
                    if (!library)
                        throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal compilation failed");
                    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:generated.entry_point.c_str()]];
                    if (!function)
                        throw std::runtime_error("Metal entry point missing");
                    if (function.functionType != (generated.stage == shader::metal::Stage::Vertex ? MTLFunctionTypeVertex : MTLFunctionTypeFragment))
                        throw std::runtime_error("Wrong Metal function stage");
                    ++passed;
                    fetch += generated.uses_framebuffer_fetch;
                    std::cout << "PASS " << relative_path.string() << '\n';
                } catch (const std::exception &error) {
                    ++failed;
                    std::cerr << "FAIL " << relative_path.string() << " " << error.what() << '\n';
                }
            }
        }
        std::cout << "RESULT passed=" << passed << " failed=" << failed << " framebuffer_fetch=" << fetch << '\n';
        return failed || !passed ? 1 : 0;
    }
}
