// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later

#include <shader/msl_recompiler.h>
#include <shader/spirv_recompiler.h>

#include <stdexcept>

namespace shader::metal {

Program convert_gxp(const SceGxmProgram &program, const std::string &hash,
    const FeatureState &features, const Hints &hints, bool maskupdate) {
    if (!features.support_scaled_attribute_formats && !hints.attributes)
        throw std::invalid_argument("Metal GXM shaders require vertex attribute hints (an empty list is valid)");

    // Reuse the backend-independent USSE translation and Vulkan's descriptor/
    // depth conventions as an intermediate representation, then compile to MSL.
    // No Vulkan driver or shader module is involved in this path.
    auto intermediate = shader::convert_gxp(program, hash, features, Target::SpirVMetal, hints, maskupdate);
    return convert_spirv(intermediate.spirv);
}

} // namespace shader::metal
