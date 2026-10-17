// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later

#include <shader/metal4/msl_recompiler.h>
#include <shader/metal4/spirv_recompiler.h>

#include <stdexcept>

namespace shader::metal {

Program convert_gxp(const SceGxmProgram &program, const std::string &hash,
    const FeatureState &features, const shader::metal3::Hints &hints, bool maskupdate) {
    if (!features.support_scaled_attribute_formats && !hints.attributes)
        throw std::invalid_argument("Metal GXM shaders require vertex attribute hints (an empty list is valid)");

    // The native translator owns Metal's ABI and resource conventions.
    auto intermediate = shader::metal3::convert_gxp(program, hash, features, shader::metal3::Target::SpirVMetal, hints, maskupdate);
    return convert_spirv(intermediate.spirv);
}

} // namespace shader::metal
