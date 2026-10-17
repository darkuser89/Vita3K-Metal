// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <shader/metal4/render_uniforms.h>

// The private translator and runtime share one ABI definition. Keep the
// translator's established names without maintaining a second layout.
namespace shader::metal3 {
using shader::metal::RenderVertUniformBlock;
using shader::metal::RenderFragUniformBlock;
using shader::metal::VertUniformFieldId;
using shader::metal::FragUniformFieldId;
using enum shader::metal::VertUniformFieldId;
using enum shader::metal::FragUniformFieldId;
using shader::metal::UniformBlockExtended;
using shader::metal::RenderVertUniformBlockExtended;
using shader::metal::RenderFragUniformBlockExtended;
}
