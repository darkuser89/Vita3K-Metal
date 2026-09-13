// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <shader/msl_recompiler.h>
#include <cstddef>
shader::metal::Program compile_gxp_fixture(const std::vector<uint32_t> &words, size_t size,
    const std::string &name, const std::string &mode, float lod_bias = 0.f);
