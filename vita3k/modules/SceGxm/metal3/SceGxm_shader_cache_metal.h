// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
struct MetalVertexProgramCacheKey {
    SceGxmRegisteredProgram vertex_program;
    uint32_t hash;
    Address metal_registration = 0;
};

struct MetalVertexCacheLess { bool operator()(const MetalVertexProgramCacheKey &, const MetalVertexProgramCacheKey &) const; };
using MetalVertexProgramCache = std::map<MetalVertexProgramCacheKey, Ptr<SceGxmVertexProgram>, MetalVertexCacheLess>;

struct MetalFragmentProgramCacheKey {
    SceGxmRegisteredProgram fragment_program;
    SceGxmBlendInfo blend_info;
    SceGxmOutputRegisterFormat metal_output_format = SCE_GXM_OUTPUT_REGISTER_FORMAT_DECLARED;
    SceGxmMultisampleMode metal_multisample_mode = SCE_GXM_MULTISAMPLE_NONE;
    Address metal_registration = 0;
};

struct MetalFragmentCacheLess { bool operator()(const MetalFragmentProgramCacheKey &, const MetalFragmentProgramCacheKey &) const; };
using MetalFragmentProgramCache = std::map<MetalFragmentProgramCacheKey, Ptr<SceGxmFragmentProgram>, MetalFragmentCacheLess>;


struct MetalShaderPatcherCache {
    MetalVertexProgramCache vertex_program_cache;
    MetalFragmentProgramCache fragment_program_cache;
};
