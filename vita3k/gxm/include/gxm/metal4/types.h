// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <gxm/types.h>
enum SceGxmYuvProfile : uint32_t {
    SCE_GXM_YUV_PROFILE_BT601_STANDARD = 0,
    SCE_GXM_YUV_PROFILE_BT709_STANDARD = 1,
    SCE_GXM_YUV_PROFILE_BT601_FULL_RANGE = 2,
    SCE_GXM_YUV_PROFILE_BT709_FULL_RANGE = 3,
};

inline constexpr uint32_t SCE_GXM_PROGRAM_FLAG_OUTPUT_IN_DECLARED_FORMAT = 1u << 2;
