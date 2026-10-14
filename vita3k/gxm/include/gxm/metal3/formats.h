// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <gxm/types.h>
namespace gxm {
inline bool is_alpha_only_color_format(SceGxmColorFormat format) {
    return format == SCE_GXM_COLOR_FORMAT_U8_A || format == SCE_GXM_COLOR_FORMAT_S8_A;
}
inline bool is_green_only_color_format(SceGxmColorFormat format) {
    return format == SCE_GXM_COLOR_FORMAT_U16_G || format == SCE_GXM_COLOR_FORMAT_S16_G
        || format == SCE_GXM_COLOR_FORMAT_F16_G;
}
inline bool is_red_alpha_color_format(SceGxmColorFormat format) {
    return format == SCE_GXM_COLOR_FORMAT_U8U8_RA || format == SCE_GXM_COLOR_FORMAT_U8U8_AR
        || format == SCE_GXM_COLOR_FORMAT_S8S8_RA || format == SCE_GXM_COLOR_FORMAT_S8S8_AR;
}
}
