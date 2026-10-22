// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cmath>
#include <string_view>

namespace renderer::metal {
// RCAS can replace the final sample only at a one-to-one pixel mapping.
// Fractional origins/extents need the existing bilinear sample AFTER RCAS.
// Keep position subtraction exact in FP32, including negative crop origins.
inline bool fsr_pixel_aligned(double x, double y, double width, double height) {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(width) && std::isfinite(height)
        && std::abs(x) <= 1048576 && std::abs(y) <= 1048576
        && width >= 1 && width <= 32767 && height >= 1 && height <= 32767
        && std::trunc(x) == x && std::trunc(y) == y
        && std::trunc(width) == width && std::trunc(height) == height;
}

// This wrapper calls the ORIGINAL helpers from SPIRV-Cross's RCAS translation.
// It relies on this module's CurrFilter being just FsrRcasH + half4(c, 1).
// Unknown upstream shaders must keep the compute/presentation fallback until
// their entry-point semantics have been checked; never silently bypass changes.
inline constexpr std::string_view fsr_rcas_module_sha256 =
    "c28d0e2d4da17e207b055981dbd2731cebf3aa2003e778aa114c4f71d614d8af";

inline constexpr std::string_view fsr_rcas_fragment = R"(
fragment float4 screen_rcas(float4 position [[position]],
    constant float4& presentation [[buffer(0)]],
    texture2d<float> image [[texture(0)]], sampler s [[sampler(0)]]) {
    uint2 pixel = uint2(position.xy - presentation.xy);
    float sharpening = presentation.z;
    uint4 constants;
    FsrRcasCon(constants, sharpening);
    half r, g, b;
    FsrRcasH(r, g, b, pixel, constants, image, s);
    return float4(half4(r, g, b, half(1.0)));
}
)";
}
