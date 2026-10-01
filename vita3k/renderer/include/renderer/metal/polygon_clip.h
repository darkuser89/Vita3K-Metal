// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <shader/metal_capture.h>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace renderer::metal {

// A clipped vertex is a linear combination of the original post-vertex
// outputs. The weights also reconstruct guest varyings at a new clip edge.
struct PolygonClipVertex {
    std::array<float, 4> position{};
    std::array<float, 3> weights{};
};

enum class PolygonFace { Front, Back, Degenerate };

struct ClippedPolygon {
    std::vector<PolygonClipVertex> vertices;
    PolygonFace face = PolygonFace::Degenerate;
};

using CapturedVertexOutputs = std::array<std::array<float, 4>, shader::metal::CAPTURE_OUTPUT_SLOT_COUNT>;

// Positions already contain the renderer's viewport flip. The default clips
// -w <= x,y <= w and 0 <= z <= w; depth-clamped draws instead supply the
// shader's eye/far guards among clip_distances and leave Z for the rasterizer.
ClippedPolygon clip_triangle_positions(const std::array<std::array<float, 4>, 3> &positions,
    std::span<const std::array<float, 3>> clip_distances = {}, bool depth_clamp = false);

// Reconstruct values at the clipped vertices from the three captured guest
// vertex outputs. Slot zero uses the clipped position directly.
std::vector<CapturedVertexOutputs> interpolate_polygon_outputs(
    const ClippedPolygon &polygon, const std::array<CapturedVertexOutputs, 3> &source);

// Clip an already transformed line while interpolating every GXP output.
// The returned endpoints can be replayed as a wide screen-space quad.
std::optional<std::array<CapturedVertexOutputs, 2>> clip_line_outputs(
    const std::array<CapturedVertexOutputs, 2> &source, bool depth_clamp = false);
std::array<CapturedVertexOutputs, 6> expand_line_outputs(
    const std::array<CapturedVertexOutputs, 2> &endpoints,
    float viewport_width, float viewport_height, float native_width);

enum class PolygonTopology { List, Strip, Fan };

struct RoutedPointPolygon {
    uint32_t primitive_index = 0;
    PolygonFace face = PolygonFace::Degenerate;
    std::vector<CapturedVertexOutputs> points;
};

// Preserve primitive order while clipping, classifying, and expanding each
// triangle into the points Vulkan's polygon point mode would rasterize.
std::vector<RoutedPointPolygon> route_point_polygons(PolygonTopology topology,
    std::span<const uint32_t> indices, std::span<const CapturedVertexOutputs> captured,
    bool cull_front, bool cull_back);

struct RoutedWideLine {
    uint32_t primitive_index = 0;
    PolygonFace face = PolygonFace::Front;
    std::array<CapturedVertexOutputs, 6> quad{};
};

// Keep guest primitive order for native line lists and the visible edges of
// triangle lists, strips, and fans after homogeneous clipping and culling.
std::vector<RoutedWideLine> route_wide_lines(bool triangles, PolygonTopology topology,
    std::span<const uint32_t> indices, std::span<const CapturedVertexOutputs> captured,
    bool cull_front, bool cull_back, float viewport_width, float viewport_height,
    float front_width, float back_width);

} // namespace renderer::metal
