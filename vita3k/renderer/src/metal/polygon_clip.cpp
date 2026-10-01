// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/polygon_clip.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace renderer::metal {
namespace {

float plane_distance(const std::array<float, 4> &p, unsigned plane) {
    switch (plane) {
    case 0: return p[3] + p[0]; // left
    case 1: return p[3] - p[0]; // right
    case 2: return p[3] + p[1]; // bottom
    case 3: return p[3] - p[1]; // top
    case 4: return p[2];        // near
    default: return p[3] - p[2]; // far
    }
}

PolygonClipVertex between(const PolygonClipVertex &a, const PolygonClipVertex &b, float distance_a, float distance_b) {
    const float t = std::clamp(distance_a / (distance_a - distance_b), 0.0f, 1.0f);
    PolygonClipVertex result;
    for (unsigned i = 0; i < 4; ++i)
        result.position[i] = a.position[i] + t * (b.position[i] - a.position[i]);
    for (unsigned i = 0; i < 3; ++i)
        result.weights[i] = a.weights[i] + t * (b.weights[i] - a.weights[i]);
    return result;
}

void append_vertex(std::vector<PolygonClipVertex> &vertices, const PolygonClipVertex &vertex) {
    if (!vertices.empty() && vertices.back().position == vertex.position
        && vertices.back().weights == vertex.weights) return;
    vertices.push_back(vertex);
}

auto captured_clip_distances(const std::array<CapturedVertexOutputs, 3> &source) {
    std::array<std::array<float, 3>, shader::metal::CAPTURE_CLIP_COUNT> result{};
    for (size_t plane = 0; plane < result.size(); ++plane)
        for (size_t corner = 0; corner < 3; ++corner)
            result[plane][corner] = source[corner][shader::metal::CAPTURE_CLIP_SLOT + plane][0];
    return result;
}

} // namespace

ClippedPolygon clip_triangle_positions(const std::array<std::array<float, 4>, 3> &positions,
    std::span<const std::array<float, 3>> clip_distances, bool depth_clamp) {
    ClippedPolygon result;
    for (const auto &plane : clip_distances)
        if (!std::all_of(plane.begin(), plane.end(), [](float value) { return std::isfinite(value); })) return result;
    for (unsigned i = 0; i < 3; ++i) {
        if (!std::all_of(positions[i].begin(), positions[i].end(),
                [](float value) { return std::isfinite(value); })) return result;
        PolygonClipVertex vertex;
        vertex.position = positions[i];
        vertex.weights[i] = 1.0f;
        result.vertices.push_back(vertex);
    }
    for (size_t plane = 0; plane < 6 + clip_distances.size() && !result.vertices.empty(); ++plane) {
        // With native depth clamp the captured shader guards replace Z planes.
        if (depth_clamp && (plane == 4 || plane == 5)) continue;
        const auto distance = [&](const PolygonClipVertex &vertex) {
            if (plane < 6) return plane_distance(vertex.position, unsigned(plane));
            const auto &values = clip_distances[plane - 6];
            return vertex.weights[0] * values[0] + vertex.weights[1] * values[1] + vertex.weights[2] * values[2];
        };
        std::vector<PolygonClipVertex> clipped;
        clipped.reserve(result.vertices.size() + 1);
        auto previous = result.vertices.back();
        float previous_distance = distance(previous);
        bool previous_inside = previous_distance >= 0;
        for (const auto &current : result.vertices) {
            const float current_distance = distance(current);
            const bool current_inside = current_distance >= 0;
            if (previous_inside != current_inside)
                append_vertex(clipped, between(previous, current, previous_distance, current_distance));
            if (current_inside) append_vertex(clipped, current);
            previous = current;
            previous_distance = current_distance;
            previous_inside = current_inside;
        }
        if (clipped.size() > 1 && clipped.front().position == clipped.back().position
            && clipped.front().weights == clipped.back().weights) clipped.pop_back();
        result.vertices = std::move(clipped);
    }

    // Culling uses signed area after homogeneous division. The captured
    // positions already include the Y flip used by the native Metal pipeline.
    double twice_area = 0;
    if (result.vertices.size() < 3) return result;
    for (size_t i = 0; i < result.vertices.size(); ++i) {
        const auto &a = result.vertices[i].position;
        const auto &b = result.vertices[(i + 1) % result.vertices.size()].position;
        if (a[3] <= 0 || b[3] <= 0) return result;
        twice_area += (double(a[0]) / a[3]) * (double(b[1]) / b[3])
            - (double(b[0]) / b[3]) * (double(a[1]) / a[3]);
    }
    // Vulkan classifies zero-area triangles as back-facing too.
    result.face = twice_area > 0 ? PolygonFace::Front : PolygonFace::Back;
    return result;
}

std::vector<CapturedVertexOutputs> interpolate_polygon_outputs(
    const ClippedPolygon &polygon, const std::array<CapturedVertexOutputs, 3> &source) {
    std::vector<CapturedVertexOutputs> result;
    result.reserve(polygon.vertices.size());
    for (const auto &vertex : polygon.vertices) {
        CapturedVertexOutputs outputs{};
        outputs[0] = vertex.position;
        for (size_t slot = 1; slot < outputs.size(); ++slot) {
            for (size_t component = 0; component < 4; ++component) {
                for (size_t corner = 0; corner < 3; ++corner)
                    outputs[slot][component] += vertex.weights[corner] * source[corner][slot][component];
            }
        }
        // All returned vertices passed the clip planes. Avoid clipping them
        // again in the replay shader because interpolation rounded below zero.
        for (size_t plane = 0; plane < shader::metal::CAPTURE_CLIP_COUNT; ++plane) {
            auto &distance = outputs[shader::metal::CAPTURE_CLIP_SLOT + plane][0];
            distance = std::max(distance, 0.0f);
        }
        result.push_back(outputs);
    }
    return result;
}

std::optional<std::array<CapturedVertexOutputs, 2>> clip_line_outputs(
    const std::array<CapturedVertexOutputs, 2> &source, bool depth_clamp) {
    auto endpoints = source;
    for (const auto &endpoint : endpoints)
        if (!std::all_of(endpoint[0].begin(), endpoint[0].end(),
                [](float value) { return std::isfinite(value); })) return std::nullopt;
    for (const auto &endpoint : endpoints)
        for (size_t plane = 0; plane < shader::metal::CAPTURE_CLIP_COUNT; ++plane)
            if (!std::isfinite(endpoint[shader::metal::CAPTURE_CLIP_SLOT + plane][0])) return std::nullopt;
    for (unsigned plane = 0; plane < 6 + shader::metal::CAPTURE_CLIP_COUNT; ++plane) {
        if (depth_clamp && (plane == 4 || plane == 5)) continue;
        const auto distance = [&](const CapturedVertexOutputs &vertex) {
            return plane < 6 ? plane_distance(vertex[0], plane)
                : vertex[shader::metal::CAPTURE_CLIP_SLOT + plane - 6][0];
        };
        const float first = distance(endpoints[0]);
        const float second = distance(endpoints[1]);
        if (first < 0 && second < 0) return std::nullopt;
        if ((first < 0) == (second < 0)) continue;
        const float t = std::clamp(first / (first - second), 0.0f, 1.0f);
        CapturedVertexOutputs clipped{};
        for (size_t slot = 0; slot < clipped.size(); ++slot)
            for (size_t component = 0; component < 4; ++component)
                clipped[slot][component] = endpoints[0][slot][component]
                    + t * (endpoints[1][slot][component] - endpoints[0][slot][component]);
        endpoints[first < 0 ? 0 : 1] = clipped;
    }
    if (endpoints[0][0][3] <= 0 || endpoints[1][0][3] <= 0) return std::nullopt;
    for (auto &endpoint : endpoints)
        for (size_t plane = 0; plane < shader::metal::CAPTURE_CLIP_COUNT; ++plane) {
            auto &distance = endpoint[shader::metal::CAPTURE_CLIP_SLOT + plane][0];
            distance = std::max(distance, 0.0f);
        }
    return endpoints;
}

std::array<CapturedVertexOutputs, 6> expand_line_outputs(
    const std::array<CapturedVertexOutputs, 2> &endpoints,
    float viewport_width, float viewport_height, float native_width) {
    if (!std::isfinite(viewport_width) || !std::isfinite(viewport_height)
        || !std::isfinite(native_width) || viewport_width <= 0 || viewport_height <= 0
        || native_width <= 0 || endpoints[0][0][3] <= 0 || endpoints[1][0][3] <= 0)
        throw std::invalid_argument("Invalid wide line viewport or width");
    const auto &a = endpoints[0][0], &b = endpoints[1][0];
    const double dx = (double(b[0]) / b[3] - double(a[0]) / a[3]) * viewport_width;
    const double dy = (double(b[1]) / b[3] - double(a[1]) / a[3]) * viewport_height;
    const double length = std::hypot(dx, dy);
    std::array<CapturedVertexOutputs, 4> corners{};
    if (length <= 1e-12) {
        for (size_t i = 0; i < corners.size(); ++i) {
            corners[i] = endpoints[0];
            const float sx = (i & 1) ? 0.5f : -0.5f;
            const float sy = (i & 2) ? 0.5f : -0.5f;
            corners[i][0][0] += sx * native_width / viewport_width * a[3];
            corners[i][0][1] += sy * native_width / viewport_height * a[3];
        }
    } else {
        const float normal_x = float(-dy / length);
        const float normal_y = float(dx / length);
        for (size_t i = 0; i < corners.size(); ++i) {
            const auto &endpoint = endpoints[i / 2];
            corners[i] = endpoint;
            const float side = (i & 1) ? -1.0f : 1.0f;
            corners[i][0][0] += side * normal_x * native_width / viewport_width * endpoint[0][3];
            corners[i][0][1] += side * normal_y * native_width / viewport_height * endpoint[0][3];
        }
    }
    return {corners[0], corners[1], corners[2], corners[2], corners[1], corners[3]};
}

std::vector<RoutedPointPolygon> route_point_polygons(PolygonTopology topology,
    std::span<const uint32_t> indices, std::span<const CapturedVertexOutputs> captured,
    bool cull_front, bool cull_back) {
    std::vector<RoutedPointPolygon> result;
    const size_t triangle_count = topology == PolygonTopology::List
        ? indices.size() / 3 : indices.size() < 3 ? 0 : indices.size() - 2;
    result.reserve(triangle_count);
    for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
        std::array<uint32_t, 3> corners{};
        if (topology == PolygonTopology::List) {
            corners = {indices[triangle * 3], indices[triangle * 3 + 1], indices[triangle * 3 + 2]};
        } else if (topology == PolygonTopology::Fan) {
            corners = {indices[0], indices[triangle + 1], indices[triangle + 2]};
        } else if (triangle & 1) {
            corners = {indices[triangle + 1], indices[triangle], indices[triangle + 2]};
        } else {
            corners = {indices[triangle], indices[triangle + 1], indices[triangle + 2]};
        }
        std::array<CapturedVertexOutputs, 3> source{};
        std::array<std::array<float, 4>, 3> positions{};
        for (size_t corner = 0; corner < 3; ++corner) {
            if (corners[corner] >= captured.size())
                throw std::out_of_range("Point polygon index exceeds the captured vertex buffer");
            source[corner] = captured[corners[corner]];
            positions[corner] = source[corner][0];
        }
        const auto distances = captured_clip_distances(source);
        auto polygon = clip_triangle_positions(positions, distances, true);
        if (polygon.vertices.empty()
            || (polygon.face == PolygonFace::Front && cull_front)
            || (polygon.face == PolygonFace::Back && cull_back)) continue;
        RoutedPointPolygon routed;
        routed.primitive_index = uint32_t(triangle);
        routed.face = polygon.face;
        routed.points = interpolate_polygon_outputs(polygon, source);
        result.push_back(std::move(routed));
    }
    return result;
}

std::vector<RoutedWideLine> route_wide_lines(bool triangles, PolygonTopology topology,
    std::span<const uint32_t> indices, std::span<const CapturedVertexOutputs> captured,
    bool cull_front, bool cull_back, float viewport_width, float viewport_height,
    float front_width, float back_width) {
    std::vector<RoutedWideLine> result;
    const auto source_at = [&](uint32_t index) -> const CapturedVertexOutputs & {
        if (index >= captured.size()) throw std::out_of_range("Wide line index exceeds captured vertices");
        return captured[index];
    };
    const auto add_segment = [&](uint32_t primitive_index, PolygonFace face,
                                 const CapturedVertexOutputs &a, const CapturedVertexOutputs &b) {
        auto clipped = clip_line_outputs({a, b}, true);
        if (!clipped) return;
        RoutedWideLine line;
        line.primitive_index = primitive_index;
        line.face = face;
        line.quad = expand_line_outputs(*clipped, viewport_width, viewport_height,
            face == PolygonFace::Back ? back_width : front_width);
        result.push_back(std::move(line));
    };
    if (!triangles) {
        result.reserve(indices.size() / 2);
        for (size_t primitive = 0; primitive < indices.size() / 2; ++primitive)
            add_segment(uint32_t(primitive), PolygonFace::Front,
                source_at(indices[primitive * 2]), source_at(indices[primitive * 2 + 1]));
        return result;
    }
    const size_t count = topology == PolygonTopology::List
        ? indices.size() / 3 : indices.size() < 3 ? 0 : indices.size() - 2;
    result.reserve(count * 3);
    for (size_t primitive = 0; primitive < count; ++primitive) {
        std::array<uint32_t, 3> corners{};
        if (topology == PolygonTopology::List)
            corners = {indices[primitive * 3], indices[primitive * 3 + 1], indices[primitive * 3 + 2]};
        else if (topology == PolygonTopology::Fan)
            corners = {indices[0], indices[primitive + 1], indices[primitive + 2]};
        else if (primitive & 1)
            corners = {indices[primitive + 1], indices[primitive], indices[primitive + 2]};
        else
            corners = {indices[primitive], indices[primitive + 1], indices[primitive + 2]};
        std::array<CapturedVertexOutputs, 3> source{};
        std::array<std::array<float, 4>, 3> positions{};
        for (size_t corner = 0; corner < 3; ++corner) {
            source[corner] = source_at(corners[corner]);
            positions[corner] = source[corner][0];
        }
        const auto distances = captured_clip_distances(source);
        const auto polygon = clip_triangle_positions(positions, distances, true);
        if (polygon.vertices.size() < 2
            || (polygon.face == PolygonFace::Front && cull_front)
            || (polygon.face == PolygonFace::Back && cull_back)) continue;
        const auto outputs = interpolate_polygon_outputs(polygon, source);
        for (size_t edge = 0; edge < outputs.size(); ++edge)
            add_segment(uint32_t(primitive), polygon.face,
                outputs[edge], outputs[(edge + 1) % outputs.size()]);
    }
    return result;
}

} // namespace renderer::metal
