// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include <renderer/shaders.h>

#include <renderer/vulkan/state.h>

#include <gxm/types.h>
#include <renderer/state.h>
#include <renderer/types.h>
#include <shader/metal4/spirv_recompiler.h>
#include <util/fs.h>
#include <util/log.h>

#include <string>
#include <vector>

#include <renderer/metal4/integration.h>

namespace renderer {
bool get_shaders_cache_hashs_metal3(State &renderer) {
    const std::string hash_file_name = fmt::format("hashs-{}.dat", (renderer.current_backend == Backend::OpenGL) ? "gl" : (renderer.current_backend == Backend::Metal ? "metal" : "vk"));
    const auto hash_path = renderer.shaders_path / hash_file_name;
    std::ifstream shaders_hashs(hash_path, std::ios::in | std::ios::binary);
    if (!shaders_hashs.is_open())
        return false;

    renderer.shaders_cache_hashs.clear();
    constexpr uintmax_t header_bytes = sizeof(size_t) + 2 * sizeof(uint32_t);
    constexpr uintmax_t pair_bytes = 2 * sizeof(Sha256Hash);
    constexpr size_t max_hash_pairs = 1'000'000;
    std::error_code size_error;
    const auto file_bytes = fs::file_size(hash_path, size_error);
    if (size_error || file_bytes < header_bytes) {
        LOG_WARN("Shader hash cache has an incomplete header: {}", hash_path.string());
        return false;
    }
    // Read size of hashes list
    size_t size = 0;
    if (!shaders_hashs.read(reinterpret_cast<char *>(&size), sizeof(size))
        || size > max_hash_pairs || size != (file_bytes - header_bytes) / pair_bytes
        || file_bytes - header_bytes != uintmax_t(size) * pair_bytes) {
        LOG_WARN("Shader hash cache has an invalid entry count: {}", hash_path.string());
        return false;
    }

    // Check version of cache
    uint32_t versionInFile = 0;
    uint32_t features_mask = 0;
    if (!shaders_hashs.read(reinterpret_cast<char *>(&versionInFile), sizeof(versionInFile))
        || !shaders_hashs.read(reinterpret_cast<char *>(&features_mask), sizeof(features_mask))) {
        LOG_WARN("Shader hash cache has an incomplete version header: {}", hash_path.string());
        return false;
    }
    if (versionInFile != shader::metal3::CURRENT_VERSION || features_mask != renderer.get_features_mask()) {
        shaders_hashs.close();
        if (renderer.current_backend == Backend::Metal) {
            // Metal opens its device-scoped native cache in set_app before the
            // launch path reads this shared index. Its complete variant keys
            // already include the feature mask, so discard only the stale
            // hash list and leave the live native cache directory intact.
            fs::remove(hash_path);
        } else {
            fs::remove_all(renderer.shaders_path);
            fs::remove_all(renderer.shaders_log_path);
        }
        if (versionInFile != shader::metal3::CURRENT_VERSION)
            LOG_WARN("Current version of shader hash index: {}, is outdated.", versionInFile);
        else if (renderer.current_backend == Backend::Metal)
            LOG_WARN("Incompatible Metal shader features enabled; recreating shader hash index");
        else
            LOG_WARN("Incompatible GPU features enabled, recreating shader cache");
        return false;
    }

    if (renderer.current_backend == Backend::Vulkan) {
        // Read the pipeline cache
        dynamic_cast<vulkan::VKState &>(renderer).pipeline_cache.read_pipeline_cache();
    }

    // Read Hashs info value
    for (size_t a = 0; a < size; a++) {
        auto read = [&shaders_hashs]() {
            Sha256Hash hash;
            shaders_hashs.read(reinterpret_cast<char *>(hash.data()), sizeof(Sha256Hash));
            return hash;
        };

        ShadersHash hash;
        hash.frag = read();
        hash.vert = read();

        if (!shaders_hashs) {
            renderer.shaders_cache_hashs.clear();
            LOG_WARN("Shader hash cache ended before all entries were read: {}", hash_path.string());
            return false;
        }
        renderer.shaders_cache_hashs.push_back({ hash.frag, hash.vert });
    }

    shaders_hashs.close();

    return !renderer.shaders_cache_hashs.empty();
    }

void save_shaders_cache_hashs_metal3(State &renderer, std::vector<ShadersHash> &shaders_cache_hashs) {
    fs::create_directories(renderer.shaders_path);
    std::string hash_file_name = fmt::format("hashs-{}.dat", (renderer.current_backend == Backend::OpenGL) ? "gl" : (renderer.current_backend == Backend::Metal ? "metal" : "vk"));
    std::ofstream shaders_hashs(renderer.shaders_path / hash_file_name, std::ios::out | std::ios::binary);

    if (shaders_hashs.is_open()) {
        // Write Size of shaders cache hashes list
        const auto size = shaders_cache_hashs.size();
        shaders_hashs.write((const char *)&size, sizeof(size));

        // Write version of cache
        const uint32_t versionInFile = shader::metal3::CURRENT_VERSION;
        shaders_hashs.write((const char *)&versionInFile, sizeof(uint32_t));
        const uint32_t features_mask = renderer.get_features_mask();
        shaders_hashs.write((const char *)&features_mask, sizeof(uint32_t));

        // Write shader hash list
        for (const auto &hash : shaders_cache_hashs) {
            auto write = [&shaders_hashs](const Sha256Hash &hash) {
                shaders_hashs.write(reinterpret_cast<const char *>(hash.data()), sizeof(Sha256Hash));
            };

            write(hash.frag);
            write(hash.vert);
        }
        shaders_hashs.close();
    }
    }
}
