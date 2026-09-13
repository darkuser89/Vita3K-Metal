add_library(
	renderer
	STATIC
	src/gl/attribute_formats.cpp
	src/gl/color_formats.cpp
	src/gl/compile_program.cpp
	src/gl/draw.cpp
	src/gl/fence.cpp
	src/gl/renderer.cpp
	src/gl/ring_buffer.cpp
	src/gl/overlay_renderer.cpp
	src/gl/screen_render.cpp
	src/gl/surface_cache.cpp
	src/gl/sync_state.cpp
	src/gl/texture_formats.cpp
	src/gl/texture.cpp
	src/gl/uniforms.cpp

	src/vulkan/allocator.cpp
	src/vulkan/context.cpp
	src/vulkan/creation.cpp
	src/vulkan/gxm_to_vulkan.cpp
	src/vulkan/pipeline_cache.cpp
	src/vulkan/renderer.cpp
	src/vulkan/scene.cpp
	src/vulkan/overlay_renderer.cpp
	src/vulkan/screen_filters.cpp
	src/vulkan/screen_renderer.cpp
	src/vulkan/surface_cache.cpp
	src/vulkan/sync_state.cpp
	src/vulkan/texture.cpp

	src/texture/cache.cpp
	src/texture/format.cpp
	src/texture/palette.cpp
	src/texture/pvrt-dec.cpp
	src/texture/replacement.cpp
	src/texture/yuv.cpp

	src/batch.cpp
	src/creation.cpp
	src/renderer.cpp
	src/scene.cpp
	src/shaders.cpp
	src/state_set.cpp
	src/sync.cpp
	src/transfer.cpp
)

target_include_directories(renderer PUBLIC include)
target_link_libraries(renderer PUBLIC display mem stb shader glutil threads config util vkutil overlay)
target_link_libraries(renderer PRIVATE dialog ddspp SDL3::SDL3 stb ffmpeg xxHash::xxhash concurrentqueue)

if(ANDROID)
	target_link_libraries(renderer PRIVATE android adrenotools)
endif()

if(APPLE)
    # Content-derived revision prevents stale translations after local compiler
    # edits as well as upstream recompiler or SPIRV-Cross updates.
    file(GLOB_RECURSE METAL_CACHE_INPUTS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/vita3k/shader/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/vita3k/shader/include/*.h"
        "${PROJECT_SOURCE_DIR}/vita3k/features/include/*.h"
        "${PROJECT_SOURCE_DIR}/vita3k/gxm/include/*.h"
        "${PROJECT_SOURCE_DIR}/vita3k/gxm/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/external/glslang/SPIRV/*.cpp"
        "${PROJECT_SOURCE_DIR}/external/glslang/SPIRV/*.h"
        "${PROJECT_SOURCE_DIR}/external/glslang/glslang/Include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/*.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/metal/*.h")
    file(GLOB METAL_CROSS_INPUTS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/external/SPIRV-Cross/*.cpp"
        "${PROJECT_SOURCE_DIR}/external/SPIRV-Cross/*.hpp"
        "${PROJECT_SOURCE_DIR}/external/SPIRV-Cross/*.h")
    list(APPEND METAL_CACHE_INPUTS ${METAL_CROSS_INPUTS})
    list(SORT METAL_CACHE_INPUTS)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${METAL_CACHE_INPUTS})
    set(METAL_CACHE_CONTENT_HASHES "")
    foreach(INPUT IN LISTS METAL_CACHE_INPUTS)
        file(SHA256 "${INPUT}" INPUT_HASH)
        string(APPEND METAL_CACHE_CONTENT_HASHES "${INPUT_HASH}\n")
    endforeach()
    string(SHA256 METAL_CACHE_REVISION "${METAL_CACHE_CONTENT_HASHES}")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cache-generated/metal_cache_revision.h"
        CONTENT "#pragma once\n#define METAL_CACHE_REVISION \"${METAL_CACHE_REVISION}\"\n")
    add_library(renderer-metal-runtime STATIC src/metal/device.mm src/metal/cache.mm)
    target_include_directories(renderer-metal-runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/cache-generated")
	target_include_directories(renderer-metal-runtime PUBLIC include)
	target_compile_options(renderer-metal-runtime PUBLIC "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
	target_link_libraries(renderer-metal-runtime PUBLIC shader "-framework Metal" "-framework Foundation")
	target_link_libraries(renderer PRIVATE renderer-metal-runtime)
	if(VITA3K_BUILD_METAL_TESTS)
        add_executable(metal-cache-validation tests/metal_cache_validation.mm)
        target_link_libraries(metal-cache-validation PRIVATE renderer-metal-runtime)
		add_executable(metal-render-validation tests/metal_render_validation.mm)
		add_executable(metal-gxm-validation tests/metal_gxm_validation.cpp)
		add_executable(metal-depth-alias-validation tests/metal_depth_alias_validation.cpp)
		add_executable(metal-depth-memory-validation tests/metal_depth_memory_validation.cpp)
        add_executable(metal-depth-store-validation tests/metal_depth_store_validation.mm)
        target_compile_options(metal-depth-store-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_link_libraries(metal-depth-store-validation PRIVATE renderer)
		add_executable(metal-fragment-disable-validation tests/metal_fragment_disable_validation.cpp)
		target_link_libraries(metal-fragment-disable-validation PRIVATE renderer)
		add_executable(metal-face-state-validation tests/metal_face_state_validation.cpp)
		target_link_libraries(metal-face-state-validation PRIVATE renderer)
		target_link_libraries(metal-depth-memory-validation PRIVATE renderer)
		target_link_libraries(metal-depth-alias-validation PRIVATE renderer)
		add_executable(metal-texture-validation tests/metal_texture_validation.cpp)
		add_executable(metal-swizzle-validation tests/metal_swizzle_validation.mm)
		add_executable(metal-surface-memory-validation tests/metal_surface_memory_validation.mm)
		add_executable(metal-cube-validation tests/metal_cube_validation.mm)
		add_executable(metal-visibility-validation tests/metal_visibility_validation.mm)
		add_executable(metal-msaa-validation tests/metal_msaa_validation.mm)
		add_executable(metal-msaa-scene-validation tests/metal_msaa_scene_validation.cpp)
		target_link_libraries(metal-msaa-scene-validation PRIVATE renderer)
		target_link_libraries(metal-msaa-validation PRIVATE renderer)
		target_compile_options(metal-visibility-validation PRIVATE -fobjc-arc)
		target_link_libraries(metal-visibility-validation PRIVATE renderer-metal-runtime)
		target_compile_options(metal-cube-validation PRIVATE -fobjc-arc)
		target_link_libraries(metal-cube-validation PRIVATE renderer)
		target_compile_options(metal-surface-memory-validation PRIVATE -fobjc-arc)
		target_link_libraries(metal-surface-memory-validation PRIVATE renderer)
		add_executable(metal-gbuffer-validation tests/metal_gbuffer_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-stealth-validation tests/metal_stealth_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-stealth-vertex-validation tests/metal_stealth_vertex_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-register-copy-validation tests/metal_register_copy_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-half-fetch-validation tests/metal_half_fetch_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		target_compile_options(metal-half-fetch-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-half-fetch-validation PRIVATE renderer-metal-runtime)
		target_compile_options(metal-register-copy-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-register-copy-validation PRIVATE renderer-metal-runtime)
		add_executable(metal-shadow-validation tests/metal_shadow_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-unit13-shadow-validation tests/metal_unit13_shadow_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-unit13-cube-validation tests/metal_unit13_cube_validation.mm ../shader/tests/metal_shader_fixture.cpp)
        add_executable(metal-mip-lod-validation tests/metal_mip_lod_validation.mm ../shader/tests/metal_shader_fixture.cpp)
        target_compile_options(metal-mip-lod-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_link_libraries(metal-mip-lod-validation PRIVATE renderer)
        add_executable(metal-sample-half-validation tests/metal_sample_half_validation.mm ../shader/tests/metal_shader_fixture.cpp)
        target_compile_options(metal-sample-half-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_link_libraries(metal-sample-half-validation PRIVATE renderer)
        add_executable(metal-sly-material-validation tests/metal_sly_material_validation.mm ../shader/tests/metal_shader_fixture.cpp)
        target_compile_options(metal-sly-material-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_link_libraries(metal-sly-material-validation PRIVATE renderer)
        add_executable(metal-sao-material-validation tests/metal_sao_material_validation.mm ../shader/tests/metal_shader_fixture.cpp)
        target_compile_options(metal-sao-material-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_link_libraries(metal-sao-material-validation PRIVATE renderer)
        add_executable(metal-point-validation tests/metal_point_validation.cpp)
        target_link_libraries(metal-point-validation PRIVATE renderer)
        add_executable(metal-depth-reload-validation tests/metal_depth_reload_validation.cpp)
        target_link_libraries(metal-depth-reload-validation PRIVATE renderer)
        add_executable(metal-projected-validation tests/metal_projected_validation.mm ../shader/tests/metal_shader_fixture.cpp)
        target_compile_options(metal-projected-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_link_libraries(metal-projected-validation PRIVATE renderer)
		add_executable(metal-surface-format-validation tests/metal_surface_format_validation.mm)
		target_compile_options(metal-surface-format-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-surface-format-validation PRIVATE renderer)
		add_executable(metal-cube-alias-validation tests/metal_cube_alias_validation.cpp)
		add_executable(metal-cube-capture-validation tests/metal_cube_capture_validation.mm)
		target_compile_options(metal-cube-capture-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-cube-capture-validation PRIVATE renderer)
		target_link_libraries(metal-cube-alias-validation PRIVATE renderer)
		target_compile_options(metal-unit13-cube-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-unit13-cube-validation PRIVATE renderer)
		target_compile_options(metal-unit13-shadow-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-unit13-shadow-validation PRIVATE renderer)
		add_executable(metal-lod-validation tests/metal_lod_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-lod-gxm-validation tests/metal_lod_gxm_validation.cpp)
		target_link_libraries(metal-lod-gxm-validation PRIVATE renderer)
		target_compile_options(metal-lod-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-lod-validation PRIVATE renderer)
		target_compile_options(metal-shadow-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-shadow-validation PRIVATE renderer)
		target_compile_options(metal-stealth-vertex-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-stealth-vertex-validation PRIVATE renderer-metal-runtime)
		add_executable(metal-native-color-validation tests/metal_native_color_validation.cpp)
		target_link_libraries(metal-native-color-validation PRIVATE renderer)
		target_compile_options(metal-stealth-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-stealth-validation PRIVATE renderer-metal-runtime)
		add_executable(metal-vertex-replay tests/metal_vertex_replay.mm)
		target_compile_options(metal-vertex-replay PRIVATE -fobjc-arc)
		target_link_libraries(metal-vertex-replay PRIVATE renderer-metal-runtime)
		target_compile_options(metal-gbuffer-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-gbuffer-validation PRIVATE renderer)
		target_compile_options(metal-swizzle-validation PRIVATE -fobjc-arc)
		target_link_libraries(metal-swizzle-validation PRIVATE renderer)
		target_link_libraries(metal-texture-validation PRIVATE renderer)
		add_executable(metal-overlay-validation tests/metal_overlay_validation.mm)
		add_executable(metal-buffer-validation tests/metal_buffer_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-thread-buffer-validation tests/metal_thread_buffer_validation.mm ../shader/tests/metal_shader_fixture.cpp)
		add_executable(metal-interface-validation tests/metal_interface_validation.mm)
		add_executable(metal-doa-skinning-validation tests/metal_doa_skinning_validation.mm)
		target_compile_options(metal-doa-skinning-validation PRIVATE -fobjc-arc)
		target_link_libraries(metal-doa-skinning-validation PRIVATE renderer)
		target_compile_options(metal-interface-validation PRIVATE -fobjc-arc)
		target_link_libraries(metal-interface-validation PRIVATE renderer-metal-runtime shader)
		target_link_libraries(metal-thread-buffer-validation PRIVATE renderer-metal-runtime shader)
		target_compile_options(metal-buffer-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-buffer-validation PRIVATE renderer)
		target_compile_options(metal-overlay-validation PRIVATE -fobjc-arc)
		target_link_libraries(metal-overlay-validation PRIVATE renderer dialog)
		target_link_libraries(metal-gxm-validation PRIVATE renderer)
		target_link_libraries(metal-render-validation PRIVATE renderer-metal-runtime)
		add_test(NAME metal-render-smoke COMMAND metal-render-validation)
		set_tests_properties(metal-render-smoke PROPERTIES LABELS "metal;gpu")
	endif()
	target_sources(renderer PRIVATE src/vulkan/metal_layer.mm src/metal/state.mm src/metal/overlay.mm src/metal/screen.mm src/metal/buffers.mm src/metal/textures.mm)
	target_link_libraries(renderer PRIVATE "-framework QuartzCore" "-framework Cocoa")
endif()

if(X11_FOUND)
	target_link_libraries(renderer PRIVATE ${X11_LIBRARIES})
	target_include_directories(renderer PRIVATE ${X11_INCLUDE_DIR})
endif()

if(WAYLAND_FOUND)
	target_link_libraries(renderer PRIVATE ${WAYLAND_LIBRARIES})
	target_include_directories(renderer PRIVATE ${WAYLAND_INCLUDE_DIRS})
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	# silence all vma warnings
	target_compile_options(renderer PRIVATE "-Wno-nullability-completeness")
endif()

# Marshmallow Tracy linking
if(TRACY_ENABLE_ON_CORE_COMPONENTS)
	target_link_libraries(renderer PRIVATE tracy)
endif()
