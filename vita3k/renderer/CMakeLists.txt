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
	target_sources(renderer PRIVATE src/vulkan/metal_layer.mm src/metal/state.mm src/metal/overlay.mm src/metal/screen.mm src/metal/buffers.mm src/metal/textures.mm src/metal/polygon_clip.cpp)
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
