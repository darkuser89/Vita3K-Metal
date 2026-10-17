add_library(
	shader
	STATIC
	src/translator/alu.cpp
	src/translator/ialu.cpp
	src/translator/branch_cond.cpp
	src/translator/data.cpp
	src/translator/special.cpp
	src/translator/illegal.cpp
	src/translator/texture.cpp
	src/translator/utils.cpp
	src/gxp_parser.cpp
	src/usse_disasm.cpp
	src/usse_program_analyzer.cpp
	src/usse_decode_helpers.cpp
	src/usse_translator_entry.cpp
	src/usse_utilities.cpp
	src/spirv_recompiler.cpp
)

target_include_directories(shader PUBLIC include)
target_link_libraries(shader PUBLIC features gxm util)
target_link_libraries(shader PRIVATE SPIRV spirv-cross-glsl)

if(APPLE)
	# Native Metal translation has a separate namespace and ABI. The shared
	# compiler remains the official implementation used by Vulkan and OpenGL.
	add_library(shader-metal4 STATIC
		src/metal4/gxp_parser_metal.cpp
		src/metal4/spirv_recompiler_metal.cpp
		src/metal4/usse_disasm_metal.cpp
		src/metal4/usse_program_analyzer_metal.cpp
		src/metal4/usse_decode_helpers_metal.cpp
		src/metal4/usse_translator_entry_metal.cpp
		src/metal4/usse_utilities_metal.cpp
		src/metal4/translator/alu_metal.cpp
		src/metal4/translator/ialu_metal.cpp
		src/metal4/translator/branch_cond_metal.cpp
		src/metal4/translator/data_metal.cpp
		src/metal4/translator/special_metal.cpp
		src/metal4/translator/illegal_metal.cpp
		src/metal4/translator/texture_metal.cpp
		src/metal4/translator/utils_metal.cpp)
	target_include_directories(shader-metal4 PUBLIC include)
	target_link_libraries(shader-metal4 PUBLIC features gxm util)
	target_link_libraries(shader-metal4 PRIVATE SPIRV spirv-cross-glsl)
	if(TRACY_ENABLE_ON_CORE_COMPONENTS)
		target_link_libraries(shader-metal4 PRIVATE tracy)
	endif()
	target_sources(shader-metal4 PRIVATE src/metal4/msl_recompiler_metal.cpp src/metal4/msl_gxp_metal.cpp)
	target_link_libraries(shader-metal4 PRIVATE spirv-cross-msl)
endif()

# Marshmallow Tracy linking
if(TRACY_ENABLE_ON_CORE_COMPONENTS)
	target_link_libraries(shader PRIVATE tracy)
endif()
