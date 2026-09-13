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
	target_sources(shader PRIVATE src/msl_recompiler.cpp src/msl_gxp.cpp)
	target_link_libraries(shader PRIVATE spirv-cross-msl)

	option(VITA3K_BUILD_METAL_TESTS "Build native Metal shader validation tools" OFF)
	if(VITA3K_BUILD_METAL_TESTS)
		add_executable(metal-memory-analysis tests/metal_memory_analysis.cpp)
		target_link_libraries(metal-memory-analysis PRIVATE shader spirv-cross-core)
		add_test(NAME metal-memory-analysis COMMAND metal-memory-analysis)
		add_executable(metal-shader-validation tests/metal_shader_validation.mm tests/metal_shader_fixture.cpp)
		target_compile_options(metal-shader-validation PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
		target_link_libraries(metal-shader-validation PRIVATE shader "-framework Metal" "-framework Foundation")
	endif()
endif()

# Marshmallow Tracy linking
if(TRACY_ENABLE_ON_CORE_COMPONENTS)
	target_link_libraries(shader PRIVATE tracy)
endif()
