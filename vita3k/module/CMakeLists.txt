add_library(
	module
	STATIC
	src/load_module.cpp
	src/call_result.cpp
)

target_include_directories(module PUBLIC include)
target_link_libraries(module PUBLIC config cpu emuenv kernel util)
if(TRACY_ENABLE_ON_CORE_COMPONENTS)
	target_link_libraries(module PUBLIC tracy)
endif()
