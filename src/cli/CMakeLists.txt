set(MGARD_CLI_INTERNAL_CPP "cli_internal_generated.cpp")

set(
	MGARD_CLI_CPP
	"arguments.cpp"
	"cmdline.cpp"
	"executable.cpp"
	"output.cpp"
	"cli_internal.cpp"
	"${MGARD_CLI_INTERNAL_CPP}"
)

pkg_search_module(TCLAP IMPORTED_TARGET GLOBAL tclap>=1.4)
if(NOT TCLAP_FOUND)
	find_package(TCLAP 1.4)
	if(TCLAP_FOUND)
		#Mimic target defined by `pkg_search_module`.
		add_library(PkgConfig::TCLAP ALIAS TCLAP::TCLAP)
	endif()
endif()

if(NOT TCLAP_FOUND)
	message(STATUS "Will not build executable (TCLAP not found).")
else()
	add_custom_command(
		OUTPUT
		"${MGARD_CLI_INTERNAL_CPP}"
		COMMAND "${PYTHON3}"
		ARGS
		"${MGARD_GENERATE_DIMENSION_SWITCH}"
		"--compress"
		"${MAXIMUM_DIMENSION}"
		"${MGARD_CLI_INTERNAL_CPP}"
	)

	add_executable(mgard-cli "${MGARD_CLI_CPP}")
	target_include_directories(mgard-cli PRIVATE "${PROJECT_SOURCE_DIR}/include")
	target_link_libraries(mgard-cli PRIVATE mgard-library PkgConfig::TCLAP)
	set_target_properties(mgard-cli PROPERTIES OUTPUT_NAME mgard)

	install(TARGETS mgard-cli RUNTIME)
	install(FILES man/man1/mgard.1 DESTINATION "${CMAKE_INSTALL_MANDIR}/man1")
endif()

set(TCLAP_FOUND "${TCLAP_FOUND}" PARENT_SCOPE)
