#These must be given relative to `PROJECT_SOURCE_DIR`. See the Python script.
set(
	MGARD_DOC_PAGES
	"README.md"
	"doc/MGARD-X.md"
	"doc/MDR-X.md"
	"doc/MGARD-GPU.md"
	"examples/README.md"
	"examples/linking/README.md"
	"examples/compression/README.md"
	"examples/decompression/README.md"
	"examples/mgard-x/HighLevelAPIs/README.md"
	"examples/mgard-x/LowLevelAPIs/README.md"
	"examples/mgard-x/MDR/README.md"
	"examples/mgard-x/MDR-X/README.md"
)

find_program(DOXYGEN NAMES "doxygen")
find_program(SED NAMES "sed")

if(SED STREQUAL "SED-NOTFOUND")
	message(STATUS "Will not build documentation (sed not found).")
elseif(PYTHON3 STREQUAL "PYTHON3-NOTFOUND")
	message(STATUS "Will not build documentation (Python 3 not found).")
elseif(DOXYGEN STREQUAL "DOXYGEN-NOTFOUND")
	message(STATUS "Will not build documentation (Doxygen not found).")
else()
	configure_file("${PROJECT_SOURCE_DIR}/scripts/modify_documentation_links.py.in" "scripts/modify_documentation_links.py")
	foreach(MGARD_DOC_PAGE ${MGARD_DOC_PAGES})
		set(MGARD_DOC_PAGE_FIXED "${CMAKE_CURRENT_BINARY_DIR}/${MGARD_DOC_PAGE}")
		add_custom_command(
			OUTPUT "${MGARD_DOC_PAGE_FIXED}"
			#Fix the relative links.
			COMMAND "${PYTHON3}"
			ARGS
			"${CMAKE_CURRENT_BINARY_DIR}/scripts/modify_documentation_links.py"
			"${MGARD_DOC_PAGE}"
			"${MGARD_DOC_PAGE_FIXED}"
			COMMAND "${SED}"
			ARGS
			#Get rid of the workflow status badges.
			"--expression" "'s/\\s*\\[!\\[[^]]\\+\\]\\[[^]]\\+\\]\\]\\[[^]]\\+\\]//g'"
			#Stop Doxygen from interpreting something like 'Huffman@GPU' – but not something like ' @ref' (added by the Python script above) – as a command.
			"--expression" "'s/\\([[:alnum:]]\\)@\\([[:alnum:]]\\)/\\1\\\\@\\2/g'"
			"--in-place"
			"${MGARD_DOC_PAGE_FIXED}"
			WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
			DEPENDS "${PROJECT_SOURCE_DIR}/${MGARD_DOC_PAGE}"
		)
	endforeach()
	list(TRANSFORM MGARD_DOC_PAGES PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")
	list(GET MGARD_DOC_PAGES 0 MGARD_DOC_MAINPAGE)
	list(JOIN MGARD_DOC_PAGES " " MGARD_DOC_PAGES_JOINED)
	configure_file(".doxygen.in" ".doxygen")
	add_custom_target(
		mgard-documentation
		ALL
		COMMAND "${DOXYGEN}" "${CMAKE_CURRENT_BINARY_DIR}/.doxygen"
		DEPENDS "${MGARD_DOC_PAGES}"
		WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	)
	install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doxygen/html/" TYPE DOC)
endif()
