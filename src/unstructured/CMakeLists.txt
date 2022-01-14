set(
	MGARD_MOAB_CPP
	"measure.cpp"
	"LinearOperator.cpp"
	"pcg.cpp"
	"MassMatrix.cpp"
	"MeshLevel.cpp"
	"MeshHierarchy.cpp"
	"MeshRefiner.cpp"
	"UniformMeshRefiner.cpp"
	"UniformMeshHierarchy.cpp"
	"UniformRestriction.cpp"
	"norms.cpp"
	"estimators.cpp"
	"EnumeratedMeshRange.cpp"
	"indicators.cpp"
	"IndicatorInput.cpp"
)

find_package(MOAB)
#We run into this when building MOAB 5.3.0 with `./configure && make`.
if(MOAB_FOUND AND NOT TARGET MOAB)
	message(STATUS "MOAB found, but configuration file ")
	list(APPEND CMAKE_MESSAGE_INDENT "    ")
	message(STATUS "${MOAB_CONFIG}")
	list(POP_BACK CMAKE_MESSAGE_INDENT)
	message(STATUS "did not define a library target.")
	set(MOAB_FOUND FALSE)
endif()

if(MOAB_FOUND)
	target_sources(mgard-library PRIVATE ${MGARD_MOAB_CPP})
	target_link_libraries(mgard-library PUBLIC MOAB)
else()
	message(STATUS "Will not enable experimental unstructured mesh support (MOAB not found).")
endif()

set(MOAB_FOUND "${MOAB_FOUND}" PARENT_SCOPE)
