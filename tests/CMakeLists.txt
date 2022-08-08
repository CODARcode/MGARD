set(
	MGARD_TESTS_CPP
	"src/testing_utilities.cpp"
	"src/test_utilities.cpp"
	"src/test_blas.cpp"
	"src/test_shuffle.cpp"
	"src/test_TensorMeshHierarchy.cpp"
	"src/test_TensorMeshHierarchyIteration.cpp"
	"src/test_TensorLinearOperator.cpp"
	"src/test_TensorMassMatrix.cpp"
	"src/test_TensorProlongation.cpp"
	"src/test_TensorRestriction.cpp"
	"src/test_LinearQuantizer.cpp"
	"src/test_TensorMultilevelCoefficientQuantizer.cpp"
	"src/test_TensorNorms.cpp"
	"src/test_TensorQuantityOfInterest.cpp"
	"src/test_compress.cpp"
	"src/test_decompose.cpp"
	"src/test_format.cpp"
	"src/test_quantize.cpp"
	"src/lossless_regression.cpp"
	"src/test_lossless.cpp"
	"src/test_CompressedDataset.cpp"
	"src/huffman_regression.cpp"
	"src/test_huffman.cpp"
)

if(MGARD_ENABLE_UNSTRUCTURED AND MOAB_FOUND)
	list(
		APPEND
		MGARD_TESTS_CPP
		"src/unstructured/test_estimators.cpp"
		"src/unstructured/test_indicators.cpp"
		"src/unstructured/test_norms.cpp"
		"src/unstructured/test_data.cpp"
		"src/unstructured/test_pcg.cpp"
		"src/unstructured/test_measure.cpp"
		"src/unstructured/test_MeshLevel.cpp"
		"src/unstructured/test_MeshHierarchy.cpp"
		"src/unstructured/test_MeshRefiner.cpp"
		"src/unstructured/test_LinearOperator.cpp"
		"src/unstructured/test_MassMatrix.cpp"
		"src/unstructured/test_UniformMeshHierarchy.cpp"
		"src/unstructured/test_UniformMeshRefiner.cpp"
		"src/unstructured/test_UniformEdgeFamilies.cpp"
		"src/unstructured/test_UniformRestriction.cpp"
		"src/unstructured/test_EnumeratedMeshRange.cpp"
		"src/unstructured/test_SituatedCoefficientRange.cpp"
		"src/unstructured/test_IndicatorInput.cpp"
		"src/unstructured/test_MultilevelCoefficientQuantizer.cpp"
	)
endif()

find_package(Catch2 3.0.0)

if(Catch2_FOUND)
	configure_file("include/testing_paths.hpp.in" "include/testing_paths.hpp")
	add_executable(mgard-tests ${MGARD_TESTS_CPP})
	target_include_directories(mgard-tests PRIVATE "include" "${CMAKE_CURRENT_BINARY_DIR}/include")
	target_link_libraries(mgard-tests PRIVATE mgard-library Catch2::Catch2WithMain)
	set_target_properties(mgard-tests PROPERTIES OUTPUT_NAME tests)
else()
	message(STATUS "Will not build tests (Catch2 not found).")
endif()

set(Catch2_FOUND "${Catch2_FOUND}" PARENT_SCOPE)
