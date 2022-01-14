set(
	MGARD_BENCHMARK_CPP
	"tensor.cpp"
)

if(MGARD_ENABLE_UNSTRUCTURED AND MOAB_FOUND)
	list(APPEND MGARD_BENCHMARK_CPP "unstructured.cpp")
endif()

find_package(benchmark)

if(benchmark_FOUND)
	add_executable(mgard-benchmarks ${MGARD_BENCHMARK_CPP})
	target_link_libraries(mgard-benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main mgard-library)
	set_target_properties(mgard-benchmarks PROPERTIES OUTPUT_NAME benchmarks)
else()
	message(STATUS "Will not build benchmarks (Google Benchmark not found).")
endif()

set(benchmark_FOUND "${benchmark_FOUND}" PARENT_SCOPE)
