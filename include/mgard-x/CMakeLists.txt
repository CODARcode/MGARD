add_subdirectory(DataRefactoring)
add_subdirectory(CompressionLowLevel)
add_subdirectory(CompressionHighLevel)
add_subdirectory(DomainDecomposer)
add_subdirectory(Metadata)
add_subdirectory(Config)
add_subdirectory(Hierarchy)
add_subdirectory(Lossless)
add_subdirectory(MDRHighLevel)
add_subdirectory(Quantization)
add_subdirectory(RuntimeX)
add_subdirectory(UncertaintyCollector)

if (MGARD_ENABLE_EXTERNAL_COMPRESSOR)
	add_subdirectory(ExternalCompressionLowLevel)
endif()

set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)

