CXXFLAGS += -g -fno-omit-frame-pointer

ifdef DEBUG
# Compiling with sanitizers during development is simply essential:
	CXXFLAGS += -fsanitize=address -fsanitize=undefined
	LDFLAGS += -fsanitize=address -fsanitize=undefined
else
	CXXFLAGS += -O3 -march=native -ffast-math -fno-finite-math-only
endif

ARFLAGS := -rcs

MKDIR := mkdir --parents
RMDIR := rmdir

DIR_INC := include
DIR_SRC := src
DIR_OBJ := obj
DIR_BIN := bin
DIR_LIB := lib
DIR_DOC := doc

CPPFLAGS += $(foreach DIR,$(DIRECTORIES_INCLUDE),-I$(DIR))

#Below we'll set `DIRECTORIES_INCLUDE` to one of these.
structured@DIRECTORIES_INCLUDE := $(DIR_INC)
unstructured@DIRECTORIES_INCLUDE := $(DIR_INC) $(HOME)/.local/include
benchmarks@DIRECTORIES_INCLUDE := $(DIR_INC) $(HOME)/.local/include

structured@CXXFLAGS := -std=c++11 -fPIC
unstructured@CXXFLAGS := -std=c++17 -Wfatal-errors -Wall -Wextra -fPIC

structured@LDFLAGS :=
unstructured@LDFLAGS := -L$(HOME)/.local/lib
benchmarks@LDFLAGS := $(unstructured@LDFLAGS)

structured@LDLIBS := -lz -ldl
unstructured@LDLIBS := -lMOAB -llapack -lz -lstdc++fs -lm
benchmarks@LDLIBS := -lbenchmark -lbenchmark_main -pthread $(structured@LDLIBS) $(unstructured@LDLIBS)

dirty@FILES =
dirty@DIRECTORIES =

structured@MGARD_STEMS_TESTED := interpolation mgard
structured@MGARD_STEMS_UNTESTED := mgard_api mgard_nuni mgard_compress mgard_mesh
structured@MGARD_STEMS = $(structured@MGARD_STEMS_TESTED) $(structured@MGARD_STEMS_UNTESTED)
structured@TEST_STEMS := mgard_test
structured@STEMS = $(structured@MGARD_STEMS) $(structured@TEST_STEMS)

#Tested but not compiled. `$(STEM).hpp` exists, `$(STEM).tpp` might exist, and `$(STEM).cpp` does not exist.
unstructured@HEADER_ONLY := blas utilities data UniformEdgeFamilies LinearQuantizer SituatedCoefficientRange MultilevelCoefficientQuantizer
unstructured@MGARD_STEMS := measure LinearOperator pcg MassMatrix MeshLevel MeshHierarchy MeshRefiner UniformMeshRefiner UniformMeshHierarchy UniformRestriction norms estimators EnumeratedMeshRange indicators IndicatorInput
unstructured@STEMS = $(unstructured@MGARD_STEMS)

tests@DIR_ROOT := tests
tests@DIR_INC := $(tests@DIR_ROOT)/$(DIR_INC)
tests@DIR_SRC := $(tests@DIR_ROOT)/$(DIR_SRC)
tests@STEMS := $(foreach STEM,$(structured@MGARD_STEMS_TESTED) $(unstructured@STEMS) $(unstructured@HEADER_ONLY),test_$(STEM)) testing_utilities main
tests@EXECUTABLE := $(DIR_BIN)/tests

tests@SCRIPT := $(DIR_BIN)/mgard_test
structured@LIB := $(DIR_LIB)/libmgard.a

benchmarks@DIR_SRC := benchmark
benchmarks@STEM := bench
benchmarks@EXECUTABLE := $(DIR_BIN)/speed

.PHONY: all
all: $(structured@LIB) $(tests@SCRIPT)

define create-directory
$1:
	$(MKDIR) $$@

dirty@DIRECTORIES += $1
endef

#Need the trailing '/' to match the output of `dir`.
$(foreach DIR,$(DIR_OBJ) $(DIR_BIN) $(DIR_LIB),$(eval $(call create-directory,$(DIR)/)))

define stem-to-source
$(DIR_SRC)/$1.cpp
endef

define tests@stem-to-source
$(tests@DIR_SRC)/$1.cpp
endef

define benchmarks@stem-to-source
$(benchmarks@DIR_SRC)/$1.cpp
endef

define stem-to-object
$(DIR_OBJ)/$1.o
endef

define compile-cpp
$2: $1 | $$(dir $2)
	$$(COMPILE.cpp) $$(OUTPUT_OPTION) $$<

dirty@FILES += $2
endef

define link-cpp
$2: $1 | $$(dir $2)
	$$(LINK.cpp) $$^ $$(LDLIBS) $$(OUTPUT_OPTION)

dirty@FILES += $2
endef

define archive-cpp
$2: $1 | $$(dir $2)
	$$(AR) $$(ARFLAGS) $$@ $$^

dirty@FILES += $2
endef

$(eval $(foreach STEM,$(structured@STEMS),$(call stem-to-object,$(STEM))): DIRECTORIES_INCLUDE = $(structured@DIRECTORIES_INCLUDE))
$(eval $(foreach STEM,$(unstructured@STEMS),$(call stem-to-object,$(STEM))): DIRECTORIES_INCLUDE = $(unstructured@DIRECTORIES_INCLUDE))
$(eval $(foreach STEM,$(tests@STEMS),$(call stem-to-object,$(STEM))): DIRECTORIES_INCLUDE += $(unstructured@DIRECTORIES_INCLUDE) $(tests@DIR_INC))

$(eval $(foreach STEM,$(structured@STEMS),$(call stem-to-object,$(STEM))): CXXFLAGS += $(structured@CXXFLAGS))
$(eval $(foreach STEM,$(unstructured@STEMS),$(call stem-to-object,$(STEM))): CXXFLAGS += $(unstructured@CXXFLAGS))
$(eval $(foreach STEM,$(tests@STEMS),$(call stem-to-object,$(STEM))): CXXFLAGS += $(unstructured@CXXFLAGS))

$(foreach STEM,$(structured@STEMS),$(eval $(call compile-cpp,$(call stem-to-source,$(STEM)),$(call stem-to-object,$(STEM)))))
$(foreach STEM,$(unstructured@STEMS),$(eval $(call compile-cpp,$(call stem-to-source,$(STEM)),$(call stem-to-object,$(STEM)))))
$(foreach STEM,$(tests@STEMS),$(eval $(call compile-cpp,$(call tests@stem-to-source,$(STEM)),$(call stem-to-object,$(STEM)))))

$(eval $(call archive-cpp,$(foreach STEM,$(structured@MGARD_STEMS),$(call stem-to-object,$(STEM))),$(structured@LIB)))

.PHONY: check
check: $(tests@EXECUTABLE)
	./$<

$(tests@SCRIPT): LDFLAGS += $(structured@LDFLAGS)
$(tests@SCRIPT): LDLIBS += $(structured@LDLIBS)
$(eval $(call link-cpp,$(foreach STEM,$(structured@TEST_STEMS),$(call stem-to-object,$(STEM))) $(structured@LIB),$(tests@SCRIPT)))

$(tests@EXECUTABLE): LDFLAGS += $(unstructured@LDFLAGS)
$(tests@EXECUTABLE): LDLIBS += $(unstructured@LDLIBS)
#The object files obtained from `$(structured@MGARD_STEMS_TESTED)` are already included in `$(structured@LIB)`.
$(eval $(call link-cpp,$(foreach STEM,$(unstructured@STEMS) $(tests@STEMS),$(call stem-to-object,$(STEM))) $(structured@LIB),$(tests@EXECUTABLE)))

benchmarks@SOURCE := $(call benchmarks@stem-to-source,$(benchmarks@STEM))
benchmarks@OBJECT := $(call stem-to-object,$(benchmarks@STEM))

$(benchmarks@OBJECT): DIRECTORIES_INCLUDE = $(benchmarks@DIRECTORIES_INCLUDE)
$(benchmarks@OBJECT): CXXFLAGS += $(structured@CXXFLAGS)
$(benchmarks@EXECUTABLE): LDFLAGS += $(benchmarks@LDFLAGS)
$(benchmarks@EXECUTABLE): LDLIBS += $(benchmarks@LDLIBS)

$(eval $(call compile-cpp,$(benchmarks@SOURCE),$(benchmarks@OBJECT)))
$(eval $(call link-cpp,$(benchmarks@OBJECT) $(structured@LIB) $(foreach STEM,$(unstructured@MGARD_STEMS),$(call stem-to-object,$(STEM))),$(benchmarks@EXECUTABLE)))

.PHONY: benchmarks
benchmarks: $(benchmarks@EXECUTABLE)

.PHONY: doc
doc:
	doxygen .doxygen

.PHONY: doc-clean
doc-clean:
	$(RM) --recursive $(DIR_DOC)

TAGSFILE := tags

.PHONY: tags
tags:
	ctags --langmap=C++:+.tpp --recurse -f $(TAGSFILE) $(DIR_INC) $(DIR_SRC)

.PHONY: tags-clean
tags-clean:
	$(RM) $(TAGSFILE)

.PHONY: clean
clean: doc-clean tags-clean
	$(RM) $(dirty@FILES)
	for dir in $(dirty@DIRECTORIES); do if [ -d "$$dir" ]; then $(RMDIR) "$$dir"; fi; done
