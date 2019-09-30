DEBUG ?= 0

LDFLAGS = -L$(HOME)/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib
LDLIBS = -lMOAB -lhdf5_hl -lhdf5 -llapack -lblas -lblaspp -lz -lm

DIR_SRC := src
DIR_OBJ := obj
DIR_INC := include
DIR_BIN := bin
DIR_DOC := doc

CPPFLAGS = -I$(DIR_INC) -I$(HOME)/include
CXXFLAGS = -std=c++17 -Wfatal-errors -Wall -Wextra

ifneq ($(DEBUG), 0)
CXXFLAGS += -g -fsanitize=address -fsanitize=undefined
endif

DIRTY = $(DIRTY_OBJECT_FILES) $(DIRTY_EXECUTABLE_FILES)
DIRTY_OBJECT_FILES =
DIRTY_EXECUTABLE_FILES =

#Tested but not compiled.
HEADERS_WITHOUT_IMPLEMENTATIONS := utilities
HELPER_STEMS := measure LinearOperator pcg
MGARD_STEMS := MassMatrix MeshLevel MeshHierarchy MeshRefiner UniformMeshRefiner UniformMeshHierarchy
STEMS = $(HELPER_STEMS) $(MGARD_STEMS)

TESTS@DIR_ROOT := tests
TESTS@DIR_INC := $(TESTS@DIR_ROOT)/$(DIR_INC)
TESTS@STEMS := $(foreach STEM,$(STEMS) $(HEADERS_WITHOUT_IMPLEMENTATIONS),test_$(STEM)) testing_utilities main
TESTS@EXECUTABLE := $(DIR_BIN)/tests

define stem-to-source
$(DIR_SRC)/$1.cpp
endef

define TESTS@stem-to-source
$(TESTS@DIR_ROOT)/$(call stem-to-source,$1)
endef

define stem-to-object
$(DIR_OBJ)/$1.o
endef

define compile-cpp
$2: $1
	$$(COMPILE.cpp) $$(OUTPUT_OPTION) $$<

DIRTY_OBJECT_FILES += $2
endef

define link-cpp
$2: $1
	$$(LINK.cpp) $$^ $(LDLIBS) $$(OUTPUT_OPTION)
DIRTY_EXECUTABLE_FILES += $2
endef

.PHONY: all
all: $(foreach STEM,$(STEMS),$(call stem-to-object,$(STEM)))

$(eval $(foreach STEM,$(TESTS@STEMS),$(call stem-to-object,$(STEM))): CPPFLAGS += -I$(TESTS@DIR_INC))
$(foreach STEM,$(TESTS@STEMS),$(eval $(call compile-cpp,$(call TESTS@stem-to-source,$(STEM)),$(call stem-to-object,$(STEM)))))
$(foreach STEM,$(STEMS),$(eval $(call compile-cpp,$(call stem-to-source,$(STEM)),$(call stem-to-object,$(STEM)))))

.PHONY: check
check: $(TESTS@EXECUTABLE)
	./$<

$(eval $(call link-cpp,$(foreach STEM,$(STEMS) $(TESTS@STEMS),$(call stem-to-object,$(STEM))),$(TESTS@EXECUTABLE)))

.PHONY: doc
doc:
	doxygen .doxygen

.PHONY:
doc-clean:
	$(RM) --recursive $(DIR_DOC)

.PHONY: clean
clean: doc-clean
	$(RM) $(DIRTY)
