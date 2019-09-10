LDFLAGS = -L$(HOME)/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib
LDLIBS = -lMOAB -lhdf5_hl -lhdf5 -llapack -lblas -lblaspp -lz -lm

DIR_SRC := src
DIR_OBJ := obj
DIR_INC := include
DIR_BIN := bin

CPPFLAGS = -I$(DIR_INC) -I$(HOME)/include
CXXFLAGS = -Wfatal-errors -Wall -Wextra

DIRTY = $(DIRTY_OBJECT_FILES) $(DIRTY_EXECUTABLE_FILES)
DIRTY_OBJECT_FILES =
DIRTY_EXECUTABLE_FILES =

STEMS := measure pcg MeshLevel

TESTS@DIR_ROOT := tests
TESTS@DIR_INC := $(TESTS@DIR_ROOT)/$(DIR_INC)
TESTS@STEMS := $(foreach STEM,$(STEMS),test_$(STEM)) main
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

$(eval $(foreach STEM,$(TESTS@STEMS),$(call stem-to-object,$(STEM))): CPPFLAGS += -I$(TESTS@DIR_INC))
$(foreach STEM,$(TESTS@STEMS),$(eval $(call compile-cpp,$(call TESTS@stem-to-source,$(STEM)),$(call stem-to-object,$(STEM)))))
$(foreach STEM,$(STEMS),$(eval $(call compile-cpp,$(call stem-to-source,$(STEM)),$(call stem-to-object,$(STEM)))))

.PHONY: check
check: $(TESTS@EXECUTABLE)
	./$<

$(eval $(call link-cpp,$(foreach STEM,$(STEMS) $(TESTS@STEMS),$(call stem-to-object,$(STEM))),$(TESTS@EXECUTABLE)))

.PHONY: clean
clean:
	$(RM) $(DIRTY)
