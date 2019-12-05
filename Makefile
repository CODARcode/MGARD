CXX= g++

MKDIR=mkdir
RMDIR=rmdir --ignore-fail-on-non-empty

SRC=src
INC=include
INC_PARAMS=$(foreach d, $(INC), -I$d)
OBJ=obj

CXXFLAGS= -std=c++11 $(INC_PARAMS) -fPIC

LDFLAGS = -lz -ldl
ARFLAGS = -rcs

ifdef DEBUG
# Compiling with sanitizers during development is simply essential:
	CXXFLAGS += -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fsanitize=undefined
else
	CXXFLAGS +=  -O3 -march=native 
endif


LINK.o = $(CXX) $(LDFLAGS) $(TARGET_ARCH)
COMPILE.cpp = $(CXX) $(CXXFLAGS) -c

vpath %.o $(OBJ)
vpath %.cpp $(SRC)


SOURCES=mgard_test.cpp mgard_api.cpp mgard.cpp mgard_compress.cpp mgard_nuni.cpp  mgard_api_float.cpp mgard_float.cpp mgard_nuni_float.cpp
OBJECTS=$(foreach SOURCE,$(basename $(SOURCES)),$(OBJ)/$(SOURCE).o)

EXECUTABLE=mgard_test

LIB=libmgard.a

.PHONY: all clean test

all: $(EXECUTABLE) $(LIB)

$(EXECUTABLE): $(OBJECTS)
	$(LINK.o) -o $@ $^ $(LDFLAGS)


$(OBJ)/%.o: %.cpp | $(OBJ)
	$(COMPILE.cpp) $< -o $@

$(OBJ):
	$(MKDIR) $@

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $^

clean:
	$(RM) $(EXECUTABLE) $(OBJECTS) $(LIB) $(SIRIUS_EXEC) speed.x
	if [ -d $(OBJ) ]; then $(RMDIR) $(OBJ); fi

speed.x: $(LIB) benchmark/bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ benchmark/bench.cpp  -lbenchmark -lbenchmark_main -pthread $(TARGET_ARCH) $(LDFLAGS) $(LIB)
