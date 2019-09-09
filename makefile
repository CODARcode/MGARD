CXX= g++

LINK.o = $(CXX) $(LDFLAGS) $(TARGET_ARCH)
MKDIR=mkdir
RMDIR=rmdir --ignore-fail-on-non-empty

SRC=src
IDIR1=include
#IDIR2=blosc/include
INC=$(IDIR1) #$(IDIR2) 
INC_PARAMS=$(foreach d, $(INC), -I$d)
OBJ=obj


CXXFLAGS= -std=c++11 -c  -Wall -Wfatal-errors $(INC_PARAMS)  -O3 -fPIC
CFLAGS=  -c -Wall -Wfatal-errors $(INC_PARAMS) -O3

LDFLAGS = -lz -lm -lstdc++
ARFLAGS = rcs


vpath %.o $(OBJ)
vpath %.c $(SRC)
vpath %.cpp $(SRC)


SOURCES=mgard.cpp mgard_nuni.cpp mgard_api.cpp 
OBJECTS=$(foreach SOURCE,$(basename $(SOURCES)),$(OBJ)/$(SOURCE).o)

SOURCES_SIRIUS=mgard_sirius_test.c mgard.cpp mgard_nuni.cpp mgard_api.cpp 
OBJECTS_SIRIUS=$(foreach SOURCE,$(basename $(SOURCES_SIRIUS)),$(OBJ)/$(SOURCE).o)

EXECUTABLE=mgard_test
SIRIUS_EXEC=mgard_sirius_test

LIB=libmgard.a

.PHONY: all clean test

#all: $(EXECUTABLE) $(LIB) $(SIRIUS_EXEC) test test2 test3
all: $(EXECUTABLE) $(LIB) 

$(EXECUTABLE): $(OBJECTS) 
	$(LINK.o) -o $@ $^ $(LDFLAGS)

#$(SIRIUS_EXEC): $(OBJECTS_SIRIUS) 
#	$(LINK.o) -o $@ $^ $(LDFLAGS)

$(OBJ)/%.o: %.cpp | $(OBJ)
	$(COMPILE.cpp) $< -o $@

$(OBJ)/%.o: %.c | $(OBJ)
	$(COMPILE.c) $< -o $@

$(OBJ):
	$(MKDIR) $@

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $^


clean:
	$(RM) $(EXECUTABLE) $(OBJECTS) $(LIB) $(SIRIUS_EXEC)
	if [ -d $(OBJ) ]; then $(RMDIR) $(OBJ); fi
