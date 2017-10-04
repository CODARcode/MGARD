
CXX= g++
CC = gcc
#Using $(CXX) rather than $(CC) to link against C++ standard library.
LINK.o = $(CXX) $(LDFLAGS) $(TARGET_ARCH)
MKDIR=mkdir
RMDIR=rmdir --ignore-fail-on-non-empty



CXXFLAGS= -c -std=c++11 -Wall -Wfatal-errors -I$(INC) -O3 
CFLAGS=  -c -Wall -Wfatal-errors -I$(INC) -O3

LDFLAGS = -lz
ARFLAGS = rcs

SRC=src
INC=include
OBJ=obj

vpath %.o $(OBJ)
vpath %.c $(SRC)
vpath %.cpp $(SRC)

SOURCES=cmgard.c mgard.cpp mgard_capi.cpp 
OBJECTS=$(foreach SOURCE,$(basename $(SOURCES)),$(OBJ)/$(SOURCE).o)

EXECUTABLE=cmgard
LIB=libmgard.a

.PHONY: all clean

all: $(EXECUTABLE) $(LIB)

$(EXECUTABLE): $(OBJECTS) 
	$(LINK.o) -o $@ $^

$(OBJ)/%.o: %.cpp | $(OBJ)
	$(COMPILE.cpp) $< -o $@

$(OBJ)/%.o: %.c | $(OBJ)
	$(COMPILE.c) $< -o $@

$(OBJ):
	$(MKDIR) $@

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $^

clean:
	$(RM) $(EXECUTABLE) $(OBJECTS) $(LIB)
	if [ -d $(OBJ) ]; then $(RMDIR) $(OBJ); fi
