
CXX= g++
CC = gcc

LINK.o = $(CC) $(LDFLAGS) $(TARGET_ARCH)
MKDIR=mkdir
RMDIR=rmdir --ignore-fail-on-non-empty


CXXFLAGS= -c  -Wall -Wfatal-errors -I$(INC) -O3 -fPIC
CFLAGS=  -c -Wall -Wfatal-errors -I$(INC) -O3

LDFLAGS = -lz -lm -lstdc++
ARFLAGS = rcs

SRC=src
INC=include
OBJ=obj

vpath %.o $(OBJ)
vpath %.c $(SRC)
vpath %.cpp $(SRC)


SOURCES=mgard_test.c mgard.cpp mgard_nuni.cpp mgard_capi.cpp 
OBJECTS=$(foreach SOURCE,$(basename $(SOURCES)),$(OBJ)/$(SOURCE).o)

EXECUTABLE=mgard_test
LIB=libmgard.a

.PHONY: all clean test

all: $(EXECUTABLE) $(LIB) test test2

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

test: $(EXECUTABLE)
	./$(EXECUTABLE) data/u3_513x513_orig data/u3_513x513.mgard  513 513 1e-2

test2: $(EXECUTABLE)
	./$(EXECUTABLE) data/data_600x400_orig data/data_600x400.mgard  600 400 1e-3
clean:
	$(RM) $(EXECUTABLE) $(OBJECTS) $(LIB)
	if [ -d $(OBJ) ]; then $(RMDIR) $(OBJ); fi
