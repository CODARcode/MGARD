
CXX= g++
CC = gcc

CXXFLAGS= -c -std=c++11 -Wall -Wfatal-errors -I$(INC) -O3 
CCFLAGS=  -c -Wall -Wfatal-errors -I$(INC) -O3
LDFLAGS = -lz

SRC=src
INC=include
OBJ=obj


VPATH=src
SOURCES=cmgard.c mgard.cpp mgard_capi.cpp 
OBJECTS=cmgard.o mgard.o mgard_capi.o 
OBJECTS2=$(patsubst %.o,obj/%.o, cmgard.o mgard.o mgard_capi.o)


EXECUTABLE=cmgard
LIB=libmgard.a


all: $(SOURCES) $(EXECUTABLE) $(LIB) 

$(EXECUTABLE): $(OBJECTS) 
	$(CXX) -o $@ $(OBJECTS2) $(LDFLAGS) 

.cpp.o:
	$(CXX) $(CXXFLAGS)  $< -o  $(OBJ)/$@

.c.o:
	$(CC) $(CCFLAGS)    $< -o  $(OBJ)/$@

$(LIB): $(OBJECTS)
	ar rcs $(LIB)  $(OBJECTS2)

clean:
	$(RM) *.o cmgard $(OBJ)/*.o $(LIB)

