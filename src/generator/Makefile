CXX = g++
CXXFLAGS = 

INC = -I./tinyexpr++/
SRC = string.cpp\
      loop.cpp\
      line.cpp\
      other_line.cpp\
      output.cpp\
      template.cpp\
      routine.cpp\
      subs.cpp
OBJ = $(SRC:.cpp=.o)
%.o: %.cpp gen.h
	$(CXX) $(CXXFLAGS) $(INC) -c $<

a.out: $(OBJ)
	$(CXX) $(CXXFLAGS) $(INC) $(OBJ) gen.cpp

all: a.out

clean:
	$(RM) $(OBJ)

.PHONY: clean
