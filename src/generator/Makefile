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

SRC_COMPLEX = ./input/complex/solver_lbm_collision_bgk_complex_new.agc\
              ./input/complex/solver_lbm_stream_complex_new.agc\
              ./input/complex/solver_lbm_impose_bc.agc\
              ./input/complex/solver_lbm_setic.agc\
              ./input/complex/solver_lbm_compute_forces_mea.agc\
              ./input/complex/solver_lbm_compute_forces_cv.agc
complex: $(SRC_COMPLEX)
	./a.out ./input/complex/solver_lbm_collision_bgk_complex_new.agc
	./a.out ./input/complex/solver_lbm_stream_complex_new.agc
	./a.out ./input/complex/solver_lbm_impose_bc.agc
	./a.out ./input/complex/solver_lbm_setic.agc
	./a.out ./input/complex/solver_lbm_compute_forces_mea.agc
	./a.out ./input/complex/solver_lbm_compute_forces_cv.agc

.PHONY: clean
