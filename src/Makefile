# Library paths.
VTK_VER = 9.2
EIG_VER = 3.4.0
#|
#Use this for personal installation.
#VTK_INC ?= /usr/local/include/vtk-$(VTK_VER)
#VTK_LIB ?= /usr/local/lib/
#EIG_INC ?= ../lib/eigen/eigen-$(EIG_VER)/
#|
# I used this on the Mist cluster.
#VTK_INC ?= ../lib/vtk/include/vtk-$(VTK_VER)
#VTK_LIB ?= ../lib/vtk/lib64/
#EIG_INC ?= ../lib/eigen/eigen-$(EIG_VER)/
#|
# I used this on the Narval cluster.
#VTK_INC ?= /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/gcc9/vtk/9.1.0/include/vtk-9.1/
#VTK_LIB ?= /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/gcc9/vtk/9.1.0/lib/
#EIG_INC ?= ../lib/eigen/eigen-$(EIG_VER)/
#|
# I used this for my personal laptop build.
#VTK_INC ?= ../lib/vtk/include/vtk-$(VTK_VER)
#VTK_LIB ?= ../lib/vtk/lib/
VTK_INC ?= ../../AGAL-latest/lib/vtk/include/vtk-$(VTK_VER)/
VTK_LIB ?= ../../AGAL-latest/lib/vtk/lib/
EIG_INC ?= ../../AGAL-latest/lib/eigen/eigen-$(EIG_VER)/
LIBS = -lcuda \
       -lgomp\
       -lvtkCommonCore-$(VTK_VER) \
       -lvtkCommonDataModel-$(VTK_VER) \
       -lvtksys-$(VTK_VER) \
       -lvtkIOXML-$(VTK_VER) \
       -lvtkCommonExecutionModel-$(VTK_VER) \
       -lvtkFiltersCore-$(VTK_VER) \
       -lvtkCommonMisc-$(VTK_VER) \
       -lvtkIOCore-$(VTK_VER) \
       -lvtkIOImage-$(VTK_VER) \
       -lvtkIOLegacy-$(VTK_VER) \
       -lvtkCommonColor-$(VTK_VER) \
       -lvtkRenderingCore-$(VTK_VER) \
       -lvtkRenderingContextOpenGL2-$(VTK_VER) \
       -lvtkRenderingCore-$(VTK_VER) \
       -lvtkRenderingFreeType-$(VTK_VER) \
       -lvtkRenderingGL2PSOpenGL2-$(VTK_VER) \
       -lvtkRenderingOpenGL2-$(VTK_VER)\
       -lvtkImagingCore-$(VTK_VER)


# Fixed solver parameters (defaults specified).
N_PRECISION?=0
N_DIM?=2
N_Q?=9
S_LES?=0
N_CASE?=0
P_SHOW_REFINE?=0
P_PRINT_ADVANCE?=0
N_REGEN?=0


# Compilation parameters.
CXX = g++
NVCC = nvcc
#|
CPPFLAGS = 
CXXFLAGS = -g
NVCCFLAGS =
LDFLAGS = 
#|
INC = -I../inc/ -I$(VTK_INC) -I$(EIG_INC)
LIB = -L$(VTK_LIB) --compiler-options=-fopenmp -Xcompiler \"-Wl,-rpath,$(VTK_LIB)\"
#|
SRC = read_input.cu\
      mesh_init.cu\
      mesh_dest.cu\
      mesh_comm.cu\
      mesh_amr.cu\
      mesh_advance.cu\
      mesh_restart.cu\
      mesh_output.cu\
      mesh_print_uniform.cu\
      solver_lbm_init.cu\
      solver_lbm_advance.cu\
      solver_lbm_criterion.cu\
      solver_lbm_compute.cu\
      solver_lbm_set_ic_d2q9.cu\
      solver_lbm_set_ic_d3q19.cu\
      solver_lbm_set_ic_d3q27.cu\
      solver_lbm_collide_d2q9.cu\
      solver_lbm_collide_d3q19.cu\
      solver_lbm_collide_d3q27.cu\
      solver_lbm_stream_inpl_d2q9.cu\
      solver_lbm_stream_inpl_d3q19.cu\
      solver_lbm_stream_inpl_d3q27.cu\
      solver_lbm_interp_linear_d2q9.cu\
      solver_lbm_interp_linear_d3q19.cu\
      solver_lbm_interp_linear_d3q27.cu\
      solver_lbm_average_d2q9.cu\
      solver_lbm_average_d3q19.cu\
      solver_lbm_average_d3q27.cu
      
#|
OBJ = $(SRC:.cu=.o)
#|
OPTS = -DINPUT_CONFMAKE\
       -DN_PRECISION=$(N_PRECISION)\
       -DN_DIM=$(N_DIM)\
       -DN_Q=$(N_Q)\
       -DS_LES=$(S_LES)\
       -DN_CASE=$(N_CASE)\
       -DP_SHOW_REFINE=$(P_SHOW_REFINE)\
       -DP_PRINT_ADVANCE=$(P_PRINT_ADVANCE)


#
#Targets
#

all: a.out

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<
	
a.out: $(OBJ) main.cu
	$(NVCC) $(OBJ) $(INC) $(OPTS) $(LIB) main.cu $(LIBS) -o $@ $(NVCCFLAGS)
	
clean:
	rm ./*.o
	
output.out:
	$(NVCC) $(INC) $(LIB) output_process.cpp $(LIBS) -o ../out/$@ $(NVCCFLAGS)
	
generate:
	echo "Generating LBM code..."; cd ./generators;\
	octave collide_builder.m;\
	octave stream_builder.m;\
	octave IC_builder.m;\
	octave interp_builder.m;\
	octave average_builder.m;\
	cd ..;\
