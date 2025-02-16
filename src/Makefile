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
Nqx?=1
M_LBLOCK?=16
S_LES?=0
N_CASE?=0
P_SHOW_REFINE?=0
P_SHOW_ADVANCE?=0
N_REGEN?=0
OPTS = -DINPUT_CONFMAKE\
       -DN_PRECISION=$(N_PRECISION)\
       -DN_DIM=$(N_DIM)\
       -DN_Q=$(N_Q)\
       -DNqx=$(Nqx)\
       -DM_LBLOCK=$(M_LBLOCK)\
       -DS_LES=$(S_LES)\
       -DN_CASE=$(N_CASE)\
       -DP_SHOW_REFINE=$(P_SHOW_REFINE)\
       -DP_SHOW_ADVANCE=$(P_SHOW_ADVANCE)
       
       

# Compilation parameters.
CXX = g++
NVCC = nvcc
#|
CPPFLAGS = 
CXXFLAGS = -g
NVCCFLAGS = -lineinfo
LDFLAGS = 
#|
INC = -I../inc/ -I$(VTK_INC) -I$(EIG_INC)
LIB = -L$(VTK_LIB) --compiler-options=-fopenmp -Xcompiler \"-Wl,-rpath,$(VTK_LIB)\"
VPATH = ../inc/
#|
MQ = 
ifeq ($(shell expr $(Nqx) \> 1), 1)
   MQ = _mq
endif



# Building the geometry object.
SRC_GEOMETRY = geometry_init.cu\
      geometry_dest.cu\
      geometry_add.cu\
      geometry_print.cu
OBJ_GEOMETRY = $(SRC_GEOMETRY:.cu=.o)
geometry_%.o: geometry_%.cu geometry.h cppspec.h
	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<



# Building the mesh object.
SRC_MESH = mesh_init.cu\
      mesh_dest.cu\
      mesh_comm.cu\
      mesh_criterion.cu\
      mesh_amr.cu\
      mesh_advance.cu\
      mesh_restart.cu\
      mesh_output.cu\
      mesh_print_uniform.cu\
      mesh_print_vthb.cu
OBJ_MESH = $(SRC_MESH:.cu=.o)
mesh_%.o: mesh_%.cu mesh.h solver.h cppspec.h
	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<



# Building the solver object.
SRC_SOLVER = solver_lbm_init.cu\
      solver_lbm_advance.cu\
      solver_lbm_criterion.cu\
      solver_lbm_compute.cu\
      solver_lbm_set_ic_d2q9.cu\
      solver_lbm_set_ic_d3q19.cu\
      solver_lbm_set_ic_d3q27.cu\
      solver_lbm_collide_bgk_d2q9.cu\
      solver_lbm_collide_bgk_d3q19.cu\
      solver_lbm_collide_bgk_d3q27.cu\
      solver_lbm_collide_bgk_interp_linear_d2q9.cu\
      solver_lbm_collide_bgk_interp_linear_d3q19.cu\
      solver_lbm_collide_bgk_interp_linear_d3q27.cu\
      solver_lbm_collide_bgk_interp_cubic_d2q9.cu\
      solver_lbm_collide_bgk_interp_cubic_d3q19.cu\
      solver_lbm_collide_bgk_interp_cubic_d3q27.cu\
      solver_lbm_collide_mrt_interp_linear_d2q9.cu\
      solver_lbm_collide_mrt_interp_linear_d3q19.cu\
      solver_lbm_collide_mrt_interp_cubic_d2q9.cu\
      solver_lbm_collide_mrt_interp_cubic_d3q19.cu\
      solver_lbm_vel2mom_mrt_d2q9.cu\
      solver_lbm_vel2mom_mrt_d3q19.cu\
      solver_lbm_mom2vel_mrt_d2q9.cu\
      solver_lbm_mom2vel_mrt_d3q19.cu\
      solver_lbm_collide_mrt_d2q9.cu\
      solver_lbm_collide_mrt_d3q19.cu\
      solver_lbm_enforce_symmetry_d2q9.cu\
      solver_lbm_enforce_symmetry_d3q19.cu\
      solver_lbm_enforce_symmetry_d3q27.cu\
      solver_lbm_stream_inpl_d2q9.cu\
      solver_lbm_stream_inpl_d3q19.cu\
      solver_lbm_stream_inpl_d3q27.cu\
      solver_lbm_interp_linear$(MQ)_d2q9.cu\
      solver_lbm_interp_linear$(MQ)_d3q19.cu\
      solver_lbm_interp_linear$(MQ)_d3q27.cu\
      solver_lbm_interp_cubic$(MQ)_d2q9.cu\
      solver_lbm_interp_cubic$(MQ)_d3q19.cu\
      solver_lbm_interp_cubic$(MQ)_d3q27.cu\
      solver_lbm_interp_linear_mrt_d2q9.cu\
      solver_lbm_interp_linear_mrt_d3q19.cu\
      solver_lbm_interp_cubic_mrt_d2q9.cu\
      solver_lbm_interp_cubic_mrt_d3q19.cu\
      solver_lbm_average$(MQ)_d2q9.cu\
      solver_lbm_average$(MQ)_d3q19.cu\
      solver_lbm_average$(MQ)_d3q27.cu\
      solver_lbm_average_cubic_d2q9.cu\
      solver_lbm_average_mrt_d2q9.cu\
      solver_lbm_average_mrt_d3q19.cu
OBJ_SOLVER = $(SRC_SOLVER:.cu=.o)
solver_%.o: solver_%.cu solver.h cppspec.h
	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<



# Building the application objects.
SRC_MAIN = main_read_input.cu
OBJ_MAIN = $(SRC_MAIN:.cu=.o)
main_%.o: main_%.cu geometry.h mesh.h solver.h cppspec.h
	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<



# Application.
OBJ = $(OBJ_GEOMETRY) $(OBJ_MESH) $(OBJ_SOLVER) $(OBJ_MAIN)
a.out: main.cu $(OBJ)
	$(NVCC) $(OBJ) $(INC) $(OPTS) $(LIB) main.cu $(LIBS) -o $@ $(NVCCFLAGS)
all: a.out

geometry: $(OBJ_GEOMETRY)
mesh: $(OBJ_MESH)
solver: $(OBJ_SOLVER)
clean:
	$(RM) $(OBJ)

.PHONY: clean all geometry mesh solver
