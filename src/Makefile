VTK_VER = 9.2

#Use this for personal installation.
#VTK_INC ?= /usr/local/include/vtk-$(VTK_VER)
#VTK_LIB ?= /usr/local/lib/

# I used this on the Mist cluster.
#VTK_INC ?= ../lib/vtk/include/vtk-$(VTK_VER)
#VTK_LIB ?= ../lib/vtk/lib64/

# I used this on the Narval cluster.
#VTK_INC ?= /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/gcc9/vtk/9.1.0/include/vtk-9.1/
#VTK_LIB ?= /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/gcc9/vtk/9.1.0/lib/

# I used this for my personal laptop build.
VTK_INC ?= ../lib/vtk/include/vtk-$(VTK_VER)
VTK_LIB ?= ../lib/vtk/lib/



# Solver parameters.
N_PRECISION?=0
N_DIM?=2
N_Q?=9
N_Q_max?=9
MAX_LEVELS?=1
MAX_LEVELS_INTERIOR?=MAX_LEVELS
N_LEVEL_START?=0
L_c?=1.0
L_fy?=1.0
L_fz?=1.0
v0?=1.5625E-5
Nx?=1024
S_TYPE?=1
S_INTERP_TYPE?=1
S_INIT_TYPE?=0
B_TYPE?=1
N_CASE?=0
N_RESTART?=0
PERIODIC_X?=
PERIODIC_Y?=
PERIODIC_Z?=

# Turbluence parameters.
S_LES?=0

# Refinement parameters.
P_REFINE?=16
N_REFINE_START?=-4
N_REFINE_INC?=1
N_CONN_TYPE?=1
P_SHOW_REFINE?=1
	
# Probe parameters.
N_PROBE?=1
N_PROBE_DENSITY?=4
N_PROBE_FREQUENCY?=Nx
V_PROBE_TOL?=1e-5
N_PROBE_FORCE?=1
N_PROBE_F_FREQUENCY?=16
N_PROBE_AVE?=0
N_PROBE_AVE_FREQUENCY?=4
N_PROBE_AVE_START?=100*Nx
	
# Printing parameters.
P_DIR_NAME?=../out/results/
P_DIR_NAME_=\"$(P_DIR_NAME)\"
P_RENDER?=100*Nx
P_PRINT?=100*Nx
N_PRINT?=1
N_PRINT_LEVELS?=3
P_SHOW_ADVANCE?=0
P_PRINT_ADVANCE?=0

# Generator options.
N_REGEN?=1



INC = -I../inc/ -I$(VTK_INC)

LIB = -L$(VTK_LIB) --compiler-options=-fopenmp -Xcompiler \"-Wl,-rpath,$(VTK_LIB)\"

SRC = 	./mesh/init_grid_data.cu\
	./mesh/mesh_amr.cu\
	./mesh/mesh_output.cu\
	./mesh/mesh_forces.cu\
	./mesh/mesh_restart.cu\
	./mesh/mesh_print.cu\
	./mesh/mesh_comm.cu\
	./mesh/mesh_comm_average_d2q9.cu\
	./mesh/mesh_comm_average_d3q19.cu\
	./mesh/mesh_comm_average_d3q27.cu\
	./mesh/mesh_comm_interp_linear_d2q9.cu\
	./mesh/mesh_comm_interp_linear_d3q19.cu\
	./mesh/mesh_comm_interp_linear_d3q27.cu\
	./solver/solver.cu\
	./solver/solver_lbm_criterion.cu\
	./solver/solver_lbm_init_d2q9.cu\
	./solver/solver_lbm_init_d3q19.cu\
	./solver/solver_lbm_init_d3q27.cu\
	./solver/solver_lbm_collide_d2q9.cu\
	./solver/solver_lbm_collide_d3q19.cu\
	./solver/solver_lbm_collide_d3q27.cu\
	./solver/solver_lbm_stream_inpl_d2q9.cu\
	./solver/solver_lbm_stream_inpl_d3q19.cu\
	./solver/solver_lbm_stream_inpl_d3q27.cu
	

LIBS = 	-lcuda \
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
	-lvtkRenderingOpenGL2-$(VTK_VER)

leftparen:=(
rightparen:=)
	
OPTS = 	-DN_PRECISION=$(N_PRECISION)\
	-DN_DIM=$(N_DIM)\
	-DN_Q=$(N_Q)\
	-DN_Q_max=$(N_Q_max)\
	-DMAX_LEVELS=$(MAX_LEVELS)\
	-DMAX_LEVELS_INTERIOR=$(MAX_LEVELS_INTERIOR)\
	-DN_LEVEL_START=$(N_LEVEL_START)\
	-DL_ci=$(L_c)\
	-DL_fyi=$(L_fy)\
	-DL_fzi=$(L_fz)\
	-Dv0i=$(v0)\
	-DNx=$(Nx)\
	-DS_TYPE=$(S_TYPE) \
	-DS_INIT_TYPE=$(S_INIT_TYPE)\
	-DB_TYPE=$(B_TYPE)\
	-DN_CASE=$(N_CASE)\
	-DN_RESTART=$(N_RESTART)\
	$(PERIODIC_X)\
	$(PERIODIC_Y)\
	$(PERIODIC_Z)\
	-DS_LES=$(S_LES)\
	-DP_REFINEi=$(P_REFINE)\
	-DN_REFINE_START=$(N_REFINE_START)\
	-DN_REFINE_INC=$(N_REFINE_INC)\
	-DN_CONN_TYPE=$(N_CONN_TYPE)\
	-DP_SHOW_REFINE=$(P_SHOW_REFINE)\
	-DN_PROBE=$(N_PROBE)\
	-DN_PROBE_DENSITY=$(N_PROBE_DENSITY)\
	-DN_PROBE_FREQUENCYi=$(N_PROBE_FREQUENCY)\
	-DV_PROBE_TOL=$(V_PROBE_TOL)\
	-DN_PROBE_FORCE=$(N_PROBE_FORCE)\
	-DN_PROBE_F_FREQUENCYi=$(N_PROBE_F_FREQUENCY)\
	-DN_PROBE_AVE=$(N_PROBE_AVE)\
	-DN_PROBE_AVE_FREQUENCYi=$(N_PROBE_AVE_FREQUENCY)\
	-DN_PROBE_AVE_STARTi=$(N_PROBE_AVE_START)\
	-DP_DIR_NAME=$(P_DIR_NAME_)\
	-DP_RENDERi=$(P_RENDER)\
	-DP_PRINTi=$(P_PRINT)\
	-DN_PRINT=$(N_PRINT)\
	-DN_PRINT_LEVELS=$(N_PRINT_LEVELS)\
	-DP_SHOW_ADVANCE=$(P_SHOW_ADVANCE)\
	-DP_PRINT_ADVANCE=$(P_PRINT_ADVANCE)
	
all: generate
	if ! test -d $(P_DIR_NAME) ; then mkdir $(P_DIR_NAME); echo "Creating new results directory ($(P_DIR_NAME)..."; fi
	nvcc $(OPTS) $(INC) $(SRC) $(LIB) main.cu $(LIBS) -o a.out

generate:
	if [ $(N_REGEN) = 1 ]; then \
		echo "Generating LBM code..."; cd ./generators;\
		octave collide_builder.m;\
		octave stream_builder.m;\
		octave IC_builder.m;\
		octave interp_builder.m;\
		octave average_builder.m;\
		cd ..;\
	fi
