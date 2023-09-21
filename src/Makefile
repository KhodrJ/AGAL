VTK_VER = 9.2
# VTK_INC ?= /usr/local/include/vtk-$(VTK_VER)
# VTK_LIB ?= /usr/local/lib/
# Alt:
VTK_INC ?= ../lib/vtk/include/vtk-$(VTK_VER)
VTK_LIB ?= ../lib/vtk/lib/

N_PRECISION ?= 0
N_DIM ?= 3
MAX_LEVELS ?= 4
VERBOSE ?= 0
S_LES ?= 0
S_TYPE ?= 0
P_DIR_NAME ?= \"../out/\"

INC = -I../inc/ -I$(VTK_INC)
LIB = -L$(VTK_LIB) -Xcompiler \"-Wl,-rpath,$(VTK_LIB)\"
SRC = init_grid_data.cu mesh.cu solver_lbm.cu

LIBS = 	-lcuda \
	-lvtkCommonCore-$(VTK_VER) \
	-lvtkCommonDataModel-$(VTK_VER) \
	-lvtksys-$(VTK_VER) \
	-lvtkIOXML-$(VTK_VER) \
	-lvtkCommonExecutionModel-$(VTK_VER) \
	-lvtkFiltersHyperTree-$(VTK_VER) \
	-lvtkFiltersCore-$(VTK_VER) \
	-lvtkIOCore-$(VTK_VER) \
	-lvtkIOLegacy-$(VTK_VER)

OPTS = 	-DN_PRECISION=$(N_PRECISION)\
	-DN_DIM=$(N_DIM) \
	-DMAX_LEVELS=$(MAX_LEVELS) \
	-DVERBOSE=$(VERBOSE) \
	-DS_LES=$(S_LES) \
	-DS_TYPE=$(S_TYPE) \
	-DP_DIR_NAME=$(P_DIR_NAME)

all:
	nvcc $(OPTS) $(INC) $(SRC) $(LIB) main.cu $(LIBS) -o a.out
