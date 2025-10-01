# Library paths.
EIG_VER = 3.4.0
#|
# I used this for personal installation (970M, Debian).
VTK_VER = 9.5
VTK_INC ?= ../../AGAL-latest/lib/vtk-$(VTK_VER)/include/vtk-$(VTK_VER)/
VTK_LIB ?= ../../AGAL-latest/lib/vtk-$(VTK_VER)/lib/
EIG_INC ?= ../../AGAL-latest/lib/eigen/eigen-$(EIG_VER)/
#|
# I used this for personal computer (RTX 4070, WSL).
#VTK_VER = 9.4
#VTK_INC ?= ../lib/vtk-$(VTK_VER)/include/vtk-$(VTK_VER)
#VTK_LIB ?= ../lib/vtk-$(VTK_VER)/lib/
#|
# I used this for the Narval cluster.
#VTK_VER = 9.2
#VTK_INC ?= ../lib/vtk-$(VTK_VER)/include/vtk-$(VTK_VER)
#VTK_LIB ?= ../lib/vtk-$(VTK_VER)/lib64/
# I used this for the Trillium cluster.
#VTK_VER = 9.5
#VTK_INC ?= ../lib/vtk/vtk-$(VTK_VER)/include/vtk-$(VTK_VER)
#VTK_LIB ?= ../lib/vtk/vtk-$(VTK_VER)/lib64/
LIBS = -lcuda\
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
N_CASE?=0
P_SHOW_REFINE?=1
P_SHOW_ADVANCE?=1
N_REGEN?=0
OPTS = -DN_CASE=$(N_CASE)\
       -DP_SHOW_REFINE=$(P_SHOW_REFINE)\
       -DP_SHOW_ADVANCE=$(P_SHOW_ADVANCE)
       

# Compilation parameters.
CXX = g++
NVCC = nvcc
#|
CPPFLAGS = 
CXXFLAGS = -g -Wno-unused-variable
NVCCFLAGSPTX = -lineinfo -Xptxas "-v" -arch=sm_50 -Xcompiler=\"-Wno-unused-variable\"
NVCCFLAGS = -lineinfo -Xcompiler=\"-Wno-unused-variable\"
LDFLAGS = 
#|
INC = -I../inc/ -I$(VTK_INC) -I./geometry -I./mesh -I./solver_lbm
LIB = -L. -L$(VTK_LIB) --compiler-options=-fopenmp -Xcompiler \"-Wl,-rpath,$(VTK_LIB)\"
VPATH = ../inc/:./geometry:./mesh:./solver_lbm

all: a.out


# Building the geometry object.
SRC_GEOMETRY = geometry_init.cu\
	geometry_dest.cu\
	geometry_add.cu\
	geometry_import.cu\
	geometry_convert.cu\
	geometry_bins.cu\
	geometry_bins_cpu.cu\
	geometry_bins_gpu.cu\
	geometry_bins_draw.cu\
	geometry_print.cu\
	geometry_refine.cu
OBJ_GEOMETRY = $(SRC_GEOMETRY:.cu=.o)
# geometry_%.o: geometry_%.cu geometry.h cppspec.h
# 	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<
# libgeom.a: geometry_inst.o
# 	ar rcs libgeom.a geometry_inst.o



# Building the mesh object.
SRC_MESH = mesh_init.cu\
	mesh_dest.cu\
	mesh_comm.cu\
	mesh_criterion.cu\
	mesh_voxelizer.cu\
	mesh_voxelizer_voxelize.cu\
	mesh_voxelizer_propagate.cu\
	mesh_voxelizer_refine.cu\
	mesh_voxelizer_masks.cu\
	mesh_voxelizer_links.cu\
	mesh_amr.cu\
	mesh_advance.cu\
	mesh_restart.cu\
	mesh_output.cu\
	mesh_print_uniform.cu\
	mesh_print_vthb.cu\
	mesh_print_vthb_patch.cu\
	mesh_print_image.cu
OBJ_MESH = $(SRC_MESH:.cu=.o)
# mesh_%.o: mesh_%.cu mesh.h solver.h cppspec.h
# 	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<


	
SRC_SOLVER_LBM = solver_lbm_init.cu\
	solver_lbm_advance.cu\
	solver_lbm_compute.cu\
	solver_lbm_compute_macro.cu\
	solver_lbm_criterion.cu\
	solver_lbm_kernels.cu\
	solver_lbm_kernels_comm.cu\
	solver_lbm_kernels_forces.cu\
	solver_lbm_kernels_les.cu\
	solver_lbm_kernels_debug.cu
OBJ_SOLVER_LBM = $(SRC_SOLVER_LBM:.cu=.o)
# solver_lbm_%.o: solver_lbm_%.cu solver_lbm.h solver.h cppspec.h
# 	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<


# Building the application objects.
INC_MAIN = cppspec.h\
	parser.h\
	geometry.h\
	geometry_bins.h\
	mesh.h\
	solver_lbm.h\
	solver.h\
	structs.h\
	util.h\
	util_tribin.h\
	util_tribin_old.h\
	util_math.h\
	util_interp.h\
	util_index.h\
	custom.h\
	debug_matlab.h
# SRC_MAIN = main_read_input.cu
# OBJ_MAIN = $(SRC_MAIN:.cu=.o)
# main_%.o: main_%.cu
# 	$(NVCC) $(NVCCFLAGS) $(INC) $(OPTS) -c $<


# Make a time stamp indicating the last choice of velocity set.
VSET ?= D2Q9
# Check if specified VSET is a valid choice.
ifeq (,$(filter $(VSET),D2Q9 D3Q19 D3Q27))
$(error MODE must be one of: D2Q9, D3Q19, D3Q27. Got '$(MODE)')
endif
#
VSET_STAMP := vset_$(VSET)
$(VSET_STAMP):
	@echo "VSET changed to $(VSET)"
	@rm -f vset_*     # remove old VSET stamp files to ensure rebuild next time it changes
	@touch $@


# Application.
OBJ = $(OBJ_GEOMETRY) $(OBJ_MESH) $(OBJ_SOLVER_LBM)
SRC = $(SRC_GEOMETRY) $(SRC_MESH) $(SRC_SOLVER_LBM)
a.out: main.cu $(VSET_STAMP) $(SRC) $(INC_MAIN)
	$(NVCC) -DUSE$(VSET) $(NVCCFLAGS) $(INC) $(OPTS) $(LIB) main.cu $(LIBS) -o $@

geometry: $(OBJ_GEOMETRY)
mesh: $(OBJ_MESH)
solver_lbm: $(OBJ_SOLVER_LBM)
pxtas:
	$(NVCC) -DUSE$(VSET) $(NVCCFLAGSPTX) $(INC) $(OPTS) $(LIB) main.cu $(LIBS) -o $@
clean:
	$(RM) $(OBJ) vset_*

.PHONY: clean pxtas all geometry mesh solver_lbm


# Shortcuts:
#
# make pxtas VSET=D3Q27 &> out.txt && sed -i 's/Compiling entry/\n\nCompiling entry/g' out.txt   <-] to print pxtas
#
# git fetch --all
# git reset --hard origin/develop
