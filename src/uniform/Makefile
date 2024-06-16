# Input options.
Nx ?= 128
N_Q ?= 9
N_Q_max ?= 9
N_DIM ?= 2
N_PRECISION ?= 0
N_iters ?= 1*Nx
N_CONN_TYPE ?= 0
N_SHUFFLE ?= 0
N_TREEFORM ?= 0

OPTS = 		-DNx=$(Nx)\
		-DN_Q=$(N_Q)\
		-DN_Q_max=$(N_Q_max)\
		-DN_DIM=$(N_DIM)\
		-DN_PRECISION=$(N_PRECISION)\
		-DN_iters=$(N_iters)\
		-DN_CONN_TYPE=$(N_CONN_TYPE)\
		-DN_SHUFFLE=$(N_SHUFFLE)\
		-DN_TREEFORM=$(N_TREEFORM)

all:
	nvcc $(OPTS) cell_uniform.cu -o a_cell.out
	nvcc $(OPTS) block_uniform.cu -o a_block.out
