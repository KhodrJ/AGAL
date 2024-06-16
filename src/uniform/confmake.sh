# Solver parameters.
N_PRECISION=0
N_Q=9
v0=5.00e-5
Nx=32
N_CONN_TYPE=1
N_SHUFFLE=1
N_OCTREE=1

# Output.
P_DIR_NAME=../out/
N_iters=1



# These automatically choose N_DIM and N_Q_max.
if [[ ! ($N_Q == 9 || $N_Q == 19 || $N_Q == 27) ]]; then
	echo "Invalid choice of velocity set (either 9 [2D] or 19/27 [3D])..."
	exit
fi
if [[ $N_Q == 9 ]]; then N_DIM=2; N_Q_max=9; else N_DIM=3; N_Q_max=27; fi



make\
	N_PRECISION=$N_PRECISION\
	N_DIM=$N_DIM\
	N_Q=$N_Q\
	N_Q_max=$N_Q_max\
	v0=$v0\
	Nx=$Nx\
	N_CONN_TYPE=$N_CONN_TYPE\
	N_SHUFFLE=$N_SHUFFLE\
	N_OCTREE=$N_OCTREE\
	N_iters=$N_iters
