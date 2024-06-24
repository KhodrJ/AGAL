#!/bin/bash

# REMEMBER:
# - Set IC, BC in 'generators/input' directory. Make N_REGEN=1 if changed.
# - Select N_CASE if different geometry is being employed.
# - Adjust parameters as necessary.

# To enable periodic boundary conditions along either X, Y or Z, set the corresponding
# variable to 1, otherwise leave it empty.

# Solver parameters.
N_PRECISION=0
N_Q=19
MAX_LEVELS=3
MAX_LEVELS_INTERIOR=3
N_LEVEL_START=0
L_c=1.0
L_fy=1.0
L_fz=1.0
# |
#v0=1.5625E-5
v0=5.00e-5
# |
Nx=64
PERIODIC_X=
PERIODIC_Y=
PERIODIC_Z=
B_TYPE=1
N_CASE=0
N_RESTART=0


# Turbluence parameters.
S_LES=0

# Refinement parameters.
P_REFINE=32
N_REFINE_START=-3
N_REFINE_INC=1
P_SHOW_REFINE=1
	
# Probe parameters.
N_PROBE=0
N_PROBE_DENSITY=4
N_PROBE_FREQUENCY=32
V_PROBE_TOL=1e-4
N_PROBE_FORCE=0
N_PROBE_F_FREQUENCY=16
N_PROBE_AVE=0
N_PROBE_AVE_FREQUENCY=1
N_PROBE_AVE_START=100*Nx
	
# Printing parameters.
P_DIR_NAME=../out/TEST_UPLOAD/
P_PRINT=100*Nx
N_PRINT=1
N_PRINT_LEVELS=2
P_SHOW_ADVANCE=0
P_PRINT_ADVANCE=0

# Generator options.
N_REGEN=0



if [[ ! ($N_Q == 9 || $N_Q == 19 || $N_Q == 27) ]]; then
	echo "Invalid choice of velocity set (either 9 [2D] or 19/27 [3D])..."
	exit
fi
if [[ $N_Q == 9 ]]; then N_DIM=2; N_Q_max=9; else N_DIM=3; N_Q_max=27; fi

if [[ $PERIODIC_X == 1 ]]; then
	PERIODIC_X=-DPERIODIC_X
fi
if [[ $PERIODIC_Y == 1 ]]; then
	PERIODIC_Y=-DPERIODIC_Y
fi
if [[ $PERIODIC_Z == 1 ]]; then
	PERIODIC_Z=-DPERIODIC_Z
fi



make\
	N_PRECISION=$N_PRECISION\
	N_DIM=$N_DIM\
	N_Q=$N_Q\
	N_Q_max=$N_Q_max\
	MAX_LEVELS=$MAX_LEVELS\
	MAX_LEVELS_INTERIOR=$MAX_LEVELS_INTERIOR\
	N_LEVEL_START=$N_LEVEL_START\
	L_c=$L_c\
	L_fy=$L_fy\
	L_fz=$L_fz\
	v0=$v0\
	Nx=$Nx\
	B_TYPE=$B_TYPE\
	PERIODIC_X=$PERIODIC_X\
	PERIODIC_Y=$PERIODIC_Y\
	PERIODIC_Z=$PERIODIC_Z\
	S_INTERP_TYPE=$S_INTERP_TYPE\
	S_INIT_TYPE=$S_INIT_TYPE\
	N_CASE=$N_CASE\
	N_RESTART=$N_RESTART\
	S_LES=$S_LES\
	P_REFINE=$P_REFINE\
	N_REFINE_START=$N_REFINE_START\
	N_REFINE_INC=$N_REFINE_INC\
	N_CONN_TYPE=$N_CONN_TYPE\
	P_SHOW_REFINE=$P_SHOW_REFINE\
	N_PROBE=$N_PROBE\
	N_PROBE_DENSITY=$N_PROBE_DENSITY\
	N_PROBE_FREQUENCY=$N_PROBE_FREQUENCY\
	V_PROBE_TOL=$V_PROBE_TOL\
	N_PROBE_FORCE=$N_PROBE_FORCE\
	N_PROBE_F_FREQUENCY=$N_PROBE_F_FREQUENCY\
	N_PROBE_AVE=$N_PROBE_AVE\
	N_PROBE_AVE_FREQUENCY=$N_PROBE_AVE_FREQUENCY\
	N_PROBE_AVE_START=$N_PROBE_AVE_START\
	P_DIR_NAME=$P_DIR_NAME\
	P_PRINT=$P_PRINT\
	N_PRINT=$N_PRINT\
	N_PRINT_LEVELS=$N_PRINT_LEVELS\
	P_SHOW_ADVANCE=$P_SHOW_ADVANCE\
	P_PRINT_ADVANCE=$P_PRINT_ADVANCE\
	N_REGEN=$N_REGEN
