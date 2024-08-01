#!/bin/bash

# REMEMBER:
# - Set IC, BC in 'generators/input' directory. Make N_REGEN=1 if changed.
# - Select N_CASE if different geometry is being employed.
# - Adjust parameters as necessary.

# To enable periodic boundary conditions along either X, Y or Z, set the corresponding
# variable to 1, otherwise leave it empty.

#
# Input parameters.
#

# Solver parameters.
N_PRECISION=0
N_Q=19
S_LES=1
N_CASE=0

# Refinement parameters.
P_SHOW_REFINE=0
	
# Printing parameters.
P_PRINT_ADVANCE=0


# 
# Sent to compiler. 
#

if [[ ! ($N_Q == 9 || $N_Q == 19 || $N_Q == 27) ]]; then
	echo "Invalid choice of velocity set (either 9 [2D] or 19/27 [3D])..."
	exit
fi
if [[ $N_Q == 9 ]]; then N_DIM=2; else N_DIM=3; fi

make\
	N_PRECISION=$N_PRECISION\
	N_DIM=$N_DIM\
	N_Q=$N_Q\
	S_LES=$S_LES\
	N_CASE=$N_CASE\
	P_SHOW_REFINE=$P_SHOW_REFINE\
	P_PRINT_ADVANCE=$P_PRINT_ADVANCE
