/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"
#include "solver_lbm.h"

__global__
void Cu_ComputeForcesP
(
	
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < N)
	{
		
	}
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeForcesP(int i_dev, int L)
{
	
	
	return 0;
}
