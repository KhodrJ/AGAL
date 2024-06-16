/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

int Mesh::M_Interpolate(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2)
{
#if (N_Q==9)
	M_Interpolate_Linear_d2q9(i_dev, L, var, Cscale, Cscale2);
#elif (N_Q==19)
	M_Interpolate_Linear_d3q19(i_dev, L, var, Cscale, Cscale2);
#else // (N_Q==27)
	M_Interpolate_Linear_d3q27(i_dev, L, var, Cscale, Cscale2);
#endif
	
	return 0;
}

int Mesh::M_Average(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2)
{
#if (N_Q==9)
	M_Average_d2q9(i_dev, L, var, Cscale, Cscale2);
#elif (N_Q==19)
	M_Average_d3q19(i_dev, L, var, Cscale, Cscale2);
#else // (N_Q==27)
	M_Average_d3q27(i_dev, L, var, Cscale, Cscale2);
#endif
	
	return 0;
}
