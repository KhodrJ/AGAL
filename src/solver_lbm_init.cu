#include "mesh.h"
#include "solver.h"

int Solver_LBM::S_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str)
{
	// Set solver (LBM) parameters from input.
	//S_LES   = params_int["S_LES"];
	v0                                     = params_dbl["v0"];
	s_1                                    = params_dbl["s1"];
	s_2                                    = params_dbl["s2"];
	s_3                                    = params_dbl["s3"];
	s_4                                    = params_dbl["s4"];
	s_5                                    = params_dbl["s5"];
	s_6                                    = params_dbl["s6"];
	S_COLLISION                            = params_int["S_COLLISION"];
	S_INTERP                               = params_int["S_INTERP"];
	S_INTERP_HYBRID                        = params_int["S_INTERP_HYBRID"];
	S_AVERAGE                              = params_int["S_AVERAGE"];
	S_CRITERION                            = params_int["S_CRITERION"];
	
	// Some shortcuts.
	n_maxcells                             = mesh->n_maxcells;
	n_maxcblocks                           = mesh->n_maxcblocks;
	MAX_LEVELS                             = mesh->MAX_LEVELS;
	MAX_LEVELS_INTERIOR                    = mesh->MAX_LEVELS_INTERIOR;
	N_LEVEL_START                          = mesh->N_LEVEL_START;
	
	// Update modes of interpolation and averaging in advance routine.
	V_INTERP_ADVANCE = V_INTERP_INTERFACE;
	V_AVERAGE_ADVANCE = V_AVERAGE_INTERFACE;
	if (S_INTERP==1) // If using cubic interpolation, the whole block of cells needs to be updated from fine grid.
		V_AVERAGE_ADVANCE = V_AVERAGE_BLOCK;
	
	std::cout << "Initialized solver (LBM) object..." << std::endl;
	
	// Compute the relaxation rates to be applied for all grid levels and their ratios.
	dxf_vec             = new ufloat_t[MAX_LEVELS];
	tau_vec             = new ufloat_t[MAX_LEVELS];
	tau_vec_MRT         = new ufloat_t[6*MAX_LEVELS];
	tau_ratio_vec_C2F   = new ufloat_t[MAX_LEVELS];
	tau_ratio_vec_F2C   = new ufloat_t[MAX_LEVELS];
	
	// Spatial step and BGK relaxation rate arrays.
	for (int L = 0; L < MAX_LEVELS; L++)
	{
		dxf_vec[L] = (ufloat_t)mesh->dxf_vec[L];
		tau_vec[L] = (ufloat_t)(v0*3.0 + 0.5*dxf_vec[L]);
	}
	
	// MRT arrays.
	tau_vec_MRT[0] = s_1;
	tau_vec_MRT[1] = s_2;
	tau_vec_MRT[2] = s_3;
	tau_vec_MRT[3] = s_4;
	tau_vec_MRT[4] = s_5;
	tau_vec_MRT[5] = s_6;
	for (int L = 1; L < MAX_LEVELS; L++)
	{
		for (int p = 0; p < 6; p++)
			tau_vec_MRT[p+L*6] = tau_vec_MRT[p+(L-1)*6]*( (dxf_vec[L]/dxf_vec[L-1])*(tau_vec[L-1]/tau_vec[L]) );
	}
	
	// BGK rate ratios.
	for (int L = 0; L < MAX_LEVELS-1; L++)
	{
		tau_ratio_vec_C2F[L] = tau_vec[L+1]/tau_vec[L];
		tau_ratio_vec_F2C[L] = tau_vec[L]/tau_vec[L+1];
	}

	return 0;
}
