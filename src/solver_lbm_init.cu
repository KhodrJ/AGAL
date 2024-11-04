#include "mesh.h"
#include "solver.h"

int Solver_LBM::S_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl)
{
	// Set solver (LBM) parameters from input.
	//S_LES   = params_int["S_LES"];
	v0      = params_dbl["v0"];
	
	// Some shortcuts.
	n_maxcells                             = mesh->n_maxcells;
	n_maxcblocks                           = mesh->n_maxcblocks;
	MAX_LEVELS                             = mesh->MAX_LEVELS;
	MAX_LEVELS_INTERIOR                    = mesh->MAX_LEVELS_INTERIOR;
	N_LEVEL_START                          = mesh->N_LEVEL_START;
	S_INTERP                               = mesh->S_INTERP;
	S_AVERAGE                              = mesh->S_AVERAGE;
	
	// Update modes of interpolation and averaging in advance routine.
	V_INTERP_ADVANCE = V_INTERP_INTERFACE;
	V_AVERAGE_ADVANCE = V_AVERAGE_INTERFACE;
	if (S_INTERP==1) // If using cubic interpolation, the whole block of cells needs to be updated from fine grid.
		V_AVERAGE_ADVANCE = V_AVERAGE_BLOCK;
	
	std::cout << "Initialized solver (LBM) object..." << std::endl;
	
	// Compute the relaxation rates to be applied for all grid levels and their ratios.
	dxf_vec             = new ufloat_t[MAX_LEVELS];
	tau_vec             = new ufloat_t[MAX_LEVELS];
	tau_ratio_vec_C2F   = new ufloat_t[MAX_LEVELS];
	tau_ratio_vec_F2C   = new ufloat_t[MAX_LEVELS];
	for (int L = 0; L < MAX_LEVELS; L++)
	{
		dxf_vec[L] = (ufloat_t)mesh->dxf_vec[L];
		tau_vec[L] = (ufloat_t)(v0*3.0 + 0.5*dxf_vec[L]);
	}
	for (int L = 0; L < MAX_LEVELS-1; L++)
	{
		tau_ratio_vec_C2F[L] = tau_vec[L+1]/tau_vec[L];
		tau_ratio_vec_F2C[L] = tau_vec[L]/tau_vec[L+1];
	}

	return 0;
}
