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
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		/*
		n_ids[i_dev]                   = mesh->n_ids[i_dev];
		c_cells_ID_mask[i_dev]         = mesh->c_cells_ID_mask[i_dev];
		c_cells_f_F[i_dev]             = mesh->c_cells_f_F[i_dev];
		c_cblock_f_X[i_dev]            = mesh->c_cblock_f_X[i_dev];
		c_cblock_ID_mask[i_dev]        = mesh->c_cblock_ID_mask[i_dev];
		c_cblock_ID_nbr[i_dev]         = mesh->c_cblock_ID_nbr[i_dev];
		c_cblock_ID_nbr_child[i_dev]   = mesh->c_cblock_ID_nbr_child[i_dev];
		c_cblock_ID_onb[i_dev]         = mesh->c_cblock_ID_onb[i_dev];
		c_cblock_ID_ref[i_dev]         = mesh->c_cblock_ID_ref[i_dev];
		c_cblock_level[i_dev]          = mesh->c_cblock_level[i_dev];
		*/
	}
	
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
