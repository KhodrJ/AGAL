#include "mesh.h"
#include "solver_lbm.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str)
{
	// o====================================================================================
	// | Set solver (LBM) parameters from input.
	// o====================================================================================
	
	// Lift from the mesh class.
	n_maxcells                             = mesh->n_maxcells;
	n_maxcblocks                           = mesh->n_maxcblocks;
	MAX_LEVELS                             = mesh->MAX_LEVELS;
	MAX_LEVELS_INTERIOR                    = mesh->MAX_LEVELS_INTERIOR;
	N_LEVEL_START                          = mesh->N_LEVEL_START;
	
	// Shared.
	dxf_vec = new ufloat_t[MAX_LEVELS];
	dvf_vec = new ufloat_t[MAX_LEVELS];
	for (int L = 0; L < MAX_LEVELS; L++)
	{
		dxf_vec[L] = (ufloat_t)mesh->dxf_vec[L];
		dvf_vec[L] = (ufloat_t)1.0;
		for (int d = 0; d < N_DIM; d++)
			dvf_vec[L] *= dxf_vec[L];
		std::cout << "DX/V: " << dxf_vec[L] << " " << dvf_vec[L] << std::endl;
	}
	
	// General.
	v0                                     = params_dbl["v0"];
	S_CRITERION                            = params_int["S_CRITERION"];
	S_LES                                  = params_int["S_LES"];
	S_FORCE_ORDER                          = params_int["S_FORCE_ORDER"];
	S_FORCE_TYPE                           = params_int["S_FORCE_TYPE"];
	S_BC_TYPE                              = params_int["S_BC_TYPE"];
	S_FORCEVOLUME_Xm                       = 4.0*(int(params_dbl["S_FORCEVOLUME_Xm"]*(mesh->Nxi[0]/4)))*dxf_vec[0];
	S_FORCEVOLUME_XM                       = 4.0*(int(params_dbl["S_FORCEVOLUME_XM"]*(mesh->Nxi[0]/4)))*dxf_vec[0];
	S_FORCEVOLUME_Ym                       = 4.0*(int(params_dbl["S_FORCEVOLUME_Ym"]*(mesh->Nxi[1]/4)))*dxf_vec[0];
	S_FORCEVOLUME_YM                       = 4.0*(int(params_dbl["S_FORCEVOLUME_YM"]*(mesh->Nxi[1]/4)))*dxf_vec[0];
	S_FORCEVOLUME_Zm                       = 4.0*(int(params_dbl["S_FORCEVOLUME_Zm"]*(mesh->Nxi[2]/4)))*dxf_vec[0];
	S_FORCEVOLUME_ZM                       = 4.0*(int(params_dbl["S_FORCEVOLUME_ZM"]*(mesh->Nxi[2]/4)))*dxf_vec[0];
// 	S_LES_P1                               = params_dbl["S_LES_P1"]; // TODO
// 	S_LES_P2                               = params_dbl["S_LES_P2"]; // TODO
// 	S_LES_P3                               = params_dbl["S_LES_P3"]; // TODO
	
	// BGK.
	if (CM==CM_BGK)
	{
		tau_vec = new ufloat_t[MAX_LEVELS];
		for (int L = 0; L < MAX_LEVELS; L++)
			tau_vec[L] = (ufloat_t)(v0*3.0 + 0.5*dxf_vec[L]);
	}
	
	// MRT.
	if (CM==CM_MRT)
	{
		tau_vec = new ufloat_t[6*MAX_LEVELS];
		s_vec = new double[6];
		s_vec[0] = params_dbl["s1"];
		s_vec[1] = params_dbl["s2"];
		s_vec[2] = params_dbl["s3"];
		s_vec[3] = params_dbl["s4"];
		s_vec[4] = params_dbl["s5"];
		s_vec[5] = params_dbl["s6"];
		tau_vec[0] = s_vec[0];
		tau_vec[1] = s_vec[1];
		tau_vec[2] = s_vec[2];
		tau_vec[3] = s_vec[3];
		tau_vec[4] = s_vec[4];
		tau_vec[5] = s_vec[5];
		for (int L = 1; L < MAX_LEVELS; L++)
		{
			for (int p = 0; p < 6; p++)
				tau_vec[p+L*6] = tau_vec[p+(L-1)*6]*( (dxf_vec[L]/dxf_vec[L-1])*(tau_vec[L-1]/tau_vec[L]) );
		}
	}
	
	// Update modes of interpolation and averaging in advance routine.
	V_INTERP_ADVANCE = V_INTERP_INTERFACE;
	V_AVERAGE_ADVANCE = V_AVERAGE_INTERFACE;
	if (IM != IM_LINEAR) // If not using linear interpolation, the whole block of cells needs to be updated from fine grid.
		V_AVERAGE_ADVANCE = V_AVERAGE_BLOCK;
	
	std::cout << "Initialized solver (LBM) object..." << std::endl;

	return 0;
}
