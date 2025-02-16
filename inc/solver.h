#ifndef SOLVER_H
#define SOLVER_H

#include "cppspec.h"

class Mesh;

class Solver
{
	private:
	
	virtual int S_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl) = 0;
	
	public:
	
	Mesh *mesh;
	
	// o====================================================================================
	// | Solver parameters.
	// o====================================================================================
	
	// From mesh object.
	long int   n_maxcells;              ///< Maximum number of cells that can be stored in GPU memory.
	int        n_maxcblocks;            ///< Maximum number of cell-blocks corresponding to @ref n_maxcells.
	int        MAX_LEVELS;              ///< Maximum number of grids for the domain interior and boundary.
	int        MAX_LEVELS_INTERIOR;     ///< Maximum number of grids for the domain interior alone.
	int        N_LEVEL_START;           ///< Grid level to employ as the root grid for advancement.
	
	// o====================================================================================
	// | Routines.
	// o====================================================================================
	
	virtual int S_SetIC(int i_dev, int L) = 0;
	virtual int S_Interpolate(int i_dev, int L, int var) = 0;
	virtual int S_Average(int i_dev, int L, int var) = 0;
	virtual int S_Advance(int i_dev, int L, double *tmp) = 0;
	virtual int S_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out_u) = 0;
	virtual int S_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out_u) = 0;
	virtual int S_ComputeForces(int i_dev, int L) = 0;
	virtual int S_ComputeRefCriteria(int i_dev, int L, int var) = 0;
	virtual int S_Debug(int var) = 0;
	
	Solver(Mesh *mesh_, std::map<std::string, int> params_int, std::map<std::string, double> params_dbl)
	{
		mesh = mesh_;
	}
	
	~Solver()
	{
		
	}
};

class Solver_LBM : public Solver
{
	private:
	
	int S_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl);
	
	public:
	
	// o====================================================================================
	// | LBM solver parameters and routines.
	// o====================================================================================
	
	// Input parameters.
	int        S_COLLISION;             ///< Indicates collision model (0 for BGK, 1 for MRT).
	int        S_INTERP;                ///< Indicates type of interpolation (0 for linear, 1 for cubic).
	int        S_INTERP_HYBRID;         ///< Indicates whether to use combined interpolation and collision.
	int        S_AVERAGE;               ///< Indicates type of average (0 for standard, 1 for 'cubic' which didn't work).
	int        S_CRITERION;             ///< Indicates refinement criterion (0 for |w|, 1 for Q).
	int        V_INTERP_ADVANCE;        ///< Controls interpolation parameters.
	int        V_AVERAGE_ADVANCE;       ///< Controls averaging parameters.
	
	// Grid advancement parameters.
	//int        S_LES;
	double     v0;
	double     s_1;
	double     s_2;
	double     s_3;
	double     s_4;
	double     s_5;
	double     s_6;
	ufloat_t   *dxf_vec;
	ufloat_t   *tau_vec;
	ufloat_t   *tau_vec_MRT;
	ufloat_t   *tau_ratio_vec_C2F;
	ufloat_t   *tau_ratio_vec_F2C;
	
	// LBM-specific routines.
	int S_Collide(int i_dev, int L);
	int S_Stream(int i_dev, int L);
	int S_SetInitialConditions_d2q9(int i_dev, int L);
	int S_SetInitialConditions_d3q19(int i_dev, int L);
	int S_SetInitialConditions_d3q27(int i_dev, int L);
	int S_Collide_BGK_d2q9(int i_dev, int L);
	int S_Collide_BGK_d3q19(int i_dev, int L);
	int S_Collide_BGK_d3q27(int i_dev, int L);
	int S_Collide_MRT_d2q9(int i_dev, int L);
	int S_Collide_MRT_d3q19(int i_dev, int L);
	int S_Vel2Mom_MRT_d2q9(int i_dev, int L);
	int S_Vel2Mom_MRT_d3q19(int i_dev, int L);
	int S_Mom2Vel_MRT_d2q9(int i_dev, int L);
	int S_Mom2Vel_MRT_d3q19(int i_dev, int L);
	int S_EnforceSymmetry_d2q9(int i_dev, int L);
	int S_EnforceSymmetry_d3q19(int i_dev, int L);
	int S_EnforceSymmetry_d3q27(int i_dev, int L);
	int S_Stream_Inpl_d2q9(int i_dev, int L);
	int S_Stream_Inpl_d3q19(int i_dev, int L);
	int S_Stream_Inpl_d3q27(int i_dev, int L);
	int S_Interpolate_Linear_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Linear_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Linear_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Cubic_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Cubic_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Cubic_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Linear_MRT_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Linear_MRT_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Cubic_MRT_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Cubic_MRT_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_MRT_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_MRT_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_MRT_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_Cubic_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_Cubic_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_Cubic_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	
	// LBM-specific optimized routines.
	int S_Collide_BGK_Interpolate_Linear_d2q9(int i_dev, int L);
	int S_Collide_BGK_Interpolate_Linear_d3q19(int i_dev, int L);
	int S_Collide_BGK_Interpolate_Linear_d3q27(int i_dev, int L);
	int S_Collide_BGK_Interpolate_Cubic_d2q9(int i_dev, int L);
	int S_Collide_BGK_Interpolate_Cubic_d3q19(int i_dev, int L);
	int S_Collide_BGK_Interpolate_Cubic_d3q27(int i_dev, int L);
	int S_Collide_MRT_Interpolate_Linear_d2q9(int i_dev, int L);
	int S_Collide_MRT_Interpolate_Linear_d3q19(int i_dev, int L);
	int S_Collide_MRT_Interpolate_Linear_d3q27(int i_dev, int L);
	int S_Collide_MRT_Interpolate_Cubic_d2q9(int i_dev, int L);
	int S_Collide_MRT_Interpolate_Cubic_d3q19(int i_dev, int L);
	int S_Collide_MRT_Interpolate_Cubic_d3q27(int i_dev, int L);
	
	// o====================================================================================
	// | Required routines.
	// o====================================================================================
	
	// Required.
	int S_SetIC(int i_dev, int L);
	int S_Interpolate(int i_dev, int L, int var);
	int S_Average(int i_dev, int L, int var);
	int S_Advance(int i_dev, int L, double *tmp);
	int S_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out);
	int S_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out);
	int S_ComputeForces(int i_dev, int L);
	int S_ComputeRefCriteria(int i_dev, int L, int var);
	int S_Debug(int var);
	
	Solver_LBM(Mesh *mesh_, std::map<std::string, int> params_int, std::map<std::string, double> params_dbl) : Solver(mesh_, params_int, params_dbl)
	{
		S_Init(params_int, params_dbl);
		std::cout << "[-] Finished making solver (LBM) object." << std::endl << std::endl;
	}
	
	~Solver_LBM()
	{
		delete[] dxf_vec;
		delete[] tau_vec;
		delete[] tau_ratio_vec_C2F;
		delete[] tau_ratio_vec_F2C;
	}
};

#endif
