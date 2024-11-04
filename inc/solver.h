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
	long int   n_maxcells;
	int        n_maxcblocks;
	int        MAX_LEVELS;
	int        MAX_LEVELS_INTERIOR;
	int        N_LEVEL_START;
	int        S_INTERP;
	int        S_AVERAGE;
	int        V_INTERP_ADVANCE;
	int        V_AVERAGE_ADVANCE;
	
	virtual int S_SetIC(int i_dev, int L) = 0;
	virtual int S_Interpolate(int i_dev, int L, int var) = 0;
	virtual int S_Average(int i_dev, int L, int var) = 0;
	virtual int S_Advance(int i_dev, int L, std::ofstream *file, double *tmp) = 0;
	virtual int S_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out_u) = 0;
	virtual int S_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out_u) = 0;
	virtual int S_ComputeForces(int i_dev, int L, std::ofstream *out) = 0;
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
	
	// Grid advancement parameters.
	//int        S_LES;
	ufloat_t   v0;
	ufloat_t   *dxf_vec;
	ufloat_t   *tau_vec;
	ufloat_t   *tau_ratio_vec_C2F;
	ufloat_t   *tau_ratio_vec_F2C;
	
	// LBM-specific routines.
	int S_Collide(int i_dev, int L);
	int S_Stream(int i_dev, int L);
	int S_SetInitialConditions_d2q9(int i_dev, int L);
	int S_SetInitialConditions_d3q19(int i_dev, int L);
	int S_SetInitialConditions_d3q27(int i_dev, int L);
	int S_Collide_d2q9(int i_dev, int L);
	int S_Collide_d3q19(int i_dev, int L);
	int S_Collide_d3q27(int i_dev, int L);
	int S_Stream_Inpl_d2q9(int i_dev, int L);
	int S_Stream_Inpl_d3q19(int i_dev, int L);
	int S_Stream_Inpl_d3q27(int i_dev, int L);
	int S_Interpolate_Linear_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Linear_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Linear_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Cubic_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Cubic_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Interpolate_Cubic_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_Cubic_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_Cubic_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int S_Average_Cubic_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	
	
	// Required.
	int S_SetIC(int i_dev, int L);
	int S_Interpolate(int i_dev, int L, int var);
	int S_Average(int i_dev, int L, int var);
	int S_Advance(int i_dev, int L, std::ofstream *file, double *tmp);
	int S_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out);
	int S_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out);
	int S_ComputeForces(int i_dev, int L, std::ofstream *out);
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
