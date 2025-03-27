#ifndef SOLVER_H
#define SOLVER_H

#include "cppspec.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
class Mesh;

// class Solver;
// class Solver_LBM;

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
class Solver
{
	private:
	
	virtual int S_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str) = 0;
	
	public:
	
	Mesh<ufloat_t,ufloat_g_t,AP> *mesh;
	
	// o====================================================================================
	// | Solver parameters.
	// o====================================================================================
	
	// From argument pack.
	const int N_DEV                     = AP->N_DEV;
	const int N_DIM                     = AP->N_DIM;
	const int N_Q_max                   = AP->N_Q_max;
	const int Nqx                       = AP->Nqx;
	const int N_CHILDREN                = AP->N_DIM;
	const int N_QUADS                   = AP->N_QUADS;
	const int M_TBLOCK                  = AP->M_TBLOCK;
	const int M_HBLOCK                  = AP->M_HBLOCK;
	const int M_CBLOCK                  = AP->M_CBLOCK;
	const int M_LBLOCK                  = AP->M_LBLOCK;
	const int M_BLOCK                   = AP->M_BLOCK;
	const int M_RNDOFF                  = AP->M_RNDOFF;
	
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
	virtual int S_Debug(int i_dev, int L, int var) = 0;
	
	Solver(
		Mesh<ufloat_t,ufloat_g_t,AP> *mesh_,
		std::map<std::string, int> params_int,
		std::map<std::string, double> params_dbl,
		std::map<std::string, std::string> params_str
	)
	{
		mesh = mesh_;
		mesh->solver = this;
	}
	
	~Solver()
	{
		
	}
};

#endif
