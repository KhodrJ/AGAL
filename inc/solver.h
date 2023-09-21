/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef SOLVER_H
#define SOLVER_H

#include "mesh.h"

class Solver
{	
	public:
	
	Mesh			*mesh;
	
	//
	  // ================================
	  // === Public Member Functions: ===
	  // ================================
	//
		
	//!
	virtual int		S_Init() = 0;
	virtual int		S_Initialize(int i_dev, int L) = 0;
	virtual int 		S_Advance(int i_dev, int L, std::ofstream *file, double *tmp=0) = 0;
	virtual int		S_ComputeRefCriteria(int i_dev, int L, int var) = 0;
	virtual int		S_EvaluateRefCriteria(int i_dev, int var) = 0;
	virtual void		S_Interpolate(int i_dev, int L, int var) = 0;
	virtual void		S_Average(int i_dev, int L, int var) = 0;
	virtual void		S_UpdateMesh(int var, std::ofstream *file) = 0;
	
	void			S_FreezeMesh(int var)						{ mesh->M_FreezeRefinedCells(var); }
	//void			S_Interpolate(int i_dev, int L, int var, ufloat_t Cscale)	{ mesh->M_Interpolate(i_dev, L, var, Cscale); }
	//void			S_Average(int i_dev, int L, int var, ufloat_t Cscale)		{ mesh->M_Average(i_dev, L, var, Cscale); }

	//
	  // =====================
	  // === Constructors: ===
	  // =====================
	//
	
	Solver(Mesh *mesh_)
	{
		mesh = mesh_;
	}
	
	~Solver()
	{
		
	}
};

class Solver_LBM : public Solver
{
	//
	  // =================================
	  // === Private Member Functions: ===
	  // =================================
	//

	ufloat_t	dx_vec[MAX_LEVELS];
	ufloat_t	tau_vec[MAX_LEVELS];
	
	public:
	
	//
	  // ================================
	  // === Public Member Functions: ===
	  // ================================
	//

	// TMP
	int		S_SetValuesDebug(int i_dev, int L, ufloat_t v);
	int		S_SetInitialConditions(int i_dev, int L, bool init);
	int		S_ComputeU(int i_dev, int L);
	int		S_ComputeW(int i_dev, int L);
	int		S_Collide(int i_dev, int L);
	int		S_Stream(int i_dev, int L);
	int		S_ComputeEq(int i_dev, int L);
	void		S_Interpolate(int i_dev, int L, int var) 	{ mesh->M_Interpolate(i_dev, L, var, tau_vec[L+1]/tau_vec[L]); } // 0.5
	void		S_Average(int i_dev, int L, int var)		{ mesh->M_Average(i_dev, L, var, tau_vec[L]/tau_vec[L+1]); }
	//	
	
	int		S_Init();
	int		S_Initialize(int i_dev, int L);
	int		S_Advance(int i_dev, int L, std::ofstream *file, double *tmp=0);
	int		S_ComputeRefCriteria(int i_dev, int L, int var);
	int		S_EvaluateRefCriteria(int i_dev, int var);
	void		S_UpdateMesh(int var, std::ofstream *file)	{ mesh->M_RefineAndCoarsenCells(var, tau_vec, file); }
	
	//
	  // =====================
	  // === Constructors: ===
	  // =====================
	//
	
	Solver_LBM(Mesh *mesh) : Solver(mesh)
	{
		S_Init();
	}
};

#endif
