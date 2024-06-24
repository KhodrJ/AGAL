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
	virtual void		S_Interpolate(int i_dev, int L, int var) = 0;
	virtual void		S_Average(int i_dev, int L, int var) = 0;
	virtual void		S_UpdateMesh(int var, std::ofstream *file) = 0;
	
	void			S_FreezeMesh(int var)						{ mesh->M_FreezeRefinedCells(var); }

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
	ufloat_t	tau_ratio_vec_C2F[MAX_LEVELS];
	ufloat_t	tau_ratio_vec_F2C[MAX_LEVELS];
	
	public:
	
	//
	  // ================================
	  // === Public Member Functions: ===
	  // ================================
	//

	// TMP
	int		S_SetValuesDebug(int i_dev, int L, ufloat_t v);
	int		S_SetInitialConditions_d2q9(int i_dev, int L);
	int		S_SetInitialConditions_d3q19(int i_dev, int L);
	int		S_SetInitialConditions_d3q27(int i_dev, int L);
	int		S_SetInitialConditions(int i_dev, int L, bool init);
	int		S_Collide_d2q9(int i_dev, int L);
	int		S_Collide_d3q19(int i_dev, int L);
	int		S_Collide_d3q27(int i_dev, int L);
	int		S_Collide(int i_dev, int L);
	int		S_Stream_Inpl_d2q9(int i_dev, int L);
	int		S_Stream_Inpl_d3q19(int i_dev, int L);
	int		S_Stream_Inpl_d3q27(int i_dev, int L);
	int		S_Stream(int i_dev, int L);
	void		S_Interpolate(int i_dev, int L, int var) 	{ mesh->M_Interpolate(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]); }
	void		S_Average(int i_dev, int L, int var)		{ mesh->M_Average(i_dev, L, var, tau_vec[L], tau_ratio_vec_F2C[L]); }	
	
	int		S_Init();
	int		S_Initialize(int i_dev, int L);
	int		S_Advance(int i_dev, int L, std::ofstream *file, double *tmp=0);
	int		S_ComputeRefCriteria(int i_dev, int L, int var);
	void		S_UpdateMesh(int var, std::ofstream *file)	{ mesh->M_RefineAndCoarsenCells(var, tau_ratio_vec_C2F, file); }
	
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
