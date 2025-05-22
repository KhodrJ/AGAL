#ifndef SOLVER_LBM_H
#define SOLVER_LBM_H

#include "solver.h"

constexpr int VS_D2Q9 = 0;
constexpr int VS_D3Q19 = 1;
constexpr int VS_D3Q27 = 2;

constexpr int CM_BGK = 0;
constexpr int CM_TRT = 1;
constexpr int CM_MRT = 2;

constexpr int IM_LINEAR = 0;
constexpr int IM_QUADRATIC = 1;
constexpr int IM_CUBIC = 2;

constexpr int GetLBMSize(int VS)
{
	switch (VS)
	{
		case VS_D2Q9: return 9;
		case VS_D3Q19: return 19;
		case VS_D3Q27: return 27;
		default: return 9;
	}
}

struct LBMPack
{
	const ArgsPack AP;
	const int VS;
	const int CM;
	const int IM;
	const int N_Q = GetLBMSize(VS);
	
	constexpr LBMPack(
		const ArgsPack *AP_,
		const int VS_,
		const int CM_,
		const int IM_
	) : 
		AP(*AP_),
		VS(VS_),
		CM(CM_),
		IM(IM_)
	{
	}
};

// VS is the velocity set (D2Q9, D3Q19, D3Q27).
// CM is the collision model (BGK, TRT, MRT).
// IM is the interpolation model (linear, quadratic, cubic).
template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
class Solver_LBM : public Solver<ufloat_t,ufloat_g_t,AP>
{
	private:
	
	int S_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
	
	public:
	
	// From argument pack.
	Mesh<ufloat_t,ufloat_g_t,AP>        *mesh;
	const int N_DEV                     = AP->N_DEV;
	const int N_DIM                     = AP->N_DIM;
	const int N_Q_max                   = AP->N_Q_max;
	const int Nqx                       = AP->Nqx;
	const int N_CHILDREN                = AP->N_CHILDREN;
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
	
	// Constants.
	const int VS         = LP->VS;
	const int CM         = LP->CM;
	const int IM         = LP->IM;
	
	// o====================================================================================
	// | LBM solver parameters and routines.
	// o====================================================================================
	
	// Input parameters.
	int        S_INTERP;                ///< Indicates type of interpolation (0 for linear, 1 for cubic).
	int        S_LES;                   ///< Indicates the turbulence model to employ during collision.
	int        S_FORCE_TYPE;            ///< Indicates the order of accuracy for the momentum exchange algorithm.
	int        S_CRITERION;             ///< Indicates refinement criterion (0 for |w|, 1 for Q).
	int        V_INTERP_ADVANCE;        ///< Controls interpolation parameters.
	int        V_AVERAGE_ADVANCE;       ///< Controls averaging parameters.
	
	// Grid advancement parameters.
	double     v0;
	double     *s_vec;
	ufloat_t   *dxf_vec;
	ufloat_t   *tau_vec;
	ufloat_t   *tau_ratio_vec;
	bool       compute_forces = true;
	
	// LBM: General routines.
	//int S_Collide(int i_dev, int L);
	//int S_Stream(int i_dev, int L);
	int S_Vel2Mom(int i_dev, int L);
	int S_Mom2Vel(int i_dev, int L);
	int S_ComputeMacroProperties(int i_dev, int i_kap, int i_Q, int kap_i, ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w);
	int S_IdentifyFaces(int i_dev, int L);
	
	// o====================================================================================
	// | Routines required from base class.
	// o====================================================================================
	
	// Required.
	//int S_SetIC(int i_dev, int L);
	//int S_Interpolate(int i_dev, int L, int var);
	//int S_Average(int i_dev, int L, int var);
	int S_Advance(int i_dev, int L, double *tmp);
	int S_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out);
	int S_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out);
	int S_ComputeForces(int i_dev, int L);
	int S_ComputeRefCriteria(int i_dev, int L, int var);
	int S_Debug(int i_dev, int L, int var);
	
	// o====================================================================================
	// | Parameter-specific routines.
	// o====================================================================================
	
	// Initial conditions.
	int S_SetInitialConditions_D2Q9(int i_dev, int L);
	int S_SetInitialConditions_D3Q19(int i_dev, int L);
	int S_SetInitialConditions_D3Q27(int i_dev, int L);
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D2Q9), int>::type = 0> int S_SetICW(int i_dev, int L) { S_SetInitialConditions_D2Q9(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q19), int>::type = 0> int S_SetICW(int i_dev, int L) { S_SetInitialConditions_D3Q19(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q27), int>::type = 0> int S_SetICW(int i_dev, int L) { S_SetInitialConditions_D3Q27(i_dev, L); return 0; }
	int S_SetIC(int i_dev, int L) { S_SetICW(i_dev, L); return 0; }
	
	// Collision.
	int S_Collide_BGK_D2Q9(int i_dev, int L);
	int S_Collide_TRT_D2Q9(int i_dev, int L); // TODO
	int S_Collide_MRT_D2Q9(int i_dev, int L); // TODO
	int S_Collide_BGK_D3Q19(int i_dev, int L);
	int S_Collide_TRT_D3Q19(int i_dev, int L); // TODO
	int S_Collide_MRT_D3Q19(int i_dev, int L); // TODO
	int S_Collide_BGK_D3Q27(int i_dev, int L);
	int S_Collide_TRT_D3Q27(int i_dev, int L); // TODO
	int S_Collide_MRT_D3Q27(int i_dev, int L); // TODO
	int S_Collision_Original_D2Q9(int i_dev, int L);
	int S_Collision_Original_D3Q19(int i_dev, int L);
	int S_Collision_Original_D3Q27(int i_dev, int L);
	int S_Collision_New_S1_D2Q9(int i_dev, int L);
	int S_Collision_New_S2_D2Q9(int i_dev, int L);
	int S_Collision_New_S1_D3Q19(int i_dev, int L);
	int S_Collision_New_S2_D3Q19(int i_dev, int L);
	int S_Collision_New_S1_D3Q27(int i_dev, int L);
	int S_Collision_New_S2_D3Q27(int i_dev, int L);
	int S_ImposeBC_D2Q9(int i_dev, int L);
	int S_ImposeBC_D3Q19(int i_dev, int L);
	int S_ImposeBC_D3Q27(int i_dev, int L);
	int S_ReportForces(int i_dev, int L) { mesh->M_ReportForces(i_dev, L); return 0; }
// 	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D2Q9), int>::type = 0> int S_Collide(int i_dev, int L) { S_Collision_Original_D2Q9(i_dev, L); return 0; }
// 	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q19), int>::type = 0> int S_Collide(int i_dev, int L) { S_Collision_Original_D3Q19(i_dev, L); return 0; }
// 	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q27), int>::type = 0> int S_Collide(int i_dev, int L) { S_Collision_Original_D3Q27(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D2Q9), int>::type = 0> int S_Collide(int i_dev, int L) { S_Collision_New_S1_D2Q9(i_dev, L); S_Collision_New_S2_D2Q9(i_dev, L); S_ImposeBC_D2Q9(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q19), int>::type = 0> int S_Collide(int i_dev, int L) { S_Collision_New_S1_D3Q19(i_dev, L); S_Collision_New_S2_D3Q19(i_dev, L); S_ImposeBC_D3Q19(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q27), int>::type = 0> int S_Collide(int i_dev, int L) { S_Collision_New_S1_D3Q27(i_dev, L); S_Collision_New_S2_D3Q27(i_dev, L); S_ImposeBC_D3Q27(i_dev, L); return 0; }
	
	// Streaming.
	int S_Stream_D2Q9(int i_dev, int L);
	int S_Stream_D3Q19(int i_dev, int L);
	int S_Stream_D3Q27(int i_dev, int L);
	int S_Stream_Original_D2Q9(int i_dev, int L);
	int S_Stream_Original_D3Q19(int i_dev, int L);
	int S_Stream_Original_D3Q27(int i_dev, int L);
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D2Q9), int>::type = 0> int S_Stream(int i_dev, int L) { S_Stream_Original_D2Q9(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q19), int>::type = 0> int S_Stream(int i_dev, int L) { S_Stream_Original_D3Q19(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q27), int>::type = 0> int S_Stream(int i_dev, int L) { S_Stream_Original_D3Q27(i_dev, L); return 0; }
	
	// Interpolation.
	int S_Interpolate_Linear_D2Q9(int i_dev, int L, int var);
	int S_Interpolate_Linear_D3Q19(int i_dev, int L, int var);
	int S_Interpolate_Linear_D3Q27(int i_dev, int L, int var);
	int S_Interpolate_Quadratic_D2Q9(int i_dev, int L, int var); // TODO
	int S_Interpolate_Quadratic_D3Q19(int i_dev, int L, int var); // TODO
	int S_Interpolate_Quadratic_D3Q27(int i_dev, int L, int var); // TODO
	int S_Interpolate_Cubic_D2Q9(int i_dev, int L, int var);
	int S_Interpolate_Cubic_D3Q19(int i_dev, int L, int var);
	int S_Interpolate_Cubic_D3Q27(int i_dev, int L, int var);
	int S_Interpolate_Linear_Original_D2Q9(int i_dev, int L, int var);
	int S_Interpolate_Linear_Original_D3Q19(int i_dev, int L, int var);
	int S_Interpolate_Linear_Original_D3Q27(int i_dev, int L, int var);
	int S_Interpolate_Cubic_Original_D2Q9(int i_dev, int L, int var);
	int S_Interpolate_Cubic_Original_D3Q19(int i_dev, int L, int var);
	int S_Interpolate_Cubic_Original_D3Q27(int i_dev, int L, int var);
	template <int VS=LP->VS, int IM=LP->IM, typename std::enable_if<(VS==VS_D2Q9 && IM==IM_LINEAR), int>::type = 0> int S_InterpolateW(int i_dev, int L, int var) { S_Interpolate_Linear_Original_D2Q9(i_dev, L, var); return 0; }
	template <int VS=LP->VS, int IM=LP->IM, typename std::enable_if<(VS==VS_D3Q19 && IM==IM_LINEAR), int>::type = 0> int S_InterpolateW(int i_dev, int L, int var) { S_Interpolate_Linear_Original_D3Q19(i_dev, L, var); return 0; }
	template <int VS=LP->VS, int IM=LP->IM, typename std::enable_if<(VS==VS_D3Q27 && IM==IM_LINEAR), int>::type = 0> int S_InterpolateW(int i_dev, int L, int var) { S_Interpolate_Linear_Original_D3Q27(i_dev, L, var); return 0; }
	template <int VS=LP->VS, int IM=LP->IM, typename std::enable_if<(VS==VS_D2Q9 && IM==IM_CUBIC), int>::type = 0> int S_InterpolateW(int i_dev, int L, int var) { S_Interpolate_Cubic_Original_D2Q9(i_dev, L, var); return 0; }
	template <int VS=LP->VS, int IM=LP->IM, typename std::enable_if<(VS==VS_D3Q19 && IM==IM_CUBIC), int>::type = 0> int S_InterpolateW(int i_dev, int L, int var) { S_Interpolate_Cubic_Original_D3Q19(i_dev, L, var); return 0; }
	template <int VS=LP->VS, int IM=LP->IM, typename std::enable_if<(VS==VS_D3Q27 && IM==IM_CUBIC), int>::type = 0> int S_InterpolateW(int i_dev, int L, int var) { S_Interpolate_Cubic_Original_D3Q27(i_dev, L, var); return 0; }
	int S_Interpolate(int i_dev, int L, int var) { S_InterpolateW(i_dev, L, var); return 0; }
	
	// Averaging.
	int S_Average_D2Q9(int i_dev, int L, int var);
	int S_Average_D3Q19(int i_dev, int L, int var);
	int S_Average_D3Q27(int i_dev, int L, int var);
	int S_Average_Original_D2Q9(int i_dev, int L, int var);
	int S_Average_Original_D3Q19(int i_dev, int L, int var);
	int S_Average_Original_D3Q27(int i_dev, int L, int var);
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D2Q9), int>::type = 0> int S_AverageW(int i_dev, int L, int var) { S_Average_Original_D2Q9(i_dev, L, var); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q19), int>::type = 0> int S_AverageW(int i_dev, int L, int var) { S_Average_Original_D3Q19(i_dev, L, var); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q27), int>::type = 0> int S_AverageW(int i_dev, int L, int var) { S_Average_Original_D3Q27(i_dev, L, var); return 0; }
	int S_Average(int i_dev, int L, int var) { S_AverageW(i_dev, L, var); return 0; }
	
	// Optimized combos.
	int S_Collide_BGK_Interpolate_Linear_D2Q9(int i_dev, int L); // TODO
	int S_Collide_BGK_Interpolate_Cubic_D2Q9(int i_dev, int L); // TODO
	int S_Collide_TRT_Interpolate_Linear_D2Q9(int i_dev, int L); // TODO
	int S_Collide_TRT_Interpolate_Cubic_D2Q9(int i_dev, int L); // TODO
	int S_Collide_MRT_Interpolate_Linear_D2Q9(int i_dev, int L); // TODO
	int S_Collide_MRT_Interpolate_Cubic_D2Q9(int i_dev, int L); // TODO
	
	// DEBUG: Draw geometry.
	int S_Debug_DrawGeometry_D2Q9(int i_dev, int L);
	int S_Debug_DrawGeometry_D3Q19(int i_dev, int L);
	int S_Debug_DrawGeometry_D3Q27(int i_dev, int L);
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D2Q9), int>::type = 0> int S_Debug_DrawGeometry(int i_dev, int L) { S_Debug_DrawGeometry_D2Q9(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q19), int>::type = 0> int S_Debug_DrawGeometry(int i_dev, int L) { S_Debug_DrawGeometry_D3Q19(i_dev, L); return 0; }
	template <int VS=LP->VS, typename std::enable_if<(VS==VS_D3Q27), int>::type = 0> int S_Debug_DrawGeometry(int i_dev, int L) { S_Debug_DrawGeometry_D3Q27(i_dev, L); return 0; }
	
	
	
	Solver_LBM(Mesh<ufloat_t,ufloat_g_t,AP> *mesh_, std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str) : Solver<ufloat_t,ufloat_g_t,AP>(mesh_, params_int, params_dbl, params_str)
	{
		mesh = mesh_;
		S_Init(params_int, params_dbl, params_str);
		std::cout << "[-] Finished making solver (LBM) object." << std::endl << std::endl;
	}
	
	~Solver_LBM()
	{
		delete[] dxf_vec;
		delete[] tau_vec;
		if (CM==CM_MRT)
			delete[] s_vec;
	}
};

#endif
