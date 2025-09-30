/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef SOLVER_LBM_H
#define SOLVER_LBM_H

#include "solver.h"

enum class VelocitySet
{
    D2Q9,
    D3Q19,
    D3Q27
};
enum class CollisionOperator
{
    BGK,
    TRT,
    MRT
};
enum class InterpOrder
{
    Linear,
    Quadratic,
    Cubic
};
enum class LESModel
{
    None,
    Smagorinsky,
    WALE,
    Vreman
};

constexpr int GetLBMSize(VelocitySet VS)
{
    switch (VS)
    {
        case VelocitySet::D2Q9: return 9;
        case VelocitySet::D3Q19: return 19;
        case VelocitySet::D3Q27: return 27;
        default: return 9;
    }
}

__constant__ double LBMw[27];
__constant__ int LBMpb[27];

struct LBMPack
{
    const ArgsPack AP;
    const VelocitySet VS;
    const CollisionOperator CM;
    const InterpOrder IM;
    const LESModel LM;
    const int N_Q = GetLBMSize(VS);
    
    constexpr LBMPack(
        const ArgsPack *AP_,
        const VelocitySet VS_,
        const CollisionOperator CM_,
        const InterpOrder IM_,
        const LESModel LM_=LESModel::None
    ) : 
        AP(*AP_),
        VS(VS_),
        CM(CM_),
        IM(IM_),
        LM(LM_)
    {
    }
};

// VS is the velocity set (D2Q9, D3Q19, D3Q27).
// CM is the collision model (BGK, TRT, MRT).
// IM is the interpolation model (linear, quadratic, cubic).
// LM is the LES model (none, Smagorinsky, Vreman, WALE).
template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
class Solver_LBM : public Solver<ufloat_t,ufloat_g_t,AP>
{
    private:
    
    int S_Init();
    
    public:
    
    Parser *parser;
    Mesh<ufloat_t,ufloat_g_t,AP> *mesh;    
    
    // From argument pack.
    const int N_DEV                     = AP->N_DEV;            ///< Number of GPU devices.
    const int N_DIM                     = AP->N_DIM;            ///< Number of dimensions.
    const int N_Q_max                   = AP->N_Q_max;          ///< Neighbor-halo size (including self).
    const int Nqx                       = AP->Nqx;              ///< Number of sub-blocks along one axis.
    const int N_CHILDREN                = AP->N_CHILDREN;       ///< Number of children per block.
    const int N_QUADS                   = AP->N_QUADS;          ///< Total number of sub-blocks per cell-block.
    const int M_TBLOCK                  = AP->M_TBLOCK;         ///< Number of threads per thread-block in primary-mode.
    const int M_CBLOCK                  = AP->M_CBLOCK;         ///< Number of cells per cell-block.
    const int M_LBLOCK                  = AP->M_LBLOCK;         ///< Number of cell-blocks processed per thread-block in primary-mode.
    const int M_WBLOCK                  = AP->M_WBLOCK;         ///< Number of threads working within a warp in uprimary-mode.
    const int M_LWBLOCK                 = AP->M_LWBLOCK;        ///< Number of cell-blocks processed per thread-block in uprimary-mode.
    const int M_BLOCK                   = AP->M_BLOCK;          ///< Number of threads per thread-block in secondary-mode.
    const int M_RNDOFF                  = AP->M_RNDOFF;         ///< Round-off constant for memory alignment.
    
    // From mesh object.
    long int   n_maxcells;              ///< Maximum number of cells that can be stored in GPU memory.
    int        n_maxcblocks;            ///< Maximum number of cell-blocks corresponding to @ref n_maxcells.
    int        MAX_LEVELS;              ///< Maximum number of grids for the domain interior and boundary.
    int        MAX_LEVELS_INTERIOR;     ///< Maximum number of grids for the domain interior alone.
    int        N_LEVEL_START;           ///< Grid level to employ as the root grid for advancement.
    
    // Constants.
    const VelocitySet VS               = LP->VS;
    const CollisionOperator CM         = LP->CM;
    const InterpOrder IM               = LP->IM;
    const LESModel LM                  = LP->LM;
    const int N_Q                      = LP->N_Q;
    
    // o====================================================================================
    // | LBM solver parameters and routines.
    // o====================================================================================
    
    // Input parameters.
    int        S_INTERP;                ///< Indicates type of interpolation (0 for linear, 1 for cubic).
    int        S_LES;                   ///< Indicates the turbulence model to employ during collision.
    int        S_FORCE_ORDER;           ///< Indicates the order of accuracy for the momentum exchange algorithm.
    int        S_FORCE_TYPE;            ///< Indicates the type of force calculation approach [0: MEA, 1: CV].
    int        S_BC_TYPE;               ///< Indicates the type of curved-boundary BC type to use [0: SBB, 1: IBB].
    int        S_CRITERION;             ///< Indicates refinement criterion (0 for |w|, 1 for Q).
    int        V_INTERP_ADVANCE;        ///< Controls interpolation parameters.
    int        V_AVERAGE_ADVANCE;       ///< Controls averaging parameters.
    ufloat_t   S_FORCEVOLUME_Xm;        ///< Lower x-bound on control volume for force calculation.
    ufloat_t   S_FORCEVOLUME_XM;        ///< Upper x-bound on control volume for force calculation.
    ufloat_t   S_FORCEVOLUME_Ym;        ///< Lower y-bound on control volume for force calculation.
    ufloat_t   S_FORCEVOLUME_YM;        ///< Upper y-bound on control volume for force calculation.
    ufloat_t   S_FORCEVOLUME_Zm;        ///< Lower z-bound on control volume for force calculation.
    ufloat_t   S_FORCEVOLUME_ZM;        ///< Upper z-bound on control volume for force calculation.
    
    // Grid advancement parameters.
    double     v0;
    double     *s_vec;
    ufloat_t   *dxf_vec;
    ufloat_t   *dvf_vec;
    ufloat_t   *tau_vec;
    ufloat_t   *tau_ratio_vec;
    bool       compute_forces = true;
    
    // LBM: General routines.
    int S_Collide(int i_dev, int L);
    int S_Stream(int i_dev, int L);
    int S_ImposeBC(int i_dev, int L);
    int S_ComputeForcesMEA(int i_dev, int L, int var);
    int S_ComputeForcesCV(int i_dev, int L, int var);
    int S_ReportForces(int i_dev, int L, int i, double t, int START);
    int S_ComputeMacroProperties(int i_dev, int i_kap, int i_Q, int kap_i, ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w);
    int S_ComputeEddyViscosity(int i_dev, int L);
    int S_ComputePressureOnWall(int i_dev, int L, int var);
    
    int S_Average(int i_dev, int L, int var);
    int S_Interpolate_Linear(int i_dev, int L, int var);
    int S_Interpolate_Cubic(int i_dev, int L, int var);
    int S_Debug_DrawGeometry(int i_dev, int L);
    
    // o====================================================================================
    // | Routines required from base class.
    // o====================================================================================
    
    // Required.
    int S_SetIC(int i_dev, int L);
    int S_Advance(int i_dev, int L, double *tmp);
    int S_RefreshVariables(int i_dev, int L, int var=0);
    int S_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out);
    int S_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out);
    int S_ComputeRefCriteria(int i_dev, int L, int var);
    int S_Debug(int i_dev, int L, int var);
    
    // o====================================================================================
    // | Parameter-specific routines (TODO: these will be refactored).
    // o====================================================================================
    
    // Interpolation.
    template <InterpOrder IM=LP->IM, typename std::enable_if<(IM==InterpOrder::Linear), int>::type = 0> int S_InterpolateW(int i_dev, int L, int var)
    {
        S_Interpolate_Linear(i_dev, L, var);
        return 0;
    }
    template <InterpOrder IM=LP->IM, typename std::enable_if<(IM==InterpOrder::Cubic), int>::type = 0> int S_InterpolateW(int i_dev, int L, int var) 
    {
        S_Interpolate_Cubic(i_dev, L, var);
        return 0;
    }
    int S_Interpolate(int i_dev, int L, int var)
    {
        S_InterpolateW(i_dev, L, var);
        return 0;
    }
    
    Solver_LBM(Mesh<ufloat_t,ufloat_g_t,AP> *mesh_) : Solver<ufloat_t,ufloat_g_t,AP>(mesh_)
    {
        mesh = mesh_;
        parser = mesh->parser;
        S_Init();
        std::cout << "[-] Finished making solver (LBM) object." << std::endl << std::endl;
    }
    
    ~Solver_LBM()
    {
        delete[] dxf_vec;
        delete[] tau_vec;
        if (CM==CollisionOperator::MRT)
            delete[] s_vec;
    }
};

#endif
