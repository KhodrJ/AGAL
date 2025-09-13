/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef MESH_H
#define MESH_H

#include "cppspec.h"
#include "parser.h"
#include "geometry.h"
#include "solver.h"

// Mesh refinement/coarsening.
enum class RefIds
{
    Unrefined,
    UnrefinedViolating,
    Refined,
    RefinedWChild,
    RefinedPermanent,
    RefinedWChildPermanent,
    MarkRefine,
    MarkCoarsen,
    MarkNew,
    MarkRemove,
    Inactve,
    IndeterminateEven,
    IndeterminateOdd
};
constexpr int V_REF_ID_UNREFINED             = 0;    ///< Indicates cell-block is unrefined.
constexpr int V_REF_ID_UNREFINED_VIO         = 9;    ///< Indicates cell-block is unrefined by possibly violating a quality-control criterion.
constexpr int V_REF_ID_REFINED               = 1;    ///< Indicates cell-block is refined.
constexpr int V_REF_ID_REFINED_WCHILD        = 2;    ///< Indicates cell-block is refined and has at least one refined child.
constexpr int V_REF_ID_REFINED_PERM          = 3;    ///< Indicates cell-block is refined permanently (mainly for near-wall stability).
constexpr int V_REF_ID_REFINED_WCHILD_PERM   = 10;   ///< Indicates cell-block is refined permanently (mainly for near-wall stability).
constexpr int V_REF_ID_MARK_REFINE           = 4;    ///< Indicates cell-block is marked for refinement.
constexpr int V_REF_ID_MARK_COARSEN          = 5;    ///< Indicates cell-block is marked for coarsening.
constexpr int V_REF_ID_NEW                   = 6;    ///< Indicates cell-block was newly inserted (as a child).
constexpr int V_REF_ID_REMOVE                = 7;    ///< Indicates cell-block will be removed (as a child).
constexpr int V_REF_ID_INACTIVE              = 8;    ///< Indicates cell-block is inactive in the simulation.
constexpr int V_REF_ID_INDETERMINATE_E       = 11;   ///< Indicates cell-block is an indeterminate state (even).
constexpr int V_REF_ID_INDETERMINATE_O       = 12;   ///< Indicates cell-block is an indeterminate state (odd).

// Mesh communication.
enum class MaskComm
{
    InterpolateInterface,
    InterpolateAdded,
    AverageInterface,
    AverageBlock,
    AverageGrid
};
constexpr int V_INTERP_INTERFACE             = 0;    ///< Interpolate to interface cells only.
constexpr int V_INTERP_ADDED                 = 1;    ///< Interpolate to newly-added cells.
constexpr int V_AVERAGE_INTERFACE            = 0;    ///< Average involves interface cells only.
constexpr int V_AVERAGE_BLOCK                = 1;    ///< Average involves whole masked block.
constexpr int V_AVERAGE_GRID                 = 2;    ///< Average involves whole grid.

// Mesh-geometry interaction.
enum class MaskCell
{
    Interior,
    Interface,
    Ghost,
    Solid,
    Dummy
};
enum class MaskBlock
{
    Regular,
    Interface,
    Solid,
    SolidAdjacent,
    Dummy
};
constexpr int V_CELLMASK_INTERIOR            = 0;    ///< Indicates that the cell doesn't participate in fine-coarse data transfers.
constexpr int V_CELLMASK_INTERFACE           = 1;    ///< Indicates that the cell participates in fine-to-coarse data transfers.
constexpr int V_CELLMASK_GHOST               = 2;    ///< Indicates that the cell participates in coarse-to-fine data transfers.
constexpr int V_CELLMASK_SOLID               = -1;   ///< Indicates cell-center lies within the solid.
constexpr int V_CELLMASK_SOLID_VIS           = -8;   ///< Indicates cell-center lies within the solid separately for visualization.
constexpr int V_CELLMASK_SOLID_DIFF          = -9;   ///< Indicates cell-center lies within the solid separately for visualization (2).
constexpr int V_CELLMASK_DUMMY_I             = -2;   ///< Indicates a dummy mask value.
constexpr int V_CELLMASK_SOLIDS              = -5;   ///< Indicates cell-center lies outside the solid (guard during propagation).
constexpr int V_CELLMASK_BOUNDARY            = -6;   ///< Indicates cell is adjacent to solid cell, boundary conditions are imposed therein.
constexpr int V_BLOCKMASK_REGULAR            = 0;    ///< Default state of a cell-block.
constexpr int V_BLOCKMASK_INTERFACE          = 1;    ///< This cell-block participates in the grid communication routines.
constexpr int V_BLOCKMASK_SOLID              = -3;   ///< This cell-block lies entirely within a solid object.
constexpr int V_BLOCKMASK_SOLIDB             = -1;   ///< This cell-block lies on the boundary of a solid object.
constexpr int V_BLOCKMASK_SOLIDA             = -2;   ///< This cell-block is adjacent to the boundary of a solid object.
constexpr int V_BLOCKMASK_INDETERMINATE_O    = -4;   ///< This cell-block is in an indeterminate state (odd).
constexpr int V_BLOCKMASK_INDETERMINATE_E    = -5;   ///< This cell-block is in an indeterminate state (even).

// Mesh refinements types.
constexpr int V_MESH_REF_NW_CASES            = 0;    ///< Near-wall refinement for the benchmark cases.
constexpr int V_MESH_REF_NW_GEOMETRY         = 1;    ///< Near-wall refinement for a general geometry.
constexpr int V_MESH_REF_UNIFORM             = 2;    ///< Uniform refinement of the mesh at a specified level.
constexpr int V_MESH_REF_SOLUTION            = 3;    ///< Refinement based on the numerical solution (depends on solver).

// Mesh restart.
constexpr int V_MESH_RESTART_SAVE            = 0;    ///< Save the mesh to restart later.
constexpr int V_MESH_RESTART_LOAD            = 1;    ///< Load mesh data from previous save.

// Mesh solvers.
constexpr int V_SOLVER_LBM_BGK               = 0;
constexpr int V_SOLVER_LBM_TRT               = 1;
constexpr int V_SOLVER_LBM_MRT               = 2;

bool init_conn = false;
__constant__ int V_CONN_ID[81];
__constant__ int V_CONN_MAP[27];
int V_CONN_ID_H[81];
int V_CONN_MAP_H[27];

template <int N_DIM>
inline int InitConnectivity()
{
    // o====================================================================================
    // | Load connectivity indices into (GPU) constant memory.
    // o====================================================================================
    
    int V_CONN_ID_2D[81] = {0, 1, 0, -1, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int V_CONN_ID_3D[81] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};
    
    int V_CONN_MAP_2D[27] = {7, 4, 8, 3, 0, 1, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int V_CONN_MAP_3D[27] = {20, 12, 26, 10, 6, 15, 24, 17, 21, 8, 4, 13, 2, 0, 1, 14, 3, 7, 22, 18, 23, 16, 5, 9, 25, 11, 19};
    
    if (N_DIM==2)
    {
        cudaMemcpyToSymbol(V_CONN_ID, V_CONN_ID_2D, sizeof(int)*81);
        cudaMemcpyToSymbol(V_CONN_MAP, V_CONN_MAP_2D, sizeof(int)*27);
        for (int p = 0; p < 81; p++) V_CONN_ID_H[p] = V_CONN_ID_2D[p];
        for (int p = 0; p < 81; p++) V_CONN_MAP_H[p] = V_CONN_MAP_2D[p];
    }
    if (N_DIM==3)
    {
        cudaMemcpyToSymbol(V_CONN_ID, V_CONN_ID_3D, sizeof(int)*81);
        cudaMemcpyToSymbol(V_CONN_MAP, V_CONN_MAP_3D, sizeof(int)*27);
        for (int p = 0; p < 81; p++) V_CONN_ID_H[p] = V_CONN_ID_3D[p];
        for (int p = 0; p < 81; p++) V_CONN_MAP_H[p] = V_CONN_MAP_3D[p];
    }
    init_conn = true;
    
    return 0;
}


#include "structs.h"
#include "custom.h"
#include "util.h"
#include "index_mapper.h"



template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
class Mesh
{
    public:
    
    const int       N_DEV = AP->N_DEV;
    
    struct TextOutput
    {
        int init = 0;
        std::ofstream iter_printer;
        std::ofstream ref_printer;
        std::ofstream adv_printer;
        std::ofstream force_printer;
    };
    
    private:
    
    // o====================================================================================
    // | Intermediate/temporary storage.
    // o====================================================================================
    
    //! Temporary integer array for debugging (retrieves data from GPU for printing).
    int        **tmp_1 = new int*[N_DEV];
    
    //! Temporary floating-point array for debugging (retrieves data from GPU for printing).
    ufloat_t    **tmp_2 = new ufloat_t*[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_1.
    thrust::device_ptr<int> *c_tmp_1_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_2.
    thrust::device_ptr<int> *c_tmp_2_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_3.
    thrust::device_ptr<int> *c_tmp_3_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_4.
    thrust::device_ptr<int> *c_tmp_4_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_5.
    thrust::device_ptr<int> *c_tmp_5_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_6.
    thrust::device_ptr<int> *c_tmp_6_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_7.
    thrust::device_ptr<int> *c_tmp_7_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_8.
    thrust::device_ptr<int> *c_tmp_8_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_ones.
    thrust::device_ptr<int> *c_tmp_ones_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_tmp_counting_iter.
    thrust::device_ptr<int> *c_tmp_counting_iter_dptr = new thrust::device_ptr<int>[N_DEV];
    
    
    //! Temporary GPU storage.
    /*! Currently being used in:
        @ref M_RefineAndCoarsenBlocks to temporarily store indices of cell-blocks marked for refinement.
    */
    int        **c_tmp_1 = new int*[N_DEV];
    
    //! Temporary GPU storage.
    /*! Currently being used in:
        @ref M_RefineAndCoarsenBlocks to temporarily store the scattered map for constructing new cell-blocks and updating child connectivity.
    */    
    int        **c_tmp_2 = new int*[N_DEV];
    
    //! Temporary GPU storage.
    /*! Currently being used in:
        @ref M_RefineAndCoarsenBlocks to temporarily store gap indices to be used for generating new cell-blocks.
    */
    int        **c_tmp_3 = new int*[N_DEV];
    
    //! Temporary GPU storage.
    /*! Currently being used in:
        @ref M_RefineAndCoarsenBlocks to temporarily store levels of cell-blocks to be removed for coarsening.
    */
    int        **c_tmp_4 = new int*[N_DEV];
    
    //! Temporary GPU storage.
    /*! Currently being used in:
        @ref M_RefineAndCoarsenBlocks to temporarily store indices of cell-blocks to be removed for coarsening.
    */
    int        **c_tmp_5 = new int*[N_DEV];
    
    //! Temporary GPU storage.
    /*! Currently being used in:
        @ref M_RefineAndCoarsenBlocks to temporarily store 'efficient' map of IDs involved in the connectivity update.
    */
    int        **c_tmp_6 = new int*[N_DEV];
    
    //! Temporary GPU storage.
    /*! Currently being used in:
        @ref M_RefineAndCoarsenBlocks to temporarily store 'efficient' map of IDs involved in the connectivity update.
    */
    int        **c_tmp_7 = new int*[N_DEV];
    
    //! Temporary GPU storage.
    /*! Currently being used in:
        @ref M_RefineAndCoarsenBlocks to temporarily store 'efficient' map of IDs involved in the connectivity update.
    */
    int        **c_tmp_8 = new int*[N_DEV];
    
    //! Counting iterator used for copying down indices of cell-blocks satisfying certain conditions.
    int        **c_tmp_counting_iter = new int*[N_DEV];
    
    
    // o====================================================================================
    // | Routines.
    // o====================================================================================
    
    int M_Init();
    int M_Dest();
    
    
    
    
    
    public:
    
    int mesh_init = 0;
    int geometry_init = 0;
    int solver_init = 0;
    int bdata_init = 0;
    Parser *parser;
    Geometry<ufloat_t,ufloat_g_t,AP> *geometry;
    Solver<ufloat_t,ufloat_g_t,AP> *solver;
    TextOutput to;
    
    
    
    
    // o====================================================================================
    // | Mesh parameters.
    // o====================================================================================
    
    // Constants.
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
    const int N_Q;
    
    // Domain size.
    float           Lx                      = 1.0F;         ///< Length of domain in x-axis (in meters).
    float           Ly                      = 1.0F;         ///< Length of domain in y-axis as a fraction of @ref Lx.
    float           Lz                      = 1.0F;         ///< Length of domain in z-axis as a fraction of @ref Lx.
    int             Nx                      = 1;            ///< Number of cells along x-axis of domain.
    int             Ny                      = 1;            ///< Number of cells along y-axis of domain.
    int             Nz                      = 1;            ///< Number of cells along z-axis of domain.
    int             *Nxi;                                   ///< Array storing x-, y- and z-axis resolutions.
    ufloat_t        dx                      = 1.0;          ///< Spatial step. Equal to the temporal step in the Lattice Boltzmann solver.
    ufloat_t        dx_cblock               = 1.0;          ///< Spatial step on cell-block basis.
    int             n_coarsecblocks         = 1;            ///< Total number of coarse blocks in the domain.
    long int        n_maxcells              = 1;            ///< Maximum number of cells that can be stored in GPU memory.
    int             n_maxcblocks            = 1;            ///< Maximum number of cell-blocks corresponding to @ref n_maxcells.
    long int        n_solida                = 0;            ///< Number of cells adjacent to a solid cell.
    int             n_solidb                = 0;            ///< Number of cell-blocks containing at least one cell adjacent to a solid cell.
    int             n_solidb_a              = 0;            ///< The value @ref n_solidb adjusted for alignment in device memory.
    long int        n_maxcells_b            = 0;            ///< Number of cells corresponding to @ref n_solidb.
    
    // Other parameters.
    // - Debug.
    int             use_cpu                 = 0;            ///< Indicates to use the CPU version of the routines.
    // - Grid and hierarchy.
    int             N_PRECISION             = 0;            ///< Floating-point precision to be used for storing the solution field.
    int             MAX_LEVELS              = 1;            ///< Maximum number of grids for the domain interior and boundary.
    int             MAX_LEVELS_WALL         = 1;            ///< Maximum number of grids for the domain boundary alone.
    int             MAX_LEVELS_INTERIOR     = 1;            ///< Maximum number of grids for the domain interior alone.
    int             N_ITER_TOTAL            = Nx;           ///< Total number of iterations to employ for advancement.
    int             N_LEVEL_START           = 0;            ///< Grid level to employ as the root grid for advancement.
    int             N_RESTART               = 0;            ///< Indicates whether or not to use an existing restart file.
    int             PERIODIC_X              = 0;            ///< Indicates periodicity between boundaries aligned with x-axis.
    int             PERIODIC_Y              = 0;            ///< Indicates periodicity between boundaries aligned with y-axis.
    int             PERIODIC_Z              = 0;            ///< Indicates periodicity between boundaries aligned with z-axis.
    // - Mesh refinement.
    int             P_REFINE                = 1;            ///< Indicates frequency of refinement.
    ufloat_t        N_REFINE_START          = -3.0;         ///< Tuning parameter for refinement criterion.
    ufloat_t        N_REFINE_INC            = 1.0;          ///< Tuning parameter for refinement criterion.
    ufloat_t        N_REFINE_MAX            = 1.0;          ///< Tuning parameter for refinement criterion.
    // - Probe: convergence.
    int             N_PROBE                 = 0;            ///< Indicates if probing for converged solution.
    int             N_PROBE_DENSITY         = 4;            ///< Density of points used for probing of convergence.
    int             N_PROBE_FREQUENCY       = 16;           ///< Indicates frequency of probed convergence verification.
    double          V_PROBE_TOL             = 1e-4;         ///< Tolerance to be used for probed convergence verification.
    // - Probe: forces.
    int             N_PROBE_FORCE           = 0;            ///< Indicates if probing the force exerted on an obstacle in the domain.
    int             N_PROBE_F_FREQUENCY     = 16;           ///< Indicates frequency of force calculation.
    int             N_PROBE_F_START         = 0;            ///< Indicates the iteration when force calculation begins.
    // - Probe: time average.
    int             N_PROBE_AVE             = 0;            ///< Indicates if computing time-averaged velocity.
    int             N_PROBE_AVE_FREQUENCY   = 1;            ///< Indicates how frequently the time-averaged velocity is updated.
    int             N_PROBE_AVE_START       = 0;            ///< Indicates the iteration when time-average calculation begins.
    // - Input/Output.
    int             N_PRINT_LEVELS          = 1;            ///< Number of grid levels to include when printing.
    int             N_PRINT_LEVELS_LEGACY   = 1;            ///< Number of grid levels to include when printing (legacy .vthb format).
    int             N_PRINT_LEVELS_IMAGE    = 1;            ///< Number of grid levels to include when printing (image data .vti format).
    int             P_OUTPUT                = Nx;           ///< Frequency of output calls.
    int             N_OUTPUT_START          = 0;            ///< Indicates the iteration after which to start producing output files.
    std::string     input_dir;                              ///< Input directory.
    std::string     output_dir;                             ///< Output directory.
    std::ofstream   *output_file_direct;                    ///< Direct output file (stores time-series data in binary format).
    // - Rendering.
    int             VOL_I_MIN               = 0;            ///< Lower bounding x-index for output subdomain.
    int             VOL_I_MAX               = Nx;           ///< Upper bounding x-index for output subdomain.
    int             VOL_J_MIN               = 0;            ///< Lower bounding y-index for output subdomain.
    int             VOL_J_MAX               = Nx;           ///< Upper bounding y-index for output subdomain.
    int             VOL_K_MIN               = 0;            ///< Lower bounding z-index for output subdomain.
    int             VOL_K_MAX               = Nx;           ///< Upper bounding z-index for output subdomain.
    
    
    
    // o====================================================================================
    // | CPU parameters.
    // o====================================================================================
    
    //! Array of cell masks used for correcting fine-coarse data transfers.
    ///< Takes on values (0 - interior, 1 - interface, 2 - invalid / exterior).
    int        **cells_ID_mask = new int*[N_DEV];
    
    //! Auxillary array of cell masks indicating boundary Ids for complex geometries.
    ///< The stored values (one for each direction) indicate the index of the boundary condition to be imposed on each cell.
    int        **cells_ID_mask_b = new int*[N_DEV];
    
    //! Auxillary array of cell-center-boundary distances.
    ///< Stores the distances of cell-centers to the nearest boundary along each possible direction.
    ufloat_g_t    **cells_f_X_b = new ufloat_g_t*[N_DEV];
    
    //! Solution array.
    ///< Stores the numerical solution in a structured of arrays format (i.e. f0: c0, c1, c2,..., f1: c0, c1, c2,... and so on).
    ufloat_t     **cells_f_F = new ufloat_t*[N_DEV];
    
    //! Probed solution field at tn.
    //! Stores the coarse solution at probed locations according to the specified @ref N_PROBE_DENSITY.
    ufloat_t    **cells_f_U_probed_tn = new ufloat_t*[N_DEV];
    
    //! Probed solution field at tn-dt.
    //! Stores the coarse solution at probed locations according to the specified @ref N_PROBE_DENSITY.
    ufloat_t    **cells_f_U_mean = new ufloat_t*[N_DEV];
    
    //! Array of cell-block spatial coordinates.
    //! Stores the coordinate of the lower bottom-left corner of the cell-block in a structure of arrays format (i.e. x: cb0, cb1, cb2,..., y: cb0, cb1, cb2,... and so on).
    ufloat_t     **cblock_f_X = new ufloat_t*[N_DEV];
    
    //! Array of IDs indicating whether index block participates in inteperpolation or averaging.
    //! Stores an indicator integer (0 - inactive, 1 - active) indicating whether or not the current cell-block is active in interpolation / averaging.
    int        **cblock_ID_mask = new int*[N_DEV];
    
    //! Array of cell-block neighbor IDs.
    //! Stores the IDs of neighbor cell-blocks in a structure of arrays format (i.e. ID of neighbor 0: cb0, cb1, cb2,..., ID of neighbor 1: cb0, cb1, cb2,... and so on). Directions of the neighbors are designed to match the discrete particle velocity set selected at compile time. IDs can take non-negative values (indicating valid neighbor cell-blocks) and negative values (indicating a boundary where the neighbor would normally be if not equal to @ref N_SKIPID, otherwise indicating that a neighbor does not exist but points to the interior).
    int         **cblock_ID_nbr = new int*[N_DEV];
    
    //! Array of cell-block neighbor IDs in Array of Structures format.
    int         **cblock_ID_nbr_aos = new int*[N_DEV];
    
    //! Array of cell-block neighbor's child IDs.
    //! Stores the IDs of the first child (of zero-child-index) of neighboring cell-blocks in a structure of arrays format (i.e. ID of child 0: cb0, cb1, cb2,..., ID of child 1: cb0, cb1, cb2,... and so on). Non-negative values indicate a valid child cell-block. Values equal to @ref N_SKIPID indicate that a process involving this (non-existant) child should be skipped.
    int         **cblock_ID_nbr_child = new int*[N_DEV];
    
    //! Array of cell-block Ids indicating near-boundary status (0 - not on a boundary, 1 - on a boundary).
    int        **cblock_ID_onb = new int*[N_DEV];
    
    //! Array of cell-block Ids indicating the location of solid-cell linkage data for the cell-blocks.
    int        **cblock_ID_onb_solid = new int*[N_DEV];
    
    //! Stores the force contributions of the various cell-blocks.
    ufloat_t    **cblock_f_Ff = new ufloat_t*[N_DEV];
    
    //! Array of cell-block refinement IDs.
    //! Stores the refinement ID for cell-blocks which indicate status during mesh updates. Values are defined by macros V_REF_ID_....
    int         **cblock_ID_ref = new int*[N_DEV];
    
    //! Array of cell-block levels.
    int         **cblock_level = new int*[N_DEV];
    
    //! Array of cell-block edge/face counts.
    int        **cblock_ID_face_count = new int*[N_DEV];
    
    //! Array of cell-block edge/face IDs.
    int        **cblock_ID_face = new int*[N_DEV];
    
    //! Arrays of active cell-block IDs.
    //! Stores the IDs of active cell-blocks, classified among the possible grid hierarchy levels with an array for each level.
    int        **id_set = new int*[N_DEV];
    
    //! Arrays of I indices for the coarsest grid.
    int        **coarse_I = new int*[N_DEV];
    
    //! Arrays of J indices for the coarsest grid.
    int        **coarse_J = new int*[N_DEV];
    
    //! Arrays of K indices for the coarsest grid.
    int        **coarse_K = new int*[N_DEV];
    
    //! Array of coarse cell-block Ids marked for probing.
    //! Stores the Ids for coarse cell-blocks that are to be probed and still valid (i.e. in interior of domain).
    int        **id_set_probed = new int*[N_DEV];
    
    //! Array of active cell-block counts.
    int        **n_ids = new int*[N_DEV];
    
    //! Probed cell-block counter.
    int        *n_ids_probed = new int[N_DEV];
    
    //! Array of largest cell-block IDs per grid level.
    //! Stores the maximum cell-block ID on a grid level. The element with index @ref MAX_LEVELS represents the largest ID in the array and is used to identify the limit of access in the data arrays (saves time when I don't need to go through all possible cell-cblocks).
    int        **id_max = new int*[N_DEV];
    
    //! Array of gap cell-block IDs in data arrays.
    //! Stores the cell-block IDs of gaps that are formed in the data arrays when cell-blocks are coarsened as a simulation is processed. Stored in reverse order for convenience during the refinement and coarsening routine.
    int        **gap_set = new int*[N_DEV];
    
    //! Number of available gaps.
    int        *n_gaps = new int[N_DEV];
    
    //! Vector of spatial steps for all grid levels.
    ufloat_t    *dxf_vec;
    
    //! Neighbor blocks' local indices used to establish connectivity.
    const double    c[2][27*3] =
    {
        { 0,1,0,-1,0,1,-1,-1,1,   0,0,1,0,-1,1,1,-1,-1},
        { 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0,1,-1,1,-1,1,-1,-1,1,   0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1,1,-1,1,-1,-1,1,1,-1,   0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1 }
    };
    
    // o====================================================================================
    // | GPU parameters.
    // o====================================================================================
    
    size_t          free_t = 0;                             ///< Number of free bytes in GPU memory.
    size_t          total_t = 0;                            ///< Number of total bytes in GPU memory.
    long int        N_bytes_pc;                             ///< Number of bytes required per cell.
    double          M_FRAC                  = 0.85;         ///< Fraction of free memory to use.
    cudaStream_t    *streams = new cudaStream_t[N_DEV];     ///< CUDA streams employed by mesh.

    //! GPU counterpart of @ref cells_ID_mask.
    int        **c_cells_ID_mask = new int*[N_DEV];
    
    //! GPU counterpart of @ref cells_ID_mask_b.
    int        **c_cells_ID_mask_b = new int*[N_DEV];
    
    //! GPU counterpart of @ref cells_f_X_b.
    ufloat_g_t    **c_cells_f_X_b = new ufloat_g_t*[N_DEV];
    
    //! GPU counterpart of @ref cells_f_F.
    ufloat_t     **c_cells_f_F = new ufloat_t*[N_DEV];
    
    //! GPU counterpart of @ref cblock_f_X.
    ufloat_t     **c_cblock_f_X = new ufloat_t*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_mask.
    int        **c_cblock_ID_mask = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_nbr.
    int         **c_cblock_ID_nbr = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_nbr_aos.
    int         **c_cblock_ID_nbr_aos = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_nbr_child.
    int         **c_cblock_ID_nbr_child = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_onb.
    int        **c_cblock_ID_onb = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_onb_solid.
    int        **c_cblock_ID_onb_solid = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_f_Ff.
    ufloat_t    **c_cblock_f_Ff = new ufloat_t*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_ref.
    int         **c_cblock_ID_ref = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_level.
    int         **c_cblock_level = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_face_count.
    int        **c_cblock_ID_face_count = new int*[N_DEV];
    
    //! GPU counterpart of @ref cblock_ID_face.
    int        **c_cblock_ID_face = new int*[N_DEV];
    
    //! GPU counterpart of @ref id_set.
    int        **c_id_set = new int*[N_DEV];
    
    //! GPU counterpart of @ref gap_set.
    int        **c_gap_set = new int*[N_DEV];

    //! A Thrust pointer-cast of the device array @ref c_id_set.
    thrust::device_ptr<int> *c_id_set_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_gap_set.
    thrust::device_ptr<int> *c_gap_set_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_cells_f_W.
    thrust::device_ptr<ufloat_t> *c_cells_f_W_dptr = new thrust::device_ptr<ufloat_t>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_cblock_ID_ref.
    thrust::device_ptr<int> *c_cblock_ID_ref_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_cblock_level.
    thrust::device_ptr<int> *c_cblock_level_dptr = new thrust::device_ptr<int>[N_DEV];
    
    //! A Thrust pointer-cast of the device array @ref c_cblock_f_Ff.
    thrust::device_ptr<ufloat_t> *c_cblock_f_Ff_dptr = new thrust::device_ptr<ufloat_t>[N_DEV];
    
    // o====================================================================================
    // | Routines.
    // o====================================================================================
    
    //! Restart the solver by reading binary data prepared from a previous call to @M_Restart with argument @V_MESH_RESTART_SAVE.
    /*! Restarts the solver or prepares for future restart.
            @param i_dev is the ID of the device to be processed.
            @param var indicates whether to save data for later ( @V_MESH_RESTART_SAVE ) or to load a previously-saved file ( @V_MESH_RESTART_LOAD ).
            @param iter indicates the last iteration (if saving for later).
    */
    int             M_Restart(int i_dev, int var, int *iter=0);
    
    //! Update the mean velocity vectors at the coarsest level. Most relevant for LES simulations for comparison with reference data.
    /*! Updates the time average for velocity data on the coarsest level.
            @param i_dev is the ID of the device to be processed.
    */
    int             M_UpdateMeanVelocities(int i_dev, int N_iters_ave);
    
    //! Checks convergence according to probe properties @N_PROBE and @N_PROBE_FREQ.
    /*! Checks convergence of the solver accroding to a fixed set of probed locations.
            @param i_dev is the ID of the device to be processed.
    */
    ufloat_t        M_CheckConvergence(int i_dev);
    
    //! Computes macroscopic properties from DDFs retrieved from GPU memory.
    /*! Prints the mesh using VTK's hierarchical box format (.vthb).
            @param i_dev is the ID of the device to be processed.
            @param i_Q is the ID of the cell-block quadrant whose properties are being computed.
            @param i_kap is the ID of the cell-block whose properties are being computed.
            @param dx_L is the spatial step size in the block.
        @param out is the output array storing the macroscopic properties for all cells in a single cell-block.
    */
    int             M_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out_u);
    
    //! Computes the appropriate properties for output. Intermediate step in @M_RenderAndPrint_Uniform routine. 
    /*! Prints the mesh using VTK's hierarchical box format (.vthb).
            @param i_dev is the ID of the device to be processed.
            @param i_Q is the ID of the cell-block quadrant whose properties are being computed.
            @param i_kap is the ID of the cell-block whose properties are being computed.
            @param dx_L is the spatial step size in the block.
        @param out is the output array storing the macroscopic properties for all cells in a single cell-block.
    */
    int             M_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out_u);
    
    //! Constructs a complex geometry in the form of a set of edges/triangles in 2D/3D. These can be read from a file or hard-coded for simpler cases like flow past a circle/sphere or an airfoil.
    /*! Builds a complex geometry from a file or a hard-coded script.
            @param i_dev is the ID of the device to be processed.
    */
    int             M_AcquireBoundaries(int i_dev);
    
    //! An intermediate routine to check if an edge/face interects with a cell-block.
    /*! Check if an edge/face intersects with a cell-block.
            @param i_dev is the ID of the device to be processed.
    */
    int             M_CheckFaceIntersection(int i_dev);
    
    //! Print forces along and perpendicular to flow direction to a text file for the flow-past-square-cylinder case studies. This is a temporary function for the manuscript, will be refined later.
    /*! Checks neighbors on the coarsest level and computes forces via momentum exchange algorithm.
            @param i_dev is the ID of the device to be processed.
            @param out is the target file for output.
    */
    int             M_ComputeForces(int i_dev, int L);
    
    //! Print the mesh.
    /*! Prints the mesh using VTK's structured grid format (.vtk).
        @param i_dev is the ID of the device to be processed.
        @param iter is the current iteration being printed.
        @param params defines the limits of a cubic portion of the domain (i.e. x1 = params[0], x2 = params[1], y1 = params[2]...).
    */
    int             M_RenderAndPrint_Uniform(int i_dev, int iter);
    
    //! Add a geometry.
    /*! Add a pointer to the geometry object.
        @param geometry is the geometry object.
    */
    int             M_AddGeometry(Geometry<ufloat_t,ufloat_g_t,AP> *geometry);
    
    //! Add a solver.
    /*! Add a pointer to the solver object.
        @param solver is the solver object.
    */
    int             M_AddSolver(Solver<ufloat_t,ufloat_g_t,AP> *solver);
    
    //! Recursively traverses the hierarchy for printing..
    /*! Starting from a block on the root grid, the hierarchy is traversed. Cells of leaf blocks are printed while branches are traversed further.
        @param i_dev is the ID of the device to be processed.
        @param i_kap is the Id of the current block being processed.
        @param Is is a pointer to the child indices being tracker during traversal.
        @param L is the grid level of the current block being processed.
        @param dx_f is the spatial step for the output grid.
        @param mult_f is a pointer to the spatial resolution multipliers.
        @param vol is the volume of the printed domain (computed from @Nxi_f beforehand).
        @param Nxi_f is a pointer to the finest spatial resolutions.
        @param tmp_data is a pointer to the organized mesh data to be inserted in the vtkUniformGrid data arrays.
    */
    int             M_FillBlock(int i_dev, int *Is, int i_kap, int L, double dx_f, int *mult_f, int vol, int *Nxi_f, double *tmp_data);
    
    //! Print the mesh.
    /*! Prints the mesh using VTK's hierarchical box format (.vthb).
        @param i_dev is the ID of the device to be processed.
        @param iter is the current iteration being printed.
    */
    int             M_Print_VTHB(int i_dev, int iter);
    
    //! Transfer data structures from CPU to GPU.
    //! All data structures pertaining to the mesh (i.e. all cells_{f/ID}_X, cblock_{f/ID}_X) are copied from host to their respective devices. This will overwrite what is on the GPU if it has not been retrieved before.
    int             M_LoadToGPU();
    
    //! Transfer data structures from GPU to CPU.
    //! All data structures pertaining to the mesh (i.e. all cells_{f/ID}_X, cblock_{f/ID}_X) are copied from their respective devices to the host. This will overwrite what is on the CPU if it has not been loaded before.
    int             M_RetrieveFromGPU();
    
    //! Freeze (or unfreeze) the current status of refinement in the grid.
    //! @param var is the indicator to freeze (0) or unfreeze (1).
    int             M_FreezeRefinedCells(int var);
    
    //! Perform refinement and coarsening wherever marked.
    /*! @param var is a debugging indicator.
        @param file is the file to output step execution times.
    */
    int             M_RefineAndCoarsenBlocks(int var);
    
    //! Interpolate values between grid levels.
    /*! @param i_dev is the ID of the device to be processed.
        @param L is the level on which interpolant is being computed. L+1 is the level on which computed values lie.
        @param var is an indicator variable (@ref V_INTERP_INTERFACE to interpolate to ghost cells only, @ref V_INTERP_ADDED to interpolate to all cells).
    */
    int             M_Interpolate(int i_dev, int L, int var);
    
    //! Average between grid levels.
    /*! @param i_dev is the ID of the device to be processed.
        @param L is the level on which average is being computed. L+1 is the level on which retrieved values lie.
        @param var is an indicator variable (@ref V_AVERAGE_INTERFACE to average data from interface cells, @ref V_AVERAGE_BLOCK to average data from blocks near a coarse-fine interface, @ref V_AVERAGE_GRID to average from all cells).
    */
    int             M_Average(int i_dev, int L, int var);
    
    //! Advance the grid using the specified solver and input parameters (for now, only BGK-LBM).
    /*! @param i_dev is the ID of the device to be processed.
        @param L is the grid level being advanced.
        @param file is a pointer to the output file storing recorded advancement times (if P_SHOW_ADVANCE is set to 1).
        @param tmp is a pointer to an array temporarily storing the recorded advancement times.
    */
    int             M_Advance(int i_dev, int L, double *tmp);
    
    // TODO: Documentation.
    int             M_ComputeRefCriteria(int i_dev, int L, int var);
    int             M_ComputeRefCriteria_Geometry_Naive(int i_dev, int L);
    int             M_ComputeRefCriteria_Geometry_Binned(int i_dev, int L);
    int             M_Geometry_Voxelize_S1(int i_dev, int L);
    int             M_Geometry_Voxelize_S2(int i_dev, int L);
    int             M_Geometry_Voxelize_S3(int i_dev);
    int             M_UpdateMasks_Vis(int i_dev, int L);
    int             M_Geometry_FillBinned3D(int i_dev);
    int             M_IdentifyFaces(int i_dev, int L);
    int             M_Advance_InitTextOutput();
    int             M_Advance_RefineNearWall();
    int             M_Advance_LoadRestartFile(int &iter_s);
    int             M_Advance_PrintIter(int i, int iter_s);
    int             M_Advance_Step(int i, int iter_s, int iter_mult);
    int             M_Advance_RefineWithSolution(int i, int iter_s);
    int             M_Advance_Probe(int i, int iter_s);
    int             M_Advance_ProbeAverage(int i, int iter_s, int &N_iter_ave);
    int             M_Advance_PrintData(int i, int iter_s);
    int             M_Advance_PrintForces(int i, int iter_s, int START);
    int             M_ReportForces(int i_dev, int L, int i, double t, int START);
    int             M_Print_VTHB_Patch(int i_dev, int iter);
    int             M_Print_ImageData(int i_dev, int iter);
    int             M_ResetIntermediateAMRArraysV1(int i_dev);
    int             M_ResetIntermediateAMRArraysV2(int i_dev);
    
    //! Sets up a solver loop and advances the grid for a specified number of iterations (equal to @ref P_PRINT * @ref N_PRINT) with mesh refinement/coarsening, rendering and printing every @ref P_REFINE, @ref P_RENDER and @ref P_PRINT iterations, respectively.
    int             M_AdvanceLoop();
    
    //! [DEBUG] Modify specific values in data arrays on the GPU.
    //! @param var is an indicator for the subroutine to apply.
    int             M_CudaModVals(int var);
    
    //! [DEBUG] Print the last N gaps from the gap set.
    /*! @param var i_dev is the ID of the device to be processed.
        @param N is the number of gaps to print.
    */
    int             M_PrintLastNGaps(int i_dev, int N)
    {
        std::cout << "Gaps:" << std::endl;
        for (int i = 0; i < N; i++)
            std::cout << gap_set[i_dev][n_gaps[i_dev]-N+i] << " ";
        std::cout << std::endl;
        
        return 0;
    }
    
    
    
    
    
    // NOTE: This will be redone to incorporate proper intialization.
    Mesh(
        Parser *parser_,
        const int &N_Q_
    ) : parser(parser_), N_Q(N_Q_)
    {
        M_Init();
        std::cout << "[-] Finished making mesh object." << std::endl << std::endl;
    }

    ~Mesh()
    {
        M_Dest();
        std::cout << "[-] Finished deleting mesh object." << std::endl << std::endl;
    }
};

#endif
