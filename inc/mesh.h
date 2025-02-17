#ifndef MESH_H
#define MESH_H

// Mesh refinement/coarsening.
#define V_REF_ID_UNREFINED             0    ///< Indicates cell-block is unrefined.
#define V_REF_ID_UNREFINED_VIO         9    ///< Indicates cell-block is unrefined by possibly violating a quality-control criterion.
#define V_REF_ID_REFINED               1    ///< Indicates cell-block is refined.
#define V_REF_ID_REFINED_WCHILD        2    ///< Indicates cell-block is refined and has at least one refined child.
#define V_REF_ID_REFINED_PERM          3    ///< Indicates cell-block is refined permanently (mainly for near-wall stability).
#define V_REF_ID_REFINED_WCHILD_PERM   10   ///< Indicates cell-block is refined permanently (mainly for near-wall stability).
#define V_REF_ID_MARK_REFINE           4    ///< Indicates cell-block is marked for refinement.
#define V_REF_ID_MARK_COARSEN          5    ///< Indicates cell-block is marked for coarsening.
#define V_REF_ID_NEW                   6    ///< Indicates cell-block was newly inserted (as a child).
#define V_REF_ID_REMOVE                7    ///< Indicates cell-block will be removed (as a child).
#define V_REF_ID_INACTIVE              8    ///< Indicates cell-block is inactive in the simulation.

// Mesh communication.
#define V_INTERP_INTERFACE             0    ///< Interpolate to interface cells only.
#define V_INTERP_ADDED                 1    ///< Interpolate to newly-added cells.
#define V_AVERAGE_INTERFACE            0    ///< Average involves interface cells only.
#define V_AVERAGE_BLOCK                1    ///< Average involves whole masked block.
#define V_AVERAGE_GRID                 2    ///< Average involves whole grid.

// Mesh refinements types.
#define V_MESH_REF_NW_CASES            0    ///< Near-wall refinement for the benchmark cases.
#define V_MESH_REF_NW_GEOMETRY         1    ///< Near-wall refinement for a general geometry.
#define V_MESH_REF_UNIFORM             2    ///< Uniform refinement of the mesh at a specified level.
#define V_MESH_REF_SOLUTION            3    ///< Refinement based on the numerical solution (depends on solver).

// Mesh restart.
#define V_MESH_RESTART_SAVE            0    ///< Save the mesh to restart later.
#define V_MESH_RESTART_LOAD            1    ///< Load mesh data from previous save.

// Mesh solvers.
#define V_SOLVER_LBM_BGK               0
#define V_SOLVER_LBM_TRT               1
#define V_SOLVER_LBM_MRT               2

#include "cppspec.h"
#include "geometry.h"
#include "solver.h"

class Mesh
{
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
	int		*tmp_1[N_DEV];
	
	//! Temporary floating-point array for debugging (retrieves data from GPU for printing).
	ufloat_t	*tmp_2[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_1.
	thrust::device_ptr<int> c_tmp_1_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_2.
	thrust::device_ptr<int> c_tmp_2_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_3.
	thrust::device_ptr<int> c_tmp_3_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_4.
	thrust::device_ptr<int> c_tmp_4_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_5.
	thrust::device_ptr<int> c_tmp_5_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_6.
	thrust::device_ptr<int> c_tmp_6_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_7.
	thrust::device_ptr<int> c_tmp_7_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_8.
	thrust::device_ptr<int> c_tmp_8_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_ones.
	thrust::device_ptr<int> c_tmp_ones_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_tmp_counting_iter.
	thrust::device_ptr<int> c_tmp_counting_iter_dptr[N_DEV];
	
	//! Temporary GPU storage.
	/*! Currently being used in:
	    @ref M_RefineAndCoarsenCells to temporarily store indices of cell-blocks marked for refinement.
	*/
	int		*c_tmp_1[N_DEV];
	
	//! Temporary GPU storage.
	/*! Currently being used in:
	    @ref M_RefineAndCoarsenCells to temporarily store the scattered map for constructing new cell-blocks and updating child connectivity.
	*/	
	int		*c_tmp_2[N_DEV];
	
	//! Temporary GPU storage.
	/*! Currently being used in:
	    @ref M_RefineAndCoarsenCells to temporarily store gap indices to be used for generating new cell-blocks.
	*/
	int		*c_tmp_3[N_DEV];
	
	//! Temporary GPU storage.
	/*! Currently being used in:
	    @ref M_RefineAndCoarsenCells to temporarily store levels of cell-blocks to be removed for coarsening.
	*/
	int		*c_tmp_4[N_DEV];
	
	//! Temporary GPU storage.
	/*! Currently being used in:
	    @ref M_RefineAndCoarsenCells to temporarily store indices of cell-blocks to be removed for coarsening.
	*/
	int		*c_tmp_5[N_DEV];
	
	//! Temporary GPU storage.
	/*! Currently being used in:
	    @ref M_RefineAndCoarsenCells to temporarily store 'efficient' map of IDs involved in the connectivity update.
	*/
	int		*c_tmp_6[N_DEV];
	
	//! Temporary GPU storage.
	/*! Currently being used in:
	    @ref M_RefineAndCoarsenCells to temporarily store 'efficient' map of IDs involved in the connectivity update.
	*/
	int		*c_tmp_7[N_DEV];
	
	//! Temporary GPU storage.
	/*! Currently being used in:
	    @ref M_RefineAndCoarsenCells to temporarily store 'efficient' map of IDs involved in the connectivity update.
	*/
	int		*c_tmp_8[N_DEV];
	
	//! Counting iterator used for copying down indices of cell-blocks satisfying certain conditions.
	int		*c_tmp_counting_iter[N_DEV];
	
	
	// o====================================================================================
	// | Routines.
	// o====================================================================================
	
	int M_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
	int M_Dest();
	
	
	
	
	
	public:
	
	int mesh_init = 0;
	int geometry_init = 0;
	int solver_init = 0;
	Geometry *geometry;
	Solver *solver;
	TextOutput to;
	
	
	
	
	// o====================================================================================
	// | Mesh parameters.
	// o====================================================================================
	
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
	
	// Other parameters.
	// - Grid and hierarchy.
	int             MAX_LEVELS              = 1;            ///< Maximum number of grids for the domain interior and boundary.
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
	int		*cells_ID_mask[N_DEV];
	
	//! Array of cell-centered density distribution functions (DDFs).
	///< Stores the @ref N_Q density distribution functions in a structured of arrays format (i.e. f0: c0, c1, c2,..., f1: c0, c1, c2,... and so on).
	ufloat_t 	*cells_f_F[N_DEV];
	
	//! Probed solution field at tn.
	//! Stores the coarse solution at probed locations according to the specified @ref N_PROBE_DENSITY.
	ufloat_t	*cells_f_U_probed_tn[N_DEV];
	
	//! Probed solution field at tn-dt.
	//! Stores the coarse solution at probed locations according to the specified @ref N_PROBE_DENSITY.
	ufloat_t	*cells_f_U_mean[N_DEV];
	
	//! Array of cell-block spatial coordinates.
	//! Stores the coordinate of the lower bottom-left corner of the cell-block in a structure of arrays format (i.e. x: cb0, cb1, cb2,..., y: cb0, cb1, cb2,... and so on).
	ufloat_t 	*cblock_f_X[N_DEV];
	
	//! Array of IDs indicating whether index block participates in inteperpolation or averaging.
	//! Stores an indicator integer (0 - inactive, 1 - active) indicating whether or not the current cell-block is active in interpolation / averaging.
	int		*cblock_ID_mask[N_DEV];
	
	//! Array of cell-block neighbor IDs.
	//! Stores the IDs of neighbor cell-blocks in a structure of arrays format (i.e. ID of neighbor 0: cb0, cb1, cb2,..., ID of neighbor 1: cb0, cb1, cb2,... and so on). Directions of the neighbors are designed to match the discrete particle velocity set selected at compile time. IDs can take non-negative values (indicating valid neighbor cell-blocks) and negative values (indicating a boundary where the neighbor would normally be if not equal to @ref N_SKIPID, otherwise indicating that a neighbor does not exist but points to the interior).
	int 		*cblock_ID_nbr[N_DEV];
	
	//! Array of cell-block neighbor's child IDs.
	//! Stores the IDs of the first child (of zero-child-index) of neighboring cell-blocks in a structure of arrays format (i.e. ID of child 0: cb0, cb1, cb2,..., ID of child 1: cb0, cb1, cb2,... and so on). Non-negative values indicate a valid child cell-block. Values equal to @ref N_SKIPID indicate that a process involving this (non-existant) child should be skipped.
	int 		*cblock_ID_nbr_child[N_DEV];
	
	//! Array of cell-block Ids indicating near-boundary status (0 - not on a boundary, 1 - on a boundary).
	int		*cblock_ID_onb[N_DEV];
	
	//! Array of cell-block refinement IDs.
	//! Stores the refinement ID for cell-blocks which indicate status during mesh updates. Values are defined by macros V_REF_ID_....
	int 		*cblock_ID_ref[N_DEV];
	
	//! Array of cell-block levels.
	int 		*cblock_level[N_DEV];
	
	//! Array of cell-block edge/face counts.
	int		*cblock_ID_face_count[N_DEV];
	
	//! Array of cell-block edge/face IDs.
	int		*cblock_ID_face[N_DEV];
	
	//! Array of geometry node locations. [DEPRECATED]
	//double		*geom_f_node_X[N_DEV];
	
	//! Array of face indices. [DEPRECATED]
	//int		*geom_ID_face[N_DEV];
	
	//! Array of geometry face attributes attributes. [DEPRECATED]
	//double		*geom_ID_face_attr[N_DEV];
	
	//! Arrays of active cell-block IDs.
	//! Stores the IDs of active cell-blocks, classified among the possible grid hierarchy levels with an array for each level.
	int		*id_set[N_DEV];
	
	//! Arrays of I indices for the coarsest grid.
	int		*coarse_I[N_DEV];
	
	//! Arrays of J indices for the coarsest grid.
	int		*coarse_J[N_DEV];
	
	//! Arrays of K indices for the coarsest grid.
	int		*coarse_K[N_DEV];
	
	//! Array of coarse cell-block Ids marked for probing.
	//! Stores the Ids for coarse cell-blocks that are to be probed and still valid (i.e. in interior of domain).
	int		*id_set_probed[N_DEV];
	
	//! Array of active cell-block counts.
	int		*n_ids[N_DEV];
	
	//! Number of nodes. [DEPRECATED]
	//int		n_nodes[N_DEV];
	
	//! Number of faces. [DEPRECATED]
	//int		n_faces[N_DEV];
	
	//! Probed cell-block counter.
	int		n_ids_probed[N_DEV];
	
	//! Array of largest cell-block IDs per grid level.
	//! Stores the maximum cell-block ID on a grid level. The element with index @ref MAX_LEVELS represents the largest ID in the array and is used to identify the limit of access in the data arrays (saves time when I don't need to go through all possible cell-cblocks).
	int		*id_max[N_DEV];
	
	//! Array of gap cell-block IDs in data arrays.
	//! Stores the cell-block IDs of gaps that are formed in the data arrays when cell-blocks are coarsened as a simulation is processed. Stored in reverse order for convenience during the refinement and coarsening routine.
	int		*gap_set[N_DEV];
	
	//! Number of available gaps.
	int		n_gaps[N_DEV];
	
	//! Vector of spatial steps for all grid levels.
	ufloat_t	*dxf_vec;
	
	//! Neighbor blocks' local indices used to establish connectivity.
	const double	c[2][27*3] =
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
	double          M_FRAC                  = 0.75;         ///< Fraction of free memory to use.
	cudaStream_t    streams[N_DEV];                         ///< CUDA streams employed by mesh.

	//! GPU counterpart of @ref cells_ID_mask.
	int		*c_cells_ID_mask[N_DEV];
	
	//! GPU counterpart of @ref cells_f_F.
	ufloat_t 	*c_cells_f_F[N_DEV];
	
	//! GPU counterpart of @ref cblock_f_X.
	ufloat_t 	*c_cblock_f_X[N_DEV];
	
	//! GPU counterpart of @ref cblock_ID_mask.
	int		*c_cblock_ID_mask[N_DEV];
	
	//! GPU counterpart of @ref cblock_ID_nbr.
	int 		*c_cblock_ID_nbr[N_DEV];
	
	//! GPU counterpart of @ref cblock_ID_nbr_child.
	int 		*c_cblock_ID_nbr_child[N_DEV];
	
	//! GPU counterpart of @ref cblock_ID_onb.
	int		*c_cblock_ID_onb[N_DEV];
	
	//! GPU counterpart of @ref cblock_ID_ref.
	int 		*c_cblock_ID_ref[N_DEV];
	
	//! GPU counterpart of @ref cblock_level.
	int 		*c_cblock_level[N_DEV];
	
	//! GPU counterpart of @ref cblock_ID_face_count.
	int		*c_cblock_ID_face_count[N_DEV];
	
	//! GPU counterpart of @ref cblock_ID_face.
	int		*c_cblock_ID_face[N_DEV];
	
	//! GPU counterpart of @ref geom_f_node_X. [DEPRECATED]
	//double	*c_geom_f_node_X[N_DEV];
	
	//! GPU counterpart of @ref geom_ID_face. [DEPRECATED]
	//int		*c_geom_ID_face[N_DEV];
	
	//! GPU counterpart of @ref geom_ID_face_attr. [DEPRECATED]
	//double	*c_geom_ID_face_attr[N_DEV];
	
	//! GPU counterpart of @ref id_set.
	int		*c_id_set[N_DEV];
	
	//! GPU counterpart of @ref gap_set.
	int		*c_gap_set[N_DEV];

	//! A Thrust pointer-cast of the device array @ref c_id_set.
	thrust::device_ptr<int> c_id_set_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_gap_set.
	thrust::device_ptr<int> c_gap_set_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_cells_f_W.
	thrust::device_ptr<ufloat_t> c_cells_f_W_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_cblock_ID_ref.
	thrust::device_ptr<int> c_cblock_ID_ref_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_cblock_level.
	thrust::device_ptr<int> c_cblock_level_dptr[N_DEV];
	
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
	int             M_RefineAndCoarsenCells(int var);
	
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
	
	
	int             M_ComputeRefCriteria(int i_dev, int L, int var);
	int             M_Advance_InitTextOutput();
	int             M_Advance_RefineNearWall();
	int             M_Advance_LoadRestartFile(int &iter_s);
	int             M_Advance_PrintIter(int i, int iter_s);
	int             M_Advance_Step(int i, int iter_s, int iter_mult);
	int             M_Advance_RefineWithSolution(int i, int iter_s);
	int             M_Advance_Probe(int i, int iter_s);
	int             M_Advance_ProbeAverage(int i, int iter_s, int &N_iter_ave);
	int             M_Advance_PrintData(int i, int iter_s);
	int             M_Advance_PrintForces(int i, int iter_s);
	int             M_ComplexGeom();
	
	int             M_NewSolver_LBM_BGK(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
	
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
	Mesh(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str)
	{
		M_Init(params_int, params_dbl, params_str);
		std::cout << "[-] Finished making mesh object." << std::endl << std::endl;
	}

	~Mesh()
	{
		if (geometry_init)
		{
			delete geometry;
			std::cout << "[-] Finished deleting geometry object." << std::endl;
		}
		if (solver_init)
		{
			delete solver;
			std::cout << "[-] Finished deleting solver object." << std::endl;
		}
		
		M_Dest();
		std::cout << "[-] Finished deleting mesh object." << std::endl << std::endl;
	}
};

#endif
