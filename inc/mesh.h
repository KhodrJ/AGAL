/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

// TODO
// - Make sure that the connectivity established by the set of velocity vectors 'c' is decoupled in final implementation from the lbm.h so that the mesh class can be used completely separelty with other solvers.

#ifndef MESH_H
#define MESH_H

#define V_REF_ID_UNREFINED		0			///< Indicates cell-block is unrefined.
#define V_REF_ID_UNREFINED_VIO		9			///< Indicates cell-block is unrefined by possibly violating a quality-control criterion.
#define V_REF_ID_REFINED		1			///< Indicates cell-block is refined.
#define V_REF_ID_REFINED_WCHILD		2			///< Indicates cell-block is refined and has at least one refined child.
#define V_REF_ID_REFINED_PERM		3			///< Indicates cell-block is refined permanently (mainly for near-wall stability).
#define V_REF_ID_REFINED_WCHILD_PERM	10			///< Indicates cell-block is refined permanently (mainly for near-wall stability).
#define V_REF_ID_MARK_REFINE		4			///< Indicates cell-block is marked for refinement.
#define V_REF_ID_MARK_COARSEN		5			///< Indicates cell-block is marked for coarsening.
#define V_REF_ID_NEW			6			///< Indicates cell-block was newly inserted (as a child).
#define V_REF_ID_REMOVE			7			///< Indicates cell-block will be removed (as a child).
#define V_REF_ID_INACTIVE		8			///< Indicates cell-block is inactive in the simulation.

#define V_INTERP_INTERFACE		0			///< Interpolate to interface cells only.
#define V_INTERP_ADDED			1			///< Interpolate to newly-added cells.

#define V_AVERAGE_INTERFACE		0			///< Average involves interface cells only.
#define V_AVERAGE_BLOCK			1			///< Average involves whole masked block.
#define V_AVERAGE_GRID			2			///< Average involves whole grid.

#define V_ADV_TYPE_STANDARD		0			///< Standard advancement (requires rescaling of DDFs during communication).
#define V_ADV_TYPE			V_ADV_TYPE_CUBIC	///< Selected advancement type.

#define V_MESH_RESTART_SAVE		0			///< Save the mesh to restart later.
#define V_MESH_RESTART_LOAD		1			///< Load mesh data from previous save.

#define V_RESCALE_TYPE			0
#if (V_ADV_TYPE == V_ADV_TYPE_CUBIC)
	#undef V_RESCALE_TYPE
	#define V_RESCALE_TYPE		1
#endif

#include "cppspec.h"
#include "lbm.h"

//
  // ========================
  // === CUDA Operations: ===
  // ========================
//

//! Reset values of GPU array.
/*! Reset the values of an array in GPU memory to a particular value.
    @param N is the length of the array.
    @param arr is a pointer to the array.
    @param val is the value being set.
*/
template<class T>
__global__
void Cu_ResetToValue(int N, T *arr, T val);

//! Contract array elements by a specified amount.
/*! Retrieve elements of an array and store only a fraction of them (skipping over others) in a destination array not necessarily the same as the input.
    @param N is the length of the input array.
    @param arr is the input array.
    @param frac is the integer inverse of the fraction (i.e. to skip every N entries, frac should be set to N for a fraction of 1/N).
    @param arr2 is the destination array.
*/
template <class T>
__global__
void Cu_ContractByFrac(int N, T *arr, int frac, T *arr2);

//! Fill an array with a value equal to the index.
/*! @param N is the length of the input array.
    @param arr is the input array.
*/
template <class T>
__global__
void Cu_FillLinear(int N, T *arr);



//! The mesh class.
/*!
	Main header file for the "Mesh" object whose functions are divided among two .cu files: mesh.cu and solver.cu. These two files contain those routines responsible for managing the mesh (i.e. refinement, coarsening, connectivity etc.) and the solver (i.e. advancement, interpolation, averaging etc.), respectively.

	A structure of arrays (SoA) approach is undertaken where cell data is split among a number of arrays and stored such that the data is ordered by cell (so for two arrays A and B, A[i] and B[i] can represent data points A_i and B_i for cell i). For storing vectors v_i, the array is expanded to length n_maxcells*length(v_i) and each component is represented so that contiguous sections of memory refer to that particular component in contiguous cells (v_00 v_01 v_02 ...,  v_10 v11 v12 ... refer to the first component of v in cells 0,1,2 then the second component in cells 0,1,2 etc.).

	The cell-based approach requires the following arrays:
	- cells_f_F             Array of density distribution functions
	- cells_f_X             Array of cell spatial locations
	- cells_f_U             Array of macroscopic variables p, u, v (and w if N_DIM=3)
	- cells_ID_par          Array of parent IDs
	- cells_ID_child        Array of child IDs
	- cells_ID_nbr          Array of cell neighbours
	- cells_ID_ref          Array of refinement IDs (to indicate status of the cell
				for mesh management purposes
	- cells_ID_aschild      Array of child indices for each cell (probably will be removed)          

	The block-based approach requries the following arrays:
	- cells_f_F             Array of density distribution functions
	- cells_f_U             Array of macroscopic variables p, u, v (and w if N_DIM=3)
	- blocks_f_X            Array of cell-block spatial locations
	- blocks_ID_child       Array of cell-block child IDs
	- blocks_ID_ref         Array of refinement IDs (to indicate status of the block
				for mesh management purposes
	- blocks_level          Array of integers representing the level each block occupies
	- blocks_ID_nbr         Array of cell-block neighbors
	- blocks_ID_nbr_child   Array of child IDs of zero-index for the cell-block neighbors

	Some remarks:
	- The block-based approach is managed like a tree in the sense that one cell-block can branch out to four/eight other finer blocks in 2D/3D, respectively. This ensures that the lengths of cells of a block on level L+1 are always half those on L. It also allows for re-use of the connectivity properties developed for the cell-based approach (wherein the 'cells' are replaced with cell-blocks in the framework).
	- The id_set refers to cell and cell-block IDs in the cell- and block-base approaches, respectively. A consequence is that a lack of order in the block-based approach will NOT harm coalescence in any way since cell-blocks are designed to be multiples of 32 in size.
	- Advantages of the block-based approach are: 1) there is a natural local structure provided on the cell-block level (which can be specified depending on user preference), 2) some of the processes required during refinement and coarsening in the cell-based approach become implicit such as ghost-padding / ghost-redundancy, 3) over-refinement becomes an advantage in that threads can make full use of coalescence to read cells in a block (as opposed to the cell-based approach where we can only read up to eight cells contiguously - a heavy price to pay indeed). Actually, the cell-based approach, which would normally provide the advantage of keeping a minimal number of fine cells, loses pretty much any advantage over the block-based approach when implemented in a GPU-native approach.
	- The block-based approach requires storage of spatial locations for the blocks for reconstruction of the cell spatial locations during the refinement step. The unique x,y,z values assigned to a block correspond to the coordinates of the first cell in the block. Other cell coordinates are reconstructed based on incrementation of the spatial step.
	- cells_f_W is separated from cells_f_U since the refinement criterion depends on the maximum vorticity which can be obtained efficiently using the Thrust library on the array (but we want only vorticity magnitudes and not the other macroscopic properties).
	- I am writing cell-block to differentiate between blocks of cells and blocks of threads on the GPU (when block appears alone, it can safely be assumed to refer to blocks of threads).
*/
class Mesh
{
//	public:
	
	//
	  // =============================
	  // === Intermediate Members: ===
	  // =============================
	//
	
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
	
	//! Array of ones stored on the GPU.
	//! [DEPRECATED] Currently unused, will likely remove.
	//int		*c_tmp_ones[N_DEV];
	
	//! Counting iterator used for copying down indices of cell-blocks satisfying certain conditions.
	int		*c_tmp_counting_iter[N_DEV];
	
	
	
	//
	  // =================================
	  // === Private Member Functions: ===
	  // =================================
	//

	//! Initialize the mesh.
	int		M_Init();
	
	//! [DEPRECATED] Update the @ref c_id_set and @ref c_gap_set arrays.
	/*! The routine called within old versions of the code for concatenation / stream compaction operations applied to the id_set and gap_set.
	    @param var the indicator for the subroutine to be applied.
	    @param i_dev the ID of device under consideration.
	    @param L the level of the grid hierarchy being modified.
	    @param N1 the first array length parameter.
	    @param N2 the second array length parameter.
	    @param N3 the third array length parameter.
	*/
	int		M_PlayWithVectors(int var, int i_dev, int L, int N1, int N2, int N3);
	
	//! Update connectivity in the mesh.
	/*! Called internally after refinement and coarsening. Updates connectivity at the child level to account for insert and removed children. Loops over all levels in the grid hierarchy.
	    @param var is a debugging indicator.
	*/
	int		M_UpdateConnectivity(int var);
	
	
	
	public:
	
		
		
	//
	  // ========================
	  // === Mesh Parameters: ===
	  // ========================
	//

	int		n_coarse		= Nx;		///< Number of coarse cells along one axis.
	int		n_coarsecblocks 	= 1;		///< Total number of coarse blocks in the domain.
	int		n_cells			= 1;		///< Total number of cells.
	long int	n_maxcells		= 1;		///< Maximum number of cells that can be stored in GPU memory.
	int		n_maxcblocks 		= 1;		///< Maximum number of cell-blocks corresponding to @ref n_maxcells.
	int		n_maxgroups		= 1;		///< Maximum number of cell-block groups (for in-place streaming [TODO]).
	int		*Nxi;					///< Array of mesh resolutions along the @ref N_DIM axes.
	ufloat_t	dx			= 1.0;		///< Spatial step. Equal to the temporal step in the Lattice Boltzmann solver.
	ufloat_t	dx_cblock 		= 1.0;		///< Spatial step on cell-block basis.
	std::chrono::steady_clock::time_point   begin;		///< Starting time point for recording execution time.
	std::chrono::steady_clock::time_point   end;		///< Ending time point for recording execution time.
	
	
	
	//
	  // =======================
	  // === CPU Parameters: ===
	  // =======================
	//
	
	//! Array of cell masks used for correcting fine-coarse data transfers.
	///< Takes on values (0 - interior, 1 - interface, 2 - invalid / exterior).
	int		*cells_ID_mask[N_DEV];
	
	//! Array of cell-centered density distribution functions (DDFs).
	///< Stores the @ref N_Q density distribution functions in a structured of arrays format (i.e. f0: c0, c1, c2,..., f1: c0, c1, c2,... and so on).
	ufloat_t 	*cells_f_F[N_DEV];
	
#if (S_TYPE==0)
	//! Temporary storage for DDF updates.
	//! Stores the @ref N_Q temporary density distribution functions required during solver updates in a structured of arrays format (i.e. f0: c0, c1, c2,..., f1: c0, c1, c2,... and so on).
	ufloat_t 	*cells_f_Fs[N_DEV];
#endif
	
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
	
	//! Arrays of active cell-block IDs.
	//! Stores the IDs of active cell-blocks, classified among the possible grid hierarchy levels with an array for each level.
	int		*id_set[N_DEV][MAX_LEVELS];
	
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
	int		n_ids[N_DEV][MAX_LEVELS+1];
	
	//! Probed cell-block counter.
	int		n_ids_probed[N_DEV];
	
	//! Array of largest cell-block IDs per grid level.
	//! Stores the maximum cell-block ID on a grid level. The element with index @ref MAX_LEVELS represents the largest ID in the array and is used to identify the limit of access in the data arrays (saves time when I don't need to go through all possible cell-cblocks).
	int		id_max[N_DEV][MAX_LEVELS+1];
	
	//! Array of gap cell-block IDs in data arrays.
	//! Stores the cell-block IDs of gaps that are formed in the data arrays when cell-blocks are coarsened as a simulation is processed. Stored in reverse order for convenience during the refinement and coarsening routine.
	int		*gap_set[N_DEV];
	
	//! Number of available gaps.
	int		n_gaps[N_DEV];
	
	//! Vector of spatial steps for all grid levels.
	ufloat_t	dxf_vec[MAX_LEVELS];
	
	//! [DEPRECATED] Can't remember what this is for.
	int		*pb_adjusted;
	
	
	
	//
	  // =======================
	  // === GPU Parameters: ===
	  // =======================
	//
	
	size_t		free_t;					///< Number of free bytes in GPU memory.
	size_t		total_t;				///< Number of total bytes in GPU memory.
	int		V_t_bytes;				///< Number of bytes required per cell.
	double		M_frac			= 0.75;		///< Fraction of free memory to use.
	cudaStream_t 	streams[N_DEV];				///< CUDA streams employed by mesh.

	//! GPU counterpart of @ref cells_ID_mask.
	int		*c_cells_ID_mask[N_DEV];
	
	//! GPU counterpart of @ref cells_f_F.
	ufloat_t 	*c_cells_f_F[N_DEV];
	
#if (S_TYPE==0)
	//! GPU counterpart of @ref cells_f_Fs.
	ufloat_t 	*c_cells_f_Fs[N_DEV];
#endif
	
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
	
	//! GPU counterpart of @ref id_set.
	int		*c_id_set[N_DEV][MAX_LEVELS];
	
	//! GPU counterpart of @ref gap_set.
	int		*c_gap_set[N_DEV];

	//! A Thrust pointer-cast of the device array @ref c_id_set.
	thrust::device_ptr<int> c_id_set_dptr[N_DEV][MAX_LEVELS];
	
	//! A Thrust pointer-cast of the device array @ref c_gap_set.
	thrust::device_ptr<int> c_gap_set_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_cells_f_W.
	thrust::device_ptr<ufloat_t> c_cells_f_W_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_cblock_ID_ref.
	thrust::device_ptr<int> c_cblock_ID_ref_dptr[N_DEV];
	
	//! A Thrust pointer-cast of the device array @ref c_cblock_level.
	thrust::device_ptr<int> c_cblock_level_dptr[N_DEV];
		
	double	*c_mat_interp[N_DEV];	///< Copy of interpolation matrix (auto-selected based on @ref N_DIM).
	double	*c_mat_fd[N_DEV];	///< Copy of finite-difference matrix (auto-selected based on @ref N_DIM).
		
		
		
	//
	  // ================================
	  // === Public Member Functions: ===
	  // ================================
	//
	
	//! Restart the solver by reading binary data prepared from a previous call to @M_Restart with argument @V_MESH_RESTART_SAVE.
	/*! Restarts the solver or prepares for future restart.
            @param i_dev is the ID of the device to be processed.
            @param var indicates whether to save data for later ( @V_MESH_RESTART_SAVE ) or to load a previously-saved file ( @V_MESH_RESTART_LOAD ).
            @param iter indicates the last iteration (if saving for later).
	*/
	int		M_Restart(int i_dev, int var, int *iter=0);
	
	//! Update the mean velocity vectors at the coarsest level. Most relevant for LES simulations for comparison with reference data.
	/*! Updates the time average for velocity data on the coarsest level.
            @param i_dev is the ID of the device to be processed.
	*/
	int		M_UpdateMeanVelocities(int i_dev, int N_iters_ave);
	
	//! Print forces along and perpendicular to flow direction to a text file for the flow-past-square-cylinder case studies. This is a temporary function for the manuscript, will be refined later.
	/*! Checks neighbors on the coarsest level and computes forces via momentum exchange algorithm.
            @param i_dev is the ID of the device to be processed.
            @param out is the target file for output.
	*/
	int		M_PrintForces(int i_dev, int L, std::ofstream *out);
	
	//! Checks convergence according to probe properties @N_PROBE and @N_PROBE_FREQ.
	/*! Checks convergence of the solver accroding to a fixed set of probed locations.
            @param i_dev is the ID of the device to be processed.
	*/
	ufloat_t	M_CheckConvergence(int i_dev);
	
	//! Computes macroscopic properties from DDFs retrieved from GPU memory. Intermediate step in @M_Print routine.
	/*! Prints the mesh using VTK's hierarchical box format (.vthb).
            @param i_dev is the ID of the device to be processed.
            @param i_kap is the ID of the cell-block whose properties are being computed.
            @param dx_L is the spatial step size in the block.
	    @param out is the output array storing the macroscopic properties for all cells in a single cell-block.
	    @param out2 is the output array storing the y+ values for all cells in a single cell-block (if @S_LES is 1).
	*/
	int		M_ComputeProperties(int i_dev, int i_kap, ufloat_t dx_L, double *out_u, double *out_yplus);
	
	//! Print the mesh.
	/*! Prints the mesh using VTK's structured grid format (.vtk).
	    @param i_dev is the ID of the device to be processed.
	    @param iter is the current iteration being printed.
	*/
	int		M_Print(int i_dev, int iter);
	
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
	int		M_Print_FillBlock(int i_dev, int *Is, int i_kap, int L, double dx_f, int *mult_f, int vol, int *Nxi_f, double *tmp_data);
	
	//! Print the mesh.
	/*! Prints the mesh using VTK's hierarchical box format (.vthb).
	    @param i_dev is the ID of the device to be processed.
	    @param iter is the current iteration being printed.
	*/
	int		M_Print_VTHB(int i_dev, int iter);
	
	//! [DEBUG] Print connectivity output.
	/*! Prints the @ref cblock_ID_nbr and @ref cblock_ID_nbr_child arrays divided among the cell-blocks for visualization of connectivity. Used only for small @ref Nx since it is an expensive process.
	    @param i_dev is the ID of the device to be processed.
	*/
	int		M_PrintConnectivity(int i_dev);
	
	//! Transfer data structures from CPU to GPU.
	//! All data structures pertaining to the mesh (i.e. all cells_{f/ID}_X, cblock_{f/ID}_X) are copied from host to their respective devices. This will overwrite what is on the GPU if it has not been retrieved before.
	int		M_LoadToGPU();
	
	//! Transfer data structures from GPU to CPU.
	//! All data structures pertaining to the mesh (i.e. all cells_{f/ID}_X, cblock_{f/ID}_X) are copied from their respective devices to the host. This will overwrite what is on the CPU if it has not been loaded before.
	int		M_RetrieveFromGPU();
	
	//! Freeze (or unfreeze) the current status of refinement in the grid.
	//! @param var is the indicator to freeze (0) or unfreeze (1).
	int		M_FreezeRefinedCells(int var);
	
	//! Perform refinement and coarsening wherever marked.
	/*! @param var is a debugging indicator.
	    @param scale_vec is a vector of scaling factors for rescaling variables during interpolation to new cells.
	    @param file is the file to output step execution times.
	*/
	int		M_RefineAndCoarsenCells(int var, ufloat_t *scale_vec, std::ofstream *file);
	
	int		M_Interpolate_Linear_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int		M_Interpolate_Linear_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int		M_Interpolate_Linear_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	//! Interpolate values between grid levels.
	/*! @param i_dev is the ID of the device to be processed.
	    @param L is the level on which interpolant is being computed. L+1 is the level on which computed values lie.
	    @param var is an indicator variable (0 - uniformly distribute @ref c_cells_f_F, 1 - uniformly distribute @ref c_cells_f_Fs, 2 - cubic interpolation of @ref c_cells_f_F, 3 - cubic interpolation of @ref c_cells_f_Fs, 4 - cubic interpolation of @ref c_cells_f_F to newly-added cell-blocks).
	    @param Cscale is the scaling factor (mainly for the LBM solver).
	*/
	int		M_Interpolate(int i_dev, int L, int var, ufloat_t Cscale=0, ufloat_t Cscale2=0);
	
	int		M_Average_d2q9(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int		M_Average_d3q19(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	int		M_Average_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2);
	//! Average between grid levels.
	/*! @param i_dev is the ID of the device to be processed.
	    @param L is the level on which average is being computed. L+1 is the level on which retrieved values lie.
	    @param var is an indicator variable (0 - average @ref c_cells_f_F, 1 - average @ref c_cells_f_Fs).
	*/
	int		M_Average(int i_dev, int L, int var, ufloat_t Cscale=0, ufloat_t Cscale2=0);
	
	//! [DEBUG] Modify specific values in data arrays on the GPU.
	//! @param var is an indicator for the subroutine to apply.
	int		M_CudaModVals(int var);
	
	//! [DEBUG] Print the last N gaps from the gap set.
	/*! @param var i_dev is the ID of the device to be processed.
	    @param N is the number of gaps to print.
	*/
	int		M_PrintLastNGaps(int i_dev, int N)
	{
		std::cout << "Gaps:" << std::endl;
		for (int i = 0; i < N; i++)
			std::cout << gap_set[i_dev][n_gaps[i_dev]-N+i] << " ";
		std::cout << std::endl;
		
		return 0;
	}
	
	
	
	//
	  // =====================
	  // === Constructors: ===
	  // =====================
	//
	
	//! Default constructor.
	Mesh()
	{

	}
	
	//! Main constructor.
	/*! Builds a mesh from scratch and allocates as much space on the GPU as possible.
	    @param Nxi_ the resolution array (i.e. number of cells per dimension).
	    @param dx_ the spatial step. Equal to the temporal step when using the Lattice Boltzmann Solver.
	 */
	Mesh(int *Nxi_, double dx_)
	{
		// Set mesh parameters from input.
		Nxi = new int[3]; Nxi[2] = 1;
		for (int d = 0; d < N_DIM; d++)
			Nxi[d] = Nxi_[d];
		dx = dx_;
		dx_cblock = Nbx*dx;

		
		
		// Get free and total memory from GPU(s). Get max. no. cells and round to nearest 1024.
		cudaMemGetInfo(&free_t, &total_t);
		V_t_bytes = std::ceil( 
			sizeof(int)*(1) + sizeof(ufloat_t)*(N_Q + (S_TYPE==0?N_Q:0) + 1) + 
			(sizeof(float)*(N_DIM) + sizeof(int)*(2*N_Q_max + 1+1+1) + 
			sizeof(int)*(2*N_Q_max + 10)
		)/(double)(M_CBLOCK));
		n_maxcells = (long int)free_t*M_frac / V_t_bytes;
		n_maxcells = ((n_maxcells + M_maxcells_roundoff/2) / M_maxcells_roundoff) * M_maxcells_roundoff;
		n_maxcblocks = n_maxcells / M_CBLOCK;
		n_maxgroups = n_maxcblocks / N_CHILDREN;
		std::cout << "[-]\tBefore allocations:\n";
		std::cout << "[-]\tFree: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
		std::cout << "[-]\tFree bytes: " << free_t << ", Maximum number of cells: " << n_maxcells/(1e6) << "M, Maximum number of cell blocks: " << n_maxcblocks << ", Maximum number of groups: " << n_maxgroups << std::endl;
		std::cout << "[-]\tCalculated bytes required per cell: " << V_t_bytes << std::endl << std::endl;

		
		
		// Allocate CPU memory.
		std::cout << "[-]\tInitializing CPU data...\n";
		for (int i_dev = 0; i_dev < N_DEV; i_dev++)
		{
			// Allocate memory for pointers.
			cells_ID_mask[i_dev] = new int[n_maxcells]{1};
			cells_f_F[i_dev] = new ufloat_t[n_maxcells*N_Q]{0};
#if (S_TYPE==0)
			cells_f_Fs[i_dev] = new ufloat_t[n_maxcells*N_Q]{0};
#endif
			cblock_f_X[i_dev] = new ufloat_t[n_maxcblocks*N_DIM]{0};
			cblock_ID_mask[i_dev] = new int[n_maxcblocks]{0};
			cblock_ID_nbr[i_dev] = new int[n_maxcblocks*N_Q_max]{0};
			cblock_ID_nbr_child[i_dev] = new int[n_maxcblocks*N_Q_max]{0};
			cblock_ID_onb[i_dev] = new int[n_maxcblocks]{0};
			cblock_ID_ref[i_dev] = new int[n_maxcblocks]{0};
			cblock_level[i_dev] = new int[n_maxcblocks]{0};
			
			tmp_1[i_dev] = new int[n_maxcblocks]{0};
			tmp_2[i_dev] = new ufloat_t[n_maxcblocks]{0};
			
			// Allocate memory for id_set and reset to 0.
			for (int L = 0; L < MAX_LEVELS; L++)
			{
				id_set[i_dev][L] = new int[n_maxcblocks];
				n_ids[i_dev][L] = 0;
			}
			gap_set[i_dev] = new int[n_maxcblocks];
			n_coarsecblocks = (Nxi[0]*Nxi[1]*Nxi[2])/M_CBLOCK;

			// Probe arrays.
			id_set_probed[i_dev] = new int[n_coarsecblocks];
			n_ids_probed[i_dev] = 0;
			
			for (int L = 0; L < MAX_LEVELS; L++)
				dxf_vec[L] = (ufloat_t)(dx / pow(2.0, (ufloat_t)L));
			
			std::cout << "[-]\t[->] Data arrays for holding GPU " << i_dev << "'s data allocated on the CPU." << std::endl;
		}
		std::cout << std::endl;


		
		// Allocate and initialize arrays on GPU(s).
		std::cout << "[-]\tInitializing GPU data...\n";
		for (int i_dev = 0; i_dev < N_DEV; i_dev++)
		{
			cudaSetDevice(i_dev);
			cudaStreamCreate(&streams[i_dev]);



			// Cell data.
			gpuErrchk( cudaMalloc((void **)&c_cells_ID_mask[i_dev], n_maxcells*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_cells_f_F[i_dev], n_maxcells*N_Q*sizeof(ufloat_t)) );
			gpuErrchk( cudaMalloc((void **)&c_cblock_f_X[i_dev], n_maxcblocks*N_DIM*sizeof(ufloat_t)) );
			gpuErrchk( cudaMalloc((void **)&c_cblock_ID_mask[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_cblock_ID_nbr[i_dev], n_maxcblocks*N_Q_max*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_cblock_ID_nbr_child[i_dev], n_maxcblocks*N_Q_max*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_cblock_ID_onb[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_cblock_ID_ref[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_cblock_level[i_dev], n_maxcblocks*sizeof(int)) );
			
			// ID sets.
			for (int L = 0; L < MAX_LEVELS; L++)
				gpuErrchk( cudaMalloc((void **)&c_id_set[i_dev][L], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_gap_set[i_dev], n_maxcblocks*sizeof(int)) );
			
			// Temp.
			gpuErrchk( cudaMalloc((void **)&c_tmp_1[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_tmp_2[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_tmp_3[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_tmp_4[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_tmp_5[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_tmp_6[i_dev], n_maxcblocks*N_Q_max*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_tmp_7[i_dev], n_maxcblocks*N_Q_max*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_tmp_8[i_dev], n_maxcblocks*sizeof(int)) );
			gpuErrchk( cudaMalloc((void **)&c_tmp_counting_iter[i_dev], n_maxcblocks*sizeof(int)) );

			// Thrust pointer casts.
			for (int L = 0; L < MAX_LEVELS; L++)
				c_id_set_dptr[i_dev][L] =  thrust::device_pointer_cast(c_id_set[i_dev][L]);
			c_gap_set_dptr[i_dev] = thrust::device_pointer_cast(c_gap_set[i_dev]);
			c_cblock_ID_ref_dptr[i_dev] = thrust::device_pointer_cast(c_cblock_ID_ref[i_dev]);
			c_cblock_level_dptr[i_dev] = thrust::device_pointer_cast(c_cblock_level[i_dev]);
			c_tmp_1_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_1[i_dev]);
			c_tmp_2_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_2[i_dev]);
			c_tmp_3_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_3[i_dev]);
			c_tmp_4_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_4[i_dev]);
			c_tmp_5_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_5[i_dev]);
			c_tmp_6_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_6[i_dev]);
			c_tmp_7_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_7[i_dev]);
			c_tmp_8_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_8[i_dev]);
			c_tmp_counting_iter_dptr[i_dev] = thrust::device_pointer_cast(c_tmp_counting_iter[i_dev]);
			
			// Value setting.
				// Reset masks to 1.
			Cu_ResetToValue<<<(M_BLOCK+n_maxcells-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcells, c_cells_ID_mask[i_dev], 1);
				// Reset active IDs to 0.
			Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_cblock_ID_mask[i_dev], 0);
				// Reset nbr IDs to N_SKIPID.
			Cu_ResetToValue<<<(M_BLOCK+N_Q_max*n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(N_Q_max*n_maxcblocks, c_cblock_ID_nbr[i_dev], N_SKIPID);
				// Reset nbr-child IDs to N_SKIPID.
			Cu_ResetToValue<<<(M_BLOCK+N_Q_max*n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(N_Q_max*n_maxcblocks, c_cblock_ID_nbr_child[i_dev], N_SKIPID);
				// Reset refinement IDs to 'unrefined'.
			Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_cblock_ID_ref[i_dev], V_REF_ID_INACTIVE);
				// Reset levels to 0.
			Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_cblock_level[i_dev], 0);
				// Fill the counting iterator used in refinement/coarsening.
			Cu_FillLinear<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(n_maxcblocks, c_tmp_counting_iter[i_dev]);


			
			std::cout << "[-]\t[->] Memory allocated for GPU " << i_dev << "." << std::endl;
		}
		std::cout << std::endl;

		
		
		cudaMemGetInfo(&free_t, &total_t);
		std::cout << "[-]\tAfter allocations:\n";
		std::cout << "[-]\tFree: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;

		
		
		// Set up coarse grid on all GPUs.
		std::cout << "[-]\tInitializing course grid..." << std::endl;
		M_Init();
	}

	//! Default destructor.
	~Mesh()
	{
		// Check memory currently being used before freeing for validation.
		cudaMemGetInfo(&free_t, &total_t);
		std::cout << "[-]\tBefore freeing:\n";
		std::cout << "[-]\tFree: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
		
		// Free all the memory allocated on the GPU.
		for (int i_dev = 0; i_dev < N_DEV; i_dev++)
		{
			delete[] cells_ID_mask[i_dev];
			delete[] cells_f_F[i_dev];
			delete[] cblock_f_X[i_dev];
			delete[] cblock_ID_mask[i_dev];
			delete[] cblock_ID_nbr[i_dev];
			delete[] cblock_ID_nbr_child[i_dev];
			delete[] cblock_ID_onb[i_dev];
			delete[] cblock_ID_ref[i_dev];
			delete[] cblock_level[i_dev];
			
			delete[] tmp_1[i_dev];
			delete[] tmp_2[i_dev];
			
			delete[] cells_f_U_probed_tn[i_dev];
			delete[] cells_f_U_mean[i_dev];
			
			// Allocate memory for id_set and reset to 0.
			for (int L = 0; L < MAX_LEVELS; L++)
				delete[] id_set[i_dev][L];
			delete[] gap_set[i_dev];
			delete[] coarse_I[i_dev];
			delete[] coarse_J[i_dev];
			delete[] coarse_K[i_dev];

			// Probe arrays.
			delete[] id_set_probed[i_dev];
		}
		
		// Free all the memory allocated on the GPU.
		for (int i_dev = 0; i_dev < N_DEV; i_dev++)
		{
			// Cell data.
			gpuErrchk( cudaFree(c_cells_ID_mask[i_dev]) );
			gpuErrchk( cudaFree(c_cells_f_F[i_dev]) );
			gpuErrchk( cudaFree(c_cblock_f_X[i_dev]) );
			gpuErrchk( cudaFree(c_cblock_ID_mask[i_dev]) );
			gpuErrchk( cudaFree(c_cblock_ID_nbr[i_dev]) );
			gpuErrchk( cudaFree(c_cblock_ID_nbr_child[i_dev]) );
			gpuErrchk( cudaFree(c_cblock_ID_onb[i_dev]) );
			gpuErrchk( cudaFree(c_cblock_ID_ref[i_dev]) );
			gpuErrchk( cudaFree(c_cblock_level[i_dev]) );
			
			// ID sets.
			for (int L = 0; L < MAX_LEVELS; L++)
				gpuErrchk( cudaFree(c_id_set[i_dev][L]) );
			gpuErrchk( cudaFree(c_gap_set[i_dev]) );

			// Temp.
			gpuErrchk( cudaFree(c_tmp_1[i_dev]) );
			gpuErrchk( cudaFree(c_tmp_2[i_dev]) );
			gpuErrchk( cudaFree(c_tmp_3[i_dev]) );
			gpuErrchk( cudaFree(c_tmp_4[i_dev]) );
			gpuErrchk( cudaFree(c_tmp_5[i_dev]) );
			gpuErrchk( cudaFree(c_tmp_6[i_dev]) );
			gpuErrchk( cudaFree(c_tmp_7[i_dev]) );
			gpuErrchk( cudaFree(c_tmp_8[i_dev]) );
			gpuErrchk( cudaFree(c_tmp_counting_iter[i_dev]) );
		}
		
		// Verify that GPU memory has been recovered.
		cudaMemGetInfo(&free_t, &total_t);
		std::cout << "[-]\tAfter freeing:\n";
		std::cout << "[-]\tFree: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
	}
};

#endif
