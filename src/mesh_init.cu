/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

int Mesh::M_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::string output_dir_)
{
	std::cout << " o====================================================================================" << std::endl;
	std::cout << " | New: Mesh Object                                                                   " << std::endl;
	std::cout << " o====================================================================================" << std::endl;
	
	// o====================================================================================
	// | Initialize mesh data.
	// o====================================================================================
	
	// Set mesh parameters from input.
	Lx                      = params_dbl["L_c"];
	Ly                      = params_dbl["L_fy"]*Lx;
	Lz                      = params_dbl["L_fz"]*Lx;
	Nx                      = params_int["Nx"];
	Ny                      = (int)(Nx*(Ly/Lx));
	Nz                      = (int)(Nx*(Lz/Lx));
	Nxi                     = new int[3]{Nx, Ny, Nz};
	dx                      = Lx/Nx;
	dx_cblock               = Nbx*Nqx*dx;
	MAX_LEVELS              = params_int["MAX_LEVELS"];
	MAX_LEVELS_INTERIOR     = params_int["MAX_LEVELS_INTERIOR"];
	N_ITER_TOTAL            = params_int["N_ITER_TOTAL"];
	N_LEVEL_START           = params_int["N_LEVEL_START"];
	N_RESTART               = params_int["N_RESTART"];
	PERIODIC_X              = params_int["PERIODIC_X"];
	PERIODIC_Y              = params_int["PERIODIC_Y"];
	PERIODIC_Z              = params_int["PERIODIC_Z"];
	S_INTERP                = params_int["S_INTERP"];
	S_AVERAGE               = params_int["S_AVERAGE"];
	P_REFINE                = params_int["P_REFINE"];
	N_REFINE_START          = params_int["N_REFINE_START"];
	N_REFINE_INC            = params_int["N_REFINE_INC"];
	N_PROBE                 = params_int["N_PROBE"];
	N_PROBE_DENSITY         = params_int["N_PROBE_DENSITY"];
	N_PROBE_FREQUENCY       = params_int["N_PROBE_FREQUENCY"];
	V_PROBE_TOL             = params_dbl["V_PROBE_TOL"];
	N_PROBE_FORCE           = params_int["N_PROBE_FORCE"];
	N_PROBE_F_FREQUENCY     = params_int["N_PROBE_F_FREQUENCY"];
	N_PROBE_AVE             = params_int["N_PROBE_AVE"];
	N_PROBE_AVE_FREQUENCY   = params_int["N_PROBE_AVE_FREQUENCY"];
	N_PROBE_AVE_START       = params_int["N_PROBE_AVE_START"];
	N_PRINT_LEVELS          = params_int["N_PRINT_LEVELS"];
	N_PRINT_LEVELS_LEGACY   = params_int["N_PRINT_LEVELS_LEGACY"];
	P_OUTPUT                = params_int["P_OUTPUT"];
	N_OUTPUT_START          = params_int["N_OUTPUT_START"];
	VOL_I_MIN               = params_int["VOL_I_MIN"]/(Nbx*Nqx);
	VOL_I_MAX               = params_int["VOL_I_MAX"]/(Nbx*Nqx);
	VOL_J_MIN               = params_int["VOL_J_MIN"]/(Nbx*Nqx);
	VOL_J_MAX               = params_int["VOL_J_MAX"]/(Nbx*Nqx);
	VOL_K_MIN               = params_int["VOL_K_MIN"]/(Nbx*Nqx);
	VOL_K_MAX               = params_int["VOL_K_MAX"]/(Nbx*Nqx);
	output_dir              = output_dir_;
	
	
	// Additional checks on input parameters.
	if (N_DIM == 2) Nz = 1;
	if (N_DIM == 2) Nxi[2] = 1;
	if (N_LEVEL_START > MAX_LEVELS-1)
		N_LEVEL_START = MAX_LEVELS-1;
	if (N_PRINT_LEVELS > MAX_LEVELS)
		N_PRINT_LEVELS = MAX_LEVELS;
	if (N_PRINT_LEVELS_LEGACY > MAX_LEVELS)
		N_PRINT_LEVELS_LEGACY = MAX_LEVELS;
	
	
	// Make the output directory if it doesn't already exist.
	if (output_dir.back() != '/')
		output_dir = output_dir + "/";
	std::string debug_string = "echo Checking existence of output directory...";
	std::string file_check_string = std::string("if ! test -d ") + output_dir + std::string(" ; then mkdir ") + output_dir + std::string("; mkdir ") + output_dir + std::string("/img; mkdir ") + output_dir + std::string("input; echo \"[-] Creating new results directory (") + output_dir + std::string(")...\"; fi");
	system(file_check_string.c_str());
	
	
	// Make a copy of the input and configuration files for replicability.
	std::string copy_string = "cp ../input/* " + output_dir + "input/.; cp ./confmake.sh " + output_dir + "input/.;";
	system(copy_string.c_str());
	
	
	// Create the direct-output binary file and initialize metadata.
	// 
	// Format Details:
	// - Little Endian.
	//
	// Format:
	// - n_frames (int)     Number of frames in the time series.
	// - N{x/y/z} (int)     Resolution in {x/y/z}
	std::string of_name = output_dir + std::string("out_direct.dat");
	output_file_direct = new std::ofstream(of_name, std::ios::binary);
	int *output_int_params = new int[100];
	double *output_dbl_params = new double[100];
	for (int p = 0; p < 100; p++)
	{
		output_int_params[p] = 0;
		output_dbl_params[p] = 0.0;
	}
	output_int_params[0] = (N_ITER_TOTAL - N_OUTPUT_START)/(P_OUTPUT);
	output_int_params[1] = P_OUTPUT;
	output_int_params[2] = N_OUTPUT_START;
	output_int_params[3] = Nx;
	output_int_params[4] = Ny;
	output_int_params[5] = Nz;
	output_int_params[6] = N_LEVEL_START;
	output_int_params[7] = N_PRINT_LEVELS;
	output_int_params[8] = N_PRECISION;
	output_int_params[9] = VOL_I_MIN;
	output_int_params[10] = VOL_I_MAX*Nqx;
	output_int_params[11] = VOL_J_MIN*Nqx;
	output_int_params[12] = VOL_J_MAX*Nqx;
	output_int_params[13] = VOL_K_MIN*Nqx;
	output_int_params[14] = VOL_K_MAX*Nqx;
	output_dbl_params[0] = (double)Lx;
	output_dbl_params[1] = (double)Ly;
	output_dbl_params[2] = (double)Lz;
	(*output_file_direct).write((char *)&output_int_params[0], 100*sizeof(int));
	(*output_file_direct).write((char *)&output_dbl_params[0], 100*sizeof(double));
	delete[] output_int_params;
	delete[] output_dbl_params;
	
	
	// Get free and total memory from GPU(s). Get max. no. cells and round to nearest 1024.
	gpuErrchk( cudaMemGetInfo(&free_t, &total_t) );
	cudaDeviceSynchronize();
	N_bytes_pc = std::ceil( 
		sizeof(int)*(1) + sizeof(ufloat_t)*(N_Q + 1) + 
		(sizeof(float)*(N_DIM) + sizeof(int)*(2*N_Q_max + 1+1+1) + 
		sizeof(int)*(2*N_Q_max + 10)
	)/(double)(M_CBLOCK));
	n_maxcells = (long int)free_t*M_FRAC / N_bytes_pc;
	n_maxcells = ((n_maxcells + M_RNDFF/2) / M_RNDFF) * M_RNDFF;
	n_maxcblocks = n_maxcells / M_CBLOCK;
	std::cout << "[-] Before allocations:\n";
	std::cout << "    Free: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
	std::cout << "    Free bytes: " << free_t << ", Maximum number of cells: " << n_maxcells << ", Maximum number of cell blocks: " << n_maxcblocks << std::endl;
	std::cout << "    Calculated bytes required per cell: " << N_bytes_pc << std::endl << std::endl;
	
	
	// Allocate CPU memory.
	std::cout << "[-] Initializing CPU data...\n";
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		// Allocate memory for pointers.
		cells_ID_mask[i_dev] = new int[n_maxcells]{1};
		cells_f_F[i_dev] = new ufloat_t[n_maxcells*N_Q]{0};
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
		n_ids[i_dev] = new int[MAX_LEVELS+1];
		id_max[i_dev] = new int[MAX_LEVELS+1];
		for (int L = 0; L < MAX_LEVELS; L++)
		{
			//id_set[i_dev][L] = new int[n_maxcblocks];
			n_ids[i_dev][L] = 0;
		}
		id_set[i_dev] = new int[MAX_LEVELS*n_maxcblocks];
		gap_set[i_dev] = new int[n_maxcblocks];
		n_coarsecblocks = (Nxi[0]*Nxi[1]*Nxi[2])/M_CBLOCK;

		// Probe arrays.
		id_set_probed[i_dev] = new int[n_coarsecblocks];
		n_ids_probed[i_dev] = 0;
		
		// Grid spatial steps.
		dxf_vec = new ufloat_t[MAX_LEVELS];
		for (int L = 0; L < MAX_LEVELS; L++)
			dxf_vec[L] = (ufloat_t)(dx / pow(2.0, (ufloat_t)L));
		
		std::cout << "    Data arrays for holding GPU " << i_dev << "'s data allocated on the CPU." << std::endl;
	}
	
	
	// Allocate and initialize arrays on GPU(s).
	std::cout << "[-] Initializing GPU data...\n";
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
		//for (int L = 0; L < MAX_LEVELS; L++)
		gpuErrchk( cudaMalloc((void **)&c_id_set[i_dev], MAX_LEVELS*n_maxcblocks*sizeof(int)) );
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
		//for (int L = 0; L < MAX_LEVELS; L++)
		c_id_set_dptr[i_dev] =  thrust::device_pointer_cast(c_id_set[i_dev]);
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
			// Reset cell-block boundary IDs to 0.
		Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_cblock_ID_onb[i_dev], 0);
			// Fill the counting iterator used in refinement/coarsening.
		Cu_FillLinear<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(n_maxcblocks, c_tmp_counting_iter[i_dev]);
		
		std::cout << "    Data arrays allocated on GPU " << i_dev << "." << std::endl << std::endl;;
	}
	
	
	cudaMemGetInfo(&free_t, &total_t);
	std::cout << "[-] After allocations:\n";
	std::cout << "    Free: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
	
	
	// Set up coarse grid on all GPUs.
	std::cout << "[-] Initializing root grid..." << std::endl;
	
	
	// o====================================================================================
	// | Build the root grid.
	// o====================================================================================
	
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		// Define a padded grid from which block neighbors and groups will be built.
		int block_length_x = Nxi[0]/(Nbx*Nqx);
		int block_length_y = Nxi[1]/(Nbx*Nqx);
		int grid_height = N_DIM==2 ? 1 : Nxi[2]/(Nbx*Nqx);
		int grid_height_multiplier = N_DIM==2 ? 0 : 1;
		int grid_IDs[block_length_x+2][block_length_y+2][grid_height+2];
		
		// Coarse block neighbor ID and spatial coordinate arrays. Access pattern is through structured grid.
		int *init_grid_cblock_ID_nbr = new int[N_Q_max*n_coarsecblocks];
		int *init_grid_cblock_ID_nbr_child = new int[N_Q_max*n_coarsecblocks];
		int *init_grid_cblock_ID_onb = new int[n_coarsecblocks];
		ufloat_t *init_grid_cblock_f_X = new ufloat_t[N_DIM*n_coarsecblocks];

		// Loop over the grid_ID structure and set IDs of blocks.
		int ID_counter = 0;
		for (int k = 1*grid_height_multiplier; k < grid_height*grid_height_multiplier+2 - 1; k++)
		{
			for (int j = 1; j < block_length_y+2 - 1; j++)
			{
				for (int i = 1; i < block_length_x+2 - 1; i++)
				{
					bool in_interior = true;
					
#if (N_CASE==1)
					double D = 1.0/32.0;
					double x_bi = (i-1)*dx_cblock + dx_cblock*0.5;
					double y_bi = (j-1)*dx_cblock + dx_cblock*0.5;
					
					// Establishes the square square cylinder by removing corresponding coarse blocks.
					if (x_bi >= 0.3125-(D/2.0) && x_bi <= 0.3125-(D/2.0) + D && y_bi >= ( (Ly-D)/2.0) && y_bi <= ( (Ly-D)/2.0 ) + D)
					{
						in_interior = false;
						grid_IDs[i][j][k] = -8;
						
						n_coarsecblocks--;
					}
#endif
					
					if (in_interior)
					{
						// Structured grid.
						grid_IDs[i][j][k] = ID_counter;
						
						// Check probe density and add to list if so.
						if ( (i-1)%N_PROBE_DENSITY==0 && (j-1)%N_PROBE_DENSITY==0 && ((k-1)%N_PROBE_DENSITY==0 || k-1 == -1) )
						{
							id_set_probed[i_dev][n_ids_probed[i_dev]] = ID_counter;
							n_ids_probed[i_dev]++;
						}
						
						ID_counter++;
					}
				}
			}
		};
		
		// Allocate memory for coarse Id arrays.
		coarse_I[i_dev] = new int[n_coarsecblocks];
		coarse_J[i_dev] = new int[n_coarsecblocks];
		coarse_K[i_dev] = new int[n_coarsecblocks];
		
		// Fix boundary conditions on the edges of this padded grid.
#if (N_DIM==3)
				// Z=0, Z=L
			for (int j = 0; j < block_length_y+2; j++)
			{
				for (int i = 0; i < block_length_x+2; i++)
				{
					if (PERIODIC_Z==1)
					{
						grid_IDs[i][j][0] = grid_IDs[i][j][grid_height+2 - 1 - 1];
						grid_IDs[i][j][grid_height+2 - 1] = grid_IDs[i][j][1];
					}
					else
					{
						grid_IDs[i][j][0] = -5;
						grid_IDs[i][j][grid_height+2 - 1] = -6;
					}
				}
			}
#endif
			// Y=0, Y=L
		for (int k = 0; k < grid_height+2*grid_height_multiplier; k++)
		{
			for (int i = 0; i < block_length_x+2; i++)
			{
				if (PERIODIC_Y==1)
				{
					grid_IDs[i][0][k] = grid_IDs[i][block_length_y+2 - 1 - 1][k];
					grid_IDs[i][block_length_y+2 - 1][k] = grid_IDs[i][1][k];
				}
				else
				{
					grid_IDs[i][0][k] = -3;
					grid_IDs[i][block_length_y+2 - 1][k] = -4;
				}
			}
		}
			// X=0, X=L
		for (int k = 0; k < grid_height+2*grid_height_multiplier; k++)
		{
			for (int j = 0; j < block_length_y+2; j++)
			{
				if (PERIODIC_X==1)
				{
					grid_IDs[0][j][k] = grid_IDs[block_length_x+2 - 1 - 1][j][k];
					grid_IDs[block_length_x+2 - 1][j][k] = grid_IDs[1][j][k];
				}
				else
				{
					grid_IDs[0][j][k] = -1;
					grid_IDs[block_length_x+2 - 1][j][k] = -2;
				}
			}
		}
		
		// Now loop over the structure to build the spatial coordinates and set neighbors.
		for (int k = 1*grid_height_multiplier; k < grid_height*grid_height_multiplier+2 - 1; k++)
		{
			for (int j = 1; j < block_length_y+2 - 1; j++)
			{
				for (int i = 1; i < block_length_x+2 - 1; i++)
				{
					int block_ID = grid_IDs[i][j][k];
					if (block_ID >= 0)
					{
					
						init_grid_cblock_f_X[block_ID + 0*n_coarsecblocks] = (i-1)*dx_cblock;
						init_grid_cblock_f_X[block_ID + 1*n_coarsecblocks] = (j-1)*dx_cblock;
						if (N_DIM==3)
							init_grid_cblock_f_X[block_ID + 2*n_coarsecblocks] = (k-1)*dx_cblock;
						
						int block_on_boundary = 0;
						for (int p = 0; p < N_Q_max; p++)
						{
							// Neighor.
							init_grid_cblock_ID_nbr[block_ID + p*n_coarsecblocks] = grid_IDs[ (int)(i+c[N_DIM-2][p+0*N_Q_max]) ][ (int)(j+c[N_DIM-2][p+1*N_Q_max]) ][ (int)(grid_height_multiplier*(k+c[N_DIM-2][p+2*N_Q_max])) ];
								
							// Mark if on boundary.
							if (init_grid_cblock_ID_nbr[block_ID + p*n_coarsecblocks] < 0)
								block_on_boundary = 1;
							
							// Neighbor child.
							init_grid_cblock_ID_nbr_child[block_ID + p*n_coarsecblocks] = init_grid_cblock_ID_nbr[block_ID + p*n_coarsecblocks] < 0 ? init_grid_cblock_ID_nbr[block_ID + p*n_coarsecblocks]:N_SKIPID;
							
						}
						init_grid_cblock_ID_onb[block_ID] = block_on_boundary;
						
						
						coarse_I[i_dev][block_ID] = i-1;
						coarse_J[i_dev][block_ID] = j-1;
						coarse_K[i_dev][block_ID] = N_DIM==2?0:(k-1);
					}
				}
			}
		}
		
		// Reset the first n_coarsecblocks ref IDs. This will subsequently be copied into the GPU.
		for (int kap = 0; kap < n_coarsecblocks; kap++)
			cblock_ID_ref[i_dev][kap] = V_REF_ID_UNREFINED;
		
		// Build Id and gap sets.
		// -   Initialize id_set and id_max to structured grid.
		n_ids[i_dev][0] = n_coarsecblocks;
		id_max[i_dev][0] = id_max[i_dev][MAX_LEVELS] = n_ids[i_dev][0];
		for (int xi = 0; xi < n_ids[i_dev][0]; xi++)
			id_set[i_dev][xi] = xi;
		// -   Initialize gap_set. 
		for (int xi = 0; xi < n_maxcblocks-n_ids[i_dev][0]; xi++)
			gap_set[i_dev][xi] = (n_maxcblocks-1) - xi;
		n_gaps[i_dev] = n_maxcblocks-n_ids[i_dev][0];
		
		// Initialize the probed solution fields.
		cells_f_U_probed_tn[i_dev] = new ufloat_t[n_ids_probed[i_dev]*M_CBLOCK*N_DIM]{0};
		cells_f_U_mean[i_dev] = new ufloat_t[n_ids[i_dev][0]*M_CBLOCK*4]{0};
		
		
		// o====================================================================================
		// | Copy to GPU.
		// o====================================================================================
		// | Copy arrays into GPU memory and finish. A section of size n_coarsecblocks is added
		// | to each p (nbrs) and d (locations), although these sections are separated by
		// | n_maxcblocks in GPU memory.
		// o____________________________________________________________________________________
		
		// Ref. Ids
		gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], n_coarsecblocks*sizeof(int), cudaMemcpyHostToDevice) );
		for (int p = 0; p < N_Q_max; p++)
		{
			// Nbr Ids.
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr[i_dev][p*n_maxcblocks], &init_grid_cblock_ID_nbr[p*n_coarsecblocks], n_coarsecblocks*sizeof(int), cudaMemcpyHostToDevice) );
			// Nbr child Ids.
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &init_grid_cblock_ID_nbr_child[p*n_coarsecblocks], n_coarsecblocks*sizeof(int), cudaMemcpyHostToDevice) );
		}
		
		// On-Boundary Ids.
		gpuErrchk( cudaMemcpy(c_cblock_ID_onb[i_dev], init_grid_cblock_ID_onb, n_coarsecblocks*sizeof(int), cudaMemcpyHostToDevice) );
		
		// Spatial coordinates.
		for (int d = 0; d < N_DIM; d++)
			gpuErrchk( cudaMemcpy(&c_cblock_f_X[i_dev][d*n_maxcblocks], &init_grid_cblock_f_X[d*n_coarsecblocks], n_coarsecblocks*sizeof(ufloat_t), cudaMemcpyHostToDevice) );
		
		// Id sets.
		gpuErrchk( cudaMemcpy(c_id_set[i_dev], id_set[i_dev], n_ids[i_dev][0]*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_gap_set[i_dev], gap_set[i_dev], n_gaps[i_dev]*sizeof(int), cudaMemcpyHostToDevice) );
		
		
		// o====================================================================================
		// | Free memory allocations.
		// o====================================================================================
		
		// Delete temporary arrays.
		delete[] init_grid_cblock_ID_nbr;
		delete[] init_grid_cblock_ID_nbr_child;
		delete[] init_grid_cblock_ID_onb;
		delete[] init_grid_cblock_f_X;
	}
	
	return 0;
}
