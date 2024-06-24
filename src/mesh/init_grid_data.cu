#include "mesh.h"



int Mesh::M_Init()
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		//
		  // ===============================
		  // === Coarse Grid Definition: ===
		  // ===============================
		//
		
		// Define a padded grid from which block neighbors and groups will be built.
		int block_length_x = Nxi[0]/Nbx;
		int block_length_y = Nxi[1]/Nbx;
		int grid_height = N_DIM==2 ? 1 : Nxi[2]/Nbx;
		int grid_height_multiplier = N_DIM==2 ? 0 : 1;
		int grid_IDs[block_length_x+2][block_length_y+2][grid_height+2];
		
		// Coarse block neighbor ID and spatial coordinate arrays. Access pattern is through structured grid.
		int *init_grid_cblock_ID_nbr = new int[N_Q_max*n_coarsecblocks];
		int *init_grid_cblock_ID_nbr_child = new int[N_Q_max*n_coarsecblocks];
		int *init_grid_cblock_ID_onb = new int[n_coarsecblocks];
		ufloat_t *init_grid_cblock_f_X = new ufloat_t[N_DIM*n_coarsecblocks];

		// Loop over the grid_ID structure and set IDs of blocks.
		// 	Interior defined from (1->block_length_xi+2-1)^3.
		//	Switch i_k++ to i_k+=2 for the tiled grid again.
		int ID_counter = 0;
		double D = 1.0/32.0;
		for (int k = 1*grid_height_multiplier; k < grid_height*grid_height_multiplier+2 - 1; k++)
		{
			for (int j = 1; j < block_length_y+2 - 1; j++)
			{
				for (int i = 1; i < block_length_x+2 - 1; i++)
				{
					bool in_interior = true;
					
					double x_bi = (i-1)*dx_cblock + dx_cblock*0.5;
					double y_bi = (j-1)*dx_cblock + dx_cblock*0.5;
					double z_bi = (k-1)*dx_cblock + dx_cblock*0.5;
					
#if (N_CASE==1)
					// Establishes the square square cylinder by removing corresponding coarse blocks.
					if (x_bi >= 0.3125-(D/2.0) && x_bi <= 0.3125-(D/2.0) + D && y_bi >= ( (L_fy-D)/2.0) && y_bi <= ( (L_fy-D)/2.0 ) + D)
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
#ifdef PERIODIC_Z
				grid_IDs[i][j][0] = grid_IDs[i][j][grid_height+2 - 1 - 1];
				grid_IDs[i][j][grid_height+2 - 1] = grid_IDs[i][j][1];
#else
				grid_IDs[i][j][0] = -5;
				grid_IDs[i][j][grid_height+2 - 1] = -6;
#endif
			}
		}
#endif
			// Y=0, Y=L
		for (int k = 0; k < grid_height+2*grid_height_multiplier; k++)
		{
			for (int i = 0; i < block_length_x+2; i++)
			{
#ifdef PERIODIC_Y
				grid_IDs[i][0][k] = grid_IDs[i][block_length_y+2 - 1 - 1][k];
				grid_IDs[i][block_length_y+2 - 1][k] = grid_IDs[i][1][k];				
#else
				grid_IDs[i][0][k] = -3;
				grid_IDs[i][block_length_y+2 - 1][k] = -4;
#endif
			}
		}
			// X=0, X=L
		for (int k = 0; k < grid_height+2*grid_height_multiplier; k++)
		{
			for (int j = 0; j < block_length_y+2; j++)
			{
#ifdef PERIODIC_X
				grid_IDs[0][j][k] = grid_IDs[block_length_x+2 - 1 - 1][j][k];
				grid_IDs[block_length_x+2 - 1][j][k] = grid_IDs[1][j][k];
#else
				grid_IDs[0][j][k] = -1;
				grid_IDs[block_length_x+2 - 1][j][k] = -2;
#endif
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
			#if (N_DIM==3)
						init_grid_cblock_f_X[block_ID + 2*n_coarsecblocks] = (k-1)*dx_cblock;
			#endif
						
						int block_on_boundary = 0;
						for (int p = 0; p < N_Q_max; p++)
						{
							// Neighor.
							init_grid_cblock_ID_nbr[block_ID + p*n_coarsecblocks] = grid_IDs[ (int)(i+c[N_DIMc][p+0*N_Q_max]) ][ (int)(j+c[N_DIMc][p+1*N_Q_max]) ][ (int)(grid_height_multiplier*(k+c[N_DIMc][p+2*N_Q_max])) ];
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
			// Initialize id_set and id_max to structured grid.
		n_ids[i_dev][0] = n_coarsecblocks;
		id_max[i_dev][0] = id_max[i_dev][MAX_LEVELS] = n_ids[i_dev][0];
		for (int xi = 0; xi < n_ids[i_dev][0]; xi++)
			id_set[i_dev][0][xi] = xi;
			// Initialize gap_set. 
		for (int xi = 0; xi < n_maxcblocks-n_ids[i_dev][0]; xi++)
			gap_set[i_dev][xi] = (n_maxcblocks-1) - xi;
		n_gaps[i_dev] = n_maxcblocks-n_ids[i_dev][0];
		
		// Initialize the probed solution fields.
		cells_f_U_probed_tn[i_dev] = new ufloat_t[n_ids_probed[i_dev]*M_CBLOCK*N_DIM]{0};
		cells_f_U_mean[i_dev] = new ufloat_t[n_ids[i_dev][0]*M_CBLOCK*3]{0};
		
		//
		  // ====================
		  // === Copy to GPU: ===
		  // ====================
		//
		
		// Copy arrays into GPU memory and finish. A section of size n_coarsecblocks is added to each p (nbrs) and d (locations), although these
		// sections are separated by n_maxcblocks in GPU memory.
		
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
		gpuErrchk( cudaMemcpy(c_id_set[i_dev][0], id_set[i_dev][0], n_ids[i_dev][0]*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_gap_set[i_dev], gap_set[i_dev], n_gaps[i_dev]*sizeof(int), cudaMemcpyHostToDevice) );
		
		//
		  // ====================
		  // === Free Memory: ===
		  // ====================
		//
		
		// Delete temporary arrays.
		delete[] init_grid_cblock_ID_nbr;
		delete[] init_grid_cblock_ID_nbr_child;
		delete[] init_grid_cblock_ID_onb;
		delete[] init_grid_cblock_f_X;
	}
	
	return 0;
}
