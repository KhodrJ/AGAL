#include "mesh.h"



int Mesh::M_Init()
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		// Define a padded grid from which block neighbors and groups will be built.
		int block_length_x = Nx/Nbx;
		int block_length_y = Nx/Nbx;
		int grid_height = N_DIM==2 ? 1 : Nx/Nbx;
		int grid_height_multiplier = N_DIM==2 ? 0 : 1;
		int grid_IDs[block_length_x+2][block_length_y+2][grid_height+2];
		
		// Coarse block neighbor ID and spatial coordinate arrays. Access pattern is through structured grid.
		int *init_grid_cblock_ID_nbr = new int[l_dq_max*n_coarsecblocks];
		int *init_grid_cblock_ID_nbr_child =new int[l_dq_max*n_coarsecblocks];
		ufloat_t *init_grid_cblock_f_X = new ufloat_t[N_DIM*n_coarsecblocks];
		
		// Fix boundary conditions on the edges of this padded grid.
			// X=0, X=L
		for (int k = 0; k < grid_height+2*grid_height_multiplier; k++)
		{
			for (int j = 0; j < block_length_y+2; j++)
			{
				grid_IDs[0][j][k] = -1;
				grid_IDs[block_length_x+2 - 1][j][k] = -2;
			}
		}
			// Y=0, Y=L
		for (int k = 0; k < grid_height+2*grid_height_multiplier; k++)
		{
			for (int i = 0; i < block_length_x+2; i++)
			{
				grid_IDs[i][0][k] = -3;
				grid_IDs[i][block_length_y+2 - 1][k] = -4;
			}
		}
#if (N_DIM==3)
		for (int j = 0; j < block_length_y+2; j++)
		{
			for (int i = 0; i < block_length_x+2; i++)
			{
				grid_IDs[i][j][0] = -5;
				grid_IDs[i][j][grid_height+2 - 1] = -6;
			}
		}
#endif

		// Loop over the grid_ID structure and set IDs of blocks.
		// 	Interior defined from (1->block_length_xi+2-1)^3.
		//	Switch i_k++ to i_k+=2 for the tiled grid again.
		int ID_counter = 0;
		for (int k = 1*grid_height_multiplier; k < grid_height*grid_height_multiplier+2 - 1; k++)
		{
			for (int j = 1; j < block_length_y+2 - 1; j++)
			{
				for (int i = 1; i < block_length_x+2 - 1; i++)
				{
				// Structured grid.
				grid_IDs[i][j][k] = (i-1) + (j-1)*block_length_x + (k-1)*block_length_x*block_length_y*grid_height_multiplier;
			
				// Tiled grid.
				//for (int sub_k=0; sub_k<1+1*grid_height_multiplier; sub_k++) {for (int sub_j=0; sub_j<2; sub_j++) {for (int sub_i=0; sub_i<2; sub_i++) {
				//	int tile_ID_kap = sub_i + sub_j*2 + sub_k*2*2*grid_height_multiplier;
				//	grid_IDs[i+sub_i][j+sub_j][k+sub_k] = ID_counter + tile_ID_kap;
				//}}};
				
				// Increment.
				ID_counter += N_CHILDREN;
				
				//std::cout << "Inserted Block " << grid_IDs[i][j][k] << " with coords (" << (i-1) << "," << (j-1) << "," << std::max(0,k-1) << ")..." << std::endl;
				}
			}
		};
		
		// Now loop over the structure to build the spatial coordinates and set neighbors.
		for (int k = 1*grid_height_multiplier; k < grid_height*grid_height_multiplier+2 - 1; k++)
		{
			for (int j = 1; j < block_length_y+2 - 1; j++)
			{
				for (int i = 1; i < block_length_x+2 - 1; i++)
				{
					int block_ID = grid_IDs[i][j][k];
					
					init_grid_cblock_f_X[block_ID + 0*n_coarsecblocks] = (i-1)*dx_cblock;
					init_grid_cblock_f_X[block_ID + 1*n_coarsecblocks] = (j-1)*dx_cblock;
		#if (N_DIM==3)
					init_grid_cblock_f_X[block_ID + 2*n_coarsecblocks] = (k-1)*dx_cblock;
		#endif
					
					for (int p = 0; p < l_dq_max; p++)
					{
						init_grid_cblock_ID_nbr[block_ID + p*n_coarsecblocks] = grid_IDs[ (int)(i+c[N_DIMc][p+0*l_dq_max]) ][ (int)(j+c[N_DIMc][p+1*l_dq_max]) ][ (int)(grid_height_multiplier*(k+c[N_DIMc][p+2*l_dq_max])) ];
						init_grid_cblock_ID_nbr_child[block_ID + p*n_coarsecblocks] = init_grid_cblock_ID_nbr[block_ID + p*n_coarsecblocks] < 0 ? init_grid_cblock_ID_nbr[block_ID + p*n_coarsecblocks]:N_SKIPID;
						
					}
				}
			}
		};
		
		// Reset the first n_coarsecblocks ref IDs. This will subsequently be copied into the GPU.
		for (int kap = 0; kap < n_coarsecblocks; kap++)
			cblock_ID_ref[i_dev][kap] = V_REF_ID_UNREFINED;
		
		
		
		
		
		// Copy arrays into GPU memory and finish. A section of size n_coarsecblocks is added to each p (nbrs) and d (locations), although these
		// sections are separated by n_maxcblocks in GPU memory.
		gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], n_coarsecblocks*sizeof(int), cudaMemcpyHostToDevice) );
		for (int p = 0; p < l_dq_max; p++)
		{
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr[i_dev][p*n_maxcblocks], &init_grid_cblock_ID_nbr[p*n_coarsecblocks], n_coarsecblocks*sizeof(int), cudaMemcpyHostToDevice) );
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &init_grid_cblock_ID_nbr_child[p*n_coarsecblocks], n_coarsecblocks*sizeof(int), cudaMemcpyHostToDevice) );
		}
		for (int d = 0; d < N_DIM; d++)
		{
			gpuErrchk( cudaMemcpy(&c_cblock_f_X[i_dev][d*n_maxcblocks], &init_grid_cblock_f_X[d*n_coarsecblocks], n_coarsecblocks*sizeof(ufloat_t), cudaMemcpyHostToDevice) );
		}
		
		
		
		
		
		// Load the interpolation matrices into GPU memory.
		double tmp_mat_input[M_CBLOCK*M_CBLOCK*N_CHILDREN];
		double tmp_mat_fd_input[M_CBLOCK*M_CBLOCK*N_CHILDREN];
		int amount_to_load = 0;
#if (N_DIM==2)
		#if (S_INTERP_TYPE==0)
		amount_to_load = 16*16;
		std::ifstream mat_input; mat_input.open("./init_grid_data/interp_mat_2d.txt");
		#else
		amount_to_load = 16*16*4;
		std::ifstream mat_input; mat_input.open("./init_grid_data/interp_mat_2d_alt.txt");
		#endif
#else
		#if (S_INTERP_TYPE==0)
		amount_to_load = 64*64;
		std::ifstream mat_input; mat_input.open("./init_grid_data/interp_mat_3d.txt");
		#else
		amount_to_load = 64*64*8;
		std::ifstream mat_input; mat_input.open("./init_grid_data/interp_mat_3d_alt.txt");
		#endif
#endif
		
		for (int k = 0; k < amount_to_load; k++)
			mat_input >> tmp_mat_input[k];
		mat_input.close();
		
		for (int i_dev = 0; i_dev < N_DEV; i_dev++)
			gpuErrchk( cudaMemcpy(c_mat_interp[i_dev], tmp_mat_input, amount_to_load*sizeof(double), cudaMemcpyHostToDevice) );
		
		
		
		
		
		// Delete temporary arrays.
		delete[] init_grid_cblock_ID_nbr;
		delete[] init_grid_cblock_ID_nbr_child;
		delete[] init_grid_cblock_f_X;
	}
	
	return 0;
}
