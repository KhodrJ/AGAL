/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"


/*
 .d8888b.           888                           
d88P  Y88b          888                           
Y88b.               888                           
 "Y888b.    .d88b.  888 888  888  .d88b.  888d888 
    "Y88b. d88""88b 888 888  888 d8P  Y8b 888P"   
      "888 888  888 888 Y88  88P 88888888 888     
Y88b  d88P Y88..88P 888  Y8bd8P  Y8b.     888     
 "Y8888P"   "Y88P"  888   Y88P    "Y8888  888     
*/


int Solver_LBM::S_Init()
{
	// Compute the relaxation rate to be applied for all grid levels.
	for (int L = 0; L < MAX_LEVELS; L++)
	{
		dx_vec[L] = (ufloat_t)mesh->dxf_vec[L];
		tau_vec[L] = (ufloat_t)(v0*3.0 + 0.5*dx_vec[L]);
	}

	return 0;
}

int Solver_LBM::S_Initialize(int i_dev, int L)
{
	S_SetInitialConditions(i_dev, L, true);
	S_ComputeEq(0, L);
	for (int k = 0; k < N_INIT_ITER; k++)
	{
		S_SetInitialConditions(i_dev, L, false);
		S_Stream(i_dev, L);
	}
	
	return 0;
}

#if (V_ADV_TYPE == V_ADV_TYPE_UNIFORM)
int Solver_LBM::S_Advance(int i_dev, int L, std::ofstream *file, double *tmp)
{
#if (MAX_LEVELS>1)
	
	#if (P_SHOW_ADVANCE == 1)
	std::cout << "Collide and stream on level " << L << std::endl;
	#endif
	S_Collide(i_dev, L);
	S_Stream(i_dev, L); //
	
	if (L != MAX_LEVELS-1)
	{
		S_Advance(i_dev, L+1, file, tmp);
	}
	if (L != 0)
	{
		#if (P_SHOW_ADVANCE == 1)
		std::cout << "Interpolate from level " << L-1 << " to " << L << std::endl;
		#endif
		S_Interpolate(i_dev, L-1, 1);
	}
	
	#if (P_SHOW_ADVANCE == 1)
	//std::cout << "Stream on level " << L << std::endl;
	#endif
	//S_Stream(i_dev, L);
	if (L != MAX_LEVELS-1)
	{
		#if (P_SHOW_ADVANCE == 1)
		std::cout << "Average values from level " << L+1 << " to " << L << std::endl;
		#endif
		
		//std::cout << "Global average values from level " << L+1 << " to " << L << std::endl;
		//S_Average(i_dev, L, 0);
	}
		
	if (L == 0)
		return 0;

	#if (P_SHOW_ADVANCE == 1)
	std::cout << "Collide and stream on level " << L << std::endl;
	#endif
	S_Collide(i_dev, L);
	S_Stream(i_dev, L); //
	
	if (L != MAX_LEVELS-1)
	{
		S_Advance(i_dev, L+1, file, tmp);
	}
	
	#if (P_SHOW_ADVANCE == 1)
	//std::cout << "Stream on level " << L << std::endl;
	#endif
	//S_Stream(i_dev, L); //
	if (L != MAX_LEVELS-1)
	{
		#if (P_SHOW_ADVANCE == 1)
		std::cout << "Average values from level " << L+1 << " to " << L << std::endl;
		#endif
		
		//std::cout << "Global average values from level " << L+1 << " to " << L << std::endl;
		//S_Average(i_dev, L, 0);
	}
#else
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		S_Collide(i_dev, 0);
		S_Stream(i_dev, 0);
	}
#endif

	return 0;	
}
#else
int Solver_LBM::S_Advance(int i_dev, int L, std::ofstream *file, double *tmp)
{
#if (MAX_LEVELS>1)
	if (L == 0)
	{
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Interpolating from level " << 0 << " to " << 1 << "..." << std::endl;
		#endif
		tic_simple("");
		S_Interpolate(i_dev, 0, V_ADV_TYPE) ;//, N_Pf(0.5)*tau_vec[1]/tau_vec[0]);
		cudaDeviceSynchronize();
		tmp[0] += toc_simple("",T_US,0);
		
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Colliding and streaming on level " << 0 << "..." << std::endl;
		#endif
		tic_simple("");
		S_Collide(i_dev, 0);
		cudaDeviceSynchronize();
		tmp[1] += toc_simple("",T_US,0);
		tic_simple("");
		S_Stream(i_dev, 0);
		cudaDeviceSynchronize();
		tmp[2] += toc_simple("",T_US,0);
		
		S_Advance(i_dev, 1, file, tmp);
		
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Averaging from level " << 1 << " to " << 0 << "..." << std::endl;
		#endif
		tic_simple("");
		S_Average(i_dev, 0, V_ADV_TYPE) ;//, N_Pf(2.0)*tau_vec[L]/tau_vec[L+1]);
		cudaDeviceSynchronize();
		tmp[3] += toc_simple("",T_US,0);
		
		for (int Lp = 0; Lp < 4*MAX_LEVELS; Lp++)
			*file << tmp[Lp] << " ";
		*file << std::endl;
	}
	else
	{
		if (L < MAX_LEVELS-1)
		{
			#if (P_SHOW_ADVANCE==1)
			std::cout << "Interpolating from level " << L << " to " << L+1 << "..." << std::endl;
			#endif
			tic_simple("");
			S_Interpolate(i_dev, L, V_ADV_TYPE) ;//, N_Pf(0.5)*tau_vec[L+1]/tau_vec[L]);
			cudaDeviceSynchronize();
			tmp[0 + L*4] += toc_simple("",T_US,0);
		}
		
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Colliding and streaming on level " << L << "..." << std::endl;
		#endif
		tic_simple("");
		S_Collide(i_dev, L);
		cudaDeviceSynchronize();
		tmp[1 + L*4] += toc_simple("",T_US,0);
		tic_simple("");
		S_Stream(i_dev, L);
		cudaDeviceSynchronize();
		tmp[2 + L*4] += toc_simple("",T_US,0);
		
		if (L < MAX_LEVELS-1)
		{
			S_Advance(i_dev, L+1, file, tmp);
			
			#if (P_SHOW_ADVANCE==1)
			std::cout << "Averaging from level " << L+1 << " to " << L << "..." << std::endl;
			#endif
			tic_simple("");
			S_Average(i_dev, L, V_ADV_TYPE); //, N_Pf(2.0)*tau_vec[L]/tau_vec[L+1]);
			cudaDeviceSynchronize();
			tmp[3 + 4*L] += toc_simple("",T_US,0);
			
			#if (P_SHOW_ADVANCE==1)
			std::cout << "Interpolating from level " << L << " to " << L+1 << "..." << std::endl;
			#endif
			tic_simple("");
			S_Interpolate(i_dev, L, V_ADV_TYPE); //, N_Pf(0.5)*tau_vec[L+1]/tau_vec[L]);
			cudaDeviceSynchronize();
			tmp[0 + 4*L] += toc_simple("",T_US,0);
		}
		
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Colliding and streaming on level " << L << "..." << std::endl;
		#endif
		tic_simple("");
		S_Collide(i_dev, L);
		cudaDeviceSynchronize();
		tmp[1 + 4*L] += toc_simple("",T_US,0);
		tic_simple("");
		S_Stream(i_dev, L);
		cudaDeviceSynchronize();
		tmp[2 + 4*L] += toc_simple("",T_US,0);
		
		if (L < MAX_LEVELS-1)
		{
			S_Advance(i_dev, L+1, file, tmp);
			
			#if (P_SHOW_ADVANCE==1)
			std::cout << "Averaging from level " << L+1 << " to " << L << "..." << std::endl;
			#endif
			tic_simple("");
			S_Average(i_dev, L, V_ADV_TYPE); //, N_Pf(2.0)*tau_vec[L]/tau_vec[L+1]);
			cudaDeviceSynchronize();
			tmp[3 + 4*L] += toc_simple("",T_US,0);
		}
	}
#else
	tic_simple("");
	S_Collide(i_dev, 0);
	cudaDeviceSynchronize();
	tmp[1] += toc_simple("",T_US,0);
	tic_simple("");
	S_Stream(i_dev, 0);
	cudaDeviceSynchronize();
	tmp[2] += toc_simple("",T_US,0);
	
	for (int Lp = 0; Lp < 4; Lp++)
		*file << tmp[Lp] << " ";
	*file << std::endl;
#endif
	
	return 0;
}
#endif

__global__
void Cu_ComputeRefCriteria_V1
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dxb_L, int L,
	int *cblock_ID_ref, ufloat_t *cblock_f_X
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		// Evaluate only if current cell-block is not refined already.
		if (cblock_ID_ref[i_kap] == V_REF_ID_UNREFINED)
		{
			// Get the coordinates of the block.
			ufloat_t x_k_plus = cblock_f_X[i_kap + 0*n_maxcblocks] + N_Pf(0.5)*dxb_L;
			ufloat_t y_k_plus = cblock_f_X[i_kap + 1*n_maxcblocks] + N_Pf(0.5)*dxb_L;
#if (N_DIM==3)
			ufloat_t z_k_plus = cblock_f_X[i_kap + 2*n_maxcblocks] + N_Pf(0.5)*dxb_L;
#endif
			
			// Loop over cavity walls and identify the closest one.
			// If this closest wall is within a certain threshhold, mark for refinement.
			ufloat_t dist_min = N_Pf(1.0);
			ufloat_t dist_tmp = N_Pf(1.0);
				// xM
			//dist_min = x_k_plus - N_Pf(0.0);
				// xP
			//dist_tmp = N_Pf(1.0) - x_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
#if (N_DIM==2)
				// yM
			//dist_tmp = y_k_plus - N_Pf(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yP
			dist_tmp = N_Pf(1.0) - y_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
#else
				// zM
			//dist_tmp = z_k_plus - N_Pf(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// zP
			dist_tmp = N_Pf(1.0) - z_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
#endif
			
			// Evaluate criterion based on dist_min.
			if ( dist_min <= N_Pf(0.2)/( (ufloat_t)(1<<L) ) )
			//if ( dist_min <= N_Pf(0.2)/( (ufloat_t)(1<<1) ) )
				cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
			
			
			
			
			
			// DEBUG
			//if (L == 0 && x_k_plus > N_Pf(0.3) && x_k_plus < N_Pf(0.7) && y_k_plus > N_Pf(0.3) && y_k_plus < N_Pf(0.7) && z_k_plus > N_Pf(0.3) && z_k_plus < N_Pf(0.8))
			//	cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
			//if (L == 0 && x_k_plus > N_Pf(0.3) && x_k_plus < N_Pf(0.7) && y_k_plus > N_Pf(0.3) && y_k_plus <= N_Pf(0.85))
			//	cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
		}
	}
}

__global__
void Cu_ComputeRefCriteria_V2
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_level, int *cblock_ID_nbr,
	ufloat_t *cells_f_W
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_W[M_CBLOCK];
	__shared__ ufloat_t s_Wmax[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	bool eligible = true;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	s_Wmax[threadIdx.x] = -1;
	if (kap < id_max_curr)
	{
		//int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		if (i_kap_b > -1 && cblock_ID_ref[i_kap_b] != V_REF_ID_INACTIVE)
		//if (i_kap_b > -1)
		{
			// Load local vorticity magnitudes into shared memory.
			s_W[threadIdx.x] = cells_f_W[i_kap_b*M_CBLOCK + threadIdx.x];
			__syncthreads();
			
			// Block reduction for maximum.
			for (int s=blockDim.x/2; s>0; s>>=1)
			{
				if (threadIdx.x < s)
				{
					s_W[threadIdx.x] = max( s_W[threadIdx.x],s_W[threadIdx.x + s] );
				}
				__syncthreads();
			}
			
			// Store maximum.
			if (threadIdx.x == 0)
				s_Wmax[k] = s_W[0];
			__syncthreads();
			
			cells_f_W[i_kap_b*M_CBLOCK + threadIdx.x] = s_W[0];
			__syncthreads();
		}
	}
	__syncthreads();
	
	//if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
	//	cells_f_W[kap] = s_Wmax[threadIdx.x];
	
	// Evaluate criterion.
	if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
	{
		// If vorticity is very large, cap at 1.0 to indicate maximum needed refinement.
		if (s_Wmax[threadIdx.x] > N_Pf(1.0))
			s_Wmax[threadIdx.x] = N_Pf(1.0);
		
		int ref_kap = cblock_ID_ref[kap];
		int level_kap = cblock_level[kap];
		
#if (N_DIM==2)
		int L_desired = MAX_LEVELS-1;
		for (int p = 1; p <= MAX_LEVELS-1; p++)
		{
			if (s_Wmax[threadIdx.x] < 0-p)
				L_desired = (MAX_LEVELS-1)-p;
		}
		
		/*
		int L_desired = 4;
		if (s_Wmax[threadIdx.x] < -1) L_desired = 3;
		if (s_Wmax[threadIdx.x] < -2) L_desired = 2;
		if (s_Wmax[threadIdx.x] < -3) L_desired = 1;
		if (s_Wmax[threadIdx.x] < -4) L_desired = 0;
		*/
#else
		int L_desired = MAX_LEVELS-1;
		for (int p = 1; p <= MAX_LEVELS-1; p++)
		{
			if (s_Wmax[threadIdx.x] < 0-p)
				L_desired = (MAX_LEVELS-1)-p;
		}

		//int L_desired = 2;
		//if (s_Wmax[threadIdx.x] < -2) L_desired = 1;
		//if (s_Wmax[threadIdx.x] < -5) L_desired = 0;
#endif
		
		// Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
		for (int p = 0; p < l_dq_max; p++)
		{
			if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
				eligible = false;
		}
		
		// If cell-block is unrefined but desired level is higher than current, mark for refinement.
		if (eligible && level_kap != MAX_LEVELS-1 && ref_kap == V_REF_ID_UNREFINED && L_desired > level_kap)
			cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
		
		// If cell-block is refined (leaf) but desired level is less than current, mark for coarsening.
		if (ref_kap == V_REF_ID_REFINED && L_desired < level_kap+1)
			cblock_ID_ref[kap] = V_REF_ID_MARK_COARSEN;
	}
}

int Solver_LBM::S_ComputeRefCriteria(int i_dev, int L, int var)
{
	if (var == 0)
	{
		if (mesh->n_ids[i_dev][L] > 0)
		{
			Cu_ComputeRefCriteria_V1<<<(M_BLOCK+mesh->n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,mesh->streams[i_dev]>>>(
				mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcblocks, mesh->dxf_vec[L]*Nbx, L,
				mesh->c_cblock_ID_ref[i_dev], mesh->c_cblock_f_X[i_dev]
			);
		}
	}
	if (var == 1)
	{
		Cu_ComputeRefCriteria_V2<<<(M_CBLOCK+mesh->id_max[i_dev][MAX_LEVELS]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->id_max[i_dev][MAX_LEVELS], mesh->n_maxcblocks,
			mesh->c_cblock_ID_ref[i_dev], mesh->c_cblock_level[i_dev], mesh->c_cblock_ID_nbr[i_dev],
			mesh->c_cells_f_W[i_dev]
		);
	}
	
	return 0;
}

__global__
void Cu_EvaluateRefCritera
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_level, int *cblock_ID_nbr,
	ufloat_t *cells_f_W,
	int var
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	bool eligible = true;
	ufloat_t s_Wmax = cells_f_W[kap];
	
	if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
	{
		// If vorticity is very large, cap at 1.0 to indicate maximum needed refinement.
		if (s_Wmax > N_Pf(1.0))
			s_Wmax = N_Pf(1.0);
		
		// Compute polynomial to get the desired level.
		int ref_kap = cblock_ID_ref[kap];
		int level_kap = cblock_level[kap];
		
		int L_desired = 2;
		if (s_Wmax < -1) L_desired = 1;
		if (s_Wmax < -2) L_desired = 0;
		
		// DEBUG
		//if (L_desired > 1)
		//	printf("vM: %f, poly: %f, L_d: %i, L_c: %i\n", s_Wmax[threadIdx.x], poly, L_desired, cblock_level[kap]);
		
		// Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
		for (int p = 0; p < l_dq_max; p++)
		{
			if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
				eligible = false;
		}
		
		// If cell-block is unrefined but desired level is higher than current, mark for refinement.
		if (var == 0 && eligible && level_kap != MAX_LEVELS-1 && ref_kap == V_REF_ID_UNREFINED && L_desired > level_kap)
		{
			cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
		}
		
		// If cell-block is refined (leaf) but desired level is less than current, mark for coarsening.
		if (var == 1 && ref_kap == V_REF_ID_REFINED && L_desired < level_kap)
		{
			cblock_ID_ref[kap] = V_REF_ID_MARK_COARSEN;
		}
	}
}

int Solver_LBM::S_EvaluateRefCriteria(int i_dev, int var)
{
	Cu_EvaluateRefCritera<<<(M_CBLOCK+mesh->id_max[i_dev][MAX_LEVELS]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
		mesh->id_max[i_dev][MAX_LEVELS], mesh->n_maxcblocks,
		mesh->c_cblock_ID_ref[i_dev], mesh->c_cblock_level[i_dev], mesh->c_cblock_ID_nbr[i_dev],
		mesh->c_cells_f_W[i_dev],
		var
	);
	
	return 0;
}



/*
         8888888          d8b 888    d8b          888          
           888            Y8P 888    Y8P          888          
           888                888                 888          
           888   88888b.  888 888888 888  8888b.  888          
           888   888 "88b 888 888    888     "88b 888          
           888   888  888 888 888    888 .d888888 888          
           888   888  888 888 Y88b.  888 888  888 888          
88888888 8888888 888  888 888  "Y888 888 "Y888888 888 88888888 
*/



__global__
void Cu_SetInitialConditions_Iter
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, ufloat_t dx_L, ufloat_t tau_L,
	ufloat_t *cells_f_F, ufloat_t *cells_f_Fs,
	ufloat_t rho_t0, ufloat_t u_t0, ufloat_t v_t0, ufloat_t w_t0
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Compute factors.
	ufloat_t omega = dx_L / (dx_L); // dx_L / tau_L
	ufloat_t omegam1 = N_Pf(1.0) - omega;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		ufloat_t udotu = u_t0*u_t0 + v_t0*v_t0 + w_t0*w_t0;
		ufloat_t cdotu = N_Pf(0.0);
		if (i_kap_b > -1)
		{
#if (l_dq==9)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8;
			ufloat_t u_kap = u_t0;
			ufloat_t v_kap = v_t0;
			udotu = u_kap*u_kap + v_kap*v_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0*omegam1 + ( N_Pf(0.444444444444444)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1*omegam1 + ( N_Pf(0.111111111111111)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2*omegam1 + ( N_Pf(0.111111111111111)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3*omegam1 + ( N_Pf(0.111111111111111)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4*omegam1 + ( N_Pf(0.111111111111111)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
#endif
#if (l_dq==19)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18;
			ufloat_t u_kap = u_t0;
			ufloat_t v_kap = v_t0;
			ufloat_t w_kap = w_t0;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0*omegam1 + ( N_Pf(0.333333333333333)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1*omegam1 + ( N_Pf(0.055555555555556)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2*omegam1 + ( N_Pf(0.055555555555556)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3*omegam1 + ( N_Pf(0.055555555555556)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4*omegam1 + ( N_Pf(0.055555555555556)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5*omegam1 + ( N_Pf(0.055555555555556)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6*omegam1 + ( N_Pf(0.055555555555556)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_9*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_10*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_11*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_12*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_13*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_14*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_15*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_16*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_17*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_18*omegam1 + ( N_Pf(0.027777777777778)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
#endif
#if (l_dq==27)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			ufloat_t f_19 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells];
			ufloat_t f_20 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells];
			ufloat_t f_21 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells];
			ufloat_t f_22 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells];
			ufloat_t f_23 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells];
			ufloat_t f_24 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells];
			ufloat_t f_25 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells];
			ufloat_t f_26 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18 + f_19 + f_20 + f_21 + f_22 + f_23 + f_24 + f_25 + f_26;
			ufloat_t u_kap = u_t0;
			ufloat_t v_kap = v_t0;
			ufloat_t w_kap = w_t0;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0*omegam1 + ( N_Pf(0.296296296296296)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1*omegam1 + ( N_Pf(0.074074074074074)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2*omegam1 + ( N_Pf(0.074074074074074)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3*omegam1 + ( N_Pf(0.074074074074074)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4*omegam1 + ( N_Pf(0.074074074074074)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5*omegam1 + ( N_Pf(0.074074074074074)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6*omegam1 + ( N_Pf(0.074074074074074)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_9*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_10*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_11*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_12*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_13*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_14*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_15*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_16*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_17*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_18*omegam1 + ( N_Pf(0.018518518518519)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = f_19*omegam1 + ( N_Pf(0.004629629629630)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = f_20*omegam1 + ( N_Pf(0.004629629629630)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = f_21*omegam1 + ( N_Pf(0.004629629629630)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = f_22*omegam1 + ( N_Pf(0.004629629629630)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = f_23*omegam1 + ( N_Pf(0.004629629629630)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = f_24*omegam1 + ( N_Pf(0.004629629629630)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = -u_kap +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = f_25*omegam1 + ( N_Pf(0.004629629629630)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
			cdotu = +u_kap -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = f_26*omegam1 + ( N_Pf(0.004629629629630)*(rho_kap + rho_t0*(N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) )*omega;
#endif
		}
	}
}

__global__
void Cu_SetInitialConditions_Naive(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells,
	ufloat_t *cells_f_F, ufloat_t *cells_f_Fs,
	ufloat_t rho_t0, ufloat_t u_t0, ufloat_t v_t0, ufloat_t w_t0
)
{
	__shared__ int s_ID_cblock[M_CBLOCK]; 
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		ufloat_t udotu = u_t0*u_t0 + v_t0*v_t0 + w_t0*w_t0;
		ufloat_t cdotu = N_Pf(0.0);
		if (i_kap_b > -1)
		{
#if (l_dq==9)
			cdotu = N_Pf(0.0);   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.444444444444444)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.111111111111111)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.111111111111111)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.111111111111111)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.111111111111111)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#elif (l_dq==19)
			cdotu = N_Pf(0.0);   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.333333333333333)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#else
			cdotu = N_Pf(0.0);   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.296296296296296)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 -v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 +v_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 +v_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 -v_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 +v_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 -v_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 -v_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 +v_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0 +v_t0 +w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0 -v_t0 -w_t0 ;   cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#endif
		}
	}
}

int Solver_LBM::S_SetInitialConditions(int i_dev, int L, bool init)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		if (init)
		{
			Cu_SetInitialConditions_Naive<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
				mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells,
				mesh->c_cells_f_F[i_dev], mesh->c_cells_f_Fs[i_dev],
				N_Pf(1.0), N_Pf(0.0), N_Pf(0.0), N_Pf(0.0)
			);
		}
		else
		{
			Cu_SetInitialConditions_Iter<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
				mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells, dx_vec[L], tau_vec[L],
				mesh->c_cells_f_F[i_dev], mesh->c_cells_f_Fs[i_dev],
				N_Pf(1.0), N_Pf(0.0), N_Pf(0.0), N_Pf(0.0)
			);
		}
	}
	
	return 0;
}



/*
         .d8888b.                                           888                    
        d88P  Y88b                                          888                    
        888    888                                          888                    
        888         .d88b.  88888b.d88b.  88888b.  888  888 888888 .d88b.          
        888        d88""88b 888 "888 "88b 888 "88b 888  888 888   d8P  Y8b         
        888    888 888  888 888  888  888 888  888 888  888 888   88888888         
        Y88b  d88P Y88..88P 888  888  888 888 d88P Y88b 888 Y88b. Y8b.             
88888888 "Y8888P"   "Y88P"  888  888  888 88888P"   "Y88888  "Y888 "Y8888 88888888 
                                          888                                      
                                          888                                      
                                          888                                      
*/



__global__
void Cu_ComputeU(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells,
	ufloat_t *cells_f_F, ufloat_t *cells_f_U
)
{
	__shared__ int s_ID_cblock[M_CBLOCK]; 
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		if (i_kap_b > -1)
		{
#if (l_dq==9)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8;
			ufloat_t u_kap = + f_1 - f_3 + f_5 - f_6 - f_7 + f_8;
			ufloat_t v_kap = + f_2 - f_4 + f_5 + f_6 - f_7 - f_8;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = rho_kap;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = u_kap / rho_kap;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = v_kap / rho_kap;
#elif (l_dq==19)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18;
			ufloat_t u_kap = + f_1 - f_2 + f_7 - f_8 + f_9 - f_10 + f_13 - f_14 + f_15 - f_16;
			ufloat_t v_kap = + f_3 - f_4 + f_7 - f_8 + f_11 - f_12 - f_13 + f_14 + f_17 - f_18;
			ufloat_t w_kap = + f_5 - f_6 + f_9 - f_10 + f_11 - f_12 - f_15 + f_16 - f_17 + f_18;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = rho_kap;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = u_kap / rho_kap;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = v_kap / rho_kap;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = w_kap / rho_kap;
#else
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			ufloat_t f_19 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells];
			ufloat_t f_20 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells];
			ufloat_t f_21 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells];
			ufloat_t f_22 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells];
			ufloat_t f_23 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells];
			ufloat_t f_24 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells];
			ufloat_t f_25 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells];
			ufloat_t f_26 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18 + f_19 + f_20 + f_21 + f_22 + f_23 + f_24 + f_25 + f_26;
			ufloat_t u_kap = + f_1 - f_2 + f_7 - f_8 + f_9 - f_10 + f_13 - f_14 + f_15 - f_16 + f_19 - f_20 + f_21 - f_22 + f_23 - f_24 - f_25 + f_26;
			ufloat_t v_kap = + f_3 - f_4 + f_7 - f_8 + f_11 - f_12 - f_13 + f_14 + f_17 - f_18 + f_19 - f_20 + f_21 - f_22 - f_23 + f_24 + f_25 - f_26;
			ufloat_t w_kap = + f_5 - f_6 + f_9 - f_10 + f_11 - f_12 - f_15 + f_16 - f_17 + f_18 + f_19 - f_20 - f_21 + f_22 + f_23 - f_24 + f_25 - f_26;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = rho_kap;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = u_kap / rho_kap;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = v_kap / rho_kap;
			cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = w_kap / rho_kap;
#endif
			__syncthreads();
		}
	}
}

int Solver_LBM::S_ComputeU(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_ComputeU<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells,
			mesh->c_cells_f_F[i_dev], mesh->c_cells_f_U[i_dev]
		);
	}
		
	return 0;
}

__global__
void Cu_ComputeW(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, ufloat_t dx_L,
	ufloat_t *cells_f_U, ufloat_t *cells_f_W
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_U[M_CBLOCK];
	__shared__ ufloat_t s_V[M_CBLOCK];
	__shared__ ufloat_t s_W[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
#if (N_DIM==3)
	int K_kap = (threadIdx.x / Nbx) / Nbx;
#endif
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	s_U[threadIdx.x] = N_Pf(0.0);
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	ufloat_t dU = N_Pf(0.0);
	ufloat_t dV = N_Pf(0.0);
#if (N_DIM==3)
	ufloat_t dW = N_Pf(0.0);
	ufloat_t vortX = N_Pf(0.0);
	ufloat_t vortY = N_Pf(0.0);
#endif
	ufloat_t vortZ = N_Pf(0.0);
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		if (i_kap_b > -1)
		{
#if (N_DIM==2)
			s_U[threadIdx.x] = cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			s_V[threadIdx.x] = cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			__syncthreads();
			
			if (I_kap < Nbx-1)
				dU = s_U[(I_kap+1)+Nbx*(J_kap)] - s_U[threadIdx.x];
			else
				dU = s_U[threadIdx.x] - s_U[(I_kap-1)+Nbx*(J_kap)];
			
			if (J_kap < Nbx-1)
				dV = s_V[(I_kap+1)+Nbx*(J_kap)] - s_V[threadIdx.x];
			else
				dV = s_V[threadIdx.x] - s_V[(I_kap-1)+Nbx*(J_kap)];
			
			vortZ = (dV-dU)/dx_L;
			cells_f_W[i_kap_b*M_CBLOCK + threadIdx.x] = floor(log2( sqrt(vortZ*vortZ) ));
			__syncthreads();
#else
			s_U[threadIdx.x] = cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			s_V[threadIdx.x] = cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			s_W[threadIdx.x] = cells_f_U[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			__syncthreads();
			
			if (I_kap < Nbx-1)
				dU = s_U[(I_kap+1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap)] - s_U[threadIdx.x];
			else
				dU = s_U[threadIdx.x] - s_U[(I_kap-1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap)];
			
			if (J_kap < Nbx-1)
				dV = s_V[(I_kap)+Nbx*(J_kap+1)+Nbx*Nbx*(K_kap)] - s_V[threadIdx.x];
			else
				dV = s_V[threadIdx.x] - s_V[(I_kap)+Nbx*(J_kap-1)+Nbx*Nbx*(K_kap)];
			
			if (K_kap < Nbx-1)
				dW = s_W[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap+1)] - s_W[threadIdx.x];
			else
				dW = s_W[threadIdx.x] - s_W[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap-1)];
			
			vortX = (dW-dV)/dx_L;
			vortY = (dU-dW)/dx_L;
			vortZ = (dV-dU)/dx_L;
			cells_f_W[i_kap_b*M_CBLOCK + threadIdx.x] = floor(log2( sqrt(vortX*vortX + vortY*vortY + vortZ*vortZ) ));
			__syncthreads();
#endif
		}
	}
}

int Solver_LBM::S_ComputeW(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_ComputeW<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells, mesh->dxf_vec[L],
			mesh->c_cells_f_U[i_dev], mesh->c_cells_f_W[i_dev]
		);
	}
		
	return 0;
}

__global__
void Cu_SetValuesDebug
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells,
	ufloat_t *cells_f_F
)
{
	__shared__ int s_ID_cblock[M_CBLOCK]; 
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		if (i_kap_b > -1)
		{
			#pragma unroll
			for (int p = 0; p < l_dq; p++)
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = (ufloat_t)(1.0/(double)l_dq);
		}
	}
}

int Solver_LBM::S_SetValuesDebug(int i_dev, int L, ufloat_t v)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_SetValuesDebug<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells,
			mesh->c_cells_f_F[i_dev]
		);
	}
		
	return 0;
}



/*
         .d8888b.           888 888 d8b      888                  
        d88P  Y88b          888 888 Y8P      888                  
        888    888          888 888          888                  
        888         .d88b.  888 888 888  .d88888  .d88b.          
        888        d88""88b 888 888 888 d88" 888 d8P  Y8b         
        888    888 888  888 888 888 888 888  888 88888888         
        Y88b  d88P Y88..88P 888 888 888 Y88b 888 Y8b.             
88888888 "Y8888P"   "Y88P"  888 888 888  "Y88888  "Y8888 88888888 
*/



__global__
void Cu_Collide_Naive(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, ufloat_t dx_L, ufloat_t tau_L,
	int *cblock_ID_nbr_child, int *cells_ID_mask,
	ufloat_t *cells_f_F, ufloat_t *cells_f_Fs
)
{
	__shared__ int s_ID_cblock[M_CBLOCK]; 
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Compute factors.
	ufloat_t omega = dx_L / tau_L;
	ufloat_t omegam1 = N_Pf(1.0) - omega;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		ufloat_t udotu = N_Pf(0.0);
		ufloat_t cdotu = N_Pf(0.0);
		
#if (V_ADV_TYPE == V_ADV_TYPE_UNIFORM)
		if (i_kap_b > -1 && cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] == 0)
		{
#else
		if (i_kap_b > -1)// && cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] == 0)
		{
#endif
			
#if (l_dq==9)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8;
			ufloat_t u_kap = (+ f_1 - f_3 + f_5 - f_6 - f_7 + f_8) / rho_kap;
			ufloat_t v_kap = (+ f_2 - f_4 + f_5 + f_6 - f_7 - f_8) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0*omegam1 + ( N_Pf(0.444444444444444)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1*omegam1 + ( N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2*omegam1 + ( N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3*omegam1 + ( N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4*omegam1 + ( N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
#endif
#if (l_dq==19)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18;
			ufloat_t u_kap = (+ f_1 - f_2 + f_7 - f_8 + f_9 - f_10 + f_13 - f_14 + f_15 - f_16) / rho_kap;
			ufloat_t v_kap = (+ f_3 - f_4 + f_7 - f_8 + f_11 - f_12 - f_13 + f_14 + f_17 - f_18) / rho_kap;
			ufloat_t w_kap = (+ f_5 - f_6 + f_9 - f_10 + f_11 - f_12 - f_15 + f_16 - f_17 + f_18) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0*omegam1 + ( N_Pf(0.333333333333333)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1*omegam1 + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2*omegam1 + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3*omegam1 + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4*omegam1 + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5*omegam1 + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6*omegam1 + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_9*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_10*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_11*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_12*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_13*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_14*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_15*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_16*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_17*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_18*omegam1 + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
#endif
#if (l_dq==27)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			ufloat_t f_19 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells];
			ufloat_t f_20 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells];
			ufloat_t f_21 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells];
			ufloat_t f_22 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells];
			ufloat_t f_23 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells];
			ufloat_t f_24 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells];
			ufloat_t f_25 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells];
			ufloat_t f_26 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18 + f_19 + f_20 + f_21 + f_22 + f_23 + f_24 + f_25 + f_26;
			ufloat_t u_kap = (+ f_1 - f_2 + f_7 - f_8 + f_9 - f_10 + f_13 - f_14 + f_15 - f_16 + f_19 - f_20 + f_21 - f_22 + f_23 - f_24 - f_25 + f_26) / rho_kap;
			ufloat_t v_kap = (+ f_3 - f_4 + f_7 - f_8 + f_11 - f_12 - f_13 + f_14 + f_17 - f_18 + f_19 - f_20 + f_21 - f_22 - f_23 + f_24 + f_25 - f_26) / rho_kap;
			ufloat_t w_kap = (+ f_5 - f_6 + f_9 - f_10 + f_11 - f_12 - f_15 + f_16 - f_17 + f_18 + f_19 - f_20 - f_21 + f_22 + f_23 - f_24 + f_25 - f_26) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0*omegam1 + ( N_Pf(0.296296296296296)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1*omegam1 + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2*omegam1 + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3*omegam1 + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4*omegam1 + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5*omegam1 + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6*omegam1 + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_9*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_10*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_11*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_12*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_13*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_14*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_15*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_16*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_17*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_18*omegam1 + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = f_19*omegam1 + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = f_20*omegam1 + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = f_21*omegam1 + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = f_22*omegam1 + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = f_23*omegam1 + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = f_24*omegam1 + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = -u_kap +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = f_25*omegam1 + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
			cdotu = +u_kap -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = f_26*omegam1 + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omega;
#endif
		}
	}
}

int Solver_LBM::S_Collide(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collide_Naive<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells, dx_vec[L], tau_vec[L],
			mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cells_ID_mask[i_dev],
			mesh->c_cells_f_F[i_dev], mesh->c_cells_f_Fs[i_dev]
		);
	}
	
	return 0;
}



/*
         .d8888b.  888                                                    
        d88P  Y88b 888                                                    
        Y88b.      888                                                    
         "Y888b.   888888 888d888 .d88b.   8888b.  88888b.d88b.           
            "Y88b. 888    888P"  d8P  Y8b     "88b 888 "888 "88b          
              "888 888    888    88888888 .d888888 888  888  888          
        Y88b  d88P Y88b.  888    Y8b.     888  888 888  888  888          
88888888 "Y8888P"   "Y888 888     "Y8888  "Y888888 888  888  888 88888888 
*/



__global__
void Cu_Stream_Naive
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, ufloat_t dx_L, ufloat_t tau_L,
	int *cblock_ID_nbr, int n_maxcblocks,
	ufloat_t *cells_f_F, ufloat_t *cells_f_Fs, ufloat_t *cells_f_U
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_F[(Nbx+2)*(Nbx+2)*(Nbx+2)];

	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
#if (N_DIM==3)
	int K_kap = (threadIdx.x / Nbx) / Nbx;
#endif
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	for (int q = 0; q < 4; q++)
	{
		if (threadIdx.x + q*M_CBLOCK < (Nbx+2)*(Nbx+2)*(Nbx+2))
			s_F[threadIdx.x + q*M_CBLOCK] = N_Pf(0.0);
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		int nbr_kap_b = i_kap_b;
		ufloat_t cdotu = N_Pf(0.0);
		if (i_kap_b > -1)
		{		
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x];
__syncthreads();
#if (l_dq==9)
//
// p = 1.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
__syncthreads();
//
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +3*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(0) + (Nbx+2)*(J_kap+1)] -= N_Pf(2.0*0.111111111111111)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +1*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)];
__syncthreads();

//
// p = 2.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
__syncthreads();
//
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +4*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+1) + (Nbx+2)*(0)] -= N_Pf(2.0*0.111111111111111)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +2*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)];
__syncthreads();

//
// p = 3.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
__syncthreads();
//
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +1*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] -= N_Pf(2.0*0.111111111111111)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +3*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)];
__syncthreads();

//
// p = 4.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
__syncthreads();
//
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +2*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] -= N_Pf(2.0*0.111111111111111)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +4*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)];
__syncthreads();

//
// p = 5.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
__syncthreads();
//
	//Get from nbr 7.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == 0) )
	{
		s_F[(0) + (Nbx+2)*(0)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(0) + (Nbx+2)*(0)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap+1 == Nbx) )
	{
		s_F[(0) + (Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
	}
}
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+1) + (Nbx+2)*(0)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
	}
}
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(0) + (Nbx+2)*(J_kap+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)];
__syncthreads();

//
// p = 6.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
__syncthreads();
//
	//Get from nbr 8.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == 0) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(0)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(Nbx+2 - 1) + (Nbx+2)*(0)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap+1 == Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
	}
}
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+1) + (Nbx+2)*(0)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
	}
}
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)];
__syncthreads();

//
// p = 7.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
__syncthreads();
//
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == Nbx-1) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap-1 == -1) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)];
__syncthreads();

//
// p = 8.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
__syncthreads();
//
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == Nbx-1) )
	{
		s_F[(0) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(0) + (Nbx+2)*(Nbx+2 - 1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap-1 == -1) )
	{
		s_F[(0) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(0) + (Nbx+2)*(J_kap+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
		if (nbr_kap_b == -4)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)];
__syncthreads();
#elif (l_dq==19)
//
// p = 1.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
__syncthreads();
//
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +2*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.055555555555556)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +1*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 2.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
__syncthreads();
//
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +1*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.055555555555556)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +2*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 3.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
__syncthreads();
//
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +4*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.055555555555556)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +3*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 4.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
__syncthreads();
//
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +3*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.055555555555556)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +4*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 5.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
__syncthreads();
//
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.055555555555556)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 6.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
__syncthreads();
//
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.055555555555556)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 7.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
__syncthreads();
//
	//Get from nbr 8.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 8.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
__syncthreads();
//
	//Get from nbr 7.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 9.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
__syncthreads();
//
	//Get from nbr 10.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 10.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
__syncthreads();
//
	//Get from nbr 9.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 11.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
__syncthreads();
//
	//Get from nbr 12.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 12.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
__syncthreads();
//
	//Get from nbr 11.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 13.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
__syncthreads();
//
	//Get from nbr 14.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 14.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
__syncthreads();
//
	//Get from nbr 13.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 15.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
__syncthreads();
//
	//Get from nbr 16.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 16.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
__syncthreads();
//
	//Get from nbr 15.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 17.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
__syncthreads();
//
	//Get from nbr 18.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 18.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
__syncthreads();
//
	//Get from nbr 17.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.027777777777778)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();
#else
//
// p = 1.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
__syncthreads();
//
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +2*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.074074074074074)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +1*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 2.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
__syncthreads();
//
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +1*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.074074074074074)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +2*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 3.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
__syncthreads();
//
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +4*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.074074074074074)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +3*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 4.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
__syncthreads();
//
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +3*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.074074074074074)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +4*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 5.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
__syncthreads();
//
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.074074074074074)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 6.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
__syncthreads();
//
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +5*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.074074074074074)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +6*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 7.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
__syncthreads();
//
	//Get from nbr 8.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 8.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
__syncthreads();
//
	//Get from nbr 7.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +7*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +8*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 9.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
__syncthreads();
//
	//Get from nbr 10.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 10.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
__syncthreads();
//
	//Get from nbr 9.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +9*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +10*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 11.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
__syncthreads();
//
	//Get from nbr 12.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 12.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
__syncthreads();
//
	//Get from nbr 11.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +11*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +12*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 13.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
__syncthreads();
//
	//Get from nbr 14.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 14.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
__syncthreads();
//
	//Get from nbr 13.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +13*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +14*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(0)+1)];
__syncthreads();

//
// p = 15.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
__syncthreads();
//
	//Get from nbr 16.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 16.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
__syncthreads();
//
	//Get from nbr 15.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +15*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +16*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(0) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 17.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
__syncthreads();
//
	//Get from nbr 18.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 18.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
__syncthreads();
//
	//Get from nbr 17.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +17*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(0.000000)*u_lid;
			s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.018518518518519)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +18*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[(I_kap+(0) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 19.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells];
__syncthreads();
//
	//Get from nbr 20.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 20*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == 0) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap+1 == Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(0) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 12.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 10.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 8.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 20.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells];
__syncthreads();
//
	//Get from nbr 19.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 19*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == Nbx-1) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap-1 == -1) && (K_kap-1 == -1) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 11.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 9.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 7.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +19*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +20*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 21.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells];
__syncthreads();
//
	//Get from nbr 22.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 22*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == 0) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap+1 == Nbx) && (K_kap-1 == -1) )
	{
		s_F[(0) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 18.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 16.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 8.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 22.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells];
__syncthreads();
//
	//Get from nbr 21.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 21*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == Nbx-1) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap-1 == -1) && (K_kap+1 == Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 17.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 15.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 7.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +21*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +22*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 23.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells];
__syncthreads();
//
	//Get from nbr 24.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 24*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == Nbx-1) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap-1 == -1) && (K_kap+1 == Nbx) )
	{
		s_F[(0) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 17.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 14.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 10.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 24.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells];
__syncthreads();
//
	//Get from nbr 23.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 23*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == 0) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap+1 == Nbx) && (K_kap-1 == -1) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 18.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 13.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 9.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +23*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +24*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();

//
// p = 25.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells];
__syncthreads();
//
	//Get from nbr 26.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 26*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == 0) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap+1 == Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 15.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 13.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 12.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 6.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == 0) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap+1 == Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(0)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 4.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == 0) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap+1 == Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(0) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 1.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == Nbx-1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(1.000000)*u_lid;
			s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap-1 == -1) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[(I_kap+(1) + 1)+(Nbx+2)*(J_kap+(-1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(-1)+1)];
__syncthreads();

//
// p = 26.
//
s_F[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells];
__syncthreads();
//
	//Get from nbr 25.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 25*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == Nbx-1) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap-1 == -1) && (K_kap-1 == -1) )
	{
		s_F[(0) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 16.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 14.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 11.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 5.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap == Nbx-1) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap-1 == -1) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 3.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap == Nbx-1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap>=0 && I_kap<Nbx) && (J_kap-1 == -1) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
	}
}
__syncthreads();
	//Get from nbr 2.
nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
if (nbr_kap_b < 0) // BC
{
	if ( (I_kap == 0) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x +25*n_maxcells];
		if (nbr_kap_b == -6)
		{
			cdotu = N_Pf(-1.000000)*u_lid;
			s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)] -= N_Pf(2.0*0.004629629629630)*N_Pf(1.0)*N_Pf(3.0)*cdotu;
		}
	}
}
else
{
	if ( (I_kap+1 == Nbx) && (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx) )
	{
		s_F[(0) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_Fs[nbr_kap_b*M_CBLOCK + threadIdx.x +26*n_maxcells];
	}
}
__syncthreads();
cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[(I_kap+(-1) + 1)+(Nbx+2)*(J_kap+(1) + 1)+(Nbx+2)*(Nbx+2)*(K_kap+(1)+1)];
__syncthreads();
#endif
		}
	}
}

__global__
void Cu_ComputeEq(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells,
	ufloat_t *cells_f_F, ufloat_t *cells_f_Fs
)
{
	__shared__ int s_ID_cblock[M_CBLOCK]; 
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		ufloat_t udotu = N_Pf(0.0);
		ufloat_t cdotu = N_Pf(0.0);
		if (i_kap_b > -1)
		{
#if (l_dq==9)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8;
			ufloat_t u_kap = (+ f_1 - f_3 + f_5 - f_6 - f_7 + f_8) / rho_kap;
			ufloat_t v_kap = (+ f_2 - f_4 + f_5 + f_6 - f_7 - f_8) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.444444444444444)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
#endif
#if (l_dq==19)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18;
			ufloat_t u_kap = (+ f_1 - f_2 + f_7 - f_8 + f_9 - f_10 + f_13 - f_14 + f_15 - f_16) / rho_kap;
			ufloat_t v_kap = (+ f_3 - f_4 + f_7 - f_8 + f_11 - f_12 - f_13 + f_14 + f_17 - f_18) / rho_kap;
			ufloat_t w_kap = (+ f_5 - f_6 + f_9 - f_10 + f_11 - f_12 - f_15 + f_16 - f_17 + f_18) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.333333333333333)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
#endif
#if (l_dq==27)
			ufloat_t f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			ufloat_t f_19 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells];
			ufloat_t f_20 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells];
			ufloat_t f_21 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells];
			ufloat_t f_22 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells];
			ufloat_t f_23 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells];
			ufloat_t f_24 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells];
			ufloat_t f_25 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells];
			ufloat_t f_26 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells];
			ufloat_t rho_kap = + f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18 + f_19 + f_20 + f_21 + f_22 + f_23 + f_24 + f_25 + f_26;
			ufloat_t u_kap = (+ f_1 - f_2 + f_7 - f_8 + f_9 - f_10 + f_13 - f_14 + f_15 - f_16 + f_19 - f_20 + f_21 - f_22 + f_23 - f_24 - f_25 + f_26) / rho_kap;
			ufloat_t v_kap = (+ f_3 - f_4 + f_7 - f_8 + f_11 - f_12 - f_13 + f_14 + f_17 - f_18 + f_19 - f_20 + f_21 - f_22 - f_23 + f_24 + f_25 - f_26) / rho_kap;
			ufloat_t w_kap = (+ f_5 - f_6 + f_9 - f_10 + f_11 - f_12 - f_15 + f_16 - f_17 + f_18 + f_19 - f_20 - f_21 + f_22 + f_23 - f_24 + f_25 - f_26) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
			
			cdotu = N_Pf(0.0);   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.296296296296296)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap -v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap +v_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap -v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap +v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = -u_kap +v_kap +w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
			cdotu = +u_kap -v_kap -w_kap ;   cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) ;
#endif
		}
	}
}

int Solver_LBM::S_Stream(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Stream_Naive<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells, dx_vec[L], tau_vec[L],
			mesh->c_cblock_ID_nbr[i_dev], mesh->n_maxcblocks,
			mesh->c_cells_f_F[i_dev], mesh->c_cells_f_Fs[i_dev], mesh->c_cells_f_U[i_dev]
		);
		
#if (V_ADV_TYPE == V_ADV_TYPE_CUBIC)
		Cu_ComputeEq<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells,
			mesh->c_cells_f_F[i_dev], mesh->c_cells_f_Fs[i_dev]
		);
#endif
	}

	return 0;
}

int Solver_LBM::S_ComputeEq(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_ComputeEq<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells,
			mesh->c_cells_f_F[i_dev], mesh->c_cells_f_Fs[i_dev]
		);
	}
	
	return 0;
}
