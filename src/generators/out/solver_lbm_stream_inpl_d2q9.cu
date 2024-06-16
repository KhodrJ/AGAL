#include "solver.h"

#if (N_Q==9)

__global__
void Cu_Stream_Inpl_d2q9
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, ufloat_t dx_L, ufloat_t tau_L,
	int *cblock_ID_onb, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int n_maxcblocks,
	ufloat_t *cells_f_F
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_F_p[(Nbx+2)*(Nbx+2)];
	__shared__ ufloat_t s_F_pb[(Nbx+2)*(Nbx+2)];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int block_on_boundary = 0;
	int i_kap = -1;
	int i_kap_bc = -1;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;

	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		i_kap = id_set_idev_L[kap];
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();

	// Loop over block Ids.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		int nbr_kap_b = -1;
		i_kap_bc = -1;
		block_on_boundary = 0;

		if (i_kap_b > -1)
		{
			i_kap_bc = cblock_ID_nbr_child[i_kap_b];
			block_on_boundary = cblock_ID_mask[i_kap_b];
		}

		if ( i_kap_b > -1 && (i_kap_bc < 0 || block_on_boundary == 1) )
		{
			// Load required neighbors.
			int nbr_id_1 = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
			int nbr_id_2 = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
			int nbr_id_5 = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];

			// 
			// 
			// 
			// p = 1, 3
			// 
			// 
			// 
			for (int q = 0; q < 3; q++)
			{
				if (threadIdx.x + q*16 < 36)
				{
					s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
				// nbr 1 (p = 3)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_1;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) && (J_kap>=0 && J_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (0) + 1)];
			}
			if (s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (0) + 1)];
			}
				// Writing p = 1 to nbr 1 in slot p = 3
			nbr_kap_b = nbr_id_1;
			if ((nbr_kap_b>=0) && (I_kap == 0) && (J_kap>=0 && J_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 2, 4
			// 
			// 
			// 
			for (int q = 0; q < 3; q++)
			{
				if (threadIdx.x + q*16 < 36)
				{
					s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
				// nbr 2 (p = 4)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_2;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) && (J_kap == 0))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (1) + 1)];
			}
			if (s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (-1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (-1) + 1)];
			}
				// Writing p = 2 to nbr 2 in slot p = 4
			nbr_kap_b = nbr_id_2;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) && (J_kap == 0))
			{
				if (s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 5, 7
			// 
			// 
			// 
			for (int q = 0; q < 3; q++)
			{
				if (threadIdx.x + q*16 < 36)
				{
					s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
				// nbr 5 (p = 7)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_5;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) && (J_kap == 0))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
					}
				}
			}
				// nbr 2 (p = 7)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_2;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) && (J_kap == 0))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
					}
				}
			}
				// nbr 1 (p = 7)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_1;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) && (J_kap>=0 && J_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (1) + 1)];
			}
			if (s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (-1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (-1) + 1)];
			}
				// Writing p = 5 to nbr 5 in slot p = 7
			nbr_kap_b = nbr_id_5;
			if ((nbr_kap_b>=0) && (I_kap == 0) && (J_kap == 0))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 5 to nbr 2 in slot p = 7
			nbr_kap_b = nbr_id_2;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) && (J_kap == 0))
			{
				if (s_F_p[(I_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F_p[(I_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 5 to nbr 1 in slot p = 7
			nbr_kap_b = nbr_id_1;
			if ((nbr_kap_b>=0) && (I_kap == 0) && (J_kap>=0 && J_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 8, 6
			// 
			// 
			// 
			for (int q = 0; q < 3; q++)
			{
				if (threadIdx.x + q*16 < 36)
				{
					s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
				// nbr 2 (p = 8)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_2;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) && (J_kap == 0))
					{
						s_F_p[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
					}
				}
			}
				// nbr 1 (p = 6)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_1;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) && (J_kap>=0 && J_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (-1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (-1) + 1)];
			}
			if (s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (1) + 1)];
			}
				// Writing p = 6 to nbr 2 in slot pb = 8
			nbr_kap_b = nbr_id_2;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) && (J_kap == 0))
			{
				if (s_F_pb[(I_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F_pb[(I_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 8 to nbr 1 in slot p = 6
			nbr_kap_b = nbr_id_1;
			if ((nbr_kap_b>=0) && (I_kap == 0) && (J_kap>=0 && J_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (1) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (1) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

		}
	}
}


int Solver_LBM::S_Stream_Inpl_d2q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Stream_Inpl_d2q9<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>
		(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells, dx_vec[L], tau_vec[L],
			mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->n_maxcblocks,
			mesh->c_cells_f_F[i_dev]
		);
	}

	return 0;
}

#endif
