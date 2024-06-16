#include "solver.h"

#if (N_Q==19)

__global__
void Cu_Stream_Inpl_d3q19
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, ufloat_t dx_L, ufloat_t tau_L,
	int *cblock_ID_onb, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int n_maxcblocks,
	ufloat_t *cells_f_F
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_F_p[(Nbx+2)*(Nbx+2)*(Nbx+2)];
	__shared__ ufloat_t s_F_pb[(Nbx+2)*(Nbx+2)*(Nbx+2)];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int block_on_boundary = 0;
	int i_kap = -1;
	int i_kap_bc = -1;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	int K_kap = (threadIdx.x / Nbx) / Nbx;

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
			int nbr_id_3 = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
			int nbr_id_5 = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
			int nbr_id_7 = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
			int nbr_id_9 = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
			int nbr_id_11 = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];

			// 
			// 
			// 
			// p = 1, 2
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
				// nbr 1 (p = 2)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_1;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)];
			}
			if (s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)];
			}
				// Writing p = 1 to nbr 1 in slot p = 2
			nbr_kap_b = nbr_id_1;
			if ((nbr_kap_b>=0) && (I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 3, 4
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
				// nbr 3 (p = 4)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_3;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)];
			}
			if (s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)];
			}
				// Writing p = 3 to nbr 3 in slot p = 4
			nbr_kap_b = nbr_id_3;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 5, 6
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
				// nbr 5 (p = 6)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_5;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)];
			}
			if (s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)];
			}
				// Writing p = 5 to nbr 5 in slot p = 6
			nbr_kap_b = nbr_id_5;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
			{
				if (s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 7, 8
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
				// nbr 7 (p = 8)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_7;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
					}
				}
			}
				// nbr 3 (p = 8)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_3;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
					}
				}
			}
				// nbr 1 (p = 8)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_1;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)];
			}
			if (s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)];
			}
				// Writing p = 7 to nbr 7 in slot p = 8
			nbr_kap_b = nbr_id_7;
			if ((nbr_kap_b>=0) && (I_kap == 0) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)];
				}
			}
				// Writing p = 7 to nbr 3 in slot p = 8
			nbr_kap_b = nbr_id_3;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F_p[(I_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)];
				}
			}
				// Writing p = 7 to nbr 1 in slot p = 8
			nbr_kap_b = nbr_id_1;
			if ((nbr_kap_b>=0) && (I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 9, 10
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
				// nbr 9 (p = 10)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_9;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
					}
				}
			}
				// nbr 5 (p = 10)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_5;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
					}
				}
			}
				// nbr 1 (p = 10)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_1;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)];
			}
			if (s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)];
			}
				// Writing p = 9 to nbr 9 in slot p = 10
			nbr_kap_b = nbr_id_9;
			if ((nbr_kap_b>=0) && (I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 9 to nbr 5 in slot p = 10
			nbr_kap_b = nbr_id_5;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
			{
				if (s_F_p[(I_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F_p[(I_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 9 to nbr 1 in slot p = 10
			nbr_kap_b = nbr_id_1;
			if ((nbr_kap_b>=0) && (I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 11, 12
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
				// nbr 11 (p = 12)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_11;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap == 0))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
					}
				}
			}
				// nbr 5 (p = 12)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_5;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
					}
				}
			}
				// nbr 3 (p = 12)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_3;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)];
			}
			if (s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)];
			}
				// Writing p = 11 to nbr 11 in slot p = 12
			nbr_kap_b = nbr_id_11;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap == 0))
			{
				if (s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 11 to nbr 5 in slot p = 12
			nbr_kap_b = nbr_id_5;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
			{
				if (s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 11 to nbr 3 in slot p = 12
			nbr_kap_b = nbr_id_3;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 13, 14
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
				// nbr 3 (p = 13)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_3;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_p[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
					}
				}
			}
				// nbr 1 (p = 14)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_1;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)];
			}
			if (s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + 1)];
			}
				// Writing p = 14 to nbr 3 in slot pb = 13
			nbr_kap_b = nbr_id_3;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_pb[(I_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F_pb[(I_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)];
				}
			}
				// Writing p = 13 to nbr 1 in slot p = 14
			nbr_kap_b = nbr_id_1;
			if ((nbr_kap_b>=0) && (I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (0) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 15, 16
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
				// nbr 5 (p = 15)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_5;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
					{
						s_F_p[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
					}
				}
			}
				// nbr 1 (p = 16)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_1;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(Nbx+2 - 1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F_pb[(I_kap + (1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)];
			}
			if (s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F_p[(I_kap + (-1) + 1) + (Nbx+2)*(J_kap + (0) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)];
			}
				// Writing p = 16 to nbr 5 in slot pb = 15
			nbr_kap_b = nbr_id_5;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
			{
				if (s_F_pb[(I_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F_pb[(I_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 15 to nbr 1 in slot p = 16
			nbr_kap_b = nbr_id_1;
			if ((nbr_kap_b>=0) && (I_kap == 0) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F_p[(I_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(J_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

			// 
			// 
			// 
			// p = 17, 18
			// 
			// 
			// 
			for (int q = 0; q < 4; q++)
			{
				if (threadIdx.x + q*64 < 216)
				{
					s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
					s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
				}
			}
			__syncthreads();
			s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
				// nbr 5 (p = 17)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_5;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
					{
						s_F_p[(I_kap+1) + (Nbx+2)*(J_kap+1) + (Nbx+2)*(Nbx+2)*(Nbx+2 - 1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
					}
				}
			}
				// nbr 3 (p = 18)
				// This nbr participates in a regular streaming exchange.
			{
				nbr_kap_b = nbr_id_3;
				if (nbr_kap_b >= 0)
				{
					if ((I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
					{
						s_F_pb[(I_kap+1) + (Nbx+2)*(Nbx+2 - 1) + (Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
					}
				}
			}
			__syncthreads();
				// Main writes within current block.
			if (s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F_pb[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + 1)];
			}
			if (s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)] >= 0)
			{
				cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F_p[(I_kap + (0) + 1) + (Nbx+2)*(J_kap + (-1) + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + 1)];
			}
				// Writing p = 18 to nbr 5 in slot pb = 17
			nbr_kap_b = nbr_id_5;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap>=0 && J_kap<Nbx) && (K_kap == 0))
			{
				if (s_F_pb[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F_pb[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (1) + (0)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (-1) + (1)*Nbx + 1)];
				}
			}
				// Writing p = 17 to nbr 3 in slot p = 18
			nbr_kap_b = nbr_id_3;
			if ((nbr_kap_b>=0) && (I_kap>=0 && I_kap<Nbx) &&  (J_kap == 0) && (K_kap>=0 && K_kap<Nbx))
			{
				if (s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + (0)*Nbx + 1)] >= 0)
				{
					cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F_p[(I_kap + (0) + (0)*Nbx + 1) + (Nbx+2)*(J_kap + (-1) + (1)*Nbx + 1) + (Nbx+2)*(Nbx+2)*(K_kap + (1) + (0)*Nbx + 1)];
				}
			}
			__syncthreads();

		}
	}
}


int Solver_LBM::S_Stream_Inpl_d3q19(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Stream_Inpl_d3q19<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>
		(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells, dx_vec[L], tau_vec[L],
			mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->n_maxcblocks,
			mesh->c_cells_f_F[i_dev]
		);
	}

	return 0;
}

#endif
