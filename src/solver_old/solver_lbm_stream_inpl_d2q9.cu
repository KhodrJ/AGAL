/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

#if (N_Q==9)

__global__
void Cu_Stream_Inpl_d2q9
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ ufloat_t s_F_p[(Nbx+2)*(Nbx+2)];
	__shared__ ufloat_t s_F_pb[(Nbx+2)*(Nbx+2)];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	int i_kap_b = -1;
	int i_kap_bc = -1;
	int i_Q = -1;
	int nbr_i_q = -1;
	int nbr_j_q = -1;
	int nbr_kap_b = -1;
	int nbr_Q_b = -1;
	int nbr_id_global_1 = -1;
	int nbr_id_global_2 = -1;
	int nbr_id_global_5 = -1;
	int nbr_id_1 = -1;
	int nbr_id_2 = -1;
	int nbr_id_5 = -1;
	int nbr_Q_1 = -1;
	int nbr_Q_2 = -1;
	int nbr_Q_5 = -1;
	int block_on_boundary = -1;
	size_t index;
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;

	s_ID_cblock[threadIdx.x] = -1;
	if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
	{
		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
	}
	__syncthreads();

	// Loop over block Ids.
	for (int k = 0; k < M_LBLOCK; k += 1)
	{
		i_kap_b = s_ID_cblock[k];

		// This part is included if n>0 only.
		if (i_kap_b > -1)
		{
			i_kap_bc=cblock_ID_nbr_child[i_kap_b];
			block_on_boundary=cblock_ID_mask[i_kap_b];
		}

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && ((i_kap_bc<0)||(block_on_boundary==1)))
		{
			nbr_id_global_1 = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
			nbr_id_global_2 = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
			nbr_id_global_5 = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
			for (int j_q = 0; j_q < Nqx; j_q += 1)
			{
				for (int i_q = 0; i_q < Nqx; i_q += 1)
				{
					i_Q = i_q + Nqx*j_q;
					nbr_id_1 = i_kap_b;
					nbr_i_q = i_q+1; if (nbr_i_q == Nqx) nbr_i_q = 0; 
					nbr_j_q = j_q+0; if (nbr_j_q == Nqx) nbr_j_q = 0; 
					nbr_Q_1 = nbr_i_q + Nqx*nbr_j_q;
					nbr_id_2 = i_kap_b;
					nbr_i_q = i_q+0; if (nbr_i_q == Nqx) nbr_i_q = 0; 
					nbr_j_q = j_q+1; if (nbr_j_q == Nqx) nbr_j_q = 0; 
					nbr_Q_2 = nbr_i_q + Nqx*nbr_j_q;
					nbr_id_5 = i_kap_b;
					nbr_i_q = i_q+1; if (nbr_i_q == Nqx) nbr_i_q = 0; 
					nbr_j_q = j_q+1; if (nbr_j_q == Nqx) nbr_j_q = 0; 
					nbr_Q_5 = nbr_i_q + Nqx*nbr_j_q;
					// Nbr 1.
					// Consider nbr 1 (100).
					if ((i_q==Nqx-1))
						nbr_id_1 = nbr_id_global_1;
					// Nbr 2.
					// Consider nbr 2 (010).
					if ((j_q==Nqx-1))
						nbr_id_2 = nbr_id_global_2;
					// Nbr 5.
					// Consider nbr 1 (100).
					if ((i_q==Nqx-1)and(j_q<Nqx-1))
						nbr_id_5 = nbr_id_global_1;
					// Consider nbr 2 (010).
					if ((i_q<Nqx-1)and(j_q==Nqx-1))
						nbr_id_5 = nbr_id_global_2;
					// Consider nbr 5 (110).
					if ((i_q==Nqx-1)and(j_q==Nqx-1))
						nbr_id_5 = nbr_id_global_5;

					//
					// DDFs p=1, pb=3.
					//
					for (int q = 0; q < 3; q += 1)
					{
						if (threadIdx.x + q*16 < 36)
						{
							s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
							s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
						}
					}
					__syncthreads();
					index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells;
					s_F_p[(I_kap+1+0)+(J_kap+1+0)*(Nbx+2)] = cells_f_F[index];
					index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells;
					s_F_pb[(I_kap+1+0)+(J_kap+1+0)*(Nbx+2)] = cells_f_F[index];
					//	 Nbr 1 participates in an exchange for p=1.
					nbr_Q_b = nbr_Q_1;
					nbr_kap_b = nbr_id_1;
					if (nbr_kap_b >= 0)
					{
						if ((I_kap==0))
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 3*n_maxcells;
							s_F_pb[(Nbx+1)+(J_kap+1)*(Nbx+2)] = cells_f_F[index];
						}
					}
					__syncthreads();
					// Main writes.
					if (s_F_pb[(I_kap+1+1)+(J_kap+1+0)*(Nbx+2)] >= 0)
					{
						index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells;
						cells_f_F[index] = s_F_pb[(I_kap+1+1)+(J_kap+1+0)*(Nbx+2)];
					}
					if (s_F_p[(I_kap+1+-1)+(J_kap+1+0)*(Nbx+2)] >= 0)
					{
						index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells;
						cells_f_F[index] = s_F_p[(I_kap+1+-1)+(J_kap+1+0)*(Nbx+2)];
					}
					// Neighbor writes.
					//	 Nbr 1 participates in an exchange for p=1.
					nbr_Q_b = nbr_Q_1;
					nbr_kap_b = nbr_id_1;
					if ((nbr_kap_b >= 0)and(I_kap==0))
					{
						if (s_F_p[(I_kap+1+(-1)+(1)*Nbx)+(J_kap+1+(0)+(0)*Nbx)*(Nbx+2)] >= 0)
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 3*n_maxcells;
							cells_f_F[index] = s_F_p[(I_kap+1+(-1)+(1)*Nbx)+(J_kap+1+(0)+(0)*Nbx)*(Nbx+2)];
						}
					}
					__syncthreads();



					//
					// DDFs p=2, pb=4.
					//
					for (int q = 0; q < 3; q += 1)
					{
						if (threadIdx.x + q*16 < 36)
						{
							s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
							s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
						}
					}
					__syncthreads();
					index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells;
					s_F_p[(I_kap+1+0)+(J_kap+1+0)*(Nbx+2)] = cells_f_F[index];
					index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells;
					s_F_pb[(I_kap+1+0)+(J_kap+1+0)*(Nbx+2)] = cells_f_F[index];
					//	 Nbr 2 participates in an exchange for p=2.
					nbr_Q_b = nbr_Q_2;
					nbr_kap_b = nbr_id_2;
					if (nbr_kap_b >= 0)
					{
						if ((J_kap==0))
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 4*n_maxcells;
							s_F_pb[(I_kap+1)+(Nbx+1)*(Nbx+2)] = cells_f_F[index];
						}
					}
					__syncthreads();
					// Main writes.
					if (s_F_pb[(I_kap+1+0)+(J_kap+1+1)*(Nbx+2)] >= 0)
					{
						index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells;
						cells_f_F[index] = s_F_pb[(I_kap+1+0)+(J_kap+1+1)*(Nbx+2)];
					}
					if (s_F_p[(I_kap+1+0)+(J_kap+1+-1)*(Nbx+2)] >= 0)
					{
						index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells;
						cells_f_F[index] = s_F_p[(I_kap+1+0)+(J_kap+1+-1)*(Nbx+2)];
					}
					// Neighbor writes.
					//	 Nbr 2 participates in an exchange for p=2.
					nbr_Q_b = nbr_Q_2;
					nbr_kap_b = nbr_id_2;
					if ((nbr_kap_b >= 0)and(J_kap==0))
					{
						if (s_F_p[(I_kap+1+(0)+(0)*Nbx)+(J_kap+1+(-1)+(1)*Nbx)*(Nbx+2)] >= 0)
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 4*n_maxcells;
							cells_f_F[index] = s_F_p[(I_kap+1+(0)+(0)*Nbx)+(J_kap+1+(-1)+(1)*Nbx)*(Nbx+2)];
						}
					}
					__syncthreads();



					//
					// DDFs p=5, pb=7.
					//
					for (int q = 0; q < 3; q += 1)
					{
						if (threadIdx.x + q*16 < 36)
						{
							s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
							s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
						}
					}
					__syncthreads();
					index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells;
					s_F_p[(I_kap+1+0)+(J_kap+1+0)*(Nbx+2)] = cells_f_F[index];
					index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells;
					s_F_pb[(I_kap+1+0)+(J_kap+1+0)*(Nbx+2)] = cells_f_F[index];
					//	 Nbr 1 participates in an exchange for p=5.
					nbr_Q_b = nbr_Q_1;
					nbr_kap_b = nbr_id_1;
					if (nbr_kap_b >= 0)
					{
						if ((I_kap==0))
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells;
							s_F_pb[(Nbx+1)+(J_kap+1)*(Nbx+2)] = cells_f_F[index];
						}
					}
					//	 Nbr 2 participates in an exchange for p=5.
					nbr_Q_b = nbr_Q_2;
					nbr_kap_b = nbr_id_2;
					if (nbr_kap_b >= 0)
					{
						if ((J_kap==0))
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells;
							s_F_pb[(I_kap+1)+(Nbx+1)*(Nbx+2)] = cells_f_F[index];
						}
					}
					//	 Nbr 5 participates in an exchange for p=5.
					nbr_Q_b = nbr_Q_5;
					nbr_kap_b = nbr_id_5;
					if (nbr_kap_b >= 0)
					{
						if ((I_kap==0)and(J_kap==0))
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells;
							s_F_pb[(Nbx+1)+(Nbx+1)*(Nbx+2)] = cells_f_F[index];
						}
					}
					__syncthreads();
					// Main writes.
					if (s_F_pb[(I_kap+1+1)+(J_kap+1+1)*(Nbx+2)] >= 0)
					{
						index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells;
						cells_f_F[index] = s_F_pb[(I_kap+1+1)+(J_kap+1+1)*(Nbx+2)];
					}
					if (s_F_p[(I_kap+1+-1)+(J_kap+1+-1)*(Nbx+2)] >= 0)
					{
						index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells;
						cells_f_F[index] = s_F_p[(I_kap+1+-1)+(J_kap+1+-1)*(Nbx+2)];
					}
					// Neighbor writes.
					//	 Nbr 1 participates in an exchange for p=5.
					nbr_Q_b = nbr_Q_1;
					nbr_kap_b = nbr_id_1;
					if ((nbr_kap_b >= 0)and(I_kap==0))
					{
						if (s_F_p[(I_kap+1+(-1)+(1)*Nbx)+(J_kap+1+(-1)+(0)*Nbx)*(Nbx+2)] >= 0)
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells;
							cells_f_F[index] = s_F_p[(I_kap+1+(-1)+(1)*Nbx)+(J_kap+1+(-1)+(0)*Nbx)*(Nbx+2)];
						}
					}
					//	 Nbr 2 participates in an exchange for p=5.
					nbr_Q_b = nbr_Q_2;
					nbr_kap_b = nbr_id_2;
					if ((nbr_kap_b >= 0)and(J_kap==0))
					{
						if (s_F_p[(I_kap+1+(-1)+(0)*Nbx)+(J_kap+1+(-1)+(1)*Nbx)*(Nbx+2)] >= 0)
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells;
							cells_f_F[index] = s_F_p[(I_kap+1+(-1)+(0)*Nbx)+(J_kap+1+(-1)+(1)*Nbx)*(Nbx+2)];
						}
					}
					//	 Nbr 5 participates in an exchange for p=5.
					nbr_Q_b = nbr_Q_5;
					nbr_kap_b = nbr_id_5;
					if ((nbr_kap_b >= 0)and(I_kap==0)and(J_kap==0))
					{
						if (s_F_p[(I_kap+1+(-1)+(1)*Nbx)+(J_kap+1+(-1)+(1)*Nbx)*(Nbx+2)] >= 0)
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells;
							cells_f_F[index] = s_F_p[(I_kap+1+(-1)+(1)*Nbx)+(J_kap+1+(-1)+(1)*Nbx)*(Nbx+2)];
						}
					}
					__syncthreads();



					//
					// DDFs p=8, pb=6.
					//
					for (int q = 0; q < 3; q += 1)
					{
						if (threadIdx.x + q*16 < 36)
						{
							s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
							s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
						}
					}
					__syncthreads();
					index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells;
					s_F_p[(I_kap+1+0)+(J_kap+1+0)*(Nbx+2)] = cells_f_F[index];
					index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells;
					s_F_pb[(I_kap+1+0)+(J_kap+1+0)*(Nbx+2)] = cells_f_F[index];
					//	 Nbr 1 participates in an exchange for p=8.
					nbr_Q_b = nbr_Q_1;
					nbr_kap_b = nbr_id_1;
					if (nbr_kap_b >= 0)
					{
						if ((I_kap==0))
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 6*n_maxcells;
							s_F_pb[(Nbx+1)+(J_kap+1)*(Nbx+2)] = cells_f_F[index];
						}
					}
					//	 Nbr 2 participates in an exchange for pb=6.
					nbr_Q_b = nbr_Q_2;
					nbr_kap_b = nbr_id_2;
					if (nbr_kap_b >= 0)
					{
						if ((J_kap==0))
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 8*n_maxcells;
							s_F_p[(I_kap+1)+(Nbx+1)*(Nbx+2)] = cells_f_F[index];
						}
					}
					__syncthreads();
					// Main writes.
					if (s_F_pb[(I_kap+1+1)+(J_kap+1+-1)*(Nbx+2)] >= 0)
					{
						index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells;
						cells_f_F[index] = s_F_pb[(I_kap+1+1)+(J_kap+1+-1)*(Nbx+2)];
					}
					if (s_F_p[(I_kap+1+-1)+(J_kap+1+1)*(Nbx+2)] >= 0)
					{
						index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells;
						cells_f_F[index] = s_F_p[(I_kap+1+-1)+(J_kap+1+1)*(Nbx+2)];
					}
					// Neighbor writes.
					//	 Nbr 1 participates in an exchange for p=8.
					nbr_Q_b = nbr_Q_1;
					nbr_kap_b = nbr_id_1;
					if ((nbr_kap_b >= 0)and(I_kap==0))
					{
						if (s_F_p[(I_kap+1+(-1)+(1)*Nbx)+(J_kap+1+(1)+(0)*Nbx)*(Nbx+2)] >= 0)
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 6*n_maxcells;
							cells_f_F[index] = s_F_p[(I_kap+1+(-1)+(1)*Nbx)+(J_kap+1+(1)+(0)*Nbx)*(Nbx+2)];
						}
					}
					//	 Nbr 2 participates in an exchange for pb=6.
					nbr_Q_b = nbr_Q_2;
					nbr_kap_b = nbr_id_2;
					if ((nbr_kap_b >= 0)and(J_kap==0))
					{
						if (s_F_pb[(I_kap+1+(1)+(0)*Nbx)+(J_kap+1+(-1)+(1)*Nbx)*(Nbx+2)] >= 0)
						{
							index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 8*n_maxcells;
							cells_f_F[index] = s_F_pb[(I_kap+1+(1)+(0)*Nbx)+(J_kap+1+(-1)+(1)*Nbx)*(Nbx+2)];
						}
					}
					__syncthreads();



				}
			}
		}
	}
}

int Solver_LBM::S_Stream_Inpl_d2q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Stream_Inpl_d2q9<<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

#endif