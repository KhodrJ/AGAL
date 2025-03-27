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
void Cu_EnforceSymmetry_d2q9
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, int *id_set_idev_L, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_onb
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	int I_kap_p = -1;
	int J_kap_p = -1;
	ufloat_t x_kap = N_Pf(0.0);
	ufloat_t y_kap = N_Pf(0.0);
	int i_kap_b = -1;
	int i_kap_bc = -1;
	int nbr_kap_b = -1;
	int block_on_boundary = -1;
	int nbr_id_1 = -1;
	int nbr_id_2 = -1;
	int nbr_id_3 = -1;
	int nbr_id_4 = -1;
	ufloat_t f_p = N_Pf(0.0);
	ufloat_t f_pb = N_Pf(0.0);
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
			block_on_boundary=cblock_ID_onb[i_kap_b];
		}

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && ((i_kap_bc<0)||(block_on_boundary==1)))
		{

			// Impose symmetry / free-slip boundary condition.
			nbr_id_1 = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
			nbr_id_2 = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
			nbr_id_3 = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
			nbr_id_4 = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
			nbr_kap_b = i_kap_b;
			I_kap_p = I_kap+1;
			J_kap_p = J_kap;
			if (I_kap_p==Nbx)
			{
				nbr_kap_b = nbr_id_1;
				I_kap_p = 0;
			}
			if ((nbr_id_2 == N_SYMMETRY)and(nbr_id_1 > -1))
			{
				if (J_kap==Nbx-1)
				{
					f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
					f_pb = cells_f_F[nbr_kap_b*M_CBLOCK + (I_kap_p+Nbx*J_kap_p) + 6*n_maxcells];
					cells_f_F[nbr_kap_b*M_CBLOCK + (I_kap_p+Nbx*J_kap_p) + 6*n_maxcells] = f_p;
					cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_pb;
				}
			}
			if ((nbr_id_4 == N_SYMMETRY)and(nbr_id_1 > -1))
			{
				if (J_kap==0)
				{
					f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
					f_pb = cells_f_F[nbr_kap_b*M_CBLOCK + (I_kap_p+Nbx*J_kap_p) + 7*n_maxcells];
					cells_f_F[nbr_kap_b*M_CBLOCK + (I_kap_p+Nbx*J_kap_p) + 7*n_maxcells] = f_p;
					cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_pb;
				}
			}
			nbr_kap_b = i_kap_b;
			I_kap_p = I_kap;
			J_kap_p = J_kap+1;
			if (J_kap_p==Nbx)
			{
				nbr_kap_b = nbr_id_2;
				J_kap_p = 0;
			}
			if ((nbr_id_1 == N_SYMMETRY)and(nbr_id_2 > -1))
			{
				if (I_kap==Nbx-1)
				{
					f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
					f_pb = cells_f_F[nbr_kap_b*M_CBLOCK + (I_kap_p+Nbx*J_kap_p) + 6*n_maxcells];
					cells_f_F[nbr_kap_b*M_CBLOCK + (I_kap_p+Nbx*J_kap_p) + 6*n_maxcells] = f_p;
					cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_pb;
				}
			}
			if ((nbr_id_3 == N_SYMMETRY)and(nbr_id_2 > -1))
			{
				if (I_kap==0)
				{
					f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
					f_pb = cells_f_F[nbr_kap_b*M_CBLOCK + (I_kap_p+Nbx*J_kap_p) + 7*n_maxcells];
					cells_f_F[nbr_kap_b*M_CBLOCK + (I_kap_p+Nbx*J_kap_p) + 7*n_maxcells] = f_p;
					cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_pb;
				}
			}
		}
	}
}

int Solver_LBM::S_EnforceSymmetry_d2q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_EnforceSymmetry_d2q9<<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

#endif