/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

#if (N_Q==27)

__global__
void Cu_SetInitialConditions_d3q27
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb, ufloat_t rho_t0, ufloat_t u_t0, ufloat_t v_t0, ufloat_t w_t0
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	int i_kap_b = -1;
	ufloat_t cdotu = N_Pf(0.0);
	ufloat_t udotu = u_t0*u_t0 + v_t0*v_t0 + w_t0*w_t0;
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

		// Latter condition is added only if n>0.
		if (i_kap_b > -1)
		{
			for (int i_Q = 0; i_Q < N_QUADS; i_Q += 1)
			{
				// Compute IC.
				cdotu = N_Pf(0.0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)0*n_maxcells] = N_Pf(0.296296296296296)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)2*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)1*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = v_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)4*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-v_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)3*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)6*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)5*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0+v_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)8*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0)+(-v_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)7*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0+w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)10*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0)+(-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)9*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = v_t0+w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)12*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-v_t0)+(-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)11*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0+(-v_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)14*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0)+v_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)13*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0+(-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)16*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0)+w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)15*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = v_t0+(-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)18*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-v_t0)+w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)17*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0+v_t0+w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)20*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0)+(-v_t0)+(-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)19*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0+v_t0+(-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)22*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0)+(-v_t0)+w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)21*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0+(-v_t0)+w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)24*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0)+v_t0+(-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)23*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = (-u_t0)+v_t0+w_t0;
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)26*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				cdotu = u_t0+(-v_t0)+(-w_t0);
				cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)25*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			}
		}
	}
}

int Solver_LBM::S_SetInitialConditions_d3q27(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_SetInitialConditions_d3q27<<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], N_Pf(1.0), N_Pf(0.05), N_Pf(0.0), N_Pf(0.0));
	}

	return 0;
}

#endif