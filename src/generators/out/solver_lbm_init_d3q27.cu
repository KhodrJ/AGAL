#include "solver.h"

#if (N_Q==27)

__global__
void Cu_SetInitialConditions_d3q27
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells,
	ufloat_t *cells_f_F,
	ufloat_t rho_t0, ufloat_t u_t0, ufloat_t v_t0, ufloat_t w_t0
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int i_kap = -1;
	ufloat_t cdotu = N_Pf(0.0);
	ufloat_t udotu = u_t0*u_t0 + v_t0*v_t0 + w_t0*w_t0;

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

		if ( i_kap_b > -1 )
		{
			// Compute IC.
			cdotu = N_Pf(0.0);
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.296296296296296)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0+v_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0-v_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0+w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0-w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0+w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0-w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0-v_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0+v_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0-w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0+w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +v_t0-w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -v_t0+w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0+v_t0+w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0-v_t0-w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0+v_t0-w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0-v_t0+w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0-v_t0+w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0+v_t0-w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = -u_t0+v_t0+w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			cdotu = +u_t0-v_t0-w_t0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);

		}
	}
}


int Solver_LBM::S_SetInitialConditions_d3q27(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_SetInitialConditions_d3q27<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>
		(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells,
			mesh->c_cells_f_F[i_dev],
			N_Pf(1.000000000000000), N_Pf(0.050000000000000), N_Pf(0.000000000000000), N_Pf(0.000000000000000)
		);
	}

	return 0;
}

#endif
