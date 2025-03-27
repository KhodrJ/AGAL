/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

#if (N_Q==27)

template <int ave_type=1>
__global__
void Cu_Average_MRT_d3q27
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, ufloat_t tau_L, ufloat_t tau_ratio, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_ID_mask_child[M_TBLOCK];
	__shared__ ufloat_t s_Fc[M_TBLOCK];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	int K_kap = (threadIdx.x / Nbx) / Nbx;
	int xc = -1;
	int i_kap_b = -1;
	int i_kap_bc = -1;
	int child0_IJK = 2*((threadIdx.x % Nbx)%2) + Nbx*(2*(((threadIdx.x / Nbx) % Nbx)%2)) + Nbx*Nbx*(2*(((threadIdx.x / Nbx) / Nbx)%2));
	int block_on_interface = -1;
	ufloat_t m_0 = N_Pf(0.0);
	ufloat_t m_1 = N_Pf(0.0);
	ufloat_t m_2 = N_Pf(0.0);
	ufloat_t m_3 = N_Pf(0.0);
	ufloat_t m_4 = N_Pf(0.0);
	ufloat_t m_5 = N_Pf(0.0);
	ufloat_t m_6 = N_Pf(0.0);
	ufloat_t m_7 = N_Pf(0.0);
	ufloat_t m_8 = N_Pf(0.0);
	ufloat_t m_9 = N_Pf(0.0);
	ufloat_t m_10 = N_Pf(0.0);
	ufloat_t m_11 = N_Pf(0.0);
	ufloat_t m_12 = N_Pf(0.0);
	ufloat_t m_13 = N_Pf(0.0);
	ufloat_t m_14 = N_Pf(0.0);
	ufloat_t m_15 = N_Pf(0.0);
	ufloat_t m_16 = N_Pf(0.0);
	ufloat_t m_17 = N_Pf(0.0);
	ufloat_t m_18 = N_Pf(0.0);
	ufloat_t m_19 = N_Pf(0.0);
	ufloat_t m_20 = N_Pf(0.0);
	ufloat_t m_21 = N_Pf(0.0);
	ufloat_t m_22 = N_Pf(0.0);
	ufloat_t m_23 = N_Pf(0.0);
	ufloat_t m_24 = N_Pf(0.0);
	ufloat_t m_25 = N_Pf(0.0);
	ufloat_t m_26 = N_Pf(0.0);
	ufloat_t tmp_i = N_Pf(0.0);
	ufloat_t rho_kap = N_Pf(0.0);
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
	ufloat_t w_kap = N_Pf(0.0);
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
			block_on_interface=cblock_ID_mask[i_kap_b];
		}

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && ((i_kap_bc>-1)and((ave_type==2)or(block_on_interface==1))))
		{
			for (int xc_k = 0; xc_k < 2; xc_k += 1)
			{
				for (int xc_j = 0; xc_j < 2; xc_j += 1)
				{
					for (int xc_i = 0; xc_i < 2; xc_i += 1)
					{
						xc = xc_i + 2*xc_j + 4*xc_k;

						// Load DDFs and compute macroscopic properties.
						m_0 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 0*n_maxcells];
						m_1 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 2*n_maxcells];
						m_2 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 1*n_maxcells];
						m_3 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 4*n_maxcells];
						m_4 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 3*n_maxcells];
						m_5 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 6*n_maxcells];
						m_6 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 5*n_maxcells];
						m_7 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 8*n_maxcells];
						m_8 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 7*n_maxcells];
						m_9 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 10*n_maxcells];
						m_10 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 9*n_maxcells];
						m_11 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 12*n_maxcells];
						m_12 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 11*n_maxcells];
						m_13 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 14*n_maxcells];
						m_14 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 13*n_maxcells];
						m_15 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 16*n_maxcells];
						m_16 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 15*n_maxcells];
						m_17 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 18*n_maxcells];
						m_18 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 17*n_maxcells];
						m_19 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 20*n_maxcells];
						m_20 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 19*n_maxcells];
						m_21 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 22*n_maxcells];
						m_22 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 21*n_maxcells];
						m_23 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 24*n_maxcells];
						m_24 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 23*n_maxcells];
						m_25 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 26*n_maxcells];
						m_26 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 25*n_maxcells];
						rho_kap = m_0;
						u_kap = m_3 / m_0;
						v_kap = m_5 / m_0;
						w_kap = m_7 / m_0;
						udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;

						// Average rescaled fi to parent if applicable.
						s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x];
						if ((ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2))
						{
							s_ID_mask_child[threadIdx.x] = 1;
						}
						__syncthreads();

						//	 p = 0
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_0 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 0*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 1
						tmp_i = (-N_Pf(11.0)*rho_kap+N_Pf(19.0)*rho*(u_kap*u_kap+v_kap*v_kap*w_kap*w_kap));
						s_Fc[threadIdx.x] = tmp_i + (m_1 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 2*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 2
						tmp_i = (N_Pf(3.0)*rho_kap-N_Pf(5.5)*rho_kap*(u_kap*u_kap+v_kap*v_kap+w_kap*w_kap));
						s_Fc[threadIdx.x] = tmp_i + (m_2 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 1*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 3
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_3 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 4*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 4
						tmp_i = (-N_Pf(0.666666666666667)*rho_kap*u_kap);
						s_Fc[threadIdx.x] = tmp_i + (m_4 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 3*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 5
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_5 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 6*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 6
						tmp_i = (-N_Pf(0.666666666666667)*rho_kap*v_kap);
						s_Fc[threadIdx.x] = tmp_i + (m_6 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 5*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 7
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_7 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 8*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 8
						tmp_i = (-N_Pf(0.666666666666667)*rho_kap*w_kap);
						s_Fc[threadIdx.x] = tmp_i + (m_8 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 7*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 9
						tmp_i = (N_Pf(2.0)*rho_kap*u_kap*u_kap;
						s_Fc[threadIdx.x] = tmp_i + (m_9 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 10*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 10
						tmp_i = -;
						s_Fc[threadIdx.x] = tmp_i + (m_10 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 9*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 11
						tmp_i = rho_kap*(v_kap*v_kap+w_kap*w_kap));
						s_Fc[threadIdx.x] = tmp_i + (m_11 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 12*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 12
						tmp_i = (-N_Pf(0.5)*(N_Pf(2.0)*rho_kap*u_kap*u_kap-rho_kap*(v_kap*v_kap+w_kap*w_kap)));
						s_Fc[threadIdx.x] = tmp_i + (m_12 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 11*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 13
						tmp_i = (rho_kap*(v_kap*v_kap-w_kap*w_kap));
						s_Fc[threadIdx.x] = tmp_i + (m_13 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 14*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 14
						tmp_i = (-N_Pf(0.5)*rho_kap*(v_kap*v_kap-w_kap*w_kap));
						s_Fc[threadIdx.x] = tmp_i + (m_14 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 13*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 15
						tmp_i = (rho_kap*u_kap*v_kap);
						s_Fc[threadIdx.x] = tmp_i + (m_15 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 16*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 16
						tmp_i = (rho_kap*v_kap*w_kap);
						s_Fc[threadIdx.x] = tmp_i + (m_16 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 15*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 17
						tmp_i = (rho_kap*u_kap*w_kap);
						s_Fc[threadIdx.x] = tmp_i + (m_17 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 18*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 18
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_18 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 17*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 19
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_19 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 20*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 20
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_20 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 19*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 21
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_21 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 22*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 22
						tmp_i = (-N_Pf(11.0)*rho_kap+N_Pf(19.0)*rho*(u_kap*u_kap+v_kap*v_kap*w_kap*w_kap));
						s_Fc[threadIdx.x] = tmp_i + (m_22 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 21*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 23
						tmp_i = (N_Pf(3.0)*rho_kap-N_Pf(5.5)*rho_kap*(u_kap*u_kap+v_kap*v_kap+w_kap*w_kap));
						s_Fc[threadIdx.x] = tmp_i + (m_23 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 24*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 24
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_24 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 23*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 25
						tmp_i = (-N_Pf(0.666666666666667)*rho_kap*u_kap);
						s_Fc[threadIdx.x] = tmp_i + (m_25 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 26*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();

						//	 p = 26
						tmp_i = 0;
						s_Fc[threadIdx.x] = tmp_i + (m_26 - tmp_i)*tau_ratio;
						__syncthreads();
						if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j)and(K_kap >= 2*xc_k)and(K_kap < 2+2*xc_k))
						{
							cells_f_F[i_kap_b*M_CBLOCK  + threadIdx.x + 25*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
						__syncthreads();
					}
				}
			}
		}
	}
}

int Solver_LBM::S_Average_MRT_d3q27(int i_dev, int L, int var, ufloat_t tau_L, ufloat_t tau_ratio_L)
{
	if (mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_INTERFACE)
	{
		Cu_Average_MRT_d3q27<0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_BLOCK)
	{
		Cu_Average_MRT_d3q27<1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_GRID)
	{
		Cu_Average_MRT_d3q27<2><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

#endif