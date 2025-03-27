/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

#if (N_Q==19)

template <int interp_type=0>
__global__
void Cu_Interpolate_Linear_MRT_d3q19
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, ufloat_t dx_L, ufloat_t tau_L, ufloat_t tau_ratio, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ ufloat_t s_F[M_TBLOCK];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	int K_kap = (threadIdx.x / Nbx) / Nbx;
	int i_kap_b = -1;
	int i_kap_bc = -1;
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
	ufloat_t tmp_i = N_Pf(0.0);
	ufloat_t rho_kap = N_Pf(0.0);
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
	ufloat_t w_kap = N_Pf(0.0);
	ufloat_t cdotu = N_Pf(0.0);
	ufloat_t udotu = N_Pf(0.0);
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
		if (i_kap_b > -1 && (((interp_type==0)and(block_on_interface==1))or((interp_type==1)and(cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE))))
		{
			// Load DDFs and compute macroscopic properties.
			m_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			m_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			m_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			m_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			m_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			m_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			m_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			m_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			m_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			m_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			m_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			m_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			m_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			m_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			m_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			m_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			m_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			m_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			m_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			rho_kap = m_0;
			u_kap = m_3 / m_0;
			v_kap = m_5 / m_0;
			w_kap = m_7 / m_0;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;

			// Interpolate rescaled fi to children if applicable.
			//
			// DDF 0.
			//
			tmp_i = (rho_kap);
			s_F[threadIdx.x] = tmp_i + (m_0 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 1.
			//
			tmp_i = (-N_Pf(11.0)*rho_kap+N_Pf(19.0)*rho*(u_kap*u_kap+v_kap*v_kap*w_kap*w_kap));
			s_F[threadIdx.x] = tmp_i + (m_1 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 2.
			//
			tmp_i = (N_Pf(3.0)*rho_kap-N_Pf(5.5)*rho_kap*(u_kap*u_kap+v_kap*v_kap+w_kap*w_kap));
			s_F[threadIdx.x] = tmp_i + (m_2 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 3.
			//
			tmp_i = (rho_kap*u_kap);
			s_F[threadIdx.x] = tmp_i + (m_3 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 4.
			//
			tmp_i = (-N_Pf(0.666666666666667)*rho_kap*u_kap);
			s_F[threadIdx.x] = tmp_i + (m_4 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 5.
			//
			tmp_i = (rho*v_kap);
			s_F[threadIdx.x] = tmp_i + (m_5 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 6.
			//
			tmp_i = (-N_Pf(0.666666666666667)*rho_kap*v_kap);
			s_F[threadIdx.x] = tmp_i + (m_6 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 7.
			//
			tmp_i = (rho_kap*w_kap);
			s_F[threadIdx.x] = tmp_i + (m_7 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 8.
			//
			tmp_i = (-N_Pf(0.666666666666667)*rho_kap*w_kap);
			s_F[threadIdx.x] = tmp_i + (m_8 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 9.
			//
			tmp_i = (N_Pf(2.0)*rho_kap*u_kap*u_kap;
			s_F[threadIdx.x] = tmp_i + (m_9 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 10.
			//
			tmp_i = -;
			s_F[threadIdx.x] = tmp_i + (m_10 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 11.
			//
			tmp_i = rho_kap*(v_kap*v_kap+w_kap*w_kap));
			s_F[threadIdx.x] = tmp_i + (m_11 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 12.
			//
			tmp_i = (-N_Pf(0.5)*(N_Pf(2.0)*rho_kap*u_kap*u_kap-rho_kap*(v_kap*v_kap+w_kap*w_kap)));
			s_F[threadIdx.x] = tmp_i + (m_12 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 13.
			//
			tmp_i = (rho_kap*(v_kap*v_kap-w_kap*w_kap));
			s_F[threadIdx.x] = tmp_i + (m_13 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 14.
			//
			tmp_i = (-N_Pf(0.5)*rho_kap*(v_kap*v_kap-w_kap*w_kap));
			s_F[threadIdx.x] = tmp_i + (m_14 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 15.
			//
			tmp_i = (rho_kap*u_kap*v_kap);
			s_F[threadIdx.x] = tmp_i + (m_15 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 16.
			//
			tmp_i = (rho_kap*v_kap*w_kap);
			s_F[threadIdx.x] = tmp_i + (m_16 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 17.
			//
			tmp_i = (rho_kap*u_kap*w_kap);
			s_F[threadIdx.x] = tmp_i + (m_17 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

			//
			// DDF 18.
			//
			tmp_i = 0;
			s_F[threadIdx.x] = tmp_i + (m_18 - tmp_i)*(tau_ratio);
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
			}
			__syncthreads();

		}
	}
}

int Solver_LBM::S_Interpolate_Linear_MRT_d3q19(int i_dev, int L, int var, ufloat_t tau_L, ufloat_t tau_ratio_L)
{
	if (mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_INTERFACE)
	{
		Cu_Interpolate_Linear_MRT_d3q19<0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_ADDED)
	{
		Cu_Interpolate_Linear_MRT_d3q19<1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

#endif