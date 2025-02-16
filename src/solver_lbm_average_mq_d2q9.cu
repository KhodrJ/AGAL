/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/
#include "solver.h"
#include "mesh.h"

#if (N_Q==9)

template <int ave_type=1>
__global__
void Cu_Average_d2q9
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, ufloat_t tau_L, ufloat_t tau_ratio, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_ID_mask_child[M_TBLOCK];
	__shared__ ufloat_t s_Fc[M_TBLOCK];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	int i_Q = -1;
	int i_Qc = -1;
	int i_Qcp = -1;
	int i_kap_b = -1;
	int i_kap_bc = -1;
	int child0_IJK = 2*((threadIdx.x % Nbx)%2) + Nbx*(2*(((threadIdx.x / Nbx) % Nbx)%2));
	int block_on_boundary = -1;
	ufloat_t f_0 = N_Pf(0.0);
	ufloat_t f_1 = N_Pf(0.0);
	ufloat_t f_2 = N_Pf(0.0);
	ufloat_t f_3 = N_Pf(0.0);
	ufloat_t f_4 = N_Pf(0.0);
	ufloat_t f_5 = N_Pf(0.0);
	ufloat_t f_6 = N_Pf(0.0);
	ufloat_t f_7 = N_Pf(0.0);
	ufloat_t f_8 = N_Pf(0.0);
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
			block_on_boundary=cblock_ID_mask[i_kap_b];
		}

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && ((i_kap_bc>-1)and((ave_type==2)or(block_on_boundary==1))))
		{
			for (int j_q = 0; j_q < Nqx/2; j_q += 1)
			{
				for (int i_q = 0; i_q < Nqx/2; i_q += 1)
				{
					//
					// Child block 0.
					// 
					//
					i_Q = (i_q+0*Nqx/2) + Nqx*(j_q+0*Nqx/2);
					i_Qc = 2*i_q + 2*Nqx*j_q;
					for (int xc_j = 0; xc_j < 2; xc_j += 1)
					{
						for (int xc_i = 0; xc_i < 2; xc_i += 1)
						{
							i_Qcp = i_Qc + xc_i + Nqx*xc_j;

							// Load DDFs and compute macroscopic properties.
							f_0 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
							f_1 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
							f_2 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
							f_3 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
							f_4 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
							f_5 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
							f_6 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
							f_7 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
							f_8 = cells_f_F[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
							rho_kap = f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8;
							u_kap = (N_Pf(0.0)*f_0+N_Pf(1.0)*f_1+N_Pf(0.0)*f_2+N_Pf(-1.0)*f_3+N_Pf(0.0)*f_4+N_Pf(1.0)*f_5+N_Pf(-1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(1.0)*f_8) / rho_kap;
							v_kap = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(1.0)*f_2+N_Pf(0.0)*f_3+N_Pf(-1.0)*f_4+N_Pf(1.0)*f_5+N_Pf(1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(-1.0)*f_8) / rho_kap;
							udotu = u_kap*u_kap + v_kap*v_kap;

							// Average rescaled fi to parent if applicable.
							s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
							if ((ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2))
							{
								s_ID_mask_child[threadIdx.x] = 1;
							}
							__syncthreads();

							//	 p = 0
							cdotu = N_Pf(0.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.444444444444444)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 1
							cdotu = N_Pf(1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 2
							cdotu = N_Pf(0.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 3
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 4
							cdotu = N_Pf(0.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 5
							cdotu = N_Pf(1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 6
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 7
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 8
							cdotu = N_Pf(1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();
						}
					}
					//
					// Child block 1.
					// 
					//
					i_Q = (i_q+1*Nqx/2) + Nqx*(j_q+0*Nqx/2);
					i_Qc = 2*i_q + 2*Nqx*j_q;
					for (int xc_j = 0; xc_j < 2; xc_j += 1)
					{
						for (int xc_i = 0; xc_i < 2; xc_i += 1)
						{
							i_Qcp = i_Qc + xc_i + Nqx*xc_j;

							// Load DDFs and compute macroscopic properties.
							f_0 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
							f_1 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
							f_2 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
							f_3 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
							f_4 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
							f_5 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
							f_6 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
							f_7 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
							f_8 = cells_f_F[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
							rho_kap = f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8;
							u_kap = (N_Pf(0.0)*f_0+N_Pf(1.0)*f_1+N_Pf(0.0)*f_2+N_Pf(-1.0)*f_3+N_Pf(0.0)*f_4+N_Pf(1.0)*f_5+N_Pf(-1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(1.0)*f_8) / rho_kap;
							v_kap = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(1.0)*f_2+N_Pf(0.0)*f_3+N_Pf(-1.0)*f_4+N_Pf(1.0)*f_5+N_Pf(1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(-1.0)*f_8) / rho_kap;
							udotu = u_kap*u_kap + v_kap*v_kap;

							// Average rescaled fi to parent if applicable.
							s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
							if ((ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2))
							{
								s_ID_mask_child[threadIdx.x] = 1;
							}
							__syncthreads();

							//	 p = 0
							cdotu = N_Pf(0.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.444444444444444)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 1
							cdotu = N_Pf(1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 2
							cdotu = N_Pf(0.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 3
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 4
							cdotu = N_Pf(0.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 5
							cdotu = N_Pf(1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 6
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 7
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 8
							cdotu = N_Pf(1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();
						}
					}
					//
					// Child block 2.
					// 
					//
					i_Q = (i_q+0*Nqx/2) + Nqx*(j_q+1*Nqx/2);
					i_Qc = 2*i_q + 2*Nqx*j_q;
					for (int xc_j = 0; xc_j < 2; xc_j += 1)
					{
						for (int xc_i = 0; xc_i < 2; xc_i += 1)
						{
							i_Qcp = i_Qc + xc_i + Nqx*xc_j;

							// Load DDFs and compute macroscopic properties.
							f_0 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
							f_1 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
							f_2 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
							f_3 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
							f_4 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
							f_5 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
							f_6 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
							f_7 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
							f_8 = cells_f_F[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
							rho_kap = f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8;
							u_kap = (N_Pf(0.0)*f_0+N_Pf(1.0)*f_1+N_Pf(0.0)*f_2+N_Pf(-1.0)*f_3+N_Pf(0.0)*f_4+N_Pf(1.0)*f_5+N_Pf(-1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(1.0)*f_8) / rho_kap;
							v_kap = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(1.0)*f_2+N_Pf(0.0)*f_3+N_Pf(-1.0)*f_4+N_Pf(1.0)*f_5+N_Pf(1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(-1.0)*f_8) / rho_kap;
							udotu = u_kap*u_kap + v_kap*v_kap;

							// Average rescaled fi to parent if applicable.
							s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
							if ((ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2))
							{
								s_ID_mask_child[threadIdx.x] = 1;
							}
							__syncthreads();

							//	 p = 0
							cdotu = N_Pf(0.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.444444444444444)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 1
							cdotu = N_Pf(1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 2
							cdotu = N_Pf(0.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 3
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 4
							cdotu = N_Pf(0.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 5
							cdotu = N_Pf(1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 6
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 7
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 8
							cdotu = N_Pf(1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();
						}
					}
					//
					// Child block 3.
					// 
					//
					i_Q = (i_q+1*Nqx/2) + Nqx*(j_q+1*Nqx/2);
					i_Qc = 2*i_q + 2*Nqx*j_q;
					for (int xc_j = 0; xc_j < 2; xc_j += 1)
					{
						for (int xc_i = 0; xc_i < 2; xc_i += 1)
						{
							i_Qcp = i_Qc + xc_i + Nqx*xc_j;

							// Load DDFs and compute macroscopic properties.
							f_0 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
							f_1 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
							f_2 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
							f_3 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
							f_4 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
							f_5 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
							f_6 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
							f_7 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
							f_8 = cells_f_F[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
							rho_kap = f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8;
							u_kap = (N_Pf(0.0)*f_0+N_Pf(1.0)*f_1+N_Pf(0.0)*f_2+N_Pf(-1.0)*f_3+N_Pf(0.0)*f_4+N_Pf(1.0)*f_5+N_Pf(-1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(1.0)*f_8) / rho_kap;
							v_kap = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(1.0)*f_2+N_Pf(0.0)*f_3+N_Pf(-1.0)*f_4+N_Pf(1.0)*f_5+N_Pf(1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(-1.0)*f_8) / rho_kap;
							udotu = u_kap*u_kap + v_kap*v_kap;

							// Average rescaled fi to parent if applicable.
							s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
							if ((ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2))
							{
								s_ID_mask_child[threadIdx.x] = 1;
							}
							__syncthreads();

							//	 p = 0
							cdotu = N_Pf(0.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.444444444444444)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 1
							cdotu = N_Pf(1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 2
							cdotu = N_Pf(0.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 3
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 4
							cdotu = N_Pf(0.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 5
							cdotu = N_Pf(1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 6
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 7
							cdotu = N_Pf(-1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();

							//	 p = 8
							cdotu = N_Pf(1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
							tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
							s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
							__syncthreads();
							if ((s_ID_mask_child[child0_IJK] == 1)and(I_kap >= 2*xc_i)and(I_kap < 2+2*xc_i)and(J_kap >= 2*xc_j)and(J_kap < 2+2*xc_j))
							{
								cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
							}
							__syncthreads();
						}
					}
				}
			}
		}
	}
}

int Solver_LBM::S_Average_d2q9(int i_dev, int L, int var, ufloat_t tau_L, ufloat_t tau_ratio_L)
{
	if (mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_INTERFACE)
	{
		Cu_Average_d2q9<0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_BLOCK)
	{
		Cu_Average_d2q9<1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_GRID)
	{
		Cu_Average_d2q9<2><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

#endif