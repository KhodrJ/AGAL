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
void Cu_Interpolate_Linear_Construct_d3q19
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, ufloat_t v0, ufloat_t dx_L, ufloat_t tau_L, ufloat_t tau_ratio, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ ufloat_t s_rho[M_TBLOCK];
	__shared__ ufloat_t s_u[M_TBLOCK];
	__shared__ ufloat_t s_v[M_TBLOCK];
	__shared__ ufloat_t s_w[M_TBLOCK];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	ufloat_t x_kap = N_Pf(-0.25) + N_Pf(0.5)*I_kap;
	ufloat_t y_kap = N_Pf(-0.25) + N_Pf(0.5)*J_kap;
	int K_kap = (threadIdx.x / Nbx) / Nbx;
	ufloat_t z_kap = N_Pf(-0.25) + N_Pf(0.5)*K_kap;
	int i_kap_b = -1;
	int i_kap_bc = -1;
	int block_on_interface = -1;
	ufloat_t f_0 = N_Pf(0.0);
	ufloat_t f_1 = N_Pf(0.0);
	ufloat_t f_2 = N_Pf(0.0);
	ufloat_t f_3 = N_Pf(0.0);
	ufloat_t f_4 = N_Pf(0.0);
	ufloat_t f_5 = N_Pf(0.0);
	ufloat_t f_6 = N_Pf(0.0);
	ufloat_t f_7 = N_Pf(0.0);
	ufloat_t f_8 = N_Pf(0.0);
	ufloat_t f_9 = N_Pf(0.0);
	ufloat_t f_10 = N_Pf(0.0);
	ufloat_t f_11 = N_Pf(0.0);
	ufloat_t f_12 = N_Pf(0.0);
	ufloat_t f_13 = N_Pf(0.0);
	ufloat_t f_14 = N_Pf(0.0);
	ufloat_t f_15 = N_Pf(0.0);
	ufloat_t f_16 = N_Pf(0.0);
	ufloat_t f_17 = N_Pf(0.0);
	ufloat_t f_18 = N_Pf(0.0);
	ufloat_t tmp_i = N_Pf(0.0);
	ufloat_t rho = N_Pf(0.0);
	ufloat_t rho_x = N_Pf(0.0);
	ufloat_t rho_y = N_Pf(0.0);
	ufloat_t rho_z = N_Pf(0.0);
	ufloat_t u = N_Pf(0.0);
	ufloat_t u_x = N_Pf(0.0);
	ufloat_t u_y = N_Pf(0.0);
	ufloat_t u_z = N_Pf(0.0);
	ufloat_t u_xx = N_Pf(0.0);
	ufloat_t u_xy = N_Pf(0.0);
	ufloat_t u_zx = N_Pf(0.0);
	ufloat_t u_yy = N_Pf(0.0);
	ufloat_t u_yz = N_Pf(0.0);
	ufloat_t u_zz = N_Pf(0.0);
	ufloat_t v = N_Pf(0.0);
	ufloat_t v_x = N_Pf(0.0);
	ufloat_t v_y = N_Pf(0.0);
	ufloat_t v_z = N_Pf(0.0);
	ufloat_t v_xx = N_Pf(0.0);
	ufloat_t v_xy = N_Pf(0.0);
	ufloat_t v_xz = N_Pf(0.0);
	ufloat_t v_yy = N_Pf(0.0);
	ufloat_t v_yz = N_Pf(0.0);
	ufloat_t v_zz = N_Pf(0.0);
	ufloat_t w = N_Pf(0.0);
	ufloat_t w_x = N_Pf(0.0);
	ufloat_t w_y = N_Pf(0.0);
	ufloat_t w_z = N_Pf(0.0);
	ufloat_t w_xx = N_Pf(0.0);
	ufloat_t w_xy = N_Pf(0.0);
	ufloat_t w_xz = N_Pf(0.0);
	ufloat_t w_yy = N_Pf(0.0);
	ufloat_t w_yz = N_Pf(0.0);
	ufloat_t w_zz = N_Pf(0.0);
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
			f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			rho = f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 ;
			u = (N_Pf(0.0)*f_0 +N_Pf(1.0)*f_1 +N_Pf(-1.0)*f_2 +N_Pf(0.0)*f_3 +N_Pf(0.0)*f_4 +N_Pf(0.0)*f_5 +N_Pf(0.0)*f_6 +N_Pf(1.0)*f_7 +N_Pf(-1.0)*f_8 +N_Pf(1.0)*f_9 +N_Pf(-1.0)*f_10 +N_Pf(0.0)*f_11 +N_Pf(0.0)*f_12 +N_Pf(1.0)*f_13 +N_Pf(-1.0)*f_14 +N_Pf(1.0)*f_15 +N_Pf(-1.0)*f_16 +N_Pf(0.0)*f_17 +N_Pf(0.0)*f_18 ) / rho;
			v = (N_Pf(0.0)*f_0 +N_Pf(0.0)*f_1 +N_Pf(0.0)*f_2 +N_Pf(1.0)*f_3 +N_Pf(-1.0)*f_4 +N_Pf(0.0)*f_5 +N_Pf(0.0)*f_6 +N_Pf(1.0)*f_7 +N_Pf(-1.0)*f_8 +N_Pf(0.0)*f_9 +N_Pf(0.0)*f_10 +N_Pf(1.0)*f_11 +N_Pf(-1.0)*f_12 +N_Pf(-1.0)*f_13 +N_Pf(1.0)*f_14 +N_Pf(0.0)*f_15 +N_Pf(0.0)*f_16 +N_Pf(1.0)*f_17 +N_Pf(-1.0)*f_18 ) / rho;
			w = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(0.0)*f_2+N_Pf(0.0)*f_3+N_Pf(0.0)*f_4+N_Pf(1.0)*f_5+N_Pf(-1.0)*f_6+N_Pf(0.0)*f_7+N_Pf(0.0)*f_8+N_Pf(1.0)*f_9+N_Pf(-1.0)*f_10+N_Pf(1.0)*f_11+N_Pf(-1.0)*f_12+N_Pf(0.0)*f_13+N_Pf(0.0)*f_14+N_Pf(-1.0)*f_15+N_Pf(1.0)*f_16+N_Pf(-1.0)*f_17+N_Pf(1.0)*f_18) / rho;
			s_u[threadIdx.x] = u;
			s_v[threadIdx.x] = v;
			s_rho[threadIdx.x] = rho;
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{

				//
				// u
				//
				A00 = s_u[0];
				A10 = s_u[1]-s_u[0];
				A01 = s_u[4]-s_u[0];
				A11 = s_u[5]-s_u[1]-s_u[4]+s_u[0];
				u = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				u_x = (A10) + y_kap*(A11);
				u_y = A01 + A11*x_kap;
				u_xx = N_Pf(0.0);
				u_xy = A11;
				u_yy = N_Pf(0.0);

				// v
				A00 = s_v[0];
				A10 = s_v[1]-s_v[0];
				A01 = s_v[4]-s_v[0];
				A11 = s_v[5]-s_v[1]-s_v[4]+s_v[0];
				v = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				v_x = (A10) + y_kap*(A11);
				v_y = A01 + A11*x_kap;
				v_xx = N_Pf(0.0);
				v_xy = A11;
				v_yy = N_Pf(0.0);

				// rho
				A00 = s_rho[0];
				A10 = s_rho[1]-s_rho[0];
				A01 = s_rho[4]-s_rho[0];
				A11 = s_rho[5]-s_rho[1]-s_rho[4]+s_rho[0];
				rho = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				rho_x = (A10) + y_kap*(A11);
				rho_y = A01 + A11*x_kap;

				// aux
				aux_0 = (0.3333333333333333)*(u_x + v_y);
				aux_1xx = v0*(u_xx + u_xx) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_x));
				aux_1xy = v0*(v_xx + u_xy) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_y));
				aux_1yx = v0*(u_yy + v_xy) + (v0/rho)*rho_y*(N_Pf(0.5)*(u_y+u_x));
				aux_1yy = v0*(v_yy + v_yy) + (v0/rho)*rho_y*(N_Pf(0.5)*(v_y+v_y));

				f_0 = N_Pf(0.0);
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
				f_1 = N_Pf(0.0);
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
				f_2 = N_Pf(0.0);
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
				f_3 = N_Pf(0.0);
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
				f_4 = N_Pf(0.0);
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
				f_5 = N_Pf(0.0);
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
				f_6 = N_Pf(0.0);
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
				f_7 = N_Pf(0.0);
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
				f_8 = N_Pf(0.0);
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
				f_9 = N_Pf(0.0);
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
				f_10 = N_Pf(0.0);
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
				f_11 = N_Pf(0.0);
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
				f_12 = N_Pf(0.0);
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
				f_13 = N_Pf(0.0);
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
				f_14 = N_Pf(0.0);
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
				f_15 = N_Pf(0.0);
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
				f_16 = N_Pf(0.0);
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
				f_17 = N_Pf(0.0);
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
				f_18 = N_Pf(0.0);
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{

				//
				// u
				//
				A00 = s_u[2];
				A10 = s_u[3]-s_u[2];
				A01 = s_u[6]-s_u[2];
				A11 = s_u[7]-s_u[3]-s_u[6]+s_u[2];
				u = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				u_x = (A10) + y_kap*(A11);
				u_y = A01 + A11*x_kap;
				u_xx = N_Pf(0.0);
				u_xy = A11;
				u_yy = N_Pf(0.0);

				// v
				A00 = s_v[2];
				A10 = s_v[3]-s_v[2];
				A01 = s_v[6]-s_v[2];
				A11 = s_v[7]-s_v[3]-s_v[6]+s_v[2];
				v = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				v_x = (A10) + y_kap*(A11);
				v_y = A01 + A11*x_kap;
				v_xx = N_Pf(0.0);
				v_xy = A11;
				v_yy = N_Pf(0.0);

				// rho
				A00 = s_rho[2];
				A10 = s_rho[3]-s_rho[2];
				A01 = s_rho[6]-s_rho[2];
				A11 = s_rho[7]-s_rho[3]-s_rho[6]+s_rho[2];
				rho = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				rho_x = (A10) + y_kap*(A11);
				rho_y = A01 + A11*x_kap;

				// aux
				aux_0 = (0.3333333333333333)*(u_x + v_y);
				aux_1xx = v0*(u_xx + u_xx) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_x));
				aux_1xy = v0*(v_xx + u_xy) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_y));
				aux_1yx = v0*(u_yy + v_xy) + (v0/rho)*rho_y*(N_Pf(0.5)*(u_y+u_x));
				aux_1yy = v0*(v_yy + v_yy) + (v0/rho)*rho_y*(N_Pf(0.5)*(v_y+v_y));

				f_0 = N_Pf(0.0);
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
				f_1 = N_Pf(0.0);
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
				f_2 = N_Pf(0.0);
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
				f_3 = N_Pf(0.0);
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
				f_4 = N_Pf(0.0);
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
				f_5 = N_Pf(0.0);
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
				f_6 = N_Pf(0.0);
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
				f_7 = N_Pf(0.0);
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
				f_8 = N_Pf(0.0);
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
				f_9 = N_Pf(0.0);
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
				f_10 = N_Pf(0.0);
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
				f_11 = N_Pf(0.0);
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
				f_12 = N_Pf(0.0);
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
				f_13 = N_Pf(0.0);
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
				f_14 = N_Pf(0.0);
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
				f_15 = N_Pf(0.0);
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
				f_16 = N_Pf(0.0);
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
				f_17 = N_Pf(0.0);
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
				f_18 = N_Pf(0.0);
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{

				//
				// u
				//
				A00 = s_u[8];
				A10 = s_u[9]-s_u[8];
				A01 = s_u[12]-s_u[8];
				A11 = s_u[13]-s_u[9]-s_u[12]+s_u[8];
				u = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				u_x = (A10) + y_kap*(A11);
				u_y = A01 + A11*x_kap;
				u_xx = N_Pf(0.0);
				u_xy = A11;
				u_yy = N_Pf(0.0);

				// v
				A00 = s_v[8];
				A10 = s_v[9]-s_v[8];
				A01 = s_v[12]-s_v[8];
				A11 = s_v[13]-s_v[9]-s_v[12]+s_v[8];
				v = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				v_x = (A10) + y_kap*(A11);
				v_y = A01 + A11*x_kap;
				v_xx = N_Pf(0.0);
				v_xy = A11;
				v_yy = N_Pf(0.0);

				// rho
				A00 = s_rho[8];
				A10 = s_rho[9]-s_rho[8];
				A01 = s_rho[12]-s_rho[8];
				A11 = s_rho[13]-s_rho[9]-s_rho[12]+s_rho[8];
				rho = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				rho_x = (A10) + y_kap*(A11);
				rho_y = A01 + A11*x_kap;

				// aux
				aux_0 = (0.3333333333333333)*(u_x + v_y);
				aux_1xx = v0*(u_xx + u_xx) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_x));
				aux_1xy = v0*(v_xx + u_xy) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_y));
				aux_1yx = v0*(u_yy + v_xy) + (v0/rho)*rho_y*(N_Pf(0.5)*(u_y+u_x));
				aux_1yy = v0*(v_yy + v_yy) + (v0/rho)*rho_y*(N_Pf(0.5)*(v_y+v_y));

				f_0 = N_Pf(0.0);
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
				f_1 = N_Pf(0.0);
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
				f_2 = N_Pf(0.0);
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
				f_3 = N_Pf(0.0);
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
				f_4 = N_Pf(0.0);
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
				f_5 = N_Pf(0.0);
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
				f_6 = N_Pf(0.0);
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
				f_7 = N_Pf(0.0);
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
				f_8 = N_Pf(0.0);
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
				f_9 = N_Pf(0.0);
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
				f_10 = N_Pf(0.0);
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
				f_11 = N_Pf(0.0);
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
				f_12 = N_Pf(0.0);
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
				f_13 = N_Pf(0.0);
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
				f_14 = N_Pf(0.0);
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
				f_15 = N_Pf(0.0);
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
				f_16 = N_Pf(0.0);
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
				f_17 = N_Pf(0.0);
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
				f_18 = N_Pf(0.0);
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{

				//
				// u
				//
				A00 = s_u[10];
				A10 = s_u[11]-s_u[10];
				A01 = s_u[14]-s_u[10];
				A11 = s_u[15]-s_u[11]-s_u[14]+s_u[10];
				u = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				u_x = (A10) + y_kap*(A11);
				u_y = A01 + A11*x_kap;
				u_xx = N_Pf(0.0);
				u_xy = A11;
				u_yy = N_Pf(0.0);

				// v
				A00 = s_v[10];
				A10 = s_v[11]-s_v[10];
				A01 = s_v[14]-s_v[10];
				A11 = s_v[15]-s_v[11]-s_v[14]+s_v[10];
				v = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				v_x = (A10) + y_kap*(A11);
				v_y = A01 + A11*x_kap;
				v_xx = N_Pf(0.0);
				v_xy = A11;
				v_yy = N_Pf(0.0);

				// rho
				A00 = s_rho[10];
				A10 = s_rho[11]-s_rho[10];
				A01 = s_rho[14]-s_rho[10];
				A11 = s_rho[15]-s_rho[11]-s_rho[14]+s_rho[10];
				rho = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				rho_x = (A10) + y_kap*(A11);
				rho_y = A01 + A11*x_kap;

				// aux
				aux_0 = (0.3333333333333333)*(u_x + v_y);
				aux_1xx = v0*(u_xx + u_xx) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_x));
				aux_1xy = v0*(v_xx + u_xy) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_y));
				aux_1yx = v0*(u_yy + v_xy) + (v0/rho)*rho_y*(N_Pf(0.5)*(u_y+u_x));
				aux_1yy = v0*(v_yy + v_yy) + (v0/rho)*rho_y*(N_Pf(0.5)*(v_y+v_y));

				f_0 = N_Pf(0.0);
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
				f_1 = N_Pf(0.0);
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
				f_2 = N_Pf(0.0);
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
				f_3 = N_Pf(0.0);
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
				f_4 = N_Pf(0.0);
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
				f_5 = N_Pf(0.0);
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
				f_6 = N_Pf(0.0);
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
				f_7 = N_Pf(0.0);
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
				f_8 = N_Pf(0.0);
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
				f_9 = N_Pf(0.0);
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
				f_10 = N_Pf(0.0);
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
				f_11 = N_Pf(0.0);
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
				f_12 = N_Pf(0.0);
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
				f_13 = N_Pf(0.0);
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
				f_14 = N_Pf(0.0);
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
				f_15 = N_Pf(0.0);
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
				f_16 = N_Pf(0.0);
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
				f_17 = N_Pf(0.0);
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
				f_18 = N_Pf(0.0);
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{

				//
				// u
				//
				A00 = s_u[32];
				A10 = s_u[33]-s_u[32];
				A01 = s_u[36]-s_u[32];
				A11 = s_u[37]-s_u[33]-s_u[36]+s_u[32];
				u = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				u_x = (A10) + y_kap*(A11);
				u_y = A01 + A11*x_kap;
				u_xx = N_Pf(0.0);
				u_xy = A11;
				u_yy = N_Pf(0.0);

				// v
				A00 = s_v[32];
				A10 = s_v[33]-s_v[32];
				A01 = s_v[36]-s_v[32];
				A11 = s_v[37]-s_v[33]-s_v[36]+s_v[32];
				v = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				v_x = (A10) + y_kap*(A11);
				v_y = A01 + A11*x_kap;
				v_xx = N_Pf(0.0);
				v_xy = A11;
				v_yy = N_Pf(0.0);

				// rho
				A00 = s_rho[32];
				A10 = s_rho[33]-s_rho[32];
				A01 = s_rho[36]-s_rho[32];
				A11 = s_rho[37]-s_rho[33]-s_rho[36]+s_rho[32];
				rho = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				rho_x = (A10) + y_kap*(A11);
				rho_y = A01 + A11*x_kap;

				// aux
				aux_0 = (0.3333333333333333)*(u_x + v_y);
				aux_1xx = v0*(u_xx + u_xx) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_x));
				aux_1xy = v0*(v_xx + u_xy) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_y));
				aux_1yx = v0*(u_yy + v_xy) + (v0/rho)*rho_y*(N_Pf(0.5)*(u_y+u_x));
				aux_1yy = v0*(v_yy + v_yy) + (v0/rho)*rho_y*(N_Pf(0.5)*(v_y+v_y));

				f_0 = N_Pf(0.0);
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
				f_1 = N_Pf(0.0);
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
				f_2 = N_Pf(0.0);
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
				f_3 = N_Pf(0.0);
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
				f_4 = N_Pf(0.0);
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
				f_5 = N_Pf(0.0);
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
				f_6 = N_Pf(0.0);
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
				f_7 = N_Pf(0.0);
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
				f_8 = N_Pf(0.0);
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
				f_9 = N_Pf(0.0);
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
				f_10 = N_Pf(0.0);
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
				f_11 = N_Pf(0.0);
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
				f_12 = N_Pf(0.0);
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
				f_13 = N_Pf(0.0);
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
				f_14 = N_Pf(0.0);
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
				f_15 = N_Pf(0.0);
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
				f_16 = N_Pf(0.0);
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
				f_17 = N_Pf(0.0);
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
				f_18 = N_Pf(0.0);
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{

				//
				// u
				//
				A00 = s_u[34];
				A10 = s_u[35]-s_u[34];
				A01 = s_u[38]-s_u[34];
				A11 = s_u[39]-s_u[35]-s_u[38]+s_u[34];
				u = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				u_x = (A10) + y_kap*(A11);
				u_y = A01 + A11*x_kap;
				u_xx = N_Pf(0.0);
				u_xy = A11;
				u_yy = N_Pf(0.0);

				// v
				A00 = s_v[34];
				A10 = s_v[35]-s_v[34];
				A01 = s_v[38]-s_v[34];
				A11 = s_v[39]-s_v[35]-s_v[38]+s_v[34];
				v = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				v_x = (A10) + y_kap*(A11);
				v_y = A01 + A11*x_kap;
				v_xx = N_Pf(0.0);
				v_xy = A11;
				v_yy = N_Pf(0.0);

				// rho
				A00 = s_rho[34];
				A10 = s_rho[35]-s_rho[34];
				A01 = s_rho[38]-s_rho[34];
				A11 = s_rho[39]-s_rho[35]-s_rho[38]+s_rho[34];
				rho = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				rho_x = (A10) + y_kap*(A11);
				rho_y = A01 + A11*x_kap;

				// aux
				aux_0 = (0.3333333333333333)*(u_x + v_y);
				aux_1xx = v0*(u_xx + u_xx) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_x));
				aux_1xy = v0*(v_xx + u_xy) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_y));
				aux_1yx = v0*(u_yy + v_xy) + (v0/rho)*rho_y*(N_Pf(0.5)*(u_y+u_x));
				aux_1yy = v0*(v_yy + v_yy) + (v0/rho)*rho_y*(N_Pf(0.5)*(v_y+v_y));

				f_0 = N_Pf(0.0);
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
				f_1 = N_Pf(0.0);
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
				f_2 = N_Pf(0.0);
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
				f_3 = N_Pf(0.0);
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
				f_4 = N_Pf(0.0);
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
				f_5 = N_Pf(0.0);
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
				f_6 = N_Pf(0.0);
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
				f_7 = N_Pf(0.0);
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
				f_8 = N_Pf(0.0);
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
				f_9 = N_Pf(0.0);
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
				f_10 = N_Pf(0.0);
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
				f_11 = N_Pf(0.0);
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
				f_12 = N_Pf(0.0);
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
				f_13 = N_Pf(0.0);
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
				f_14 = N_Pf(0.0);
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
				f_15 = N_Pf(0.0);
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
				f_16 = N_Pf(0.0);
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
				f_17 = N_Pf(0.0);
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
				f_18 = N_Pf(0.0);
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{

				//
				// u
				//
				A00 = s_u[40];
				A10 = s_u[41]-s_u[40];
				A01 = s_u[44]-s_u[40];
				A11 = s_u[45]-s_u[41]-s_u[44]+s_u[40];
				u = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				u_x = (A10) + y_kap*(A11);
				u_y = A01 + A11*x_kap;
				u_xx = N_Pf(0.0);
				u_xy = A11;
				u_yy = N_Pf(0.0);

				// v
				A00 = s_v[40];
				A10 = s_v[41]-s_v[40];
				A01 = s_v[44]-s_v[40];
				A11 = s_v[45]-s_v[41]-s_v[44]+s_v[40];
				v = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				v_x = (A10) + y_kap*(A11);
				v_y = A01 + A11*x_kap;
				v_xx = N_Pf(0.0);
				v_xy = A11;
				v_yy = N_Pf(0.0);

				// rho
				A00 = s_rho[40];
				A10 = s_rho[41]-s_rho[40];
				A01 = s_rho[44]-s_rho[40];
				A11 = s_rho[45]-s_rho[41]-s_rho[44]+s_rho[40];
				rho = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				rho_x = (A10) + y_kap*(A11);
				rho_y = A01 + A11*x_kap;

				// aux
				aux_0 = (0.3333333333333333)*(u_x + v_y);
				aux_1xx = v0*(u_xx + u_xx) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_x));
				aux_1xy = v0*(v_xx + u_xy) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_y));
				aux_1yx = v0*(u_yy + v_xy) + (v0/rho)*rho_y*(N_Pf(0.5)*(u_y+u_x));
				aux_1yy = v0*(v_yy + v_yy) + (v0/rho)*rho_y*(N_Pf(0.5)*(v_y+v_y));

				f_0 = N_Pf(0.0);
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
				f_1 = N_Pf(0.0);
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
				f_2 = N_Pf(0.0);
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
				f_3 = N_Pf(0.0);
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
				f_4 = N_Pf(0.0);
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
				f_5 = N_Pf(0.0);
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
				f_6 = N_Pf(0.0);
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
				f_7 = N_Pf(0.0);
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
				f_8 = N_Pf(0.0);
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
				f_9 = N_Pf(0.0);
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
				f_10 = N_Pf(0.0);
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
				f_11 = N_Pf(0.0);
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
				f_12 = N_Pf(0.0);
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
				f_13 = N_Pf(0.0);
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
				f_14 = N_Pf(0.0);
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
				f_15 = N_Pf(0.0);
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
				f_16 = N_Pf(0.0);
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
				f_17 = N_Pf(0.0);
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
				f_18 = N_Pf(0.0);
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{

				//
				// u
				//
				A00 = s_u[42];
				A10 = s_u[43]-s_u[42];
				A01 = s_u[46]-s_u[42];
				A11 = s_u[47]-s_u[43]-s_u[46]+s_u[42];
				u = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				u_x = (A10) + y_kap*(A11);
				u_y = A01 + A11*x_kap;
				u_xx = N_Pf(0.0);
				u_xy = A11;
				u_yy = N_Pf(0.0);

				// v
				A00 = s_v[42];
				A10 = s_v[43]-s_v[42];
				A01 = s_v[46]-s_v[42];
				A11 = s_v[47]-s_v[43]-s_v[46]+s_v[42];
				v = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				v_x = (A10) + y_kap*(A11);
				v_y = A01 + A11*x_kap;
				v_xx = N_Pf(0.0);
				v_xy = A11;
				v_yy = N_Pf(0.0);

				// rho
				A00 = s_rho[42];
				A10 = s_rho[43]-s_rho[42];
				A01 = s_rho[46]-s_rho[42];
				A11 = s_rho[47]-s_rho[43]-s_rho[46]+s_rho[42];
				rho = (A00 + A10*x_kap) + y_kap*(A01 + A11*x_kap);
				rho_x = (A10) + y_kap*(A11);
				rho_y = A01 + A11*x_kap;

				// aux
				aux_0 = (0.3333333333333333)*(u_x + v_y);
				aux_1xx = v0*(u_xx + u_xx) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_x));
				aux_1xy = v0*(v_xx + u_xy) + (v0/rho)*rho_x*(N_Pf(0.5)*(u_x+u_y));
				aux_1yx = v0*(u_yy + v_xy) + (v0/rho)*rho_y*(N_Pf(0.5)*(u_y+u_x));
				aux_1yy = v0*(v_yy + v_yy) + (v0/rho)*rho_y*(N_Pf(0.5)*(v_y+v_y));

				f_0 = N_Pf(0.0);
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_0 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_0 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
				f_1 = N_Pf(0.0);
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_1 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_1 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
				f_2 = N_Pf(0.0);
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_2 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_2 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
				f_3 = N_Pf(0.0);
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_3 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_3 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
				f_4 = N_Pf(0.0);
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_4 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_4 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
				f_5 = N_Pf(0.0);
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_5 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_5 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
				f_6 = N_Pf(0.0);
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_6 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_6 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
				f_7 = N_Pf(0.0);
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_7 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_7 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
				f_8 = N_Pf(0.0);
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_8 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_8 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
				f_9 = N_Pf(0.0);
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_9 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_9 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
				f_10 = N_Pf(0.0);
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_10 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_10 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
				f_11 = N_Pf(0.0);
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_11 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_11 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
				f_12 = N_Pf(0.0);
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_12 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_12 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
				f_13 = N_Pf(0.0);
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_13 += (N_Pf(1.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_13 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
				f_14 = N_Pf(0.0);
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_14 += (N_Pf(-1.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_14 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
				f_15 = N_Pf(0.0);
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_15 += (N_Pf(1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_15 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
				f_16 = N_Pf(0.0);
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(-1.0)-u)*u_x + aux_1xx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(-1.0)-u)*v_x + aux_1xy ) - aux_0;
				f_16 += (N_Pf(-1.0)-u)*( (N_Pf(0.0)-v)*u_y + aux_1yx ) - aux_0;
				f_16 += (N_Pf(0.0)-v)*( (N_Pf(0.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
				f_17 = N_Pf(0.0);
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_17 += (N_Pf(0.0)-u)*( (N_Pf(1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_17 += (N_Pf(1.0)-v)*( (N_Pf(1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
				f_18 = N_Pf(0.0);
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(0.0)-u)*u_x + aux_1xx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(0.0)-u)*v_x + aux_1xy ) - aux_0;
				f_18 += (N_Pf(0.0)-u)*( (N_Pf(-1.0)-v)*u_y + aux_1yx ) - aux_0;
				f_18 += (N_Pf(-1.0)-v)*( (N_Pf(-1.0)-v)*v_y + aux_1yy ) - aux_0;
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
			}
		}
	}
}

int Solver_LBM::S_Interpolate_Linear_Construct_d3q19(int i_dev, int L, int var, ufloat_t tau_L, ufloat_t tau_ratio_L)
{
	if (mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_INTERFACE)
	{
		Cu_Interpolate_Linear_Construct_d3q19<0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, v0, dxf_vec[L], tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_ADDED)
	{
		Cu_Interpolate_Linear_Construct_d3q19<1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, v0, dxf_vec[L], tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

#endif