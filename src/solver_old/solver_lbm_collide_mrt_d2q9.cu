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
void Cu_Collide_MRT_d2q9
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, ufloat_t dx_L, ufloat_t tau_L, ufloat_t s_1, ufloat_t s_2, ufloat_t s_3, ufloat_t s_4, ufloat_t s_5, ufloat_t s_6, ufloat_t tau_ratio, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb, ufloat_t *cblock_f_X
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	ufloat_t x_kap = N_Pf(0.0);
	ufloat_t y_kap = N_Pf(0.0);
	int i_kap_b = -1;
	int i_kap_bc = -1;
	int nbr_kap_b = -1;
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
	ufloat_t m_0 = N_Pf(0.0);
	ufloat_t m_1 = N_Pf(0.0);
	ufloat_t m_2 = N_Pf(0.0);
	ufloat_t m_3 = N_Pf(0.0);
	ufloat_t m_4 = N_Pf(0.0);
	ufloat_t m_5 = N_Pf(0.0);
	ufloat_t m_6 = N_Pf(0.0);
	ufloat_t m_7 = N_Pf(0.0);
	ufloat_t m_8 = N_Pf(0.0);
	ufloat_t rho_kap = N_Pf(0.0);
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
	ufloat_t cdotu = N_Pf(0.0);
	ufloat_t udotu = N_Pf(0.0);
	ufloat_t omeg = dx_L / tau_L;
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
			// Retrieve DDFs and compute macroscopic properties.
			block_on_boundary = cblock_ID_onb[i_kap_b];
			x_kap = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*(N_Pf(0.5) + I_kap);
			y_kap = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*(N_Pf(0.5) + J_kap);
			m_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			m_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			m_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			m_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			m_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			m_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			m_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			m_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			m_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			rho_kap = m_0;
			u_kap = m_3 / m_0;
			v_kap = m_5 / m_0;
			udotu = u_kap*u_kap + v_kap*v_kap;

			// Eddy viscosity calculation.

			// Collision step.
			m_1 = m_1 - s_1*(m_1 - (N_Pf(-2.0)*rho_kap + N_Pf(3.0)*rho_kap*(u_kap*u_kap+v_kap*v_kap)));
			m_2 = m_2 - s_2*(m_2 - (N_Pf(9.0)*rho_kap*u_kap*u_kap*v_kap*v_kap - N_Pf(3.0)*rho_kap*(u_kap*u_kap+v_kap*v_kap) + rho_kap));
			m_4 = m_4 - s_3*(m_4 - (rho_kap*u_kap*(N_Pf(3.0)*v_kap*v_kap-N_Pf(1.0))));
			m_6 = m_6 - s_3*(m_6 - (rho_kap*v_kap*(N_Pf(3.0)*u_kap*u_kap-N_Pf(1.0))));
			m_7 = m_7 - omeg*(m_7 - (rho_kap*(u_kap*u_kap-v_kap*v_kap)));
			m_8 = m_8 - omeg*(m_8 - (rho_kap*u_kap*v_kap));
			f_0 = N_Pf(0.111111111111111)*m_0+N_Pf(-0.111111111111111)*m_1+N_Pf(0.111111111111111)*m_2+N_Pf(0.000000000000000)*m_3+N_Pf(0.000000000000000)*m_4+N_Pf(0.000000000000000)*m_5+N_Pf(0.000000000000000)*m_6+N_Pf(0.000000000000000)*m_7+N_Pf(0.000000000000000)*m_8;
			f_1 = N_Pf(0.111111111111111)*m_0+N_Pf(-0.027777777777778)*m_1+N_Pf(-0.055555555555556)*m_2+N_Pf(0.166666666666667)*m_3+N_Pf(-0.166666666666667)*m_4+N_Pf(0.000000000000000)*m_5+N_Pf(0.000000000000000)*m_6+N_Pf(0.250000000000000)*m_7+N_Pf(0.000000000000000)*m_8;
			f_2 = N_Pf(0.111111111111111)*m_0+N_Pf(-0.027777777777778)*m_1+N_Pf(-0.055555555555556)*m_2+N_Pf(0.000000000000000)*m_3+N_Pf(-0.000000000000000)*m_4+N_Pf(0.166666666666667)*m_5+N_Pf(-0.166666666666667)*m_6+N_Pf(-0.250000000000000)*m_7+N_Pf(0.000000000000000)*m_8;
			f_3 = N_Pf(0.111111111111111)*m_0+N_Pf(-0.027777777777778)*m_1+N_Pf(-0.055555555555556)*m_2+N_Pf(-0.166666666666667)*m_3+N_Pf(0.166666666666667)*m_4+N_Pf(0.000000000000000)*m_5+N_Pf(0.000000000000000)*m_6+N_Pf(0.250000000000000)*m_7+N_Pf(0.000000000000000)*m_8;
			f_4 = N_Pf(0.111111111111111)*m_0+N_Pf(-0.027777777777778)*m_1+N_Pf(-0.055555555555556)*m_2+N_Pf(0.000000000000000)*m_3+N_Pf(0.000000000000000)*m_4+N_Pf(-0.166666666666667)*m_5+N_Pf(0.166666666666667)*m_6+N_Pf(-0.250000000000000)*m_7+N_Pf(0.000000000000000)*m_8;
			f_5 = N_Pf(0.111111111111111)*m_0+N_Pf(0.055555555555556)*m_1+N_Pf(0.027777777777778)*m_2+N_Pf(0.166666666666667)*m_3+N_Pf(0.083333333333333)*m_4+N_Pf(0.166666666666667)*m_5+N_Pf(0.083333333333333)*m_6+N_Pf(0.000000000000000)*m_7+N_Pf(0.250000000000000)*m_8;
			f_6 = N_Pf(0.111111111111111)*m_0+N_Pf(0.055555555555556)*m_1+N_Pf(0.027777777777778)*m_2+N_Pf(-0.166666666666667)*m_3+N_Pf(-0.083333333333333)*m_4+N_Pf(0.166666666666667)*m_5+N_Pf(0.083333333333333)*m_6+N_Pf(0.000000000000000)*m_7+N_Pf(-0.250000000000000)*m_8;
			f_7 = N_Pf(0.111111111111111)*m_0+N_Pf(0.055555555555556)*m_1+N_Pf(0.027777777777778)*m_2+N_Pf(-0.166666666666667)*m_3+N_Pf(-0.083333333333333)*m_4+N_Pf(-0.166666666666667)*m_5+N_Pf(-0.083333333333333)*m_6+N_Pf(0.000000000000000)*m_7+N_Pf(0.250000000000000)*m_8;
			f_8 = N_Pf(0.111111111111111)*m_0+N_Pf(0.055555555555556)*m_1+N_Pf(0.027777777777778)*m_2+N_Pf(0.166666666666667)*m_3+N_Pf(0.083333333333333)*m_4+N_Pf(-0.166666666666667)*m_5+N_Pf(-0.083333333333333)*m_6+N_Pf(0.000000000000000)*m_7+N_Pf(-0.250000000000000)*m_8;

			// Impose boundary conditions.
			if (block_on_boundary == 1)
			{
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
				x_kap += dx_L*N_Pf(0.500000000000000);
				y_kap += dx_L*N_Pf(0.000000000000000);
				if ((I_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 0.0*0 + 0.0*0 );
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap;
						f_1 = -f_1 + N_Pf(2.0)*N_Pf(0.111111111111111)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 0.0*0 + 0.0*0 );
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 0.0*0 + 0.0*0 );
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.500000000000000);
				y_kap -= dx_L*N_Pf(0.000000000000000);
				x_kap += dx_L*N_Pf(0.500000000000000);
				y_kap += dx_L*N_Pf(0.500000000000000);
				if ((I_kap==Nbx-1)and(J_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap;
						f_5 = -f_5 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.500000000000000);
				y_kap -= dx_L*N_Pf(0.500000000000000);
				x_kap += dx_L*N_Pf(0.500000000000000);
				y_kap += dx_L*N_Pf(-0.500000000000000);
				if ((I_kap==Nbx-1)and(J_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap);
						f_8 = -f_8 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.500000000000000);
				y_kap -= dx_L*N_Pf(-0.500000000000000);

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
				x_kap += dx_L*N_Pf(0.000000000000000);
				y_kap += dx_L*N_Pf(0.500000000000000);
				if ((J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*0.05 + 1.0*0 + 0.0*0 );
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = v_kap;
						f_2 = -f_2 + N_Pf(2.0)*N_Pf(0.111111111111111)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0.05 + 1.0*0 + 0.0*0 );
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0.05 + 1.0*0 + 0.0*0 );
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.000000000000000);
				y_kap -= dx_L*N_Pf(0.500000000000000);
				x_kap += dx_L*N_Pf(0.500000000000000);
				y_kap += dx_L*N_Pf(0.500000000000000);
				if ((I_kap<Nbx-1)and(J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap;
						f_5 = -f_5 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.500000000000000);
				y_kap -= dx_L*N_Pf(0.500000000000000);
				x_kap += dx_L*N_Pf(-0.500000000000000);
				y_kap += dx_L*N_Pf(0.500000000000000);
				if ((I_kap>0)and(J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap;
						f_6 = -f_6 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(-0.500000000000000);
				y_kap -= dx_L*N_Pf(0.500000000000000);

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
				x_kap += dx_L*N_Pf(-0.500000000000000);
				y_kap += dx_L*N_Pf(0.000000000000000);
				if ((I_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 0.0*0 + 0.0*0 );
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap);
						f_3 = -f_3 + N_Pf(2.0)*N_Pf(0.111111111111111)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 0.0*0 + 0.0*0 );
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 0.0*0 + 0.0*0 );
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(-0.500000000000000);
				y_kap -= dx_L*N_Pf(0.000000000000000);
				x_kap += dx_L*N_Pf(-0.500000000000000);
				y_kap += dx_L*N_Pf(0.500000000000000);
				if ((I_kap==0)and(J_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap;
						f_6 = -f_6 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(-0.500000000000000);
				y_kap -= dx_L*N_Pf(0.500000000000000);
				x_kap += dx_L*N_Pf(-0.500000000000000);
				y_kap += dx_L*N_Pf(-0.500000000000000);
				if ((I_kap==0)and(J_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap);
						f_7 = -f_7 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(-0.500000000000000);
				y_kap -= dx_L*N_Pf(-0.500000000000000);

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
				x_kap += dx_L*N_Pf(0.000000000000000);
				y_kap += dx_L*N_Pf(-0.500000000000000);
				if ((J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*0.05 + -1.0*0 + 0.0*0 );
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-v_kap);
						f_4 = -f_4 + N_Pf(2.0)*N_Pf(0.111111111111111)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0.05 + -1.0*0 + 0.0*0 );
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0.05 + -1.0*0 + 0.0*0 );
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.000000000000000);
				y_kap -= dx_L*N_Pf(-0.500000000000000);
				x_kap += dx_L*N_Pf(-0.500000000000000);
				y_kap += dx_L*N_Pf(-0.500000000000000);
				if ((I_kap>0)and(J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap);
						f_7 = -f_7 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(-0.500000000000000);
				y_kap -= dx_L*N_Pf(-0.500000000000000);
				x_kap += dx_L*N_Pf(0.500000000000000);
				y_kap += dx_L*N_Pf(-0.500000000000000);
				if ((I_kap<Nbx-1)and(J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap);
						f_8 = -f_8 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.500000000000000);
				y_kap -= dx_L*N_Pf(-0.500000000000000);

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
				x_kap += dx_L*N_Pf(0.500000000000000);
				y_kap += dx_L*N_Pf(0.500000000000000);
				if ((I_kap==Nbx-1)and(J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap;
						f_5 = -f_5 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + 1.0*0 + 0.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.500000000000000);
				y_kap -= dx_L*N_Pf(0.500000000000000);

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
				x_kap += dx_L*N_Pf(-0.500000000000000);
				y_kap += dx_L*N_Pf(0.500000000000000);
				if ((I_kap==0)and(J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap;
						f_6 = -f_6 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + 1.0*0 + 0.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(-0.500000000000000);
				y_kap -= dx_L*N_Pf(0.500000000000000);

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
				x_kap += dx_L*N_Pf(-0.500000000000000);
				y_kap += dx_L*N_Pf(-0.500000000000000);
				if ((I_kap==0)and(J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap);
						f_7 = -f_7 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0.05 + -1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(-0.500000000000000);
				y_kap -= dx_L*N_Pf(-0.500000000000000);

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
				x_kap += dx_L*N_Pf(0.500000000000000);
				y_kap += dx_L*N_Pf(-0.500000000000000);
				if ((I_kap==Nbx-1)and(J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap);
						f_8 = -f_8 + N_Pf(2.0)*N_Pf(0.027777777777778)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				x_kap -= dx_L*N_Pf(0.500000000000000);
				y_kap -= dx_L*N_Pf(-0.500000000000000);

			}

			// Write fi* to global memory.
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8;
			__syncthreads();
		}
	}
}

int Solver_LBM::S_Collide_MRT_d2q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collide_MRT_d2q9<<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_vec[L], tau_vec_MRT[0+L*6], tau_vec_MRT[1+L*6], tau_vec_MRT[2+L*6], tau_vec_MRT[3+L*6], tau_vec_MRT[4+L*6], tau_vec_MRT[5+L*6], tau_ratio_vec_C2F[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_f_X[i_dev]);
	}

	return 0;
}

#endif