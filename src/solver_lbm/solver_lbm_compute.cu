#include "mesh.h"
#include "solver_lbm.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out)
{
	// Density and velocity computations.	
	for (int kap_i = 0; kap_i < M_TBLOCK; kap_i++)
	{
		ufloat_t rho = 0;
		ufloat_t u = 0;
		ufloat_t v = 0;
		ufloat_t w = 0;
		S_ComputeMacroProperties(i_dev, i_kap, i_Q, kap_i, rho, u, v, w);
		out[kap_i + 0*M_TBLOCK] = (double)rho;
		out[kap_i + 1*M_TBLOCK] = (double)u;
		out[kap_i + 2*M_TBLOCK] = (double)v;
		out[kap_i + 3*M_TBLOCK] = (double)w;
	}

	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out)
{
	S_ComputeProperties(i_dev, i_Q, i_kap, dx_L, out);
	
	// Vorticity calculation.
	for (int kap_i = 0; kap_i < M_TBLOCK; kap_i++)
	{
		int I_kap = kap_i % 4;
		int J_kap = (kap_i / 4) % 4;
		int K_kap = (kap_i / 4) / 4;
		
		out[kap_i + 4*M_TBLOCK] = 0.0;
		out[kap_i + 5*M_TBLOCK] = 0.0;
		out[kap_i + 6*M_TBLOCK] = 0.0;
if (N_DIM==2)
{
		// X
		if (I_kap < 4-1)
			out[kap_i + 6*M_TBLOCK] += (double)(out[(I_kap+1)+4*(J_kap) + 2*M_TBLOCK] - out[kap_i + 2*M_TBLOCK])/dx_L;
		else
			out[kap_i + 6*M_TBLOCK] += (double)(out[kap_i + 2*M_TBLOCK] - out[(I_kap-1)+4*(J_kap) + 2*M_TBLOCK])/dx_L;
		
		// Y
		if (J_kap < 4-1)
			out[kap_i + 6*M_TBLOCK] -= (double)(out[(I_kap)+4*(J_kap+1) + 1*M_TBLOCK] - out[kap_i + 1*M_TBLOCK])/dx_L;
		else
			out[kap_i + 6*M_TBLOCK] -= (double)(out[kap_i + 1*M_TBLOCK] - out[(I_kap)+4*(J_kap-1) + 1*M_TBLOCK])/dx_L;
}
else
{
		// X
		if (I_kap < 4-1)
		{
			out[kap_i + 5*M_TBLOCK] -= (double)(out[(I_kap+1)+4*(J_kap)+4*4*(K_kap) + 3*M_TBLOCK] - out[kap_i + 3*M_TBLOCK])/dx_L;
			out[kap_i + 6*M_TBLOCK] += (double)(out[(I_kap+1)+4*(J_kap)+4*4*(K_kap) + 2*M_TBLOCK] - out[kap_i + 2*M_TBLOCK])/dx_L;
		}
		else
		{
			out[kap_i + 5*M_TBLOCK] -= (double)(out[kap_i + 3*M_TBLOCK] - out[(I_kap-1)+4*(J_kap)+4*4*(K_kap) + 3*M_TBLOCK])/dx_L;
			out[kap_i + 6*M_TBLOCK] += (double)(out[kap_i + 2*M_TBLOCK] - out[(I_kap-1)+4*(J_kap)+4*4*(K_kap) + 2*M_TBLOCK])/dx_L;
		}
		
		// Y
		if (J_kap < 4-1)
		{
			out[kap_i + 4*M_TBLOCK] += (double)(out[(I_kap)+4*(J_kap+1)+4*4*(K_kap) + 3*M_TBLOCK] - out[kap_i + 3*M_TBLOCK])/dx_L;
			out[kap_i + 6*M_TBLOCK] -= (double)(out[(I_kap)+4*(J_kap+1)+4*4*(K_kap) + 1*M_TBLOCK] - out[kap_i + 1*M_TBLOCK])/dx_L;
		}
		else
		{
			out[kap_i + 4*M_TBLOCK] += (double)(out[kap_i + 3*M_TBLOCK] - out[(I_kap)+4*(J_kap-1)+4*4*(K_kap) + 3*M_TBLOCK])/dx_L;
			out[kap_i + 6*M_TBLOCK] -= (double)(out[kap_i + 1*M_TBLOCK] - out[(I_kap)+4*(J_kap-1)+4*4*(K_kap) + 1*M_TBLOCK])/dx_L;
		}
		
		// Z
		if (K_kap < 4-1)
		{
			out[kap_i + 4*M_TBLOCK] -= (double)(out[(I_kap)+4*(J_kap)+4*4*(K_kap+1) + 2*M_TBLOCK] - out[kap_i + 2*M_TBLOCK])/dx_L;
			out[kap_i + 5*M_TBLOCK] += (double)(out[(I_kap)+4*(J_kap)+4*4*(K_kap+1) + 1*M_TBLOCK] - out[kap_i + 1*M_TBLOCK])/dx_L;
		}
		else
		{
			out[kap_i + 4*M_TBLOCK] -= (double)(out[kap_i + 2*M_TBLOCK] - out[(I_kap)+4*(J_kap)+4*4*(K_kap-1) + 2*M_TBLOCK])/dx_L;
			out[kap_i + 5*M_TBLOCK] += (double)(out[kap_i + 1*M_TBLOCK] - out[(I_kap)+4*(J_kap)+4*4*(K_kap-1) + 1*M_TBLOCK])/dx_L;
		}
}
	}
	
	return 0;
}

/*
template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeForces_Legacy(int i_dev, int L)
{
	ufloat_t Fxp = (ufloat_t)(0.0);
	ufloat_t Fyp = (ufloat_t)(0.0);
	ufloat_t Fxm = (ufloat_t)(0.0);
	ufloat_t Fym = (ufloat_t)(0.0);	
	ufloat_t dx_L = mesh->dxf_vec[L];
	int total_blocks = 0;
	
	for (int kap = 0; kap < mesh->n_ids[i_dev][L]; kap++)
	{
		int i_kap_b = mesh->id_set[i_dev][L * n_maxcblocks + kap];
		bool block_involved = false;
		
		for (int j_q = 0; j_q < Nqx; j_q++)
		{
			for (int i_q = 0; i_q < Nqx; i_q++)
			{
				int i_Q = i_q + Nqx*j_q;
				
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					int I_kap = kap_i % 4;
					int J_kap = (kap_i / 4) % 4;
					
					ufloat_t f_1;
					ufloat_t f_2;
					ufloat_t f_3;
					ufloat_t f_4;
					ufloat_t f_5;
					ufloat_t f_6;
					ufloat_t f_7;
					ufloat_t f_8;
					
					if (CM == CM_BGK)
					{
						f_1 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 3*n_maxcells];
						f_2 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 4*n_maxcells];
						f_3 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 1*n_maxcells];
						f_4 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 2*n_maxcells];
						f_5 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 7*n_maxcells];
						f_6 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 8*n_maxcells];
						f_7 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 5*n_maxcells];
						f_8 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 6*n_maxcells];
					}
					
					if (CM == CM_MRT)
					{
						ufloat_t m_0 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 0*n_maxcells];
						ufloat_t m_1 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 3*n_maxcells];
						ufloat_t m_2 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 4*n_maxcells];
						ufloat_t m_3 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 1*n_maxcells];
						ufloat_t m_4 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 2*n_maxcells];
						ufloat_t m_5 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 7*n_maxcells];
						ufloat_t m_6 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 8*n_maxcells];
						ufloat_t m_7 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 5*n_maxcells];
						ufloat_t m_8 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 6*n_maxcells];
						f_1 = (ufloat_t)(0.111111111111111)*m_0+(ufloat_t)(-0.027777777777778)*m_1+(ufloat_t)(-0.055555555555556)*m_2+(ufloat_t)(0.166666666666667)*m_3+(ufloat_t)(-0.166666666666667)*m_4+(ufloat_t)(0.000000000000000)*m_5+(ufloat_t)(0.000000000000000)*m_6+(ufloat_t)(0.250000000000000)*m_7+(ufloat_t)(0.000000000000000)*m_8;
						f_2 = (ufloat_t)(0.111111111111111)*m_0+(ufloat_t)(-0.027777777777778)*m_1+(ufloat_t)(-0.055555555555556)*m_2+(ufloat_t)(0.000000000000000)*m_3+(ufloat_t)(-0.000000000000000)*m_4+(ufloat_t)(0.166666666666667)*m_5+(ufloat_t)(-0.166666666666667)*m_6+(ufloat_t)(-0.250000000000000)*m_7+(ufloat_t)(0.000000000000000)*m_8;
						f_3 = (ufloat_t)(0.111111111111111)*m_0+(ufloat_t)(-0.027777777777778)*m_1+(ufloat_t)(-0.055555555555556)*m_2+(ufloat_t)(-0.166666666666667)*m_3+(ufloat_t)(0.166666666666667)*m_4+(ufloat_t)(0.000000000000000)*m_5+(ufloat_t)(0.000000000000000)*m_6+(ufloat_t)(0.250000000000000)*m_7+(ufloat_t)(0.000000000000000)*m_8;
						f_4 = (ufloat_t)(0.111111111111111)*m_0+(ufloat_t)(-0.027777777777778)*m_1+(ufloat_t)(-0.055555555555556)*m_2+(ufloat_t)(0.000000000000000)*m_3+(ufloat_t)(0.000000000000000)*m_4+(ufloat_t)(-0.166666666666667)*m_5+(ufloat_t)(0.166666666666667)*m_6+(ufloat_t)(-0.250000000000000)*m_7+(ufloat_t)(0.000000000000000)*m_8;
						f_5 = (ufloat_t)(0.111111111111111)*m_0+(ufloat_t)(0.055555555555556)*m_1+(ufloat_t)(0.027777777777778)*m_2+(ufloat_t)(0.166666666666667)*m_3+(ufloat_t)(0.083333333333333)*m_4+(ufloat_t)(0.166666666666667)*m_5+(ufloat_t)(0.083333333333333)*m_6+(ufloat_t)(0.000000000000000)*m_7+(ufloat_t)(0.250000000000000)*m_8;
						f_6 = (ufloat_t)(0.111111111111111)*m_0+(ufloat_t)(0.055555555555556)*m_1+(ufloat_t)(0.027777777777778)*m_2+(ufloat_t)(-0.166666666666667)*m_3+(ufloat_t)(-0.083333333333333)*m_4+(ufloat_t)(0.166666666666667)*m_5+(ufloat_t)(0.083333333333333)*m_6+(ufloat_t)(0.000000000000000)*m_7+(ufloat_t)(-0.250000000000000)*m_8;
						f_7 = (ufloat_t)(0.111111111111111)*m_0+(ufloat_t)(0.055555555555556)*m_1+(ufloat_t)(0.027777777777778)*m_2+(ufloat_t)(-0.166666666666667)*m_3+(ufloat_t)(-0.083333333333333)*m_4+(ufloat_t)(-0.166666666666667)*m_5+(ufloat_t)(-0.083333333333333)*m_6+(ufloat_t)(0.000000000000000)*m_7+(ufloat_t)(0.250000000000000)*m_8;
						f_8 = (ufloat_t)(0.111111111111111)*m_0+(ufloat_t)(0.055555555555556)*m_1+(ufloat_t)(0.027777777777778)*m_2+(ufloat_t)(0.166666666666667)*m_3+(ufloat_t)(0.083333333333333)*m_4+(ufloat_t)(-0.166666666666667)*m_5+(ufloat_t)(-0.083333333333333)*m_6+(ufloat_t)(0.000000000000000)*m_7+(ufloat_t)(-0.250000000000000)*m_8;
					}
			
					int nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 1*n_maxcblocks];
					// nbr 1.
					if (i_q == Nqx-1)
					{
							// p = 3.
						if (((I_kap+1==4)) && nbr_kap_b == -8)
						{
							Fxm += ((+1.0))*(2.0*f_3);
							Fyp += (0)*(2.0*f_3);
						}
							// p = 7.
						if (((I_kap+1==4) && (J_kap+1< 4)) && nbr_kap_b == -8)
						{
							Fxm += ((+1.0))*(2.0*f_7);
							Fym += ((+1.0))*(2.0*f_7);
						}
							// p = 6.
						if (((I_kap+1==4) && (J_kap-1>= 0)) && nbr_kap_b == -8)
						{
							Fxm += ((+1.0))*(2.0*f_6);
							Fyp += (1.0)*(2.0*f_6);
						}
					}
					// nbr 2.
					if (j_q == Nqx-1)
					{
						nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 2*n_maxcblocks];
							// p = 4.
						if (((J_kap+1==4)) && nbr_kap_b == -8)
						{
							Fxp += (0)*(2.0*f_4);
							Fym += ((+1.0))*(2.0*f_4);
						}
							// p = 7.
						if (((I_kap+1< 4) && (J_kap+1==4)) && nbr_kap_b == -8)
						{
							Fxm += ((+1.0))*(2.0*f_7);
							Fym += ((+1.0))*(2.0*f_7);
						}
							// p = 8.
						if (((I_kap-1>= 0) && (J_kap+1==4)) && nbr_kap_b == -8)
						{
							Fxp += (1.0)*(2.0*f_8);
							Fym += ((+1.0))*(2.0*f_8);
						}
					}
					// nbr 3.
					if (i_q == 0)
					{
						nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 3*n_maxcblocks];
							// p = 1.
						if (((I_kap-1==-1)) && nbr_kap_b == -8)
						{
							Fxp += (1.0)*(2.0*f_1);
							Fyp += (0)*(2.0*f_1);
						}
							// p = 8.
						if (((I_kap-1==-1) && (J_kap+1< 4)) && nbr_kap_b == -8)
						{
							Fxp += (1.0)*(2.0*f_8);
							Fym += ((+1.0))*(2.0*f_8);
						}
							// p = 5.
						if (((I_kap-1==-1) && (J_kap-1>= 0)) && nbr_kap_b == -8)
						{
							Fxp += (1.0)*(2.0*f_5);
							Fyp += (1.0)*(2.0*f_5);
						}
					}
					// nbr 4.
					if (j_q == 0)
					{
						nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 4*n_maxcblocks];
							// p = 2.
						if (((J_kap-1==-1)) && nbr_kap_b == -8)
						{
							Fxp += (0)*(2.0*f_2);
							Fyp += (1.0)*(2.0*f_2);
						}
							// p = 5.
						if (((I_kap-1>= 0) && (J_kap-1==-1)) && nbr_kap_b == -8)
						{
							Fxp += (1.0)*(2.0*f_5);
							Fyp += (1.0)*(2.0*f_5);
						}
							// p = 6.
						if (((I_kap+1< 4) && (J_kap-1==-1)) && nbr_kap_b == -8)
						{
							Fxm += ((+1.0))*(2.0*f_6);
							Fyp += (1.0)*(2.0*f_6);
						}
					}
					// nbr 5.
					if (i_q == Nqx-1 && j_q == Nqx-1)
					{
						nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 5*n_maxcblocks];
							// p = 7.
						if (((I_kap+1==4) && (J_kap+1==4)) && nbr_kap_b == -8)
						{
							Fxm += ((+1.0))*(2.0*f_7);
							Fym += ((+1.0))*(2.0*f_7);
						}
					}
					// nbr 6.
					if (i_q == 0 && j_q == Nqx-1)
					{
						nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 6*n_maxcblocks];
							// p = 8.
						if (((I_kap-1==-1) && (J_kap+1==4)) && nbr_kap_b == -8)
						{
							Fxp += (1.0)*(2.0*f_8);
							Fym += ((+1.0))*(2.0*f_8);
						}
					}
					// nbr 7.
					if (i_q == 0 && j_q == 0)
					{
						nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 7*n_maxcblocks];
							// p = 5.
						if (((I_kap-1==-1) && (J_kap-1==-1)) && nbr_kap_b == -8)
						{
							Fxp += (1.0)*(2.0*f_5);
							Fyp += (1.0)*(2.0*f_5);
						}
					}
					// nbr 8.
					if (i_q == Nqx-1 && j_q == 0)
					{
						nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 8*n_maxcblocks];
							// p = 6.
						if (((I_kap+1==4) && (J_kap-1==-1)) && nbr_kap_b == -8)
						{
							Fxm += ((+1.0))*(2.0*f_6);
							Fyp += (1.0)*(2.0*f_6);
						}
					}
					
					for (int p = 0; p < 9; p++)
					{
						if (mesh->cblock_ID_nbr[i_dev][i_kap_b + p*n_maxcblocks] == -8)
							block_involved = true;
					}
				}
			}
		}
		
		if (block_involved)
			total_blocks++;
	}
	
	Fxp *= dx_L;
	Fyp *= dx_L;
	Fxm *= dx_L;
	Fym *= dx_L;
	ufloat_t Fx = Fxp - Fxm;
	ufloat_t Fy = Fyp - Fym;
	
	ufloat_t Dp = 1.0/32.0;
	//std::cout << std::setprecision(16) << "Fxm: " << Fxm << ", Fxp: " << Fxp << std::endl;
	//std::cout << std::setprecision(16) << "Fym: " << Fym << ", Fyp: " << Fyp << std::endl;
	//std::cout << "Fx: " << Fx << ", CD: " << 2.0*Fx / (0.05*0.05*(Dp)) << std::endl;
	//std::cout << "Fy: " << Fy << ", CL: " << 2.0*Fy / (0.05*0.05*(Dp)) << std::endl;
	mesh->to.force_printer << 2.0*Fx / (0.05*0.05*(Dp)) << " " << 2.0*Fy / (0.05*0.05*(Dp)) << std::endl;
	
	return 0;
}
*/
