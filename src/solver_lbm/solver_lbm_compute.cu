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

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ReportForcesLBM(int i_dev, int L, int iter, double t, int START)
{
	if (START == 1 && L == -1)
	{
		// Synchronize GPU.
		cudaDeviceSynchronize();
		
		double factor = 1.0;
		if (S_FORCE_TYPE==0) factor = 1.0/dxf_vec[MAX_LEVELS-1];
		if (S_FORCE_TYPE==1) factor = 1.0/dxf_vec[N_LEVEL_START];
		int cblocks_id_max = mesh->id_max[i_dev][MAX_LEVELS];
		
		double Fpx = thrust::reduce(
			thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][0*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][0*n_maxcblocks] + cblocks_id_max, 0.0
		);
		double Fmx = thrust::reduce(
			thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][1*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][1*n_maxcblocks] + cblocks_id_max, 0.0
		);
		double Fpy = thrust::reduce(
			thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][2*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][2*n_maxcblocks] + cblocks_id_max, 0.0
		);
		double Fmy = thrust::reduce(
			thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][3*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][3*n_maxcblocks] + cblocks_id_max, 0.0
		);
		double Fpz = thrust::reduce(
			thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][4*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][4*n_maxcblocks] + cblocks_id_max, 0.0
		);
		double Fmz = thrust::reduce(
			thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][5*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][5*n_maxcblocks] + cblocks_id_max, 0.0
		);
		double Fx = factor*(Fpx - Fmx);
		double Fy = factor*(Fpy - Fmy);
		double Fz = factor*(Fpz - Fmz);
		double Dp = 1.0/32.0;
		double uin = 0.05;
		std::cout << "Report:" << std::endl;
		std::cout << "CD: " << 2.0*Fx / (uin*uin*(Dp)) << "   " << 8.0*Fx / (uin*uin*(M_PI*Dp*Dp)) << std::endl;
		std::cout << "CL: " << 2.0*Fy / (uin*uin*(Dp)) << std::endl;
		mesh->to.force_printer << iter << " " << t << " " << 2.0*Fx / (uin*uin*(Dp)) << " " << 2.0*Fy / (uin*uin*(Dp)) << " " << 8.0*Fx / (uin*uin*(M_PI*Dp*Dp)) << " " << 8.0*Fy / (uin*uin*(M_PI*Dp*Dp)) << std::endl;
		
		// Reset the forces array in-case some blocks were coarsened.
		Cu_ResetToValue<<<(M_BLOCK+6*n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, mesh->streams[i_dev]>>>(6*n_maxcblocks, mesh->c_cblock_f_Ff[i_dev], (ufloat_t)0);
	}
	
	return 0;
}
