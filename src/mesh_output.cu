#include "mesh.h"
#include "solver.h"

int Mesh::M_LoadToGPU()
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		int cblocks_id_max = id_max[i_dev][MAX_LEVELS];
		long int cells_id_max = id_max[i_dev][MAX_LEVELS]*M_CBLOCK;


		// Floating point arrays.
		for (int p = 0; p < N_Q; p++)
			gpuErrchk( cudaMemcpy(&c_cells_f_F[i_dev][p*n_maxcells], &cells_f_F[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );
		for (int d = 0; d < N_DIM; d++)
			gpuErrchk( cudaMemcpy(&c_cblock_f_X[i_dev][d*n_maxcblocks], &cblock_f_X[i_dev][d*n_maxcblocks], cblocks_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );	


		// Connectivity arrays.
		for (int p = 0; p < N_Q_max; p++)
		{
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr[i_dev][p*n_maxcblocks], &cblock_ID_nbr[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &cblock_ID_nbr_child[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		}
		gpuErrchk( cudaMemcpy(c_cblock_ID_onb[i_dev], cblock_ID_onb[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );

		
		// Metadata arrays.
		gpuErrchk( cudaMemcpy(c_cells_ID_mask[i_dev], cells_ID_mask[i_dev], cells_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_ID_mask[i_dev], cblock_ID_mask[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_level[i_dev], cblock_level[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );

		for (int L = 0; L < MAX_LEVELS; L++)
			gpuErrchk( cudaMemcpy(&c_id_set[i_dev][L*n_maxcblocks], &id_set[i_dev][L*n_maxcblocks], n_ids[i_dev][L]*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_gap_set[i_dev], gap_set[i_dev], n_gaps[i_dev]*sizeof(int), cudaMemcpyHostToDevice) );
	}
	
	return 0;
}

int Mesh::M_RetrieveFromGPU()
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		int cblocks_id_max = id_max[i_dev][MAX_LEVELS];
		long int cells_id_max = id_max[i_dev][MAX_LEVELS]*M_CBLOCK;


		// Floating point arrays.
		for (int p = 0; p < N_Q; p++)
			gpuErrchk( cudaMemcpy(&cells_f_F[i_dev][p*n_maxcells], &c_cells_f_F[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );
		for (int d = 0; d < N_DIM; d++)
			gpuErrchk( cudaMemcpy(&cblock_f_X[i_dev][d*n_maxcblocks], &c_cblock_f_X[i_dev][d*n_maxcblocks], cblocks_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );


		// Connectivity arrays.
		for (int p = 0; p < N_Q_max; p++)
		{
			gpuErrchk( cudaMemcpy(&cblock_ID_nbr[i_dev][p*n_maxcblocks], &c_cblock_ID_nbr[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
			gpuErrchk( cudaMemcpy(&cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &c_cblock_ID_nbr_child[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		}
		gpuErrchk( cudaMemcpy(cblock_ID_onb[i_dev], c_cblock_ID_onb[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );


		// Metadata arrays.
		gpuErrchk( cudaMemcpy(cells_ID_mask[i_dev], c_cells_ID_mask[i_dev], cells_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_ID_mask[i_dev], c_cblock_ID_mask[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_ID_ref[i_dev], c_cblock_ID_ref[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_level[i_dev], c_cblock_level[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		
		for (int L = 0; L < MAX_LEVELS; L++)
			gpuErrchk( cudaMemcpy(&id_set[i_dev][L*n_maxcblocks], &c_id_set[i_dev][L*n_maxcblocks], n_ids[i_dev][L]*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(gap_set[i_dev], c_gap_set[i_dev], n_gaps[i_dev]*sizeof(int), cudaMemcpyDeviceToHost) );
	}

	return 0;
}

int Mesh::M_ComputeProperties(int i_dev, int i_kap, ufloat_t dx_L, double *out_u)
{
	solver->S_ComputeProperties(i_dev, i_kap, dx_L, out_u);

	return 0;
}

int Mesh::M_ComputeOutputProperties(int i_dev, int i_kap, ufloat_t dx_L, double *out_u)
{
	solver->S_ComputeOutputProperties(i_dev, i_kap, dx_L, out_u);
	
	return 0;
}

int Mesh::M_ComputeForces(int i_dev, int L, std::ofstream *out)
{
	solver->S_ComputeForces(i_dev, L, out);
	
	return 0;
}

// NOTE: Only works with N_LEVEL_START=0 for now.
int Mesh::M_UpdateMeanVelocities(int i_dev, int N_iters_ave)
{
	ufloat_t rho_kap = N_Pf(0.0);
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
#if (N_DIM==3)
	ufloat_t w_kap = N_Pf(0.0);
#endif
	double out_u[M_CBLOCK*(3+1)];
	
	for (int kap = 0; kap < n_ids[i_dev][0]; kap++)
	{
		int i_kap = id_set[i_dev][kap];
		
		// Density and velocity computations.
		M_ComputeProperties(i_dev, i_kap, dxf_vec[0], out_u);
		
		// Update mean values.
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
		{
			rho_kap = (ufloat_t)out_u[kap_i + 0*M_CBLOCK];
			u_kap = (ufloat_t)out_u[kap_i + 1*M_CBLOCK];
			v_kap = (ufloat_t)out_u[kap_i + 2*M_CBLOCK];
#if (N_DIM==3)
			w_kap = (ufloat_t)out_u[kap_i + 3*M_CBLOCK];
#endif
			
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_ids[i_dev][0]*M_CBLOCK] = 
				((ufloat_t)N_iters_ave*cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_ids[i_dev][0]*M_CBLOCK] + rho_kap) / ((ufloat_t)N_iters_ave+1.0);
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_ids[i_dev][0]*M_CBLOCK] = 
				((ufloat_t)N_iters_ave*cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_ids[i_dev][0]*M_CBLOCK] + u_kap) / ((ufloat_t)N_iters_ave+1.0);
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_ids[i_dev][0]*M_CBLOCK] = 
				((ufloat_t)N_iters_ave*cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_ids[i_dev][0]*M_CBLOCK] + v_kap) / ((ufloat_t)N_iters_ave+1.0);
#if (N_DIM==2)
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_ids[i_dev][0]*M_CBLOCK] = N_Pf(0.0);
#else
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_ids[i_dev][0]*M_CBLOCK] = 
				((ufloat_t)N_iters_ave*cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_ids[i_dev][0]*M_CBLOCK] + w_kap) / ((ufloat_t)N_iters_ave+1.0);
#endif
		}
	}
	
	return 0;
}

ufloat_t Mesh::M_CheckConvergence(int i_dev)
{
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
#if (N_DIM==3)
	ufloat_t w_kap = N_Pf(0.0);
#endif
	ufloat_t sum = N_Pf(0.0);
	ufloat_t norm = N_Pf(0.0);
	double out_u[M_CBLOCK*(7+1)];
	
	for (int kap = 0; kap < n_ids_probed[i_dev]; kap++)
	{
		int i_kap = id_set_probed[i_dev][kap];
	
		// Density and velocity computations.
		M_ComputeProperties(i_dev, i_kap, dxf_vec[0], out_u);
		
		// Check convergence of velocity magnitudes at probed locations.
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
		{
			u_kap = (ufloat_t)out_u[kap_i + 1*M_CBLOCK];
			v_kap = (ufloat_t)out_u[kap_i + 2*M_CBLOCK];
#if (N_DIM==3)
			w_kap = (ufloat_t)out_u[kap_i + 3*M_CBLOCK];
#endif
			
#if (N_DIM==2)
			ufloat_t u_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 0*n_ids_probed[i_dev]*M_CBLOCK];
			ufloat_t v_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 1*n_ids_probed[i_dev]*M_CBLOCK];
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 0*n_ids_probed[i_dev]*M_CBLOCK] = u_kap;
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 1*n_ids_probed[i_dev]*M_CBLOCK] = v_kap;
			sum += (u_kap-u_kap_prev)*(u_kap-u_kap_prev) + (v_kap-v_kap_prev)*(v_kap-v_kap_prev);
			norm += u_kap*u_kap + v_kap*v_kap;
			
			
#else
			ufloat_t u_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 0*n_ids_probed[i_dev]*M_CBLOCK];
			ufloat_t v_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 1*n_ids_probed[i_dev]*M_CBLOCK];
			ufloat_t w_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 2*n_ids_probed[i_dev]*M_CBLOCK];
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 0*n_ids_probed[i_dev]*M_CBLOCK] = u_kap;
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 1*n_ids_probed[i_dev]*M_CBLOCK] = v_kap;
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 2*n_ids_probed[i_dev]*M_CBLOCK] = w_kap;
			sum += (u_kap-u_kap_prev)*(u_kap-u_kap_prev) + (v_kap-v_kap_prev)*(v_kap-v_kap_prev) + (w_kap-w_kap_prev)*(w_kap-w_kap_prev);
			norm += u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
#endif
		}
	}

	sum = sqrt(sum/norm);
	std::cout << "Residue: " << sum << std::endl; 
	return sum;
}
