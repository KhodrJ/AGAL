/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#define APPEND_NQ(NAME) { NAME ## _d ## N_DIM ## q ## N_Q }
#include "solver.h"


/*
 .d8888b.           888                           
d88P  Y88b          888                           
Y88b.               888                           
 "Y888b.    .d88b.  888 888  888  .d88b.  888d888 
    "Y88b. d88""88b 888 888  888 d8P  Y8b 888P"   
      "888 888  888 888 Y88  88P 88888888 888     
Y88b  d88P Y88..88P 888  Y8bd8P  Y8b.     888     
 "Y8888P"   "Y88P"  888   Y88P    "Y8888  888     
*/

int Solver_LBM::S_Advance(int i_dev, int L, std::ofstream *file, double *tmp)
{
#if (MAX_LEVELS>1 && (MAX_LEVELS!=N_LEVEL_START+1))
	if (L == N_LEVEL_START)
	{
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Interpolating from level " << 0 << " to " << 1 << "..." << std::endl;
		#endif
		tic_simple("");
		// |
		S_Interpolate(i_dev, L, V_INTERP_INTERFACE) ;//, N_Pf(0.5)*tau_vec[1]/tau_vec[0]);
		// |
		cudaDeviceSynchronize();
		tmp[0 + L*4] += toc_simple("",T_US,0);
		
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Colliding and streaming on level " << 0 << "..." << std::endl;
		#endif
		tic_simple("");
		// |
		S_Collide(i_dev, L);
		// |
		cudaDeviceSynchronize();
		tmp[1 + L*4] += toc_simple("",T_US,0);
		//
		tic_simple("");
		// |
		S_Stream(i_dev, L);
		// |
		cudaDeviceSynchronize();
		tmp[2 + L*4] += toc_simple("",T_US,0);
		
		S_Advance(i_dev, L+1, file, tmp);
		
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Averaging from level " << 1 << " to " << 0 << "..." << std::endl;
		#endif
		tic_simple("");
		// |
		S_Average(i_dev, L, V_AVERAGE_INTERFACE) ;//, N_Pf(2.0)*tau_vec[L]/tau_vec[L+1]);
		// |
		cudaDeviceSynchronize();
		tmp[3 + L*4] += toc_simple("",T_US,0);
		
		double tot_time = 0.0;
		long int tot_cells = 0;
		int multip = 1;
		for (int Lp = N_LEVEL_START; Lp < 4*MAX_LEVELS; Lp++)
		{
			tot_time += tmp[Lp];
			*file << tmp[Lp] << " ";
		}
		for (int Lp = N_LEVEL_START; Lp < MAX_LEVELS-1; Lp++)
		{
			tot_cells += multip*(mesh->n_ids[i_dev][Lp] - mesh->n_ids[i_dev][Lp+1]/N_CHILDREN);
			multip *= 2;
		}
		tot_cells += multip*(mesh->n_ids[i_dev][MAX_LEVELS-1]);
		*file << (1.0 / tot_time)*(double)(tot_cells*M_CBLOCK);
		*file << std::endl;
	}
	else
	{
		if (L < MAX_LEVELS-1)
		{
			#if (P_SHOW_ADVANCE==1)
			std::cout << "Interpolating from level " << L << " to " << L+1 << "..." << std::endl;
			#endif
			tic_simple("");
			// |
			S_Interpolate(i_dev, L, V_INTERP_INTERFACE) ;//, N_Pf(0.5)*tau_vec[L+1]/tau_vec[L]);
			// |
			cudaDeviceSynchronize();
			tmp[0 + L*4] += toc_simple("",T_US,0);
		}
		
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Colliding and streaming on level " << L << "..." << std::endl;
		#endif
		tic_simple("");
		// |
		S_Collide(i_dev, L);
		// |
		cudaDeviceSynchronize();
		tmp[1 + L*4] += toc_simple("",T_US,0);
		tic_simple("");
		// |
		S_Stream(i_dev, L);
		// |
		cudaDeviceSynchronize();
		tmp[2 + L*4] += toc_simple("",T_US,0);
		
		if (L < MAX_LEVELS-1)
		{
			S_Advance(i_dev, L+1, file, tmp);
			
			#if (P_SHOW_ADVANCE==1)
			std::cout << "Averaging from level " << L+1 << " to " << L << "..." << std::endl;
			#endif
			tic_simple("");
			// |
			S_Average(i_dev, L, V_AVERAGE_INTERFACE); //, N_Pf(2.0)*tau_vec[L]/tau_vec[L+1]);
			// |
			cudaDeviceSynchronize();
			tmp[3 + 4*L] += toc_simple("",T_US,0);
			
			#if (P_SHOW_ADVANCE==1)
			std::cout << "Interpolating from level " << L << " to " << L+1 << "..." << std::endl;
			#endif
			tic_simple("");
			// |
			S_Interpolate(i_dev, L, V_INTERP_INTERFACE); //, N_Pf(0.5)*tau_vec[L+1]/tau_vec[L]);
			// |
			cudaDeviceSynchronize();
			tmp[0 + 4*L] += toc_simple("",T_US,0);
		}
		
		#if (P_SHOW_ADVANCE==1)
		std::cout << "Colliding and streaming on level " << L << "..." << std::endl;
		#endif
		tic_simple("");
		// |
		S_Collide(i_dev, L);
		// |
		cudaDeviceSynchronize();
		tmp[1 + 4*L] += toc_simple("",T_US,0);
		tic_simple("");
		// |
		S_Stream(i_dev, L);
		// |
		cudaDeviceSynchronize();
		tmp[2 + 4*L] += toc_simple("",T_US,0);
		
		if (L < MAX_LEVELS-1)
		{
			S_Advance(i_dev, L+1, file, tmp);
			
			#if (P_SHOW_ADVANCE==1)
			std::cout << "Averaging from level " << L+1 << " to " << L << "..." << std::endl;
			#endif
			tic_simple("");
			// |
			S_Average(i_dev, L, V_AVERAGE_INTERFACE); //, N_Pf(2.0)*tau_vec[L]/tau_vec[L+1]);
			// |
			cudaDeviceSynchronize();
			tmp[3 + 4*L] += toc_simple("",T_US,0);
		}
	}
#else
//#if (MAX_LEVELS==1 || (MAX_LEVELS>1 && MAX_LEVELS==N_LEVEL_START+1))
	tic_simple("");
	// |
	S_Collide(i_dev, L);
	// |
	cudaDeviceSynchronize();
	tmp[1] += toc_simple("",T_US,0);
	tic_simple("");
	// |
	S_Stream(i_dev, L);
	// |
	cudaDeviceSynchronize();
	tmp[2] += toc_simple("",T_US,0);
	
	for (int Lp = 0; Lp < 4; Lp++)
		*file << tmp[Lp] << " ";
	*file << std::endl;
#endif
	
	return 0;
}





int Solver_LBM::S_Init()
{
	// Compute the relaxation rate to be applied for all grid levels.
	for (int L = 0; L < MAX_LEVELS; L++)
	{
		dx_vec[L] = (ufloat_t)mesh->dxf_vec[L];
		tau_vec[L] = (ufloat_t)(v0*3.0 + 0.5*dx_vec[L]);
	}
	
	// Compute relaxation rate ratios. If LES is selected, store coarse physical viscosity instead for intermediate calculations.
	for (int L = 0; L < MAX_LEVELS-1; L++)
	{
		tau_ratio_vec_C2F[L] = tau_vec[L+1]/tau_vec[L];
		tau_ratio_vec_F2C[L] = tau_vec[L]/tau_vec[L+1];
	}

	return 0;
}

int Solver_LBM::S_Initialize(int i_dev, int L)
{
#if (N_Q==9)
	S_SetInitialConditions_d2q9(i_dev, L);
#elif (N_Q==19)
	S_SetInitialConditions_d3q19(i_dev, L);
#else // (N_Q==27)
	S_SetInitialConditions_d3q27(i_dev, L);
#endif
	
	return 0;
}

int Solver_LBM::S_Collide(int i_dev, int L)
{
#if (N_Q==9)
	S_Collide_d2q9(i_dev, L);
#elif (N_Q==19)
	S_Collide_d3q19(i_dev, L);
#else // (N_Q==27)
	S_Collide_d3q27(i_dev, L);
#endif
	
	return 0;
}

int Solver_LBM::S_Stream(int i_dev, int L)
{
#if (N_Q==9)
	S_Stream_Inpl_d2q9(i_dev, L);
#elif (N_Q==19)
	S_Stream_Inpl_d3q19(i_dev, L);
#else // (N_Q==27)
	S_Stream_Inpl_d3q27(i_dev, L);
#endif
	
	return 0;
}
