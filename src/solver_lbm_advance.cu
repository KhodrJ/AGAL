#include "mesh.h"
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

int Solver_LBM::S_Advance(int i_dev, int L, double *tmp)
{
	if (MAX_LEVELS>1 && (MAX_LEVELS!=N_LEVEL_START+1))
	{
		if (L == N_LEVEL_START)
		{
#if (P_SHOW_ADVANCE==1)
			tic_simple("");
#endif
			// |
			if (S_INTERP_HYBRID==0)
				S_Interpolate(i_dev, L, V_INTERP_ADVANCE);
			// |
#if (P_SHOW_ADVANCE==1)
			cudaDeviceSynchronize();
			tmp[0 + L*4] += toc_simple("",T_US,0);
			
			tic_simple("");
#endif
			// |
			S_Collide(i_dev, L);
			// |
#if (P_SHOW_ADVANCE==1)
			cudaDeviceSynchronize();
			tmp[1 + L*4] += toc_simple("",T_US,0);
			//
			tic_simple("");
#endif
			// |
			S_Stream(i_dev, L);
			// |
#if (P_SHOW_ADVANCE==1)
			cudaDeviceSynchronize();
			tmp[2 + L*4] += toc_simple("",T_US,0);
#endif
			
			S_Advance(i_dev, L+1, tmp);
			
#if (P_SHOW_ADVANCE==1)
			tic_simple("");
#endif
			// |
			S_Average(i_dev, L, V_AVERAGE_ADVANCE);
			// |
#if (P_SHOW_ADVANCE==1)
			cudaDeviceSynchronize();
			tmp[3 + L*4] += toc_simple("",T_US,0);
			
			double tot_time = 0.0;
			long int tot_cells = 0;
			int multip = 1;
			for (int Lp = N_LEVEL_START; Lp < 4*MAX_LEVELS; Lp++)
			{
				tot_time += tmp[Lp];
				mesh->to.adv_printer << tmp[Lp] << " ";
			}
			for (int Lp = N_LEVEL_START; Lp < MAX_LEVELS-1; Lp++)
			{
				tot_cells += multip*(mesh->n_ids[i_dev][Lp] - mesh->n_ids[i_dev][Lp+1]/N_CHILDREN);
				multip *= 2;
			}
			tot_cells += multip*(mesh->n_ids[i_dev][MAX_LEVELS-1]);
			mesh->to.adv_printer << (1.0 / tot_time)*(double)(tot_cells*M_CBLOCK);
			mesh->to.adv_printer << std::endl;
#endif
		}
		else
		{
			if (L < MAX_LEVELS-1)
			{
#if (P_SHOW_ADVANCE==1)
				tic_simple("");
#endif
				// |
				if (S_INTERP_HYBRID==0)
					S_Interpolate(i_dev, L, V_INTERP_ADVANCE);
				// |
#if (P_SHOW_ADVANCE==1)
				cudaDeviceSynchronize();
				tmp[0 + L*4] += toc_simple("",T_US,0);
#endif
			}
			
#if (P_SHOW_ADVANCE==1)
			tic_simple("");
#endif
			// |
			S_Collide(i_dev, L);
			// |
#if (P_SHOW_ADVANCE==1)
			cudaDeviceSynchronize();
			tmp[1 + L*4] += toc_simple("",T_US,0);
			tic_simple("");
#endif
			// |
			S_Stream(i_dev, L);
			// |
#if (P_SHOW_ADVANCE==1)
			cudaDeviceSynchronize();
			tmp[2 + L*4] += toc_simple("",T_US,0);
#endif
			
			if (L < MAX_LEVELS-1)
			{
				S_Advance(i_dev, L+1, tmp);
				
#if (P_SHOW_ADVANCE==1)
				tic_simple("");
#endif
				// |
				S_Average(i_dev, L, V_AVERAGE_ADVANCE);
				// |
#if (P_SHOW_ADVANCE==1)
				cudaDeviceSynchronize();
				tmp[3 + 4*L] += toc_simple("",T_US,0);
				
				tic_simple("");
#endif
				// |
				if (S_INTERP_HYBRID==0)
					S_Interpolate(i_dev, L, V_INTERP_ADVANCE);
				// |
#if (P_SHOW_ADVANCE==1)
				cudaDeviceSynchronize();
				tmp[0 + 4*L] += toc_simple("",T_US,0);
#endif
			}
			
#if (P_SHOW_ADVANCE==1)
			tic_simple("");
#endif
			// |
			S_Collide(i_dev, L);
			// |
#if (P_SHOW_ADVANCE==1)
			cudaDeviceSynchronize();
			tmp[1 + 4*L] += toc_simple("",T_US,0);
			tic_simple("");
#endif
			// |
			S_Stream(i_dev, L);
			// |
#if (P_SHOW_ADVANCE==1)
			cudaDeviceSynchronize();
			tmp[2 + 4*L] += toc_simple("",T_US,0);
#endif
			
			if (L < MAX_LEVELS-1)
			{
				S_Advance(i_dev, L+1, tmp);
				
#if (P_SHOW_ADVANCE==1)
				tic_simple("");
#endif
				// |
				S_Average(i_dev, L, V_AVERAGE_ADVANCE);
				// |
#if (P_SHOW_ADVANCE==1)
				cudaDeviceSynchronize();
				tmp[3 + 4*L] += toc_simple("",T_US,0);
#endif
			}
		}
	}
	else
	{
#if (P_SHOW_ADVANCE==1)
		tic_simple("");
#endif
		// |
		S_Collide(i_dev, L);
		// |
#if (P_SHOW_ADVANCE==1)
		cudaDeviceSynchronize();
		tmp[1] += toc_simple("",T_US,0);
		tic_simple("");
#endif
		// |
		S_Stream(i_dev, L);
		// |
#if (P_SHOW_ADVANCE==1)
		cudaDeviceSynchronize();
		tmp[2] += toc_simple("",T_US,0);
		
		double tot_time = 0.0;
		long int tot_cells = 0;
		int multip = 1;
		for (int Lp = N_LEVEL_START; Lp < 4*MAX_LEVELS; Lp++)
		{
			tot_time += tmp[Lp];
			mesh->to.adv_printer << tmp[Lp] << " ";
		}
		for (int Lp = N_LEVEL_START; Lp < MAX_LEVELS-1; Lp++)
		{
			tot_cells += multip*(mesh->n_ids[i_dev][Lp] - mesh->n_ids[i_dev][Lp+1]/N_CHILDREN);
			multip *= 2;
		}
		tot_cells += multip*(mesh->n_ids[i_dev][MAX_LEVELS-1]);
		mesh->to.adv_printer << (1.0 / tot_time)*(double)(tot_cells*M_CBLOCK);
		mesh->to.adv_printer << std::endl;
#endif
	}
	
	return 0;
}

int Solver_LBM::S_SetIC(int i_dev, int L)
{
#if (N_Q==9)
	S_SetInitialConditions_d2q9(i_dev, L);
#elif (N_Q==19)
	S_SetInitialConditions_d3q19(i_dev, L);
#else // N_Q==27
	S_SetInitialConditions_d3q27(i_dev, L);
#endif
	
	if (S_COLLISION==1)
	{
		/*
#if (N_Q==9)
		S_Vel2Mom_MRT_d2q9(i_dev, L);
#elif (N_Q==19)
		S_Vel2Mom_MRT_d3q19(i_dev, L);
#else // N_Q==27
		//S_Vel2Mom_MRT_d2q27(i_dev, L);
#endif	
		*/
	}

	return 0;
}

int Solver_LBM::S_Collide(int i_dev, int L)
{
	if (S_COLLISION==0)
	{
		if (S_INTERP_HYBRID==0)
		{
#if (N_Q==9)
			S_Collide_BGK_d2q9(i_dev, L);
			//S_EnforceSymmetry_d2q9(i_dev, L);
#elif (N_Q==19)
			S_Collide_BGK_d3q19(i_dev, L);
			//S_EnforceSymmetry_d3q19(i_dev, L);
#else // N_Q==27
			S_Collide_BGK_d3q27(i_dev, L);
			//S_EnforceSymmetry_d3q27(i_dev, L);
#endif
		}
		else
		{
#if (N_Q==9)
			if (S_INTERP==0)
				S_Collide_BGK_Interpolate_Linear_d2q9(i_dev, L);
			else
				S_Collide_BGK_Interpolate_Cubic_d2q9(i_dev, L);
			//S_EnforceSymmetry_d2q9(i_dev, L);
#elif (N_Q==19)
			if (S_INTERP==0)
				S_Collide_BGK_Interpolate_Linear_d3q19(i_dev, L);
			else
				S_Collide_BGK_Interpolate_Cubic_d3q19(i_dev, L);
			//S_EnforceSymmetry_d3q19(i_dev, L);
#else // N_Q==27
			if (S_INTERP==0)
				S_Collide_BGK_Interpolate_Linear_d3q27(i_dev, L);
			else
				S_Collide_BGK_Interpolate_Cubic_d3q27(i_dev, L);
			//S_EnforceSymmetry_d3q27(i_dev, L);
#endif	
		}
	}
	else
	{
		/*
		if (S_INTERP_HYBRID==0)
		{
#if (N_Q==9)
			S_Collide_MRT_d2q9(i_dev, L);
			//S_EnforceSymmetry_d2q9(i_dev, L);
#elif (N_Q==19)
			S_Collide_MRT_d3q19(i_dev, L);
			//S_EnforceSymmetry_d3q19(i_dev, L);
#else // N_Q==27
			S_Collide_BGK_d3q27(i_dev, L);
			//S_EnforceSymmetry_d3q27(i_dev, L);
#endif
		}
		else
		{
#if (N_Q==9)
			if (S_INTERP==0)
				S_Collide_MRT_Interpolate_Linear_d2q9(i_dev, L);
			else
				S_Collide_MRT_Interpolate_Cubic_d2q9(i_dev, L);
			//S_EnforceSymmetry_d2q9(i_dev, L);
#elif (N_Q==19)
			if (S_INTERP==0)
				S_Collide_MRT_Interpolate_Linear_d3q19(i_dev, L);
			else
				S_Collide_MRT_Interpolate_Cubic_d3q19(i_dev, L);
			//S_EnforceSymmetry_d3q19(i_dev, L);
#else // N_Q==27
			if (S_INTERP==0)
				S_Collide_BGK_Interpolate_Linear_d3q27(i_dev, L);
			else
				S_Collide_MRT_Interpolate_Linear_d3q27(i_dev, L);
			//S_EnforceSymmetry_d3q27(i_dev, L);
#endif
		}
		*/
	}

	return 0;
}

int Solver_LBM::S_Stream(int i_dev, int L)
{
#if (N_Q==9)
	S_Stream_Inpl_d2q9(i_dev, L);
#elif (N_Q==19)
	S_Stream_Inpl_d3q19(i_dev, L);
#else // N_Q==27
	S_Stream_Inpl_d3q27(i_dev, L);
#endif
	
	if (S_COLLISION==1)
	{
		/*
#if (N_Q==9)
		S_Vel2Mom_MRT_d2q9(i_dev, L);
#elif (N_Q==19)
		S_Vel2Mom_MRT_d3q19(i_dev, L);
#else // N_Q==27
		//S_Vel2Mom_MRT_d2q27(i_dev, L);
#endif	
		*/
	}

	return 0;
}

int Solver_LBM::S_Interpolate(int i_dev, int L, int var)
{
	if (S_COLLISION==0)
	{
#if (N_Q==9)
		if (S_INTERP==0)
			S_Interpolate_Linear_d2q9(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
		else // S_INTERP==1, or anything else for now.
			S_Interpolate_Cubic_d2q9(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
#elif (N_Q==19)
		if (S_INTERP==0)
			S_Interpolate_Linear_d3q19(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
		else
			S_Interpolate_Cubic_d3q19(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
#else // N_Q==27
		if (S_INTERP==0)
			S_Interpolate_Linear_d3q27(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
		else
			S_Interpolate_Cubic_d3q27(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
#endif
	}
	else
	{
		/*
#if (N_Q==9)
		if (S_INTERP==0)
			S_Interpolate_Linear_MRT_d2q9(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
		else
			S_Interpolate_Cubic_MRT_d2q9(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
#elif (N_Q==19)
		if (S_INTERP==0)
			S_Interpolate_Linear_MRT_d3q19(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
		else
			S_Interpolate_Cubic_MRT_d3q19(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
#else // N_Q==27
		if (S_INTERP==0)
			S_Interpolate_Linear_MRT_d3q27(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
		else
			S_Interpolate_Cubic_MRT_d3q27(i_dev, L, var, tau_vec[L], tau_ratio_vec_C2F[L]);
#endif
		*/
	}

	return 0;
}

int Solver_LBM::S_Average(int i_dev, int L, int var)
{
	if (S_COLLISION==0)
	{
#if (N_Q==9)
		if (S_AVERAGE==0)
			S_Average_d2q9(i_dev, L, var, tau_vec[L], tau_ratio_vec_F2C[L]);
		else
			S_Average_Cubic_d2q9(i_dev, L, var, tau_vec[L], tau_ratio_vec_F2C[L]);
#elif (N_Q==19)
		S_Average_d3q19(i_dev, L, var, tau_vec[L], tau_ratio_vec_F2C[L]);
#else // N_Q==27
		S_Average_d3q27(i_dev, L, var, tau_vec[L], tau_ratio_vec_F2C[L]);
#endif
	}
	else
	{
		/*
#if (N_Q==9)
		S_Average_MRT_d2q9(i_dev, L, var, tau_vec[L], tau_ratio_vec_F2C[L]);
#elif (N_Q==19)
		S_Average_MRT_d3q19(i_dev, L, var, tau_vec[L], tau_ratio_vec_F2C[L]);
#else // N_Q==27
		S_Average_MRT_d3q27(i_dev, L, var, tau_vec[L], tau_ratio_vec_F2C[L]);
#endif
		*/
	}
	
	return 0;
}

int Solver_LBM::S_Debug(int var)
{
	if (var == 0)
	{
		S_Collide(0,0);
		S_Stream(0,0);
	}
	
	return 0;
}
