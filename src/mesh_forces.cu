/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

/*
         8888888888                                                 
         888                                                        
         888                                                        
         8888888  .d88b.  888d888 .d8888b .d88b.  .d8888b           
         888     d88""88b 888P"  d88P"   d8P  Y8b 88K               
         888     888  888 888    888     88888888 "Y8888b.          
         888     Y88..88P 888    Y88b.   Y8b.          X88          
88888888 888      "Y88P"  888     "Y8888P "Y8888   88888P' 88888888 
*/                                                  

int Mesh::M_PrintForces(int i_dev, int L, std::ofstream *out)
{
	ufloat_t Fxp = N_Pf(0.0);
	ufloat_t Fyp = N_Pf(0.0);
	ufloat_t Fxm = N_Pf(0.0);
	ufloat_t Fym = N_Pf(0.0);	
	ufloat_t dx_L = dxf_vec[L];
	int total_blocks = 0;
	
	for (int kap = 0; kap < n_ids[i_dev][L]; kap++)
	{
		int i_kap_b = id_set[i_dev][L][kap];
		bool block_involved = false;
		
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
		{
			int I_kap = kap_i % Nbx;
			int J_kap = (kap_i / Nbx) % Nbx;
			
			ufloat_t f_0 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 3*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 4*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 1*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 2*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 7*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 8*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 5*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 6*n_maxcells];
	
			int nbr_kap_b = cblock_ID_nbr[i_dev][i_kap_b + 1*n_maxcblocks];
			// nbr 1.
				// p = 3.
			if (((I_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_3);
				Fyp += (0)*(2.0*f_3);
			}
				// p = 7.
			if (((I_kap+1==Nbx) && (J_kap+1< Nbx)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_7);
				Fym += ((+1.0))*(2.0*f_7);
			}
				// p = 6.
			if (((I_kap+1==Nbx) && (J_kap-1>= 0)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_6);
				Fyp += (1.0)*(2.0*f_6);
			}
			// nbr 2.
			nbr_kap_b = cblock_ID_nbr[i_dev][i_kap_b + 2*n_maxcblocks];
				// p = 4.
			if (((J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxp += (0)*(2.0*f_4);
				Fym += ((+1.0))*(2.0*f_4);
			}
				// p = 7.
			if (((I_kap+1< Nbx) && (J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_7);
				Fym += ((+1.0))*(2.0*f_7);
			}
				// p = 8.
			if (((I_kap-1>= 0) && (J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_8);
				Fym += ((+1.0))*(2.0*f_8);
			}
			// nbr 3.
			nbr_kap_b = cblock_ID_nbr[i_dev][i_kap_b + 3*n_maxcblocks];
				// p = 1.
			if (((I_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_1);
				Fyp += (0)*(2.0*f_1);
			}
				// p = 8.
			if (((I_kap-1==-1) && (J_kap+1< Nbx)) && nbr_kap_b == -8)
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
			// nbr 4.
			nbr_kap_b = cblock_ID_nbr[i_dev][i_kap_b + 4*n_maxcblocks];
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
			if (((I_kap+1< Nbx) && (J_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_6);
				Fyp += (1.0)*(2.0*f_6);
			}
			// nbr 5.
			nbr_kap_b = cblock_ID_nbr[i_dev][i_kap_b + 5*n_maxcblocks];
				// p = 7.
			if (((I_kap+1==Nbx) && (J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_7);
				Fym += ((+1.0))*(2.0*f_7);
			}
			// nbr 6.
			nbr_kap_b = cblock_ID_nbr[i_dev][i_kap_b + 6*n_maxcblocks];
				// p = 8.
			if (((I_kap-1==-1) && (J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_8);
				Fym += ((+1.0))*(2.0*f_8);
			}
			// nbr 7.
			nbr_kap_b = cblock_ID_nbr[i_dev][i_kap_b + 7*n_maxcblocks];
				// p = 5.
			if (((I_kap-1==-1) && (J_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_5);
				Fyp += (1.0)*(2.0*f_5);
			}
			// nbr 8.
			nbr_kap_b = cblock_ID_nbr[i_dev][i_kap_b + 8*n_maxcblocks];
				// p = 6.
			if (((I_kap+1==Nbx) && (J_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_6);
				Fyp += (1.0)*(2.0*f_6);
			}
			
			for (int p = 0; p < 9; p++)
			{
				if (cblock_ID_nbr[i_dev][i_kap_b + p*n_maxcblocks] == -8)
					block_involved = true;
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
	*out << 2.0*Fx / (0.05*0.05*(Dp)) << " " << 2.0*Fy / (0.05*0.05*(Dp)) << std::endl;
	
	return 0;
}
