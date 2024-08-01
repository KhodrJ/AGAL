/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

int Mesh::M_FillBlock(int i_dev, int *Is, int i_kap, int L, double dx_f, int *mult_f, int vol, int *Nxi_f, double *tmp_data)
{
	// (I,J,K) are the indices for the top parent. All child indices moving forward are built from these.
	int child_0 = cblock_ID_nbr_child[i_dev][i_kap];
	
	if (child_0 >= 0 && L+1 < N_PRINT_LEVELS) // Has children, keep traversing.
	{
#if (N_DIM==3)
		for (int xk = 0; xk < 2; xk++)
#else
		int xk = 0;
#endif
		{
			for (int xj = 0; xj < 2; xj++)
			{
				for (int xi = 0; xi < 2; xi++)
				{
					Is[L+1 + 0*N_PRINT_LEVELS] = xi;
					Is[L+1 + 1*N_PRINT_LEVELS] = xj;
					Is[L+1 + 2*N_PRINT_LEVELS] = xk;
					int xc = xi + 2*xj + 4*xk;
					M_FillBlock(i_dev, Is, child_0+xc, L+1, dx_f, mult_f, vol, Nxi_f, tmp_data);
				}
			}
		}
	}
	else // No children, print here.
	{
		// Get the macroscopic properties for this block.
		double out_u_[M_CBLOCK*(3+1)];
		M_ComputeProperties(i_dev, i_kap, dxf_vec[L], out_u_);
		
		// Modify the cell values in the region defined by the leaf block.
#if (N_DIM==3)
		for (int k = 0; k < Nbx; k++)
#else
		int k = 0;
#endif
		{
			for (int j = 0; j < Nbx; j++)
			{
				for (int i = 0; i < Nbx; i++)
				{
					int Ip = 0;
					int Jp = 0;
					int Kp = 0;
					for (int l = 0; l < L+1; l++)
					{
						Ip += mult_f[l]*Nbx*Is[l + 0*N_PRINT_LEVELS];
						Jp += mult_f[l]*Nbx*Is[l + 1*N_PRINT_LEVELS];
						Kp += mult_f[l]*Nbx*Is[l + 2*N_PRINT_LEVELS];
					}
					
#if (N_DIM==3)
					for (int kk = 0; kk < mult_f[L]; kk++)
#else
					int kk = 0;
#endif
					{
						for (int jj = 0; jj < mult_f[L]; jj++)
						{
							for (int ii = 0; ii < mult_f[L]; ii++)
							{
								int kap_i = i + Nbx*j + Nbx*Nbx*k;
								int Ipp = Ip + i*mult_f[L] + ii;
								int Jpp = Jp + j*mult_f[L] + jj;
								int Kpp = Kp + k*mult_f[L] + kk;
								long int Id = Ipp + Nxi_f[0]*Jpp + Nxi_f[0]*Nxi_f[1]*Kpp;
								
								tmp_data[Id + 0*vol] = out_u_[kap_i + 0*M_CBLOCK];
								tmp_data[Id + 1*vol] = out_u_[kap_i + 1*M_CBLOCK];
								tmp_data[Id + 2*vol] = out_u_[kap_i + 2*M_CBLOCK];
								tmp_data[Id + 3*vol] = out_u_[kap_i + 3*M_CBLOCK];
// 								tmp_data[Id + 4*vol] = out_u_[kap_i + 4*M_CBLOCK];
// 								tmp_data[Id + 5*vol] = out_u_[kap_i + 5*M_CBLOCK];
// 								tmp_data[Id + 6*vol] = out_u_[kap_i + 6*M_CBLOCK];
								//tmp_data[Id + 7*vol] = sqrt(out_u_[kap_i + 1*M_CBLOCK]*out_u_[kap_i + 1*M_CBLOCK] + out_u_[kap_i + 2*M_CBLOCK]*out_u_[kap_i + 2*M_CBLOCK] + out_u_[kap_i + 3*M_CBLOCK]*out_u_[kap_i + 3*M_CBLOCK]);
								//tmp_data[Id + 8*vol] = sqrt(out_u_[kap_i + 4*M_CBLOCK]*out_u_[kap_i + 4*M_CBLOCK] + out_u_[kap_i + 5*M_CBLOCK]*out_u_[kap_i + 5*M_CBLOCK] + out_u_[kap_i + 6*M_CBLOCK]*out_u_[kap_i + 6*M_CBLOCK]);
								tmp_data[Id + 4*vol] = L;
								tmp_data[Id + 5*vol] = i_kap;
							}
						}
					}
				}
			}
		}
	}
		
	return 0;
}

int Mesh::M_RenderAndPrint_Uniform(int i_dev, int iter)
{
	// Parameters.
		// Domain extents (w.r.t root grid, I_min <= I < I_max).
	int I_min = VOL_I_MIN;
	int I_max = VOL_I_MAX;
	int J_min = VOL_J_MIN;
	int J_max = VOL_J_MAX;
	int K_min = 0;
	int K_max = 1;
#if (N_DIM==3)
	K_min = VOL_K_MIN;
	K_max = VOL_K_MAX;
#endif
	
	
	// Resolution multiplier.
	int mult = 1;
	double dx_f = dxf_vec[0];
	for (int L = 1; L < N_PRINT_LEVELS; L++) 
	{
		mult *= 2;
		dx_f *= 0.5;
	}
	int *mult_f = new int[N_PRINT_LEVELS];
	for (int L = 0; L < N_PRINT_LEVELS; L++)
		mult_f[L] = pow(2.0, (double)(N_PRINT_LEVELS-1-L));
	
	
	// Resolution array.
	int Nxi_f[3];
	Nxi_f[0] = (I_max-I_min)*Nbx;
	Nxi_f[1] = (J_max-J_min)*Nbx;
	Nxi_f[2] = (K_max-K_min)*Nbx; if (N_DIM==2) Nxi_f[2] = 1;
	//for (int d = 0; d < 3; d++)
	//	Nxi_f[d] = Nxi[d];
	for (int d = 0; d < N_DIM; d++)
		Nxi_f[d] *= mult;
	int vol = Nxi_f[0]*Nxi_f[1]*Nxi_f[2];
		// Cell data arrays.
	int n_data = 1+3+1+1;
	double *tmp_data = new double[n_data*vol];
	double *tmp_data_b = new double[n_data*vol];
	for (long int p = 0; p < n_data*vol; p++)
		tmp_data[p] = -1.0;
	
	
	// Traverse the grid and fill data arrays.
	std::cout << "[-] Traversing grid, computing properties..." << std::endl;
	#pragma omp parallel for
	for (int kap = 0; kap < n_ids[i_dev][0]; kap++)
	{
		int Is[N_PRINT_LEVELS*3];
		for (int Ld = 0; Ld < N_PRINT_LEVELS*3; Ld++)
			Is[Ld] = 0;
		
		Is[0*N_PRINT_LEVELS] = coarse_I[i_dev][kap] - I_min;
		Is[1*N_PRINT_LEVELS] = coarse_J[i_dev][kap] - J_min;
		Is[2*N_PRINT_LEVELS] = coarse_K[i_dev][kap] - K_min;
		
		if (Is[0*N_PRINT_LEVELS] >= 0 && Is[0*N_PRINT_LEVELS] < I_max-I_min && Is[1*N_PRINT_LEVELS] >= 0 && Is[1*N_PRINT_LEVELS] < J_max-J_min && Is[2*N_PRINT_LEVELS] >= 0 && Is[2*N_PRINT_LEVELS] < K_max-K_min)
			M_FillBlock(i_dev, Is, kap, 0, dx_f, mult_f, vol, Nxi_f, tmp_data);
	}
	std::cout << "    Finished traversal..." << std::endl;
	
	
	// Direct print to binary file.
	for (int d = 0; d < 3+1; d++)
		(*output_file_direct).write((char*)&tmp_data[d*vol], vol*sizeof(double));
	(*output_file_direct).write((char*)&tmp_data[4*vol], vol*sizeof(double));
	(*output_file_direct).write((char*)&tmp_data[5*vol], vol*sizeof(double));
	
	
	// Free allocations.
	delete[] mult_f;
	delete[] tmp_data;
	delete[] tmp_data_b;
	
	return 0;
}
