/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Dest()
{
	std::cout << " o====================================================================================" << std::endl;
	std::cout << " | Deleting: Mesh Object                                                                   " << std::endl;
	std::cout << " o====================================================================================" << std::endl;
	
	// Check memory currently being used before freeing for validation.
	cudaMemGetInfo(&free_t, &total_t);
	std::cout << "[-] Before freeing:\n";
	std::cout << "    Free: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
	
	// Free all the memory allocated on the CPU.
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		delete[] cells_ID_mask[i_dev];
		delete[] cells_f_F[i_dev];
		if (enable_aux_data)
			delete[] cells_f_F_aux[i_dev];
		delete[] cblock_f_X[i_dev];
		delete[] cblock_ID_mask[i_dev];
		delete[] cblock_ID_nbr[i_dev];
		delete[] cblock_ID_nbr_child[i_dev];
		delete[] cblock_ID_onb[i_dev];
		delete[] cblock_ID_ref[i_dev];
		delete[] cblock_level[i_dev];
#if (N_CASE==2)
		delete[] cblock_ID_face_count[i_dev];
		delete[] cblock_ID_face[i_dev];
#endif
		
		delete[] tmp_1[i_dev];
		delete[] tmp_2[i_dev];
		
		delete[] cells_f_U_probed_tn[i_dev];
		delete[] cells_f_U_mean[i_dev];
		
		// Allocate memory for id_set and reset to 0.
		delete[] n_ids[i_dev];
		delete[] id_max[i_dev];
		delete[] id_set[i_dev];
		delete[] gap_set[i_dev];
		delete[] coarse_I[i_dev];
		delete[] coarse_J[i_dev];
		delete[] coarse_K[i_dev];
		delete[] dxf_vec;

		// Probe arrays.
		delete[] id_set_probed[i_dev];
		
		// Output file.
		(*output_file_direct).close();
		delete output_file_direct;
	}
	
	// Free all the memory allocated on the GPU.
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		// Cell data.
		gpuErrchk( cudaFree(c_cells_ID_mask[i_dev]) );
		gpuErrchk( cudaFree(c_cells_f_F[i_dev]) );
		if (enable_aux_data)
			gpuErrchk( cudaFree(c_cells_f_F_aux[i_dev]) );
		gpuErrchk( cudaFree(c_cblock_f_X[i_dev]) );
		gpuErrchk( cudaFree(c_cblock_f_Ff[i_dev]) );
		gpuErrchk( cudaFree(c_cblock_ID_mask[i_dev]) );
		gpuErrchk( cudaFree(c_cblock_ID_nbr[i_dev]) );
		gpuErrchk( cudaFree(c_cblock_ID_nbr_child[i_dev]) );
		gpuErrchk( cudaFree(c_cblock_ID_onb[i_dev]) );
		gpuErrchk( cudaFree(c_cblock_ID_ref[i_dev]) );
		gpuErrchk( cudaFree(c_cblock_level[i_dev]) );
		
		// ID sets.
		gpuErrchk( cudaFree(c_id_set[i_dev]) );
		gpuErrchk( cudaFree(c_gap_set[i_dev]) );

		// Temp.
		gpuErrchk( cudaFree(c_tmp_1[i_dev]) );
		gpuErrchk( cudaFree(c_tmp_2[i_dev]) );
		gpuErrchk( cudaFree(c_tmp_3[i_dev]) );
		gpuErrchk( cudaFree(c_tmp_4[i_dev]) );
		gpuErrchk( cudaFree(c_tmp_5[i_dev]) );
		gpuErrchk( cudaFree(c_tmp_6[i_dev]) );
		gpuErrchk( cudaFree(c_tmp_7[i_dev]) );
		gpuErrchk( cudaFree(c_tmp_8[i_dev]) );
		gpuErrchk( cudaFree(c_tmp_counting_iter[i_dev]) );
	}
	
	// Verify that GPU memory has been recovered.
	cudaMemGetInfo(&free_t, &total_t);
	std::cout << "[-] After freeing:\n";
	std::cout << "    Free: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
	
	return 0;
}
