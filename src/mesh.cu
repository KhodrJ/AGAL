/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"



/*
                d8888                   d8b 888 d8b                                    
               d88888                   Y8P 888 Y8P                                    
              d88P888                       888                                        
             d88P 888 888  888 888  888 888 888 888  8888b.  888d888 888  888          
            d88P  888 888  888 `Y8bd8P' 888 888 888     "88b 888P"   888  888          
           d88P   888 888  888   X88K   888 888 888 .d888888 888     888  888          
          d8888888888 Y88b 888 .d8""8b. 888 888 888 888  888 888     Y88b 888          
88888888 d88P     888  "Y88888 888  888 888 888 888 "Y888888 888      "Y88888 88888888 
                                                                          888          
                                                                     Y8b d88P          
                                                                      "Y88P"           
*/



// Resets the array to a prescribed value.
template<class T>
__global__
void Cu_ResetToValue(int N, T *arr, T val)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < N)
		arr[kap] = val;
}

// An array arr of size N is loaded into shared memory in pieces of M_BLOCK size. A fraction of this 1/frac will be reloaded into arr with a stride of frac.
// Entries in shared memory are also appended to the end array arr_p whose size is N_p. This routine is used to contract the spliced gap set so that only child 
// IDs with index 0 are recovered and loaded into the -bygroup array. While the gap set entries are already loaded, I've decided to concatenate with the ID set
// to save time.
template <class T>
__global__
void Cu_ContractByFrac(int N, T *arr, int frac, T *arr2)
{
	__shared__ int s_arr[M_BLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int new_start = blockIdx.x*blockDim.x/frac;
	
	if (kap < N)
	{
		// Read arr into shared memory first.
		s_arr[threadIdx.x] = arr[kap];
		__syncthreads();
	
		if (threadIdx.x < M_BLOCK/frac)
		{
			arr2[new_start + threadIdx.x] = s_arr[frac*threadIdx.x];
		}
	}
}

template <class T>
__global__
void Cu_Concat(int N, T *arr, int N2, T *arr2)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < N2)
	{
		arr[N+kap] = arr2[kap];
	}
}

template <class T>
__global__
void Cu_ConcatReverse(int N, T *arr, int N2, T *arr2)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < N2)
	{
		arr[N+kap] = arr2[N2-1-kap];
	}
}

template <class T>
__global__
void Cu_FillLinear(int N, T *arr)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < N)
	{
		arr[kap] = kap;
	}
}

template <class T>
__global__
void Cu_Debug_ModVals(T *arr, int start, int N, T *arr2)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < N)
	{
		arr[start+kap] = arr2[kap];
	}
}

// Used purely for debugging.
int Mesh::M_CudaModVals(int var)
{
	// Set IDs 1, 9, 14 to coarsen to check coarsening.
	if (var == 0)
	{
		int i_dev = 0;
		for (int xc = 0; xc < N_CHILDREN/2; xc++)
			cblock_ID_ref[i_dev][64 + xc] = V_REF_ID_MARK_REFINE;
		cblock_ID_ref[i_dev][9] = V_REF_ID_MARK_COARSEN;
		gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], n_maxcblocks*sizeof(int), cudaMemcpyHostToDevice) );
	}
	
	if (var == 1)
	{
		int i_dev = 0;
		cblock_ID_ref[i_dev][9] = V_REF_ID_MARK_REFINE;
		gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], n_maxcblocks*sizeof(int), cudaMemcpyHostToDevice) );
	}

	return 0;	
}



/*
         .d88888b.           888                      888             
        d88P" "Y88b          888                      888             
        888     888          888                      888             
        888     888 888  888 888888 88888b.  888  888 888888          
        888     888 888  888 888    888 "88b 888  888 888             
        888     888 888  888 888    888  888 888  888 888             
        Y88b. .d88P Y88b 888 Y88b.  888 d88P Y88b 888 Y88b.           
88888888 "Y88888P"   "Y88888  "Y888 88888P"   "Y88888  "Y888 88888888 
                                    888                               
                                    888                               
                                    888                               
*/



int Mesh::M_Print(int i_dev, int iter)
{
	// New variables.
	vtkNew<vtkOverlappingAMR> data;
	int blocks_per_level[N_PRINT_LEVELS];
	for (int L = 0; L < N_PRINT_LEVELS; L++)
		blocks_per_level[L] = n_ids[i_dev][L];
	double global_origin[3] = {0,0,0};
	
	// Get the number of levels in which there are a non-zero number of blocks.
	int n_levels_nonzero_blocks = 1;
#if (MAX_LEVELS>1)
	for (int L = 1; L < N_PRINT_LEVELS; L++)
	{
		if (n_ids[i_dev][L] > 0)
			n_levels_nonzero_blocks++;
	}
#endif
	
	// Initialize AMR object.
	data->Initialize(n_levels_nonzero_blocks, blocks_per_level);
	data->SetOrigin(global_origin);
	data->SetGridDescription(N_DIM==2?VTK_XY_PLANE:VTK_XYZ_GRID);
	
	// For each level, insert all existing blocks.
	//int n_dim_box[3] = {Nbx, Nbx, N_DIM==2?1:Nbx};
	int n_dim_lattice[3] = {Nbx+1, Nbx+1, N_DIM==2?1:Nbx+1};
	double origin_kap[3] = {0,0,0};
	for (int L = 0; L < std::min(n_levels_nonzero_blocks, N_PRINT_LEVELS); L++)
	{
		// Construct spacing array for level L.
		double dxf_L = (double)dxf_vec[L];
		double h_L_kap[3] = {dxf_L, dxf_L, N_DIM==2?(double)dx:dxf_L};
		
		// Only insert spacing array if there are actual blocks to insert on this level.
		if (n_ids[i_dev][L] > 0)
			data->SetSpacing(L, h_L_kap);
		
		// For all blocks on level L, build vtkAMRBox and insert in AMR object.
		int kap_counter = 0;
		for (int kap = 0; kap < n_ids[i_dev][L]; kap++)
		{
			// ID of the kap'th cblock.
			int i_kap = id_set[i_dev][L][kap];
			
			//if (cblock_ID_ref[i_dev][i_kap] == V_REF_ID_UNREFINED)
			{
				// Set origin of block (based on lower-left corner stored in cblock_f_X).
				for (int d = 0; d < N_DIM; d++)
					origin_kap[d] = cblock_f_X[i_dev][i_kap + d*n_maxcblocks];
				
				// Initialize uniform grid defining the block.
				vtkNew<vtkUniformGrid> grid_kap;
				grid_kap->Initialize();
				grid_kap->SetOrigin(origin_kap);
				grid_kap->SetSpacing(h_L_kap);
				grid_kap->SetDimensions(n_dim_lattice);
				
				// Fill data in the uniform grid defining the block.
					// Debug.
				vtkNew<vtkDoubleArray> data_kap_dbg;
				data_kap_dbg->SetName("debug - ref ID");
				data_kap_dbg->SetNumberOfComponents(1);
				data_kap_dbg->SetNumberOfTuples(M_CBLOCK);
				int min_nbr_id = 0;
#pragma unroll
				for (int p = 0; p < l_dq_max; p++)
				{
					int nbr_id_p = cblock_ID_nbr[i_dev][i_kap + p*n_maxcblocks];
					//if (nbr_id_p == N_SKIPID)
					//	nbr_id_p = -1;
					if (nbr_id_p < 0 && min_nbr_id == 0)
						min_nbr_id = nbr_id_p;
					if (nbr_id_p > min_nbr_id && nbr_id_p < 0)
						min_nbr_id = nbr_id_p;
				}
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					//data_kap_dbg->SetTuple1(kap_i, (double)cblock_ID_ref[i_dev][i_kap]);
					data_kap_dbg->SetTuple1(kap_i, (double)cells_ID_mask[i_dev][i_kap*M_CBLOCK + kap_i]);
					//data_kap_dbg->SetTuple1(kap_i, (double)cblock_ID_mask[i_dev][i_kap]);
					//data_kap_dbg->SetTuple1(kap_i, (double)min_nbr_id);
				}
				grid_kap->GetCellData()->AddArray(data_kap_dbg);
					// Density.
				vtkNew<vtkDoubleArray> data_kap_sc;
				data_kap_sc->SetName("density");
				data_kap_sc->SetNumberOfComponents(1);
				data_kap_sc->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					data_kap_sc->SetTuple1(kap_i, (double)cells_f_U[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells]);
				}
				grid_kap->GetCellData()->AddArray(data_kap_sc);
					// Vorticity.
				vtkNew<vtkDoubleArray> data_kap_w;
				data_kap_w->SetName("vorticity");
				data_kap_w->SetNumberOfComponents(1);
				data_kap_w->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					data_kap_w->SetTuple1(kap_i, (double)cells_f_W[i_dev][i_kap*M_CBLOCK + kap_i]);
				}
				grid_kap->GetCellData()->AddArray(data_kap_w);
					// Velocity.
				vtkNew<vtkDoubleArray> data_kap_v;
				data_kap_v->SetName("velocity");
				data_kap_v->SetNumberOfComponents(N_DIM);
				data_kap_v->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
#if (N_DIM==2)
					data_kap_v->SetTuple2(kap_i,
						(double)cells_f_U[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells],
						(double)cells_f_U[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells]
					);					
#else
					data_kap_v->SetTuple3(kap_i,
						(double)cells_f_U[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells],
						(double)cells_f_U[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells],
						(double)cells_f_U[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells]
					);
#endif
				}
				grid_kap->GetCellData()->AddArray(data_kap_v);
				
				// Create the vtkAMRBox and insert it into AMR object.
				vtkAMRBox box_kap(origin_kap, n_dim_lattice, h_L_kap, data->GetOrigin(), data->GetGridDescription());
				data->SetAMRBox(L, kap_counter, box_kap);
				data->SetDataSet(L, kap_counter, grid_kap);
				kap_counter++;
			}
		}
	}
	
	// Write the AMR object.
	std::string fileName = P_DIR_NAME + std::string("out_") + std::to_string(iter) + ".vthb";
	vtkNew<vtkXMLUniformGridAMRWriter> writer;
	writer->SetInputData(data);
	writer->SetFileName(fileName.c_str());
	writer->Write();
	
	return 0;
}

int Mesh::M_PrintConnectivity(int i_dev)
{
	std::ofstream out_conn; out_conn.open("out_conn.txt");
	
	for (int L = 0; L < MAX_LEVELS; L++)
	{
		for (int kap = 0; kap < n_ids[i_dev][L]; kap++)
		{
			int i_kap =  id_set[i_dev][L][kap];
			
			out_conn << "[ID: " << i_kap << "] Block " << kap << ", Level: " << L << std::endl;
			
			out_conn << "X: ";
			for (int d = 0; d < N_DIM; d++)
				out_conn << "x_" << d << "|" << cblock_f_X[i_dev][i_kap + d*n_maxcblocks] << " ";
			out_conn << std::endl;
			
			/*
			for (int xkap = 0; xkap < M_CBLOCK; xkap++)
			{
				out_conn << "Cell " << xkap << ": ";
				for (int p = 0; p < l_dq; p++)
					out_conn << "F(p=" << p << ")|" << cells_f_F[i_dev][i_kap*M_CBLOCK + xkap + p*n_maxcells] << "  ";
				out_conn << std::endl;
			}
			*/
		
			/*
			out_conn << "Nbrs: ";
			for (int p = 0; p < l_dq_max; p++)
				out_conn << "p=" << p << "|" << cblock_ID_nbr[i_dev][i_kap + p*n_maxcblocks] << " ";
			out_conn << std::endl;
			
			out_conn << "Child nbrs: ";
			for (int p = 0; p < l_dq_max; p++)
				out_conn << "p=" << p << "|" << cblock_ID_nbr_child[i_dev][i_kap + p*n_maxcblocks] << " ";
			out_conn << std::endl;
			
			out_conn << std::endl;
			*/
		}
	}
	
	out_conn.close();
	
	return 0;
}





int Mesh::M_LoadToGPU()
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		int cblocks_id_max = id_max[i_dev][MAX_LEVELS];
		int cells_id_max = id_max[i_dev][MAX_LEVELS]*M_CBLOCK;
		
		// Floating point arrays.
		for (int p = 0; p < l_dq; p++)
			gpuErrchk( cudaMemcpy(&c_cells_f_F[i_dev][p*n_maxcells], &cells_f_F[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );
#if (S_TYPE==0)
		for (int p = 0; p < l_dq; p++)
			gpuErrchk( cudaMemcpy(&c_cells_f_Fs[i_dev][p*n_maxcells], &cells_f_Fs[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );
#endif
		for (int d = 0; d < N_U; d++)
			gpuErrchk( cudaMemcpy(&c_cells_f_U[i_dev][d*n_maxcells], &cells_f_U[i_dev][d*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );
		for (int d = 0; d < N_DIM; d++)
			gpuErrchk( cudaMemcpy(&c_cblock_f_X[i_dev][d*n_maxcblocks], &cblock_f_X[i_dev][d*n_maxcblocks], cblocks_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cells_f_W[i_dev], cells_f_W[i_dev], cells_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );		

		
		// Connectivity arrays.
		for (int p = 0; p < l_dq_max; p++)
		{
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr[i_dev][p*n_maxcblocks], &cblock_ID_nbr[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &cblock_ID_nbr_child[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		}

		
		// Metadata arrays.
		gpuErrchk( cudaMemcpy(c_cells_ID_mask[i_dev], cells_ID_mask[i_dev], cells_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_ID_mask[i_dev], cblock_ID_mask[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_level[i_dev], cblock_level[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );

		for (int L = 0; L < MAX_LEVELS; L++)
			gpuErrchk( cudaMemcpy(c_id_set[i_dev][L], id_set[i_dev][L], n_ids[i_dev][L]*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_gap_set[i_dev], gap_set[i_dev], n_gaps[i_dev]*sizeof(int), cudaMemcpyHostToDevice) );
	}
	
	return 0;
}

int Mesh::M_RetrieveFromGPU()
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		int cblocks_id_max = id_max[i_dev][MAX_LEVELS];
		int cells_id_max = id_max[i_dev][MAX_LEVELS]*M_CBLOCK;

		// Floating point arrays.
		for (int p = 0; p < l_dq; p++)
			gpuErrchk( cudaMemcpy(&cells_f_F[i_dev][p*n_maxcells], &c_cells_f_F[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );
#if (S_TYPE==0)
		for (int p = 0; p < l_dq; p++)
			gpuErrchk( cudaMemcpy(&cells_f_Fs[i_dev][p*n_maxcells], &c_cells_f_Fs[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );
#endif
		for (int d = 0; d < N_U; d++)
			gpuErrchk( cudaMemcpy(&cells_f_U[i_dev][d*n_maxcells], &c_cells_f_U[i_dev][d*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );
		for (int d = 0; d < N_DIM; d++)
			gpuErrchk( cudaMemcpy(&cblock_f_X[i_dev][d*n_maxcblocks], &c_cblock_f_X[i_dev][d*n_maxcblocks], cblocks_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cells_f_W[i_dev], c_cells_f_W[i_dev], cells_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );

		
		// Connectivity arrays.
		for (int p = 0; p < l_dq_max; p++)
		{
			gpuErrchk( cudaMemcpy(&cblock_ID_nbr[i_dev][p*n_maxcblocks], &c_cblock_ID_nbr[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
			gpuErrchk( cudaMemcpy(&cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &c_cblock_ID_nbr_child[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		}


		// Metadata arrays.
		gpuErrchk( cudaMemcpy(cells_ID_mask[i_dev], c_cells_ID_mask[i_dev], cells_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_ID_mask[i_dev], c_cblock_ID_mask[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_ID_ref[i_dev], c_cblock_ID_ref[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_level[i_dev], c_cblock_level[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		
		for (int L = 0; L < MAX_LEVELS; L++)
			gpuErrchk( cudaMemcpy(id_set[i_dev][L], c_id_set[i_dev][L], n_ids[i_dev][L]*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(gap_set[i_dev], c_gap_set[i_dev], n_gaps[i_dev]*sizeof(int), cudaMemcpyDeviceToHost) );
	}

	return 0;
}



/*
         .d8888b.                                               888    d8b          d8b 888                      
        d88P  Y88b                                              888    Y8P          Y8P 888                      
        888    888                                              888                     888                      
        888         .d88b.  88888b.  88888b.   .d88b.   .d8888b 888888 888 888  888 888 888888 888  888          
        888        d88""88b 888 "88b 888 "88b d8P  Y8b d88P"    888    888 888  888 888 888    888  888          
        888    888 888  888 888  888 888  888 88888888 888      888    888 Y88  88P 888 888    888  888          
        Y88b  d88P Y88..88P 888  888 888  888 Y8b.     Y88b.    Y88b.  888  Y8bd8P  888 Y88b.  Y88b 888          
88888888 "Y8888P"   "Y88P"  888  888 888  888  "Y8888   "Y8888P  "Y888 888   Y88P   888  "Y888  "Y88888 88888888 
                                                                                                    888          
                                                                                               Y8b d88P          
                                                                                                "Y88P"           
*/



__global__
void Cu_UpdateBoundaries
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr)
	{
		int nbr_kap_p = 0;

		for (int p = 1; p < l_dq_max; p++)
		{
			nbr_kap_p = cblock_ID_nbr[kap + p*n_maxcblocks];
			if (nbr_kap_p < 0)
				cblock_ID_nbr_child[kap + p*n_maxcblocks] = nbr_kap_p;
		}
	}
}

__global__
void Cu_UpdateConnectivity
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child,
	int *scattered_map
)
{
	__shared__ int s_ID_child[M_BLOCK];
	__shared__ int s_ID_child_nbrs[M_BLOCK*N_CHILDREN];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int t_id_incrementor = threadIdx.x % N_CHILDREN;
	int t_id_builder = threadIdx.x / N_CHILDREN;
	int id_nbr_0_child, id_nbr_1_child, id_nbr_2_child, id_nbr_3_child, id_nbr_4_child, id_nbr_5_child,  id_nbr_6_child, id_nbr_7_child, id_nbr_8_child;
#if (N_DIM==3)
	int id_nbr_9_child, id_nbr_10_child, id_nbr_11_child, id_nbr_12_child, id_nbr_13_child, id_nbr_14_child, id_nbr_15_child, id_nbr_16_child, id_nbr_17_child, id_nbr_18_child, id_nbr_19_child, id_nbr_20_child, id_nbr_21_child, id_nbr_22_child, id_nbr_23_child, id_nbr_24_child, id_nbr_25_child, id_nbr_26_child;
#endif
	
	// Initialize shared memory.
	s_ID_child[threadIdx.x] = -1;
	s_ID_child_nbrs[threadIdx.x] = -1;
	
	if (kap < id_max_curr)
	{
		if (scattered_map[kap] > -1 && cblock_ID_nbr_child[kap + 0*n_maxcblocks] > -1)
		{
			s_ID_child[threadIdx.x] = cblock_ID_nbr_child[kap + 0*n_maxcblocks]; //scattered_map[kap];
			id_nbr_0_child = cblock_ID_nbr_child[kap + 0*n_maxcblocks];
			id_nbr_1_child = cblock_ID_nbr_child[kap + 1*n_maxcblocks];
			id_nbr_2_child = cblock_ID_nbr_child[kap + 2*n_maxcblocks];
			id_nbr_3_child = cblock_ID_nbr_child[kap + 3*n_maxcblocks];
			id_nbr_4_child = cblock_ID_nbr_child[kap + 4*n_maxcblocks];
			id_nbr_5_child = cblock_ID_nbr_child[kap + 5*n_maxcblocks];
			id_nbr_6_child = cblock_ID_nbr_child[kap + 6*n_maxcblocks];
			id_nbr_7_child = cblock_ID_nbr_child[kap + 7*n_maxcblocks];
			id_nbr_8_child = cblock_ID_nbr_child[kap + 8*n_maxcblocks];
#if (N_DIM==3)
			id_nbr_9_child = cblock_ID_nbr_child[kap + 9*n_maxcblocks];
			id_nbr_10_child = cblock_ID_nbr_child[kap + 10*n_maxcblocks];
			id_nbr_11_child = cblock_ID_nbr_child[kap + 11*n_maxcblocks];
			id_nbr_12_child = cblock_ID_nbr_child[kap + 12*n_maxcblocks];
			id_nbr_13_child = cblock_ID_nbr_child[kap + 13*n_maxcblocks];
			id_nbr_14_child = cblock_ID_nbr_child[kap + 14*n_maxcblocks];
			id_nbr_15_child = cblock_ID_nbr_child[kap + 15*n_maxcblocks];
			id_nbr_16_child = cblock_ID_nbr_child[kap + 16*n_maxcblocks];
			id_nbr_17_child = cblock_ID_nbr_child[kap + 17*n_maxcblocks];
			id_nbr_18_child = cblock_ID_nbr_child[kap + 18*n_maxcblocks];
			id_nbr_19_child = cblock_ID_nbr_child[kap + 19*n_maxcblocks];
			id_nbr_20_child = cblock_ID_nbr_child[kap + 20*n_maxcblocks];
			id_nbr_21_child = cblock_ID_nbr_child[kap + 21*n_maxcblocks];
			id_nbr_22_child = cblock_ID_nbr_child[kap + 22*n_maxcblocks];
			id_nbr_23_child = cblock_ID_nbr_child[kap + 23*n_maxcblocks];
			id_nbr_24_child = cblock_ID_nbr_child[kap + 24*n_maxcblocks];
			id_nbr_25_child = cblock_ID_nbr_child[kap + 25*n_maxcblocks];
			id_nbr_26_child = cblock_ID_nbr_child[kap + 26*n_maxcblocks];			
#endif
		}
	}
	__syncthreads();

#if (N_DIM==2)
	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 0*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 1*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 0;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 2*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 3;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 3*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 4*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 5*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 3;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 1;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 6*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 3;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 7*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 2;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 8*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();
#else
	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 0*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 4;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 6;
// 			fprintf(fileID, "Child IDs: %i %i %i %i %i %i %i %i, Nbrs 1: %i %i %i %i %i %i %i %i\n", 
// 			        s_ID_child[threadIdx.x],
// 				s_ID_child[threadIdx.x],
// 				s_ID_child[threadIdx.x],
// 				s_ID_child[threadIdx.x],
// 				s_ID_child[threadIdx.x],
// 				s_ID_child[threadIdx.x],
// 				s_ID_child[threadIdx.x],
// 				s_ID_child[threadIdx.x],
// 				s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN],
// 				s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN],
// 				s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN],
// 				s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN],
// 				s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN],
// 				s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN],
// 				s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN],
// 				s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN]
// 			);
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 1*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 3;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 5;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 7;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 2*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 0;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 4;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 5;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 3*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 6;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 7;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 4*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 1;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 2;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 3;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 5*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 4;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 5;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 6;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 7;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 6*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 0;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 6;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 5;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 4;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 7*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 3;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 7;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 6;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 5;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 8*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 4;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 6;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 1;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_9_child<0 ? id_nbr_9_child:id_nbr_9_child + 0;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 3;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_9_child<0 ? id_nbr_9_child:id_nbr_9_child + 2;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 9*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_10_child<0 ? id_nbr_10_child:id_nbr_10_child + 5;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 4;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_10_child<0 ? id_nbr_10_child:id_nbr_10_child + 7;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 6;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 3;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 10*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 4;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 5;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 2;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 3;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_11_child<0 ? id_nbr_11_child:id_nbr_11_child + 0;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_11_child<0 ? id_nbr_11_child:id_nbr_11_child + 1;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 11*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_12_child<0 ? id_nbr_12_child:id_nbr_12_child + 6;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_12_child<0 ? id_nbr_12_child:id_nbr_12_child + 7;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 4;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 5;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 12*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_13_child<0 ? id_nbr_13_child:id_nbr_13_child + 2;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 7;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_13_child<0 ? id_nbr_13_child:id_nbr_13_child + 6;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 4;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 13*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 3;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_14_child<0 ? id_nbr_14_child:id_nbr_14_child + 1;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 0;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 7;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_14_child<0 ? id_nbr_14_child:id_nbr_14_child + 5;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 4;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 14*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 5;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_15_child<0 ? id_nbr_15_child:id_nbr_15_child + 4;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 7;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_15_child<0 ? id_nbr_15_child:id_nbr_15_child + 6;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 15*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 5;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 7;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_16_child<0 ? id_nbr_16_child:id_nbr_16_child + 1;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_16_child<0 ? id_nbr_16_child:id_nbr_16_child + 3;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 2;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 16*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 6;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 7;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_17_child<0 ? id_nbr_17_child:id_nbr_17_child + 4;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_17_child<0 ? id_nbr_17_child:id_nbr_17_child + 5;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 0;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 17*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 6;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 7;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_18_child<0 ? id_nbr_18_child:id_nbr_18_child + 2;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_18_child<0 ? id_nbr_18_child:id_nbr_18_child + 3;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 1;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 18*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 6;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 5;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 4;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 3;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_9_child<0 ? id_nbr_9_child:id_nbr_9_child + 2;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_11_child<0 ? id_nbr_11_child:id_nbr_11_child + 1;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_19_child<0 ? id_nbr_19_child:id_nbr_19_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 19*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_20_child<0 ? id_nbr_20_child:id_nbr_20_child + 7;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_12_child<0 ? id_nbr_12_child:id_nbr_12_child + 6;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_10_child<0 ? id_nbr_10_child:id_nbr_10_child + 5;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 4;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 3;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 20*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 7;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_15_child<0 ? id_nbr_15_child:id_nbr_15_child + 6;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_17_child<0 ? id_nbr_17_child:id_nbr_17_child + 5;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_21_child<0 ? id_nbr_21_child:id_nbr_21_child + 4;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 21*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 7;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 6;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 5;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_22_child<0 ? id_nbr_22_child:id_nbr_22_child + 3;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_18_child<0 ? id_nbr_18_child:id_nbr_18_child + 2;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_16_child<0 ? id_nbr_16_child:id_nbr_16_child + 1;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 22*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 7;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_13_child<0 ? id_nbr_13_child:id_nbr_13_child + 6;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 4;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_18_child<0 ? id_nbr_18_child:id_nbr_18_child + 3;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_23_child<0 ? id_nbr_23_child:id_nbr_23_child + 2;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 1;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_9_child<0 ? id_nbr_9_child:id_nbr_9_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 23*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_10_child<0 ? id_nbr_10_child:id_nbr_10_child + 7;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 6;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_24_child<0 ? id_nbr_24_child:id_nbr_24_child + 5;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_17_child<0 ? id_nbr_17_child:id_nbr_17_child + 4;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 3;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_14_child<0 ? id_nbr_14_child:id_nbr_14_child + 1;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 24*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 7;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_14_child<0 ? id_nbr_14_child:id_nbr_14_child + 5;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 4;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_16_child<0 ? id_nbr_16_child:id_nbr_16_child + 3;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 2;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_25_child<0 ? id_nbr_25_child:id_nbr_25_child + 1;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_11_child<0 ? id_nbr_11_child:id_nbr_11_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 25*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();

	if (kap < id_max_curr)
	{	if (s_ID_child[threadIdx.x] > -1)
		{
			s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_12_child<0 ? id_nbr_12_child:id_nbr_12_child + 7;
			s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_26_child<0 ? id_nbr_26_child:id_nbr_26_child + 6;
			s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 5;
			s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_15_child<0 ? id_nbr_15_child:id_nbr_15_child + 4;
			s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
			s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_13_child<0 ? id_nbr_13_child:id_nbr_13_child + 2;
			s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
			s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
		}
	}
	__syncthreads();
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 26*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
	}
	__syncthreads();
#endif
}

__global__
void Cu_UpdateMasks_1
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_mask, int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int mask_ID = 0;
	
	if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
	{
		// Masking is only necessary for blocks with children.
		int i_c = cblock_ID_nbr_child[kap];
		if (i_c > -1)
		{
			// Loop over neighbor-children. A value of N_SKPID indicates that at least one child is gonna be masked.
			for (int p = 1; p < l_dq_max; p++)
			{
				// Focus only on N_SKIPID, other negative values represent domain boundaries which don't require masks.
				if (cblock_ID_nbr_child[kap + p*n_maxcblocks] == N_SKIPID)
					mask_ID = 1;
			}
		}
		
		cblock_ID_mask[kap] = mask_ID;
	}
}

__global__
void Cu_UpdateMasks_2
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_nbr, int *cblock_ID_ref, int *cells_ID_mask
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ int s_ID_mask[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
#if (N_DIM==3)
	int K_kap = (threadIdx.x / Nbx) / Nbx;
#endif
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
		s_ID_cblock[threadIdx.x] = kap;
	__syncthreads();

	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		
		if (i_kap_b > -1)
		{
			s_ID_mask[threadIdx.x] = 0;
#if (N_DIM==2)
			if (cblock_ID_nbr[i_kap_b + 1*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap >= 0 && J_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 2*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 3*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap >= 0 && J_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 4*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 5*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 6*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 7*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 8*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 1*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap >= 0 && J_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 2*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 3*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap >= 0 && J_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 4*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 5*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 6*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 7*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 8*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = s_ID_mask[threadIdx.x];
			__syncthreads();
#else
			if (cblock_ID_nbr[i_kap_b + 1*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 2*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 3*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= Nbx-4) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 4*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap < 4) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 5*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 6*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= 0 && J_kap < Nbx) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 7*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap >= Nbx-4) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 8*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap < 4) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 9*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 10*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap >= 0 && J_kap < Nbx) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 11*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= Nbx-4) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 12*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap < 4) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 13*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap < 4) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 14*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap >= Nbx-4) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 15*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap >= 0 && J_kap < Nbx) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 16*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 17*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= Nbx-4) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 18*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap < 4) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 19*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap >= Nbx-4) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 20*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap < 4) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 21*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap >= Nbx-4) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 22*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap < 4) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 23*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap < 4) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 24*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap >= Nbx-4) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 25*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 4) && (J_kap >= Nbx-4) && (K_kap >= Nbx-4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 26*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-4) && (J_kap < 4) && (K_kap < 4) )
					s_ID_mask[threadIdx.x] = 1;
			}
			if (cblock_ID_nbr[i_kap_b + 1*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 2*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 3*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= Nbx-2) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 4*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap < 2) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 5*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 6*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= 0 && J_kap < Nbx) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 7*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap >= Nbx-2) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 8*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap < 2) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 9*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 10*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap >= 0 && J_kap < Nbx) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 11*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= Nbx-2) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 12*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap < 2) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 13*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap < 2) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 14*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap >= Nbx-2) && (K_kap >= 0 && K_kap < Nbx) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 15*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap >= 0 && J_kap < Nbx) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 16*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap >= 0 && J_kap < Nbx) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 17*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap >= Nbx-2) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 18*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= 0 && I_kap < Nbx) && (J_kap < 2) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 19*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap >= Nbx-2) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 20*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap < 2) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 21*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap >= Nbx-2) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 22*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap < 2) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 23*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap < 2) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 24*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap >= Nbx-2) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 25*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap < 2) && (J_kap >= Nbx-2) && (K_kap >= Nbx-2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			if (cblock_ID_nbr[i_kap_b + 26*n_maxcblocks] == N_SKIPID)
			{
				if ( (I_kap >= Nbx-2) && (J_kap < 2) && (K_kap < 2) )
					s_ID_mask[threadIdx.x] = 2;
			}
			cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = s_ID_mask[threadIdx.x];
			__syncthreads();
#endif
		}
	}
}



/*
                d8888 888                           d8b 888    888                             
               d88888 888                           Y8P 888    888                             
              d88P888 888                               888    888                             
             d88P 888 888  .d88b.   .d88b.  888d888 888 888888 88888b.  88888b.d88b.           
            d88P  888 888 d88P"88b d88""88b 888P"   888 888    888 "88b 888 "888 "88b          
           d88P   888 888 888  888 888  888 888     888 888    888  888 888  888  888          
          d8888888888 888 Y88b 888 Y88..88P 888     888 Y88b.  888  888 888  888  888          
88888888 d88P     888 888  "Y88888  "Y88P"  888     888  "Y888 888  888 888  888  888 88888888 
                               888                                                             
                          Y8b d88P                                                             
                           "Y88P"                                                              
*/



struct is_marked_for_refinement
{
	__device__ bool operator()(const int ref_id)
	{
		return ref_id==V_REF_ID_MARK_REFINE;
	}
};

struct is_marked_for_coarsening
{
	__device__ bool operator()(const int ref_id)
	{
		return ref_id==V_REF_ID_MARK_COARSEN;
	}
};

struct is_marked_for_removal
{
	__device__ bool operator()(const int ref_id)
	{
		return ref_id==V_REF_ID_REMOVE;
	}
};

struct is_newly_generated
{
	__device__ bool operator()(const int ref_id)
	{
		return ref_id==V_REF_ID_NEW;
	}
};

struct is_equal_to
{
	is_equal_to(int level) : level_{level} {}
	__device__ bool operator()(const int level)
	{
		return level==level_;
	}
	int level_;
};

struct is_removed
{
	__device__ bool operator()(const int ID)
	{
		return ID==N_SKIPID;
	}
};

struct is_not_removed
{
	__device__ bool operator()(const int ID)
	{
		return ID!=N_SKIPID;
	}
};

struct is_nonnegative
{
	__device__ bool operator()(const int ID)
	{
		return ID>=0;
	}
};





__global__
void Cu_RefineCells_Prep
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_ID_nbr,
	int *efficient_map
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr)
	{
		int ref_kap = cblock_ID_ref[kap];
		if (ref_kap == V_REF_ID_MARK_REFINE || ref_kap == V_REF_ID_MARK_COARSEN)
		{
			for (int p = 0; p < l_dq_max; p++)
				efficient_map[kap + p*n_maxcblocks] = cblock_ID_nbr[kap + p*n_maxcblocks];
		}
	}
}

// NOTE: I'm trying to mark unrefined cell-blocks violating the quality criterion. After that, I'll go back to cell-blocks marked for coarsening and check the children - if they have at least one violating the criterion, unmark them.
__global__
void Cu_RefineCells_Q1_1
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
	{
		bool has_refined_nbr = false;
		for (int p = 1; p < l_dq_max; p++)
		{
			if (cblock_ID_nbr_child[kap + p*n_maxcblocks] >= 0)
				has_refined_nbr = true;
		}
		
		if (has_refined_nbr)
			cblock_ID_ref[kap] = V_REF_ID_UNREFINED_VIO;
	}
}

// NOTE: The coarsening corrector. Basically, if at least ONE child is near a refined neighbor AND the current block is near a boundary, do not proceed with coarsening as this creates an interface with refinement scale of 4 rather than 2.
__global__
void Cu_RefineCells_Q1_2
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
	__shared__ int s_ID_child[M_BLOCK*N_CHILDREN];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int ref_kap = -1;
	
	// Initialize shared memory array.
	for (int xc = 0; xc < N_CHILDREN; xc++)
		s_ID_child[xc + threadIdx.x*N_CHILDREN] = -1;
	__syncthreads();
	
	// Fill shared memory array with child IDs of refined cell-blocks.
	if (kap < id_max_curr)
	{
		ref_kap = cblock_ID_ref[kap];
		
		if (ref_kap == V_REF_ID_MARK_COARSEN)
		{
			int i_c = cblock_ID_nbr_child[kap];
			
			for (int xc = 0; xc < N_CHILDREN; xc++)
				s_ID_child[xc + threadIdx.x*N_CHILDREN] = i_c + xc;
		}
	}
	__syncthreads();
	
	// Go through recorded children and replace IDs with refinement IDs.
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int i_q = s_ID_child[threadIdx.x + q*M_BLOCK];
		if (i_q >= 0)
			s_ID_child[threadIdx.x + q*M_BLOCK] = cblock_ID_ref[i_q];
	}
	__syncthreads();
	
	// Now, evaluate the quality criterion.
	if (kap < id_max_curr && ref_kap == V_REF_ID_MARK_COARSEN)
	{
		bool near_boundary = false;
		bool has_violating_child = false;
		int nbr_id_p;
		
		for (int xc = 0; xc < N_CHILDREN; xc++)
		{
			if (s_ID_child[xc + threadIdx.x*N_CHILDREN] == V_REF_ID_UNREFINED_VIO || s_ID_child[xc + threadIdx.x*N_CHILDREN] == V_REF_ID_MARK_REFINE)
				has_violating_child = true;
		}
		
		for (int p = 1; p < l_dq_max; p++)
		{
			nbr_id_p = cblock_ID_nbr[kap + p*n_maxcblocks];
			if (nbr_id_p < 0)
				near_boundary = true;
		}
		
		// If both violating conditions are satisifed, revert to a regular refined cell.
		if (has_violating_child)
		//if (near_boundary || has_violating_child)
			cblock_ID_ref[kap] = V_REF_ID_REFINED;
	}
}



// Formerly: Cu_RefineCells_S1.
__global__
void Cu_AddRemoveBlocks
(
	int id_max_curr, int n_maxcblocks, ufloat_t dx,
	int *cblock_ID_nbr_child, int *cblock_ID_ref, int *cblock_level, ufloat_t *cblock_f_X,
	int *scattered_map
)
{
	// Child constructor parameters.
	unsigned int t_id_incrementor = threadIdx.x%N_CHILDREN; //threadIdx.x&(N_CHILDREN-1);
	unsigned int t_id_builder = threadIdx.x/N_CHILDREN; //threadIdx.x>>log2(N_CHILDREN);
	unsigned int t_xi_builder = threadIdx.x%2; //threadIdx.x&(1);
	unsigned int t_xj_builder = (threadIdx.x/2)%2; //(threadIdx.x>>log2(2))&(1);
#if (N_DIM==3)
	unsigned int t_xk_builder = (threadIdx.x/4)%2; //(threadIdx.x>>log2(4))&(1);
#endif
	
	__shared__ int s_ID_child[M_BLOCK];
	__shared__ int s_ref[M_BLOCK];
	__shared__ int s_level[M_BLOCK];
	__shared__ ufloat_t s_x[M_BLOCK];
	__shared__ ufloat_t s_y[M_BLOCK];
#if (N_DIM==3)
	__shared__ ufloat_t s_z[M_BLOCK];
#endif
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Initialize shared memory.
	s_ID_child[threadIdx.x] = -1;
	s_ref[threadIdx.x] = -1;
	s_level[threadIdx.x] = -1;
	s_x[threadIdx.x] = -1;
	s_y[threadIdx.x] = -1;
#if (N_DIM==3)
	s_z[threadIdx.x] = -1;
#endif
	__syncthreads();
	
	if (kap < id_max_curr)
	{
		// If this particular ID on the map is a valid ID, we need to transcribe the spatial location and level corresponding to the parent.
		// I've reset s_ID_child to -1 to know which values should be skipped.
		//if (scattered_map[kap] > -1)
		if (cblock_ID_ref[kap] == V_REF_ID_MARK_REFINE)
		{
			s_ID_child[threadIdx.x] = scattered_map[kap];
			if (scattered_map[kap] < 0)
				printf("Uh oh...(S1, scatter)\n");
			s_ref[threadIdx.x] = V_REF_ID_NEW;
			s_level[threadIdx.x] = cblock_level[kap];
			s_x[threadIdx.x] = cblock_f_X[kap + 0*n_maxcblocks];
			s_y[threadIdx.x] = cblock_f_X[kap + 1*n_maxcblocks];
#if (N_DIM==3)
			s_z[threadIdx.x] = cblock_f_X[kap + 2*n_maxcblocks];
#endif
			
			cblock_ID_nbr_child[kap] = s_ID_child[threadIdx.x];
			//for (int xc = 0; xc < N_CHILDREN; xc++)
			//	cblock_ID_child[kap + xc*n_maxcblocks] = s_ID_child[threadIdx.x]+xc;
		}
		if (cblock_ID_ref[kap] == V_REF_ID_MARK_COARSEN)
		{
			s_ID_child[threadIdx.x] = cblock_ID_nbr_child[kap];
			s_ref[threadIdx.x] = V_REF_ID_REMOVE;
			
			if (s_ID_child[threadIdx.x] >= id_max_curr)
				printf("Uh oh...(S1, ID max. violated)\n");
			
			cblock_ID_nbr_child[kap] = N_SKIPID;
			//for (int xc = 0; xc < N_CHILDREN; xc++)
			//	cblock_ID_child[kap + xc*n_maxcblocks] = N_SKIPID;
		}
	}
	__syncthreads();
	
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		int ref_q = s_ref[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
		
		// Only write if there actually is a child.
		//if (ID_child_q > -1)
		if (ref_q == V_REF_ID_NEW)
		{
			if (ID_child_q < 0)
				printf("Uh oh...(S1, new)\n");
			int level_q = s_level[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ] + 1;
			cblock_level[ID_child_q + t_id_incrementor] = level_q;
			cblock_ID_ref[ID_child_q + t_id_incrementor] = V_REF_ID_NEW;
			cblock_f_X[ID_child_q + t_id_incrementor + 0*n_maxcblocks] = s_x[t_id_builder + q*(M_BLOCK/N_CHILDREN)] + t_xi_builder*(dx/(1<<level_q));
			cblock_f_X[ID_child_q + t_id_incrementor + 1*n_maxcblocks] = s_y[t_id_builder + q*(M_BLOCK/N_CHILDREN)] + t_xj_builder*(dx/(1<<level_q));
#if (N_DIM==3)
			cblock_f_X[ID_child_q + t_id_incrementor + 2*n_maxcblocks] = s_z[t_id_builder + q*(M_BLOCK/N_CHILDREN)] + t_xk_builder*(dx/(1<<level_q));
#endif
			cblock_ID_nbr_child[ID_child_q + t_id_incrementor] = N_SKIPID;
		}
		if (ref_q == V_REF_ID_REMOVE)
		{
			if (ID_child_q < 0)
				printf("Uh oh...(S1, remove)\n");
			cblock_ID_ref[ID_child_q + t_id_incrementor] = V_REF_ID_REMOVE;
		}
	}
}

__global__
void Cu_RefineCells_S2
(
	int id_max_curr, int n_maxcblocks, int n_ids_marked_refine, int *ids_marked, int *new_child_ids, int n_ids_marked_coarsen,
	int *cblock_ID_nbr, int *cblock_ID_nbr_child,
	int *scattered_map
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (kap < id_max_curr)
	{
		bool update_scattered = false;
		for (int p = 0; p < l_dq_max; p++)
		{
			// Loop over marked IDs. Modify the nbr_child (i.e. neighbor's first child ID) using updated values.
			// Also, keep note of cblocks near other cblocks marked for refinement/coarsening - their connectivity needs to be updated.
			for (int k = 0; k < n_ids_marked_refine; k++)
			{
				if (cblock_ID_nbr[kap + p*n_maxcblocks] == ids_marked[k])
				{
					cblock_ID_nbr_child[kap + p*n_maxcblocks] = new_child_ids[k];
					update_scattered = true;
				}
			}
			for (int k = n_ids_marked_refine; k < n_ids_marked_refine+n_ids_marked_coarsen; k++)
			{
				if (cblock_ID_nbr[kap + p*n_maxcblocks] == ids_marked[k])
				{
					cblock_ID_nbr_child[kap + p*n_maxcblocks] = N_SKIPID;
					if (p != 0)
						update_scattered = true;
				}	
			}
		}
		
		// If at least one neighbor was marked for something, include this cblock in scattered map and update its child connectivity.
		// For coarsening, account for non-zero p since the coarsened cell has no children to process after coarsening.
		if (update_scattered)
			scattered_map[kap] = cblock_ID_nbr_child[kap];
	}
}

// NOTE: Here, update the child_nbr IDs according to refinement only. Update due to coarsening is deferred post correction.
__global__
void Cu_RefineCells_S2_V1_1
(
	int id_max_curr, int n_maxcblocks, int n_ids_marked_refine, int *ids_marked, int *new_child_ids,
	int *cblock_ID_nbr, int *cblock_ID_nbr_child,
	int *efficient_map
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (kap < id_max_curr && efficient_map[kap] >= 0)
	{
		for (int p = 0; p < l_dq_max; p++)
		{
			// Loop over marked IDs. Modify the nbr_child (i.e. neighbor's first child ID) using updated values.
			// Also, keep note of cblocks near other cblocks marked for refinement/coarsening - their connectivity needs to be updated.
			for (int k = 0; k < n_ids_marked_refine; k++)
			{
				if (cblock_ID_nbr[kap + p*n_maxcblocks] == ids_marked[k])
					cblock_ID_nbr_child[kap + p*n_maxcblocks] = new_child_ids[k];
			}
		}
	}
}

// NOTE: Update due to coarsening.
__global__
void Cu_RefineCells_S2_V1_2
(
	int id_max_curr, int n_maxcblocks, int n_ids_marked_coarsen, int *ids_marked,
	int *cblock_ID_nbr, int *cblock_ID_nbr_child,
	int *efficient_map
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (kap < id_max_curr && efficient_map[kap] >= 0)
	{
		for (int p = 1; p < l_dq_max; p++)
		{
			for (int k = 0; k < n_ids_marked_coarsen; k++)
			{
				if (cblock_ID_nbr[kap + p*n_maxcblocks] == ids_marked[k])
					cblock_ID_nbr_child[kap + p*n_maxcblocks] = N_SKIPID;
			}
		}
	}
}

__global__
void Cu_RefineCells_S2_V2_1
(
	int id_max_curr, int n_maxcblocks, 
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child,
	int *efficient_map, int *scattered_map
)
{
	__shared__ int s_ID_nbr[M_BLOCK*9];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	for (int p = 0; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	
#if (N_DIM==3)
	for (int k = 0; k < 3; k++)
	{
#else
	int k = 0;
#endif
	
	// First, read neighbor Ids and place in shared memory. Arrange for contiguity.
	if (kap < id_max_curr && efficient_map[kap] >= 0)
	{
		for (int p = 0; p < 9; p++)
			s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
	}
	__syncthreads();
	
	// Replace neighbor Ids with their respective marks.
	for (int p = 0; p < 9; p++)
	{
		int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
		if (i_p > -1)
		{
			s_ID_nbr[threadIdx.x + p*M_BLOCK] = cblock_ID_ref[i_p];
			if (s_ID_nbr[threadIdx.x + p*M_BLOCK] == V_REF_ID_MARK_REFINE)
				s_ID_nbr[threadIdx.x + p*M_BLOCK] = scattered_map[i_p];
			else
				s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
		}
	}
	__syncthreads();
	
	// Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
	if (kap < id_max_curr && efficient_map[kap] >= 0)
	{
		for (int p = 0; p < 9; p++)
		{
			if (s_ID_nbr[p + threadIdx.x*9] > -1)
				cblock_ID_nbr_child[kap + (k*9+p)*n_maxcblocks] = s_ID_nbr[p + threadIdx.x*9];
		}
	}
	
#if (N_DIM==3)
	__syncthreads();
	
	for (int p = 0; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	}
#endif
}

__global__
void Cu_RefineCells_S2_V2_2
(
	int id_max_curr, int n_maxcblocks, 
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child,
	int *efficient_map
)
{
	__shared__ int s_ID_nbr[M_BLOCK*9];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	for (int p = 0; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	
#if (N_DIM==3)
	for (int k = 0; k < 3; k++)
	{
#else
	int k = 0;
#endif
	
	// First, read neighbor Ids and place in shared memory. Arrange for contiguity.
	if (kap < id_max_curr && efficient_map[kap] >= 0)
	{
		for (int p = 0; p < 9; p++)
			s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
	}
	__syncthreads();
	
	// Replace neighbor Ids with their respective marks.
	for (int p = 0; p < 9; p++)
	{
		int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
		if (i_p > -1)
			s_ID_nbr[threadIdx.x + p*M_BLOCK] = cblock_ID_ref[i_p];
	}
	__syncthreads();
	
	// Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
	if (kap < id_max_curr && efficient_map[kap] >= 0)
	{
		for (int p = 0; p < 9; p++)
		{
			if ((k*9+p) > 0 && s_ID_nbr[p + threadIdx.x*9] == V_REF_ID_MARK_COARSEN)
				cblock_ID_nbr_child[kap + (k*9+p)*n_maxcblocks] = N_SKIPID;
		}
	}
	
#if (N_DIM==3)
	__syncthreads();
	
	for (int p = 0; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	}
#endif
}

__global__
void Cu_RefineCells_S3
(
	int id_max_curr_wnew,
	int *cblock_ID_ref
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr_wnew)
	{
		int ref_kap = cblock_ID_ref[kap];
		if (ref_kap == V_REF_ID_MARK_REFINE)
			cblock_ID_ref[kap] = V_REF_ID_REFINED;
		if (ref_kap == V_REF_ID_NEW || ref_kap == V_REF_ID_MARK_COARSEN)
			cblock_ID_ref[kap] = V_REF_ID_UNREFINED;
		if (ref_kap == V_REF_ID_REMOVE)
			cblock_ID_ref[kap] = V_REF_ID_INACTIVE;
		
		// If at this point this wasn't overwritten with 'V_REF_ID_REMOVE', it can be safely reverted to 'V_REF_ID_UNREFINED' instead.
		if (ref_kap == V_REF_ID_UNREFINED_VIO)
			cblock_ID_ref[kap] = V_REF_ID_UNREFINED;
	}
}

__global__
void Cu_RefineCells_Cancel
(
	int id_max_curr_wnew,
	int *cblock_ID_ref
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr_wnew)
	{
		int ref_kap = cblock_ID_ref[kap];
		if (ref_kap == V_REF_ID_MARK_REFINE)
			cblock_ID_ref[kap] = V_REF_ID_UNREFINED;
	}
}

__global__
void Cu_RefineCells_S4
(
	int id_max_curr_wnew,
	int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
	//__shared__ int s_ID_ref_parent[M_BLOCK];
	__shared__ int s_ID_child[M_BLOCK*N_CHILDREN];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int ref_kap = -1;
	int new_ref_id = -1;
	
	// Initialize shared memory array.
	//s_ID_ref_parent[M_BLOCK] = V_REF_ID_REFINED;
	for (int xc = 0; xc < N_CHILDREN; xc++)
		s_ID_child[xc + threadIdx.x*N_CHILDREN] = -1;
	__syncthreads();
	
	// Fill shared memory array with child IDs of refined cell-blocks.
	if (kap < id_max_curr_wnew)
	{
		ref_kap = cblock_ID_ref[kap];
		
		// If a cell-block is refined, loop over its children and check whether any of them have been refined too. If so, this cell-block is a branch.
		// A refined cell-block always has children with ID >= 0, no need for conditional.
		if (ref_kap == V_REF_ID_REFINED || ref_kap == V_REF_ID_REFINED_WCHILD)
		{
			int i_c = cblock_ID_nbr_child[kap];
			new_ref_id = V_REF_ID_REFINED;
			
			for (int xc = 0; xc < N_CHILDREN; xc++)
				s_ID_child[xc + threadIdx.x*N_CHILDREN] = i_c + xc;
		}
	}
	__syncthreads();
	
	// Loop over shared memory array and retrieve refinement IDs. Replace the child IDs in shared memory.
	// Doing it this way guarantees a degree of coalescence since children are all in a row in memory.
	for (int q = 0; q < N_CHILDREN; q++)
	{
		int i_q = s_ID_child[threadIdx.x + q*M_BLOCK];
		if (i_q >= 0)
		{
			s_ID_child[threadIdx.x + q*M_BLOCK] = cblock_ID_ref[i_q];
			
			//int ref_q = cblock_ID_ref[i_q];
			//if (ref_q == V_REF_ID_REFINED)
			//	s_ID_ref_parent[(threadIdx.x + q*M_BLOCK)/N_CHILDREN] = V_REF_ID_REFINED_WCHILD;
		}
	}
	__syncthreads();
	
	// Loop over children at parent level and check refinement IDs. If at least one child is refined, mark the parent as a branch instead of leaf.
	if (kap < id_max_curr_wnew)
	{
		for (int xc = 0; xc < N_CHILDREN; xc++)
		{
			if (s_ID_child[xc + threadIdx.x*N_CHILDREN] == V_REF_ID_REFINED || s_ID_child[xc + threadIdx.x*N_CHILDREN] == V_REF_ID_REFINED_WCHILD)
				new_ref_id = V_REF_ID_REFINED_WCHILD;
		}
		
		if (new_ref_id > -1)
			cblock_ID_ref[kap] = new_ref_id;
	}
}

__global__
void Cu_CoarsenCells_S1
(
	int n_ids_idev_L, int n_ids_marked_removal, int *ids_marked,
	int *id_set_idev_L
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_ids_idev_L)
	{
		// Loop over IDs.
		for (int k = 0; k < n_ids_marked_removal; k++)
		{
			if (id_set_idev_L[kap] == ids_marked[k])
			{
				id_set_idev_L[kap] = N_SKIPID;
			}
		}
	}
}

__global__
void Cu_CoarsenCells_S2
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr)
	{
		if (cblock_ID_ref[kap] == V_REF_ID_REMOVE)
		{
			for (int p = 0; p < l_dq_max; p++)
			{
				if (p > 0)
					cblock_ID_nbr[kap + p*n_maxcblocks] = N_SKIPID;
				cblock_ID_nbr_child[kap + p*n_maxcblocks] = N_SKIPID;
			}
		}
	}
}

__global__
void Cu_FreezeRefined
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr)
	{
		if (cblock_ID_ref[kap] == V_REF_ID_REFINED)
			cblock_ID_ref[kap] = V_REF_ID_REFINED_PERM;
		if (cblock_ID_ref[kap] == V_REF_ID_REFINED_WCHILD)
			cblock_ID_ref[kap] = V_REF_ID_REFINED_WCHILD_PERM;
	}
}

__global__
void Cu_UnfreezeRefined
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr)
	{
		if (cblock_ID_ref[kap] == V_REF_ID_REFINED_PERM)
			cblock_ID_ref[kap] = V_REF_ID_REFINED;
		if (cblock_ID_ref[kap] == V_REF_ID_REFINED_WCHILD_PERM)
			cblock_ID_ref[kap] = V_REF_ID_REFINED_WCHILD;
	}
}

__global__
void Cu_RefinementValidator_1
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr)
	{
		int nbr_child_0 = cblock_ID_nbr_child[kap];
		if (cblock_ID_ref[kap] == V_REF_ID_REFINED && nbr_child_0 < 0)
		{
			printf("Uh oh (refined block violated condition)...\n");
		}
		if (cblock_ID_ref[kap] == V_REF_ID_UNREFINED && nbr_child_0 > -1)
		{
			printf("Uh oh (unrefined block violated condition)...\n");
		}
	}
}

__global__
void Cu_RefinementValidator_2
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_ref, int *cblock_level, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr)
	{
		for (int p = 0; p < l_dq_max; p++)
		{
			int nbr_id_p = cblock_ID_nbr[kap + p*n_maxcblocks];
			if (nbr_id_p > -1)
			{
				int ref_kap_nbr_p = cblock_ID_ref[nbr_id_p];
				if ( (ref_kap_nbr_p == V_REF_ID_REFINED || ref_kap_nbr_p == V_REF_ID_REFINED_WCHILD || ref_kap_nbr_p == V_REF_ID_REFINED_PERM || ref_kap_nbr_p == V_REF_ID_REFINED_WCHILD_PERM || ref_kap_nbr_p == V_REF_ID_MARK_REFINE) && cblock_ID_nbr_child[kap + p*n_maxcblocks] == N_SKIPID)
					printf("Uh oh (bad connectivity data for block %i on level %i, direction %i)...\n", kap, cblock_level[kap], p);
			}
		}
	}
}

__global__
void Cu_RefinementValidator_3
(
	int id_max_curr, int n_maxcblocks,
	int *cells_ID_mask, int *cblock_ID_mask, int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
		s_ID_cblock[threadIdx.x] = kap;
	__syncthreads();

	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		
		if (i_kap_b > -1)
		{
			int mask_block_ID = cblock_ID_mask[i_kap_b];
			int child_zero_ID = cblock_ID_nbr_child[i_kap_b];
			
			if (child_zero_ID > -1)
			{
				for (int xc = 0; xc < N_CHILDREN; xc++)
				{
					int mask_cell_ID = cells_ID_mask[(child_zero_ID+xc)*M_CBLOCK + threadIdx.x];
					
					if (mask_cell_ID > 0 && mask_block_ID != 1)
						printf("Uh oh...block has a ghost layer in children but is not marked with right mask...\n");
				}
			}
		}
	}
}

__global__
void Cu_RefinementValidator_4
(
	int id_max_curr, int n_maxcblocks,
	int *cblock_ID_nbr_child
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_maxcblocks)
	{
		int i_c = cblock_ID_nbr_child[kap];
		
		if (i_c >= id_max_curr)
			printf("Hmmm...ID max [%i]. violated at prep. stage (%i, i_c=%i)\n", id_max_curr, kap, i_c);
	}
}

int Mesh::M_FreezeRefinedCells(int var)
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		// Freeze.
		if (var == 0)
		{
			Cu_FreezeRefined<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max[i_dev][MAX_LEVELS], n_maxcblocks,
				c_cblock_ID_ref[i_dev]
			);
		}
		
		// Unfreeze.
		if (var == 1)
		{
			Cu_UnfreezeRefined<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max[i_dev][MAX_LEVELS], n_maxcblocks,
				c_cblock_ID_ref[i_dev]
			);
		}
	}
	
	return 0;
}

int Mesh::M_RefineAndCoarsenCells(int var, ufloat_t *scale_vec, std::ofstream *file)
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
#if (P_SHOW_REFINE==1)
		tic_simple("[Pre]");
#endif
		// - c_tmp_1 is used to store the indices of blocks marked for refinement and those for marked for coarsening afterwards.
		// - c_tmp_2 is used to store the map.
		// - c_tmp_3 is used to store spliced gaps in reverse order.
		// - c_tmp_4 is used to store levels of the blocks marked for refinement.
		// - c_tmp_5 is used to store a copy of modified ID of the ID set before they are filtered of blocks marked for removal.
		// - Using n_ids_marked_refine, the gap set is spliced at the end and reversed in order before scattering and using in later routines.
		
		// First, collect an array of cblocks marked for refinement along with their count via thrust routines 'copy_if' and 'count_if', respectively.
		int id_max_curr = id_max[i_dev][MAX_LEVELS];
		int n_ids_marked_refine = thrust::count_if(
			thrust::device, c_cblock_ID_ref_dptr[i_dev], c_cblock_ID_ref_dptr[i_dev] + id_max_curr, is_marked_for_refinement()
		);
		int n_ids_marked_coarsen = thrust::count_if(
			thrust::device, c_cblock_ID_ref_dptr[i_dev], c_cblock_ID_ref_dptr[i_dev] + id_max_curr, is_marked_for_coarsening()
		);
		int id_max_curr_wnew = id_max_curr + N_CHILDREN*n_ids_marked_refine;
		Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
			n_maxcblocks, c_tmp_1[i_dev], -1
		);
		Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
			n_maxcblocks, c_tmp_2[i_dev], -1
		);
		Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
			n_maxcblocks, c_tmp_3[i_dev], -1
		);
		Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
			n_maxcblocks, c_tmp_4[i_dev], -1
		);
		Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
			n_maxcblocks, c_tmp_5[i_dev], -1
		);
		Cu_ResetToValue<<<(M_BLOCK+(n_maxcblocks*l_dq_max)-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
			n_maxcblocks*l_dq_max, c_tmp_6[i_dev], -1
		);
		Cu_ResetToValue<<<(M_BLOCK+(n_maxcblocks*l_dq_max)-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
			n_maxcblocks*l_dq_max, c_tmp_7[i_dev], -1
		);
		Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
			n_maxcblocks, c_tmp_8[i_dev], -1
		);	
		
		// First, collect an array of cblocks marked for refinement along with their count via thrust routines 'copy_if' and 'count_if', respectively.
		// Make sure that this step is skipped if there are not enough gaps left in the gap set.
		bool proceed_refinement = n_ids_marked_refine > 0 && N_CHILDREN*n_ids_marked_refine <= n_gaps[i_dev];
		bool proceed_coarsening = n_ids_marked_coarsen > 0;
		
		// TODO
		std::cout << "Numbers (no. ref.): " << n_ids_marked_refine << std::endl;
		
		if (!proceed_refinement && n_ids_marked_refine > 0)
		{
			std::cout << "Canceling, not enough space to refine further..." << std::endl;
			Cu_RefineCells_Cancel<<<(M_BLOCK+id_max_curr_wnew-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr_wnew, c_cblock_ID_ref[i_dev]
			);
			
			n_ids_marked_refine = 0;
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << var << " " << n_ids_marked_refine << " " << n_ids_marked_coarsen << " " << toc_simple("[Pre]",T_US) << " ";
#endif
		
		

#if (P_SHOW_REFINE==1)
		tic_simple("[S1]");
#endif		
		// Now we need to set up the scatter map for efficiently establishing connectivity. The plan is to 1) loop over cblocks and copy their neighbors down if marked for refinement/coarsening, 2) sort them and perform a unique copy in case of repitions (cblocks near each other may call on the same neighbors), 3) scatter them and use the scattered map to update connectivity only of cblocks in the vincinity of marked cblocks.
		if (proceed_refinement || proceed_coarsening)
		{
			// Loop over cblocks and copy down the neigbors IDs of marked cblocks in efficient_map. The efficient_map must be reset before this step.
			Cu_RefineCells_Prep<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr, n_maxcblocks,
				c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev],
				c_tmp_6[i_dev]
			);

			// Count the number of recorded non-negative IDs.
			int n_nonnegative_prev = thrust::count_if(
				thrust::device, c_tmp_6_dptr[i_dev], c_tmp_6_dptr[i_dev] + n_maxcblocks*l_dq_max, is_nonnegative()
			);
			
			//tic_simple("Copy");
			// Regular-copy non-negative IDs in efficient_map. Copied into c_tmp_7 since there is a (honestly low) possibility that the number of copied IDs exceeds n_maxcblocks by a good amount.
			thrust::copy_if(
				thrust::device, c_tmp_6_dptr[i_dev], c_tmp_6_dptr[i_dev] + n_maxcblocks*l_dq_max, c_tmp_7_dptr[i_dev], is_nonnegative()
			);
			/*
			int n_nonnegative = 0;
			int n_nonnegative_prev = 0;
			for (int p = 0; p < l_dq_max; p++)
			{
				n_nonnegative = thrust::count_if(
					thrust::device, &c_tmp_6_dptr[i_dev][p*n_maxcblocks], &c_tmp_6_dptr[i_dev][p*n_maxcblocks] + id_max_curr, is_nonnegative()
				);
				
				if (n_nonnegative > 0)
				{
					thrust::copy_if(
						thrust::device, &c_tmp_6_dptr[i_dev][p*n_maxcblocks], &c_tmp_6_dptr[i_dev][p*n_maxcblocks] + id_max_curr, &c_tmp_7_dptr[i_dev][n_nonnegative_prev], is_nonnegative()
					);
					
					n_nonnegative_prev += n_nonnegative;
				}
			}*/

			// Sort the copied IDs to prepare for the unique-copy.
			thrust::sort(
				thrust::device, c_tmp_7_dptr[i_dev], c_tmp_7_dptr[i_dev] + n_nonnegative_prev
			);
			
			// Reset in preparation for unique copy.
			Cu_ResetToValue<<<(M_BLOCK+(n_maxcblocks*l_dq_max)-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				n_maxcblocks*l_dq_max, c_tmp_6[i_dev], -1
			);
			
			// Perform the unique-copy. At this stage, c_tmp_6 shouldn't exceed n_maxcblocks in value so it is safe to scatter to a new c_tmp.
			thrust::unique_copy(
				thrust::device, c_tmp_7_dptr[i_dev], c_tmp_7_dptr[i_dev] + n_nonnegative_prev, c_tmp_6_dptr[i_dev]
			);
			
			// Re-count the number of recorded non-negative IDs.
			n_nonnegative_prev = thrust::count_if(
				thrust::device, c_tmp_6_dptr[i_dev], c_tmp_6_dptr[i_dev] + n_nonnegative_prev, is_nonnegative()
			);
			
			// Scatter the IDs.
			thrust::scatter(
				thrust::device, c_tmp_6_dptr[i_dev], c_tmp_6_dptr[i_dev] + n_nonnegative_prev, c_tmp_6_dptr[i_dev], c_tmp_8_dptr[i_dev]
			);
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << toc_simple("[S1]",T_US) << " ";
#endif
		
		
		
#if (P_SHOW_REFINE==1)
		tic_simple("[S2]");
#endif
		// Copy marked block indices into temporary array so that they're contiguous in memory, capture gaps required for proceeding steps.
		if (proceed_refinement)
		{
			// Copy indices of blocks marked for refinement.
			thrust::copy_if(
				//thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(id_max_curr), c_cblock_ID_ref_dptr[i_dev], c_tmp_1_dptr[i_dev], is_marked_for_refinement()
				thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max_curr, c_cblock_ID_ref_dptr[i_dev], c_tmp_1_dptr[i_dev], is_marked_for_refinement()
			);
			
			// Splice the gap set starting at (n_gaps-1) - n_ids_marked_refine, reverse its order with thrust.
			thrust::reverse_copy(
				thrust::device,  c_gap_set_dptr[i_dev] + (n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine), c_gap_set_dptr[i_dev] + (n_gaps[i_dev]), c_tmp_3_dptr[i_dev]
				//thrust::device,  &c_gap_set_dptr[i_dev][n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine], &c_gap_set_dptr[i_dev][n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine] + N_CHILDREN*n_ids_marked_refine, c_tmp_3_dptr[i_dev]
			);
			
			// Update the ID max. since new children IDs might exceed current one after previous shuffling of gaps.
			id_max_curr_wnew = std::max(id_max_curr_wnew, *thrust::max_element(thrust::device, c_tmp_3_dptr[i_dev], c_tmp_3_dptr[i_dev] + N_CHILDREN*n_ids_marked_refine) + 1);
			
			// Contract the child indices (using reversed data in temporary array, append to the end of c_tmp_3.
			Cu_ContractByFrac<<<(M_BLOCK+N_CHILDREN*n_ids_marked_refine-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
				//N_CHILDREN*n_ids_marked_refine, c_tmp_3[i_dev], N_CHILDREN, &c_gap_set[i_dev][n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine]
				
				N_CHILDREN*n_ids_marked_refine, c_tmp_3[i_dev], N_CHILDREN, &c_tmp_3[i_dev][N_CHILDREN*n_ids_marked_refine]
			);
			
			// Scatter gaps to indices of marked cells for the write process, then retrieve newly-generated blocks by level for insertion in ID sets.
			thrust::scatter(
				thrust::device, &c_tmp_3[i_dev][N_CHILDREN*n_ids_marked_refine], &c_tmp_3[i_dev][N_CHILDREN*n_ids_marked_refine] + n_ids_marked_refine, c_tmp_1_dptr[i_dev], c_tmp_2_dptr[i_dev]
			);
			
			// TEST
// 			cudaMemcpy(tmp_1[i_dev], &c_cblock_ID_nbr_child[i_dev][1*n_maxcblocks], id_max_curr_wnew*sizeof(int), cudaMemcpyDeviceToHost);
// 			std::cout << "DEBUG\n";
// 			for (int k = 0; k < id_max_curr_wnew; k++)
// 				std::cout << tmp_1[i_dev][k] << " ";
// 			std::cout << std::endl;
		
#if (N_CONN_TYPE==0)
			// Call S2 routine where new child IDs are inserted in the cblock_ID_nbr_child array.
			Cu_RefineCells_S2_V1_1<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr, n_maxcblocks, n_ids_marked_refine, c_tmp_1[i_dev], &c_tmp_3[i_dev][N_CHILDREN*n_ids_marked_refine],
				c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
				c_tmp_8[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
#else
			// Call S2 routine where new child IDs are inserted in the cblock_ID_nbr_child array.
			Cu_RefineCells_S2_V2_1<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr, n_maxcblocks,
				c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
				c_tmp_8[i_dev], c_tmp_2[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
#endif
			
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << toc_simple("[S2]",T_US) << " ";
#endif
		
		
		
#if (P_SHOW_REFINE==1)
		tic_simple("[S3]");
#endif
		// Now correct the marks for coarsening.
		if (proceed_coarsening)
		{
			// Call the first part of quality control on coarsening.
			Cu_RefineCells_Q1_1<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr, n_maxcblocks,
				c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
			
			// Call the second part of quality control on coarsening.
			Cu_RefineCells_Q1_2<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr, n_maxcblocks,
				c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
			
			// Re-evaluate whether to coarsen or not.
			n_ids_marked_coarsen = thrust::count_if(
			thrust::device, c_cblock_ID_ref_dptr[i_dev], c_cblock_ID_ref_dptr[i_dev] + id_max_curr, is_marked_for_coarsening()
			);
			proceed_coarsening = n_ids_marked_coarsen > 0;
			
			// TODO
			std::cout << "Numbers (no. coarsen.): " << n_ids_marked_coarsen << std::endl;
			
			// If we are still proceeding with coarsening...
			if (proceed_coarsening)
			{
				// Retrieve indices of blocks marked for coarsening, append to c_tmp_1. Needed for Cu_RefineCells_S2.
				thrust::copy_if(
					thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max_curr, c_cblock_ID_ref_dptr[i_dev], 	&c_tmp_1_dptr[i_dev][n_ids_marked_refine], is_marked_for_coarsening()
				);
				
#if (N_CONN_TYPE==0)
				// Call S2 routine where new child IDs are inserted in the cblock_ID_nbr_child array.
				Cu_RefineCells_S2_V1_2<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
					id_max_curr, n_maxcblocks, n_ids_marked_coarsen, &c_tmp_1[i_dev][n_ids_marked_refine],
					c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
					c_tmp_8[i_dev]
				);
				gpuErrchk( cudaPeekAtLastError() );
#else
				// Call S2 routine where new child IDs are inserted in the cblock_ID_nbr_child array.
				Cu_RefineCells_S2_V2_2<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
					id_max_curr, n_maxcblocks,
					c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
					c_tmp_8[i_dev]
				);
				gpuErrchk( cudaPeekAtLastError() );
#endif
			}
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << toc_simple("[S3]",T_US) << " " << n_ids_marked_coarsen << " ";
#endif
		
		
		
#if (P_SHOW_REFINE==1)
		tic_simple("[S4]");
#endif
		// Scatter gaps to indices of marked cells for the write process, then retrieve newly-generated blocks by level for insertion in ID sets.
		if (proceed_refinement || proceed_coarsening)
		{
			Cu_AddRemoveBlocks<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr, n_maxcblocks, dx_cblock,
				c_cblock_ID_nbr_child[i_dev], c_cblock_ID_ref[i_dev], c_cblock_level[i_dev], c_cblock_f_X[i_dev],
				c_tmp_2[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << toc_simple("[S4]",T_US) << " ";
#endif
		
		
		
#if (P_SHOW_REFINE==1)
		tic_simple("[S5]");
#endif			
		//tic("[S3]");
		// Retrieve blocks marked for removal. Run through ID sets and filter out marked blocks. Placed after Cu_AddRemoveBlocks so that children of  blocks marked for coarsening are now marked for removal.
		if (proceed_coarsening)
		{	
			// Retrieve the blocks to be removed, store in c_tmp_5.
			thrust::copy_if(
				thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max_curr_wnew, c_cblock_ID_ref_dptr[i_dev], c_tmp_5_dptr[i_dev], is_marked_for_removal()
			);
			
			// Copy the oldly-set levels of children-to-be-removed. This allows us to remove them from the correct ID set array.
			thrust::copy_if(
				thrust::device, c_cblock_level_dptr[i_dev], c_cblock_level_dptr[i_dev] + id_max_curr_wnew, c_cblock_ID_ref_dptr[i_dev], c_tmp_4_dptr[i_dev], is_marked_for_removal()
			);
			
			// Sort the levels (keys, stored in c_tmp_4) and the associated IDs for removal (values, stored in c_tmp_5). Reduce by key to get the number of gaps to insert into the ID set.
			thrust::stable_sort_by_key(
				thrust::device, c_tmp_4_dptr[i_dev], c_tmp_4_dptr[i_dev] + N_CHILDREN*n_ids_marked_coarsen, c_tmp_5_dptr[i_dev]
			);
					
			int n_rem_levels_L_prev = 0;
			int n_rem_levels_L = 0;
			for (int L = 1; L < MAX_LEVELS; L++)
			{
				// Count the number of IDs to be removed on level L.
				n_rem_levels_L = thrust::count_if(
					thrust::device, c_tmp_4_dptr[i_dev], c_tmp_4_dptr[i_dev] + N_CHILDREN*n_ids_marked_coarsen, is_equal_to(L)
				);
				
				if (n_rem_levels_L > 0)
				{
					// Copy id set to temporary array and change any IDs marked for removal to N_SKIPID.
					Cu_CoarsenCells_S1<<<(M_BLOCK+n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
						//n_ids[i_dev][L], N_CHILDREN*n_ids_marked_coarsen, c_tmp_5[i_dev],
						n_ids[i_dev][L], n_rem_levels_L, &c_tmp_5[i_dev][n_rem_levels_L_prev],
						c_id_set[i_dev][L]
					);
					gpuErrchk( cudaPeekAtLastError() );
					
					// Use remove_if to store only those IDs not marked for removal.
					thrust::remove_if(
						thrust::device, c_id_set_dptr[i_dev][L], c_id_set_dptr[i_dev][L] + n_ids[i_dev][L], is_removed()
					);
					
					n_ids[i_dev][L] -= n_rem_levels_L;
					n_rem_levels_L_prev += n_rem_levels_L;
					std::cout << "Removed " << n_rem_levels_L << " cblocks on Level " << L << "..." << std::endl;
				}
			}
			
			// Insert removed IDs into the gap set. These are the ones stored in c_tmp_5.
			// Concatenation happens at n_gaps - Nc*n_marked_refine since Nc*n_marked_refine gaps will be cut off at the end of the routine. Gaps to be used for refinement have already been processed, so this shouldn't be problematic.
			Cu_ConcatReverse<<<(M_BLOCK+n_rem_levels_L_prev-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine, c_gap_set[i_dev], n_rem_levels_L_prev, c_tmp_5[i_dev]
			);
			n_gaps[i_dev] += n_rem_levels_L_prev;
			
			gpuErrchk( cudaPeekAtLastError() );
			Cu_CoarsenCells_S2<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr, n_maxcblocks, c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
			);
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << toc_simple("[S5]",T_US) << " ";
#endif
		
		
		
		/*
#if (P_SHOW_REFINE==1)
		tic("[S4]");
#endif
		// Update neighbor-child arrays to faciitate the connectivity update to follow.
		if (proceed_refinement || proceed_coarsening)
		{
			// NOTE
			// Call S2 routine where new child IDs are inserted in the cblock_ID_nbr_child array.
			Cu_RefineCells_S2<<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				//id_max_curr, n_maxcblocks, n_ids_marked_refine, c_tmp_1[i_dev], &c_gap_set[i_dev][n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine], n_ids_marked_coarsen,
				id_max_curr, n_maxcblocks, n_ids_marked_refine, c_tmp_1[i_dev], &c_tmp_3[i_dev][N_CHILDREN*n_ids_marked_refine], n_ids_marked_coarsen,
				c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
				c_tmp_2[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		toc("[S4]",T_US);
#endif
		*/
		
		
		
#if (P_SHOW_REFINE==1)
		tic_simple("[S6]");
#endif
		//tic("[S5]");
		if (proceed_refinement)
		{
			// Copy the newly-set levels of gaps spliced from the gap set. This allows us to insert them into the correct ID set array.
			thrust::copy_if(
				thrust::device, c_cblock_level_dptr[i_dev], c_cblock_level_dptr[i_dev] + id_max_curr_wnew, c_cblock_ID_ref_dptr[i_dev], c_tmp_4_dptr[i_dev], is_newly_generated()
			);
			
			thrust::copy_if(
				thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max_curr_wnew, c_cblock_ID_ref_dptr[i_dev], c_tmp_3_dptr[i_dev], is_newly_generated()
			);
			
			// Sort the levels (keys, stored in c_tmp_4) and the associated gaps (values, stored in c_tmp_3). Reduce by key to get the number of gaps to insert into the ID set.
			thrust::stable_sort_by_key(
				thrust::device, c_tmp_4_dptr[i_dev], c_tmp_4_dptr[i_dev] + N_CHILDREN*n_ids_marked_refine, c_tmp_3_dptr[i_dev]
			);
			
			// Loop over the reduced level-keys and 1) retrieve total number to be added, 2) concatenate ID sets using total number per level, 3) update the n_ids array host-side.
			int n_new_levels_L_prev = 0;
			int n_new_levels_L = 0;
			for (int L = 1; L < MAX_LEVELS; L++)
			{
				n_new_levels_L = thrust::count_if(
					thrust::device, c_tmp_4_dptr[i_dev], c_tmp_4_dptr[i_dev] + N_CHILDREN*n_ids_marked_refine, is_equal_to(L)
				);
				
				if (n_new_levels_L > 0)
				{
					Cu_Concat<<<(M_BLOCK+n_new_levels_L-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
						n_ids[i_dev][L], c_id_set[i_dev][L], n_new_levels_L, &c_tmp_3[i_dev][n_new_levels_L_prev]
					);
					gpuErrchk( cudaPeekAtLastError() );
					n_ids[i_dev][L] += n_new_levels_L;
					
					thrust::sort(
						thrust::device, c_id_set_dptr[i_dev][L], c_id_set_dptr[i_dev][L] + n_ids[i_dev][L]
					);
					
					std::cout << "Inserted " << n_new_levels_L << " cblocks on Level " << L << "..." << std::endl;
				}
				n_new_levels_L_prev += n_new_levels_L;
			}
			n_gaps[i_dev] -= N_CHILDREN*n_ids_marked_refine;
		}
		//toc("[S5]",T_US);
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << toc_simple("[S6]",T_US) << " ";
#endif
		
		
		
#if (P_SHOW_REFINE==1)
		tic_simple("[S7]");
#endif
		// Update the ID max. Loop over refinement indices and remove marks. Update branch / leaf IDs.
		if (proceed_refinement || proceed_coarsening)
		{
			//tic("[S6p1]");
			// Update the ID max.
			for (int L = 1; L < MAX_LEVELS; L++)
				id_max[i_dev][L] = *thrust::max_element(thrust::device, c_id_set_dptr[i_dev][L], c_id_set_dptr[i_dev][L] + n_ids[i_dev][L]) + 1;
			id_max[i_dev][MAX_LEVELS] = *std::max_element(id_max[i_dev], id_max[i_dev] + MAX_LEVELS);
			//cudaDeviceSynchronize();
			//toc("[S6p1]",T_US);
			
			//tic("[S7p4]");
			Cu_UpdateMasks_1<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max[i_dev][MAX_LEVELS], n_maxcblocks,
				c_cblock_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
			//cudaDeviceSynchronize();
			//toc("[S7p3]",T_US);
			
			//tic("[S6p2]");
			// Interpolate data to newly-added cell-blocks.
			for (int L = 0; L < MAX_LEVELS-1; L++)
			{
#if (V_ADV_TYPE == V_ADV_TYPE_UNIFORM)
				M_Interpolate(i_dev, L, 4);
#else
				M_Interpolate(i_dev, L, 5, scale_vec[L+1]/scale_vec[L]);
#endif
			}
			//cudaDeviceSynchronize();
			//toc("[S6p2]",T_US);
			
			//tic("[S6p3]");
			// Call S3 routine where refinement IDs are reset for marked and newly-generated blocks.
			Cu_RefineCells_S3<<<(M_BLOCK+id_max_curr_wnew-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr_wnew, c_cblock_ID_ref[i_dev]
			);
			//cudaDeviceSynchronize();
			//toc("[S6p3]",T_US);
			
			//tic("[S6p4]");
			// Call S4 routine where branch and leaf cell-block identifications are updated.
			Cu_RefineCells_S4<<<(M_BLOCK+id_max_curr_wnew-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max_curr_wnew, c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
			);
			//cudaDeviceSynchronize();
			//toc("[S6p4]",T_US);
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << toc_simple("[S7]",T_US) << " ";
#endif
		
		
		
#if (P_SHOW_REFINE==1)
		tic_simple("[S8]");
#endif
		// Update connectivity.
		if (proceed_refinement || proceed_coarsening)
		{	
			//tic("[S7p1]");
			Cu_UpdateConnectivity<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max[i_dev][MAX_LEVELS], n_maxcblocks,
				c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
				c_tmp_8[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
			//cudaDeviceSynchronize();
			//toc("[S7p1]",T_US);
			
			//tic("[S7p2]");
			Cu_UpdateBoundaries<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				id_max[i_dev][MAX_LEVELS], n_maxcblocks,
				c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
			//cudaDeviceSynchronize();
			//toc("[S7p2]",T_US);
			
			//tic("[S7p3]");
			Cu_UpdateMasks_2<<<(M_CBLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				id_max[i_dev][MAX_LEVELS], n_maxcblocks,
				c_cblock_ID_nbr[i_dev], c_cblock_ID_ref[i_dev], c_cells_ID_mask[i_dev]
			);
			gpuErrchk( cudaPeekAtLastError() );
			//cudaDeviceSynchronize();
			//toc("[S7p3]",T_US);
		}
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		*file << toc_simple("[S8]",T_US) << "\n";
#endif
		
		
		
		//Cu_RefinementValidator_1<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
		//	n_maxcblocks, n_maxcblocks,
		//	c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
		//);
 		//Cu_RefinementValidator_2<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
 		//	n_maxcblocks, n_maxcblocks,
 		//	c_cblock_ID_ref[i_dev], c_cblock_level[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
 		//);
		//Cu_RefinementValidator_3<<<(M_CBLOCK+n_maxcblocks-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
 		//	n_maxcblocks, n_maxcblocks,
		//	c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
 		//);
	}

	return 0;
}



/*
         .d8888b.                                                         d8b                   888    d8b                            
        d88P  Y88b                                                        Y8P                   888    Y8P                            
        888    888                                                                              888                                   
        888         .d88b.  88888b.d88b.  88888b.d88b.  888  888 88888b.  888  .d8888b  8888b.  888888 888  .d88b.  88888b.           
        888        d88""88b 888 "888 "88b 888 "888 "88b 888  888 888 "88b 888 d88P"        "88b 888    888 d88""88b 888 "88b          
        888    888 888  888 888  888  888 888  888  888 888  888 888  888 888 888      .d888888 888    888 888  888 888  888          
        Y88b  d88P Y88..88P 888  888  888 888  888  888 Y88b 888 888  888 888 Y88b.    888  888 Y88b.  888 Y88..88P 888  888          
88888888 "Y8888P"   "Y88P"  888  888  888 888  888  888  "Y88888 888  888 888  "Y8888P "Y888888  "Y888 888  "Y88P"  888  888 88888888 
                                                                                                                                      
                                                                                                                                      
                                                                                                                                      
*/



// NOTE: interp_type takes on two values: (0 - interpolate to interface cells, 1 - interpolate to newly-added cells).
template <int interp_type = 0>
__global__
void Cu_Interpolate_Uniform
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, int n_maxcblocks,
	int *cblock_ID_nbr_child, int *cells_ID_mask, int *cblock_ID_mask,
	ufloat_t *cells_f_F
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_F[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int I_kap = ( threadIdx.x % Nbx         )/2;
	int J_kap = ( (threadIdx.x / Nbx) % Nbx )/2;
#if (N_DIM==3)
	int K_kap = ( (threadIdx.x / Nbx) / Nbx )/2;
#endif
	
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		int i_c0 = -1;
		if (i_kap_b > -1)
		{
			i_c0 = cblock_ID_nbr_child[i_kap_b]; // ID of first child.
			
			// Only process if children exist, skip otherwise.
			if (i_c0 >= 0)
			{
				for (int p = 0; p < l_dq; p++)
				{					
					s_F[threadIdx.x] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells];
					__syncthreads();
					
#if (N_DIM==2)
					if (interp_type == 0)
					{
						if (cells_ID_mask[(i_c0+0)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+0)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*0)];
						if (cells_ID_mask[(i_c0+1)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+1)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*0)];
						if (cells_ID_mask[(i_c0+2)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+2)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*1)];
						if (cells_ID_mask[(i_c0+3)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+3)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*1)];
					}
					if (interp_type == 1)
					{
						if (cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE)
						{
							cells_f_F[(i_c0+0)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*0)];
							cells_f_F[(i_c0+1)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*0)];
							cells_f_F[(i_c0+2)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*1)];
							cells_f_F[(i_c0+3)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*1)];
						}
					}
#else
					if (interp_type == 0)
					{
						if (cells_ID_mask[(i_c0+0)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+0)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*0) + 4*4*(K_kap + 2*0)];
						if (cells_ID_mask[(i_c0+1)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+1)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*0) + 4*4*(K_kap + 2*0)];
						if (cells_ID_mask[(i_c0+2)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+2)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*1) + 4*4*(K_kap + 2*0)];
						if (cells_ID_mask[(i_c0+3)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+3)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*1) + 4*4*(K_kap + 2*0)];
						if (cells_ID_mask[(i_c0+4)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+4)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*0) + 4*4*(K_kap + 2*1)];
						if (cells_ID_mask[(i_c0+5)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+5)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*0) + 4*4*(K_kap + 2*1)];
						if (cells_ID_mask[(i_c0+6)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+6)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*1) + 4*4*(K_kap + 2*1)];
						if (cells_ID_mask[(i_c0+7)*M_CBLOCK + threadIdx.x] > 0)
							cells_f_F[(i_c0+7)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*1) + 4*4*(K_kap + 2*1)];
					}
					if (interp_type == 1)
					{
						if (cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE)
						{
							cells_f_F[(i_c0+0)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*0) + 4*4*(K_kap + 2*0)];
							cells_f_F[(i_c0+1)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*0) + 4*4*(K_kap + 2*0)];
							cells_f_F[(i_c0+2)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*1) + 4*4*(K_kap + 2*0)];
							cells_f_F[(i_c0+3)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*1) + 4*4*(K_kap + 2*0)];
							cells_f_F[(i_c0+4)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*0) + 4*4*(K_kap + 2*1)];
							cells_f_F[(i_c0+5)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*0) + 4*4*(K_kap + 2*1)];
							cells_f_F[(i_c0+6)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*0) + 4*(J_kap + 2*1) + 4*4*(K_kap + 2*1)];
							cells_f_F[(i_c0+7)*M_CBLOCK + threadIdx.x + p*n_maxcells] = s_F[(I_kap + 2*1) + 4*(J_kap + 2*1) + 4*4*(K_kap + 2*1)];
						}
					}
#endif
					__syncthreads();
				}
			}
		}
	}
}

// NOTE: interp_type takes on two values: (0 - interpolate to interface cells, 1 - interpolate to newly-added).
//       rescale_type takes on two values: (0 - regular interpolation without rescale, 1 - interpolation with rescale).
#if (S_INTERP_TYPE==0)
template <int interp_type = 0, int rescale_type = 0>
__global__
void Cu_Interpolate_Cubic
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, int n_maxcblocks,
	int *cblock_ID_nbr_child, int *cells_ID_mask, int *cblock_ID_mask,
	ufloat_t *cells_f_F, ufloat_t *cells_f_Fs,
	double *mat_interp, ufloat_t Cscale
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ double s_V[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	double x_kap = 0.0;
	double y_kap = 0.0;
#if (N_DIM==3)
	int K_kap = (threadIdx.x / Nbx) / Nbx;
	double z_kap = 0.0;
#endif
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	s_V[threadIdx.x] = 0.0;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		int i_c0 = -1;
		//if (i_kap_b > -1)
		if ( (interp_type == 0 && i_kap_b > -1 && cblock_ID_mask[i_kap_b] == 1) || (interp_type == 1 && i_kap_b > -1 && cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE) )
		{
#if (N_DIM==2)
			i_c0 = cblock_ID_nbr_child[i_kap_b]; // ID of first child.
			
			// Only process if children exist, skip otherwise.
			if (i_c0 >= 0)
			{
				//#pragma unroll
				for (int p = 0; p < l_dq; p++)
				{
					// s_V stores the matrix multiplication of inv(A)*f where inv(A) is stored in mat_interp and f is retrieved from cells_f_F.
					for (int i_vec = 0; i_vec < M_CBLOCK; i_vec++)
					{
						s_V[threadIdx.x] += mat_interp[threadIdx.x + i_vec*M_CBLOCK]*(double)(cells_f_F[i_kap_b*M_CBLOCK + i_vec +  p*n_maxcells]);
						
						//if (threadIdx.x < 16)
						//	s_V[threadIdx.x] += (mat_interp[threadIdx.x + i_vec*16]*(double)cells_f_F[i_kap_b*M_CBLOCK + i_vec + p*n_maxcells]);
					}
					__syncthreads();
					
					// After interpolant computation, loop over child blocks and compute new values. Store immediately.
					double b0, b1, b2, b3, res;
						// xc=0
					for (int i_child_j = 0; i_child_j < 2; i_child_j++)
					{
						for (int i_child_i = 0; i_child_i < 2; i_child_i++)
						{
							x_kap = (0.5 + I_kap)*(0.125) + i_child_i*(0.5); // dx/2 + I*dx + dx(block)
							y_kap = (0.5 + J_kap)*(0.125) + i_child_j*(0.5);
							
							b0 = s_V[0 + 4*0] + x_kap*(s_V[1 + 4*0] + x_kap*(s_V[2 + 4*0] + x_kap*s_V[3 + 4*0]));
							b1 = s_V[0 + 4*1] + x_kap*(s_V[1 + 4*1] + x_kap*(s_V[2 + 4*1] + x_kap*s_V[3 + 4*1]));
							b2 = s_V[0 + 4*2] + x_kap*(s_V[1 + 4*2] + x_kap*(s_V[2 + 4*2] + x_kap*s_V[3 + 4*2]));
							b3 = s_V[0 + 4*3] + x_kap*(s_V[1 + 4*3] + x_kap*(s_V[2 + 4*3] + x_kap*s_V[3 + 4*3]));
							
							if (interp_type == 0)
							{
								if (cells_ID_mask[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x] == 2)
									cells_f_F[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = (ufloat_t)( b0 + y_kap*(b1 + y_kap*(b2 + y_kap*b3)) );
							}
							if (interp_type == 1)
							{
								cells_f_F[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = (ufloat_t)( b0 + y_kap*(b1 + y_kap*(b2 + y_kap*b3)) );
							}
						}
					}
					__syncthreads();
					
					// Reset s_V for next cell-block.
					s_V[threadIdx.x] = 0.0;
					__syncthreads();
					
					
					
					if (rescale_type == 1)
					{
						// s_V stores the matrix multiplication of inv(A)*f where inv(A) is stored in mat_interp and f is retrieved from cells_f_F.
						for (int i_vec = 0; i_vec < M_CBLOCK; i_vec++)
						{
							s_V[threadIdx.x] += mat_interp[threadIdx.x + i_vec*M_CBLOCK]*(double)(cells_f_Fs[i_kap_b*M_CBLOCK + i_vec + p*n_maxcells]);
						}
						__syncthreads();
						
						// After interpolant computation, loop over child blocks and compute new values. Store immediately.
						//double b0, b1, b2, b3, res;
							// xc=0
						for (int i_child_j = 0; i_child_j < 2; i_child_j++)
						{
							for (int i_child_i = 0; i_child_i < 2; i_child_i++)
							{
								x_kap = (0.5 + I_kap)*(0.125) + i_child_i*(0.5); // dx/2 + I*dx + dx(block)
								y_kap = (0.5 + J_kap)*(0.125) + i_child_j*(0.5);
								
								b0 = s_V[0 + 4*0] + x_kap*(s_V[1 + 4*0] + x_kap*(s_V[2 + 4*0] + x_kap*s_V[3 + 4*0]));
								b1 = s_V[0 + 4*1] + x_kap*(s_V[1 + 4*1] + x_kap*(s_V[2 + 4*1] + x_kap*s_V[3 + 4*1]));
								b2 = s_V[0 + 4*2] + x_kap*(s_V[1 + 4*2] + x_kap*(s_V[2 + 4*2] + x_kap*s_V[3 + 4*2]));
								b3 = s_V[0 + 4*3] + x_kap*(s_V[1 + 4*3] + x_kap*(s_V[2 + 4*3] + x_kap*s_V[3 + 4*3]));
								
								res = (ufloat_t)( b0 + y_kap*(b1 + y_kap*(b2 + y_kap*b3)) );
								
								if (interp_type == 0)
								{
									if (cells_ID_mask[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x] == 2)
									{
										cells_f_F[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = res + (cells_f_F[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x + p*n_maxcells] - res)*Cscale;
										
										cells_f_Fs[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = res;
									}
								}
								if (interp_type == 1)
								{
									cells_f_F[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = res  +(cells_f_F[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x + p*n_maxcells] - res)*Cscale;
									
									cells_f_Fs[(i_c0 + (i_child_i+2*i_child_j) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = res;
								}
							}
						}
						__syncthreads();
						
						// Reset s_V for next cell-block.
						s_V[threadIdx.x] = 0.0;
						__syncthreads();
					}
					
				}
			}
#else
			i_c0 = cblock_ID_nbr_child[i_kap_b]; // ID of first child.
			
			// Only process if children exist, skip otherwise.
			if (i_c0 >= 0)
			{
				//#pragma unroll
				for (int p = 0; p < l_dq; p++)
				{
					// s_V stores the matrix multiplication of inv(A)*f where inv(A) is stored in mat_interp and f is retrieved from cells_f_F.
					for (int i_vec = 0; i_vec < M_CBLOCK; i_vec++)
						s_V[threadIdx.x] += mat_interp[threadIdx.x + i_vec*M_CBLOCK]*(double)(cells_f_F[i_kap_b*M_CBLOCK + i_vec + p*n_maxcells]);
					__syncthreads();
					
					// After interpolant computation, loop over child blocks and compute new values. Store immediately.
					double b00, b10, b20, b30, b01, b11, b21, b31, b02, b12, b22, b32, b03, b13, b23, b33, c0, c1, c2, c3, res;
						// xc=0
					for (int i_child_k = 0; i_child_k < 2; i_child_k++)
					{
						for (int i_child_j = 0; i_child_j < 2; i_child_j++)
						{
							for (int i_child_i = 0; i_child_i < 2; i_child_i++)
							{
								x_kap = (0.5 + I_kap)*(0.125) + i_child_i*(0.5); // dx/2 + I*dx + dx(block)
								y_kap = (0.5 + J_kap)*(0.125) + i_child_j*(0.5);
								z_kap = (0.5 + K_kap)*(0.125) + i_child_k*(0.5);
								
								b00 = s_V[0 + 4*0 + 4*4*0] + x_kap*(s_V[1 + 4*0 + 4*4*0] + x_kap*(s_V[2 + 4*0 + 4*4*0] + x_kap*s_V[3 + 4*0 + 4*4*0]));
								b10 = s_V[0 + 4*1 + 4*4*0] + x_kap*(s_V[1 + 4*1 + 4*4*0] + x_kap*(s_V[2 + 4*1 + 4*4*0] + x_kap*s_V[3 + 4*1 + 4*4*0]));
								b20 = s_V[0 + 4*2 + 4*4*0] + x_kap*(s_V[1 + 4*2 + 4*4*0] + x_kap*(s_V[2 + 4*2 + 4*4*0] + x_kap*s_V[3 + 4*2 + 4*4*0]));
								b30 = s_V[0 + 4*3 + 4*4*0] + x_kap*(s_V[1 + 4*3 + 4*4*0] + x_kap*(s_V[2 + 4*3 + 4*4*0] + x_kap*s_V[3 + 4*3 + 4*4*0]));
								b01 = s_V[0 + 4*0 + 4*4*1] + x_kap*(s_V[1 + 4*0 + 4*4*1] + x_kap*(s_V[2 + 4*0 + 4*4*1] + x_kap*s_V[3 + 4*0 + 4*4*1]));
								b11 = s_V[0 + 4*1 + 4*4*1] + x_kap*(s_V[1 + 4*1 + 4*4*1] + x_kap*(s_V[2 + 4*1 + 4*4*1] + x_kap*s_V[3 + 4*1 + 4*4*1]));
								b21 = s_V[0 + 4*2 + 4*4*1] + x_kap*(s_V[1 + 4*2 + 4*4*1] + x_kap*(s_V[2 + 4*2 + 4*4*1] + x_kap*s_V[3 + 4*2 + 4*4*1]));
								b31 = s_V[0 + 4*3 + 4*4*1] + x_kap*(s_V[1 + 4*3 + 4*4*1] + x_kap*(s_V[2 + 4*3 + 4*4*1] + x_kap*s_V[3 + 4*3 + 4*4*1]));
								b02 = s_V[0 + 4*0 + 4*4*2] + x_kap*(s_V[1 + 4*0 + 4*4*2] + x_kap*(s_V[2 + 4*0 + 4*4*2] + x_kap*s_V[3 + 4*0 + 4*4*2]));
								b12 = s_V[0 + 4*1 + 4*4*2] + x_kap*(s_V[1 + 4*1 + 4*4*2] + x_kap*(s_V[2 + 4*1 + 4*4*2] + x_kap*s_V[3 + 4*1 + 4*4*2]));
								b22 = s_V[0 + 4*2 + 4*4*2] + x_kap*(s_V[1 + 4*2 + 4*4*2] + x_kap*(s_V[2 + 4*2 + 4*4*2] + x_kap*s_V[3 + 4*2 + 4*4*2]));
								b32 = s_V[0 + 4*3 + 4*4*2] + x_kap*(s_V[1 + 4*3 + 4*4*2] + x_kap*(s_V[2 + 4*3 + 4*4*2] + x_kap*s_V[3 + 4*3 + 4*4*2]));
								b03 = s_V[0 + 4*0 + 4*4*3] + x_kap*(s_V[1 + 4*0 + 4*4*3] + x_kap*(s_V[2 + 4*0 + 4*4*3] + x_kap*s_V[3 + 4*0 + 4*4*3]));
								b13 = s_V[0 + 4*1 + 4*4*3] + x_kap*(s_V[1 + 4*1 + 4*4*3] + x_kap*(s_V[2 + 4*1 + 4*4*3] + x_kap*s_V[3 + 4*1 + 4*4*3]));
								b23 = s_V[0 + 4*2 + 4*4*3] + x_kap*(s_V[1 + 4*2 + 4*4*3] + x_kap*(s_V[2 + 4*2 + 4*4*3] + x_kap*s_V[3 + 4*2 + 4*4*3]));
								b33 = s_V[0 + 4*3 + 4*4*3] + x_kap*(s_V[1 + 4*3 + 4*4*3] + x_kap*(s_V[2 + 4*3 + 4*4*3] + x_kap*s_V[3 + 4*3 + 4*4*3]));
								c0 = b00 + y_kap*(b10 + y_kap*(b20 + y_kap*b30));
								c1 = b01 + y_kap*(b11 + y_kap*(b21 + y_kap*b31));
								c2 = b02 + y_kap*(b12 + y_kap*(b22 + y_kap*b32));
								c3 = b03 + y_kap*(b13 + y_kap*(b23 + y_kap*b33));
								
								if (interp_type == 0)
								{
									if (cells_ID_mask[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x] == 2)
										cells_f_F[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = (ufloat_t)( c0 + z_kap*(c1 + z_kap*(c2 + z_kap*c3)) );
								}
								if (interp_type == 1)
								{
									if (cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE)
										cells_f_F[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = (ufloat_t)( c0 + z_kap*(c1 + z_kap*(c2 + z_kap*c3)) );
								}
							}
						}
					}
					__syncthreads();
					
					// Reset s_V for next cell-block.
					s_V[threadIdx.x] = 0.0;
					__syncthreads();
					
					
					
					if (rescale_type == 1)
					{
						// s_V stores the matrix multiplication of inv(A)*f where inv(A) is stored in mat_interp and f is retrieved from cells_f_F.
						for (int i_vec = 0; i_vec < M_CBLOCK; i_vec++)
							s_V[threadIdx.x] += mat_interp[threadIdx.x + i_vec*M_CBLOCK]*(double)(cells_f_Fs[i_kap_b*M_CBLOCK + i_vec + p*n_maxcells]);
						__syncthreads();
					
						// After interpolant computation, loop over child blocks and compute new values. Store immediately.
							// xc=0
						for (int i_child_k = 0; i_child_k < 2; i_child_k++)
						{
							for (int i_child_j = 0; i_child_j < 2; i_child_j++)
							{
								for (int i_child_i = 0; i_child_i < 2; i_child_i++)
								{
									x_kap = (0.5 + I_kap)*(0.125) + i_child_i*(0.5); // dx/2 + I*dx + dx(block)
									y_kap = (0.5 + J_kap)*(0.125) + i_child_j*(0.5);
									z_kap = (0.5 + K_kap)*(0.125) + i_child_k*(0.5);
									
									b00 = s_V[0 + 4*0 + 4*4*0] + x_kap*(s_V[1 + 4*0 + 4*4*0] + x_kap*(s_V[2 + 4*0 + 4*4*0] + x_kap*s_V[3 + 4*0 + 4*4*0]));
									b10 = s_V[0 + 4*1 + 4*4*0] + x_kap*(s_V[1 + 4*1 + 4*4*0] + x_kap*(s_V[2 + 4*1 + 4*4*0] + x_kap*s_V[3 + 4*1 + 4*4*0]));
									b20 = s_V[0 + 4*2 + 4*4*0] + x_kap*(s_V[1 + 4*2 + 4*4*0] + x_kap*(s_V[2 + 4*2 + 4*4*0] + x_kap*s_V[3 + 4*2 + 4*4*0]));
									b30 = s_V[0 + 4*3 + 4*4*0] + x_kap*(s_V[1 + 4*3 + 4*4*0] + x_kap*(s_V[2 + 4*3 + 4*4*0] + x_kap*s_V[3 + 4*3 + 4*4*0]));
									b01 = s_V[0 + 4*0 + 4*4*1] + x_kap*(s_V[1 + 4*0 + 4*4*1] + x_kap*(s_V[2 + 4*0 + 4*4*1] + x_kap*s_V[3 + 4*0 + 4*4*1]));
									b11 = s_V[0 + 4*1 + 4*4*1] + x_kap*(s_V[1 + 4*1 + 4*4*1] + x_kap*(s_V[2 + 4*1 + 4*4*1] + x_kap*s_V[3 + 4*1 + 4*4*1]));
									b21 = s_V[0 + 4*2 + 4*4*1] + x_kap*(s_V[1 + 4*2 + 4*4*1] + x_kap*(s_V[2 + 4*2 + 4*4*1] + x_kap*s_V[3 + 4*2 + 4*4*1]));
									b31 = s_V[0 + 4*3 + 4*4*1] + x_kap*(s_V[1 + 4*3 + 4*4*1] + x_kap*(s_V[2 + 4*3 + 4*4*1] + x_kap*s_V[3 + 4*3 + 4*4*1]));
									b02 = s_V[0 + 4*0 + 4*4*2] + x_kap*(s_V[1 + 4*0 + 4*4*2] + x_kap*(s_V[2 + 4*0 + 4*4*2] + x_kap*s_V[3 + 4*0 + 4*4*2]));
									b12 = s_V[0 + 4*1 + 4*4*2] + x_kap*(s_V[1 + 4*1 + 4*4*2] + x_kap*(s_V[2 + 4*1 + 4*4*2] + x_kap*s_V[3 + 4*1 + 4*4*2]));
									b22 = s_V[0 + 4*2 + 4*4*2] + x_kap*(s_V[1 + 4*2 + 4*4*2] + x_kap*(s_V[2 + 4*2 + 4*4*2] + x_kap*s_V[3 + 4*2 + 4*4*2]));
									b32 = s_V[0 + 4*3 + 4*4*2] + x_kap*(s_V[1 + 4*3 + 4*4*2] + x_kap*(s_V[2 + 4*3 + 4*4*2] + x_kap*s_V[3 + 4*3 + 4*4*2]));
									b03 = s_V[0 + 4*0 + 4*4*3] + x_kap*(s_V[1 + 4*0 + 4*4*3] + x_kap*(s_V[2 + 4*0 + 4*4*3] + x_kap*s_V[3 + 4*0 + 4*4*3]));
									b13 = s_V[0 + 4*1 + 4*4*3] + x_kap*(s_V[1 + 4*1 + 4*4*3] + x_kap*(s_V[2 + 4*1 + 4*4*3] + x_kap*s_V[3 + 4*1 + 4*4*3]));
									b23 = s_V[0 + 4*2 + 4*4*3] + x_kap*(s_V[1 + 4*2 + 4*4*3] + x_kap*(s_V[2 + 4*2 + 4*4*3] + x_kap*s_V[3 + 4*2 + 4*4*3]));
									b33 = s_V[0 + 4*3 + 4*4*3] + x_kap*(s_V[1 + 4*3 + 4*4*3] + x_kap*(s_V[2 + 4*3 + 4*4*3] + x_kap*s_V[3 + 4*3 + 4*4*3]));
									c0 = b00 + y_kap*(b10 + y_kap*(b20 + y_kap*b30));
									c1 = b01 + y_kap*(b11 + y_kap*(b21 + y_kap*b31));
									c2 = b02 + y_kap*(b12 + y_kap*(b22 + y_kap*b32));
									c3 = b03 + y_kap*(b13 + y_kap*(b23 + y_kap*b33));
									
									res = (ufloat_t)( c0 + z_kap*(c1 + z_kap*(c2 + z_kap*c3)) );
									
									if (interp_type == 0)
									{
										if (cells_ID_mask[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x] == 2)
										{
											cells_f_F[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = res + (cells_f_F[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] - res)*Cscale;
											
											cells_f_Fs[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = res;
										}
									}
									if (interp_type == 1)
									{
										if (cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE)
										{
											cells_f_F[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = res + (cells_f_F[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] - res)*Cscale;
											
											cells_f_Fs[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = res;
										}
									}
								}
							}
						}
						__syncthreads();
						
						// Reset s_V for next cell-block.
						s_V[threadIdx.x] = 0.0;
						__syncthreads();
					}
				}
			}
#endif
		}
	}
}
#else
template <int interp_type = 0, int rescale_type = 0>
__global__
void Cu_Interpolate_Cubic
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, int n_maxcblocks,
	int *cblock_ID_nbr_child, int *cells_ID_mask, int *cblock_ID_mask,
	ufloat_t *cells_f_F, ufloat_t *cells_f_Fs,
	double *mat_interp, ufloat_t Cscale
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_F[M_CBLOCK];
	__shared__ ufloat_t s_Fs[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	double dat_kap_1 = 0.0;
	double dat_kap_2 = 0.0;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		int i_c0 = -1;
		if ( (interp_type == 0 && i_kap_b > -1 && cblock_ID_mask[i_kap_b] == 1) || (interp_type == 1 && i_kap_b > -1 && cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE) )
		{
			i_c0 = cblock_ID_nbr_child[i_kap_b]; // ID of first child.
			
			// Only process if children exist, skip otherwise.
			if (i_c0 >= 0)
			{
				for (int p = 0; p < l_dq; p++)
				{
					// Read DDFs and place in shared memory to prepare for the upcoming matrix multiplications.
					s_F[threadIdx.x] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells];
					s_Fs[threadIdx.x] = cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells];
					__syncthreads();
					
					
					// Loop over all child blocks.
#if (N_DIM==3)
					for (int i_child_k = 0; i_child_k < 2; i_child_k++)
					{
#else
					int i_child_k = 0;
					{
#endif
						for (int i_child_j = 0; i_child_j < 2; i_child_j++)
						{
							for (int i_child_i = 0; i_child_i < 2; i_child_i++)
							{
								
								// Compute shift in interpolation matrix to retrieve weights for the correct child block.
								int mat_shift = (i_child_i + 2*i_child_j + 4*i_child_k)*M_CBLOCK*M_CBLOCK;
								
								
								// dat_kap_1 stores the interpolated DDF as a result of matrix multiplication of W_c*F where W_c is the matrix of interpolation weights for the current child block and F is the vector of parent DDFs.
								for (int i_vec = 0; i_vec < M_CBLOCK; i_vec++)
									dat_kap_1 += mat_interp[threadIdx.x + i_vec*M_CBLOCK + mat_shift]*(double)(s_F[i_vec]);
								__syncthreads();
								
								
								// dat_kap_2 stores the interpolated equilibrium DDF for rescaling if rescale_type is activated. Otherwise, dat_kap_2 is copied from dat_kap_1 to be written (Cscale would be 0, so dat_kap_1 is effectively written on its own to global memory).
								if (rescale_type == 1)
								{
									for (int i_vec = 0; i_vec < M_CBLOCK; i_vec++)
										dat_kap_2 += mat_interp[threadIdx.x + i_vec*M_CBLOCK + mat_shift]*(double)(s_Fs[i_vec]);
									__syncthreads();
								}
								else
									dat_kap_2 = dat_kap_1;
								
								
								// Write new data to global memory.
								if (interp_type == 0)
								{
									if (cells_ID_mask[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x] == 2)
										cells_f_F[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = dat_kap_2 + (dat_kap_1 - dat_kap_2)*Cscale;
									
									if (rescale_type == 1 && cells_ID_mask[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x] == 2)
										cells_f_Fs[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = dat_kap_2;
								}
								else
								{
									if (cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE)
										cells_f_F[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = dat_kap_2 + (dat_kap_1 - dat_kap_2)*Cscale;
									
									if (rescale_type == 1 && cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE)
										cells_f_Fs[(i_c0 + (i_child_i+2*i_child_j+4*i_child_k) )*M_CBLOCK + threadIdx.x + p*n_maxcells] = dat_kap_2;
								}
								
								
								// Reset register memory for next cell-block.
								dat_kap_1 = 0.0;
								dat_kap_2 = 0.0;
								__syncthreads();
								
							}
						}
					}
				}
			}
		}
	}
}
#endif

int Mesh::M_Interpolate(int i_dev, int L, int var, ufloat_t Cscale)
{
	if (n_ids[i_dev][L] > 0 && L != MAX_LEVELS-1)
	{
		if (var == 0)
		{
			Cu_Interpolate_Uniform<0><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_F[i_dev]
			);
		}
		if (var == 1)
		{
			Cu_Interpolate_Uniform<0><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_Fs[i_dev]
			);
		}
		if (var == 2)
		{
			Cu_Interpolate_Cubic<0,1><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_F[i_dev], c_cells_f_Fs[i_dev],
				c_mat_interp[i_dev], Cscale
			);
		}
		/*
		if (var == 3)
		{
			Cu_Interpolate_Cubic<0><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_Fs[i_dev],
				c_mat_interp[i_dev]
			);
		}
		if (var == 4)
		{
			Cu_Interpolate_Uniform<1><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_F[i_dev], Cscale
			);
		}
		*/
		if (var == 5)
		{
			Cu_Interpolate_Cubic<1,1><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_F[i_dev], c_cells_f_Fs[i_dev],
				c_mat_interp[i_dev], Cscale
			);
		}
		/*
		if (var == 6)
		{
			Cu_Interpolate_Uniform<1><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_Fs[i_dev]
			);
		}
		*/
	}
	
	return 0;
}


// NOTE: ave_type takes on three values: (0 - average only over interace cells, 1 - average on full masked block, 2 - average on whole grid (mainly for output)).
//       rescale_type takes on two values: (0 - regular interpolation without rescale, 1 - interpolation with rescale).
template <int ave_type = 0, int rescale_type = 0>
__global__
void Cu_Average
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, int n_maxcblocks,
	int *cblock_ID_nbr_child, int *cells_ID_mask, int *cblock_ID_mask,
	ufloat_t *cells_f_F, ufloat_t *cells_f_Fs,
	ufloat_t Cscale
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_Fc[M_CBLOCK*N_CHILDREN];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	ufloat_t dat_kap_1 = N_Pf(0.0);
	ufloat_t dat_kap_2 = N_Pf(0.0);
#if (N_DIM==2)
	int I_kap = ( threadIdx.x % Nbx         )/2;
	int J_kap = ( (threadIdx.x / Nbx) % Nbx )/2;
	int I_kap_m2 = ( threadIdx.x % Nbx         )%2;
	int J_kap_m2 = ( (threadIdx.x / Nbx) % Nbx )%2;
	int K_kap = 0;
	int K_kap_m2 = 0;
#else
	int I_kap = ( threadIdx.x % Nbx         )/2;
	int J_kap = ( (threadIdx.x / Nbx) % Nbx )/2;
	int K_kap = ( (threadIdx.x / Nbx) / Nbx )/2;
	int I_kap_m2 = ( threadIdx.x % Nbx         )%2;
	int J_kap_m2 = ( (threadIdx.x / Nbx) % Nbx )%2;
	int K_kap_m2 = ( (threadIdx.x / Nbx) / Nbx )%2;
#endif
	
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		int i_c0 = -1;
		int quad_IJK = -1;
		int child0_IJK = -1;
		//if (i_kap_b > -1)
		if (   (ave_type < 2 && i_kap_b > -1 && cblock_ID_mask[i_kap_b] == 1)   ||   (ave_type == 2 && i_kap_b > -1)   )
		//if (   (ave_type == 0 && i_kap_b > -1 && cblock_ID_mask[i_kap_b] == 1) || (ave_type > 0 && i_kap_b > -1)   )
		{
			i_c0 = cblock_ID_nbr_child[i_kap_b]; // ID of first child.
			
			// Only process if children exist, skip otherwise.
			if (i_c0 >= 0)
			{
				for (int p = 0; p < l_dq; p++)
				{
					// Load data from children into shared memory.
					for (int xc = 0; xc < N_CHILDREN; xc++)
						s_Fc[threadIdx.x + xc*M_CBLOCK] = cells_f_F[(i_c0+xc)*M_CBLOCK + threadIdx.x + p*n_maxcells];
					__syncthreads();
					
					// Compute sum at parent level, place in shared memory. Write to global memory.
					quad_IJK = I_kap + 2*J_kap + 4*K_kap;
					child0_IJK = 2*I_kap_m2 + Nbx*(2*J_kap_m2) + Nbx*Nbx*(2*K_kap_m2);
					dat_kap_1 = 
#if (N_DIM==2)
						N_Pf(0.25)*(
							s_Fc[(child0_IJK + 0 + Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 0 + Nbx*1) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*1) + quad_IJK*M_CBLOCK]
					);
#else
						N_Pf(0.125)*(
							s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK]
					);
#endif
					__syncthreads();
					
					
					
					if (rescale_type == 1)
					{
						// Load data from children into shared memory.
						for (int xc = 0; xc < N_CHILDREN; xc++)
							s_Fc[threadIdx.x + xc*M_CBLOCK] = cells_f_Fs[(i_c0+xc)*M_CBLOCK + threadIdx.x + p*n_maxcells];
						__syncthreads();
						
						// Compute sum at parent level, place in shared memory. Write to global memory.
						dat_kap_2 =
#if (N_DIM==2)
							N_Pf(0.25)*(
								s_Fc[(child0_IJK + 0 + Nbx*0) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 1 + Nbx*0) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 0 + Nbx*1) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 1 + Nbx*1) + quad_IJK*M_CBLOCK]
						);
#else
							N_Pf(0.125)*(
								s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
								s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK]
						);
#endif
						__syncthreads();
					}
					else
						dat_kap_2 = dat_kap_1;
					
					
					// Write to global memory.
					if (ave_type == 0)
					{
						if (cells_ID_mask[(i_c0 + quad_IJK)*M_CBLOCK + child0_IJK] == 1)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = dat_kap_2 + (dat_kap_1 - dat_kap_2)*Cscale;
						
							if (rescale_type == 1)
								cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = dat_kap_2;
						}
					}
					else
					{
						if (cells_ID_mask[(i_c0 + quad_IJK)*M_CBLOCK + child0_IJK] < 2)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = dat_kap_2 + (dat_kap_1 - dat_kap_2)*Cscale;
						
							if (rescale_type == 1)
								cells_f_Fs[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = dat_kap_2;
						}
					}
					
					// Reset register memory for next updates.
					dat_kap_1 = N_Pf(0.0);
					dat_kap_2 = N_Pf(0.0);
					__syncthreads();
				}
			}
		}
	}
}

/*
__global__
void Cu_Average_Global
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, int n_maxcblocks,
	int *cblock_ID_nbr_child, int *cells_ID_mask, int *cblock_ID_mask,
	ufloat_t *cells_f_F
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_Fc[M_CBLOCK*N_CHILDREN];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
#if (N_DIM==2)
	int I_kap = ( threadIdx.x % Nbx         )/2;
	int J_kap = ( (threadIdx.x / Nbx) % Nbx )/2;
	int I_kap_m2 = ( threadIdx.x % Nbx         )%2;
	int J_kap_m2 = ( (threadIdx.x / Nbx) % Nbx )%2;
#else
	int I_kap = ( threadIdx.x % Nbx         )/2;
	int J_kap = ( (threadIdx.x / Nbx) % Nbx )/2;
	int K_kap = ( (threadIdx.x / Nbx) / Nbx )/2;
	int I_kap_m2 = ( threadIdx.x % Nbx         )%2;
	int J_kap_m2 = ( (threadIdx.x / Nbx) % Nbx )%2;
	int K_kap_m2 = ( (threadIdx.x / Nbx) / Nbx )%2;
#endif
	
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		int i_c0 = -1;
		int quad_IJK = -1;
		int child0_IJK = -1;
		if (i_kap_b > -1 && cblock_ID_mask[i_kap_b] == 1)
		{
			i_c0 = cblock_ID_nbr_child[i_kap_b]; // ID of first child.
			
			// Only process if children exist, skip otherwise.
			if (i_c0 >= 0)
			{
				for (int p = 0; p < l_dq; p++)
				{
					// Load data from children into shared memory.
					for (int xc = 0; xc < N_CHILDREN; xc++)
						s_Fc[threadIdx.x + xc*M_CBLOCK] = cells_f_F[(i_c0+xc)*M_CBLOCK + threadIdx.x + p*n_maxcells];
					__syncthreads();
					
					// Compute sum at parent level, place in shared memory. Write to global memory.
#if (N_DIM==2)
					quad_IJK = I_kap + 2*J_kap;
					child0_IJK = 2*I_kap_m2 + Nbx*(2*J_kap_m2);
					if (cells_ID_mask[(i_c0 + quad_IJK)*M_CBLOCK + child0_IJK] < 2)
					{
						cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = N_Pf(0.25)*(
							s_Fc[(child0_IJK + 0 + Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 0 + Nbx*1) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*1) + quad_IJK*M_CBLOCK]
						);
					}
#else
					quad_IJK = I_kap + 2*J_kap + 4*K_kap;
					child0_IJK = 2*I_kap_m2 + Nbx*(2*J_kap_m2) + Nbx*Nbx*(2*K_kap_m2);
					if (cells_ID_mask[(i_c0 + quad_IJK)*M_CBLOCK + child0_IJK] < 2)
					{
						cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = N_Pf(0.125)*(
							s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK] +
							s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1) + quad_IJK*M_CBLOCK]
						);
					}
#endif
					__syncthreads();
				}
			}
		}
	}
}
*/

int Mesh::M_Average(int i_dev, int L, int var, ufloat_t Cscale)
{
	if (n_ids[i_dev][L] > 0 && L != MAX_LEVELS-1)
	{
		// Average F on interace cells only.
		if (var == 0)
		{
			Cu_Average<0,V_RESCALE_TYPE><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_F[i_dev], c_cells_f_Fs[i_dev],
				Cscale
			);
		}
		/*
		// Average Fs on interface cells only.
		if (var == 1)
		{
			Cu_Average<0><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_Fs[i_dev]
			);
		}
		*/
		// Average F on whole masked block.
		if (var == 2)
		{
			Cu_Average<1,V_RESCALE_TYPE><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_F[i_dev], c_cells_f_Fs[i_dev],
				Cscale
			);
		}
		/*
		// Average Fs on whole masked block.
		if (var == 3)
		{
			Cu_Average<1><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_Fs[i_dev]
			);
		}
		*/
		// Average F on whole grid.
		if (var == 4)
		{
			Cu_Average<2,V_RESCALE_TYPE><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, n_maxcblocks,
				c_cblock_ID_nbr_child[i_dev], c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev],
				c_cells_f_F[i_dev], c_cells_f_Fs[i_dev],
				Cscale
			);
		}
	}

	return 0;
}
