/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "structs.h"
#include "geometry.h"

__global__
void Cu_ResetBins(int n_faces, int n_faces_a, int G_BIN_NUM, int *bin_indicators)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_faces)
	{
		for (int j = 0; j < G_BIN_NUM; j++)
		{
			if (bin_indicators[kap + j*n_faces_a] >= 0)
				bin_indicators[kap + j*n_faces_a] = -1;
		}
	}
}

template <typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_FillBins(
	int j0, int n_faces, int n_faces_a, int G_BIN_NUM, ufloat_g_t *geom_f_face_X, int *bin_indicators,
	int rad, ufloat_g_t LxOg, ufloat_g_t LyOg, ufloat_g_t LzOg, ufloat_g_t minL0g,int G_BIN_DENSITY
)
{
	constexpr int N_DIM = AP->N_DIM;
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_faces)
	{
		// Loop over the possible bins.
		for (int j = j0; j < j0+G_BIN_NUM; j++)
		{
			int bi = j%G_BIN_DENSITY;
			int bj = (j/G_BIN_DENSITY)%G_BIN_DENSITY;
			int bk = (j/G_BIN_DENSITY)/G_BIN_DENSITY;
			
			ufloat_g_t xm = (bi - rad)*LxOg - (ufloat_g_t)0.1*LxOg;
			ufloat_g_t xM = (bi+1 + rad)*LxOg + (ufloat_g_t)0.1*LxOg;
			ufloat_g_t ym = (bj - rad)*LyOg - (ufloat_g_t)0.1*LxOg;
			ufloat_g_t yM = (bj+1 + rad)*LyOg + (ufloat_g_t)0.1*LxOg;
			ufloat_g_t zm;
			ufloat_g_t zM;
			if (N_DIM==3)
			{
				zm = (bk - rad)*LzOg - (ufloat_g_t)0.1*LxOg;
				zM = (bk+1 + rad)*LzOg + (ufloat_g_t)0.1*LxOg;
			}
			
			
			
if (N_DIM==2)
{
			// Load the current line vertices.
			ufloat_g_t vx1 = geom_f_face_X[kap + 0*n_faces_a];
			ufloat_g_t vy1 = geom_f_face_X[kap + 1*n_faces_a];
			ufloat_g_t vz1 = geom_f_face_X[kap + 2*n_faces_a];
			ufloat_g_t vx2 = geom_f_face_X[kap + 3*n_faces_a];
			ufloat_g_t vy2 = geom_f_face_X[kap + 4*n_faces_a];
			ufloat_g_t vz2 = geom_f_face_X[kap + 5*n_faces_a];
			
			// Get the bounding box.
			ufloat_g_t vBx_m = min(vx1,vx2);
			ufloat_g_t vBx_M = max(vx1,vx2);
			ufloat_g_t vBy_m = min(vy1,vy2);
			ufloat_g_t vBy_M = max(vy1,vy2);
			
			// Discretize the line and check if one of the points ends up inside the bin.
			// Only do the check by discretization if the bounding box intersects the current bin.
			if ( !( (vBx_m < xm && vBx_M < xm) || (vBx_m > xM && vBx_M > xM) || (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) ) )
			{
				ufloat_g_t L_l = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1));
				int Nu = 100;
				Nu = 2*max((L_l / 0.25*minL0g), (ufloat_g_t)Nu);
				//printf("Length of current line %i is %17.15f, minL=%17.15f\n", kap, L_l, minL0g);
				for (int p = 0; p < Nu; p++)
				{
					ufloat_g_t d_p = (p + 1.0 - 0.5)/Nu;
					ufloat_g_t vxp = vx1 + d_p*(vx2-vx1);
					ufloat_g_t vyp = vy1 + d_p*(vy2-vy1);
					
					if ( (vxp > xm) && (vxp < xM) && (vyp > ym) && (vyp < yM) )
					{
						// At the first inidication that a point lies in the bounding box, terminate the loop.
						bin_indicators[kap + (j-j0)*n_faces_a] = kap;
						break;
					}
				}
			}
		
}
else
{
			// Load the current line vertices.
			ufloat_g_t vx1 = geom_f_face_X[kap + 0*n_faces_a];
			ufloat_g_t vy1 = geom_f_face_X[kap + 1*n_faces_a];
			ufloat_g_t vz1 = geom_f_face_X[kap + 2*n_faces_a];
			ufloat_g_t vx2 = geom_f_face_X[kap + 3*n_faces_a];
			ufloat_g_t vy2 = geom_f_face_X[kap + 4*n_faces_a];
			ufloat_g_t vz2 = geom_f_face_X[kap + 5*n_faces_a];
			ufloat_g_t vx3 = geom_f_face_X[kap + 6*n_faces_a];
			ufloat_g_t vy3 = geom_f_face_X[kap + 7*n_faces_a];
			ufloat_g_t vz3 = geom_f_face_X[kap + 8*n_faces_a];
			
			// Get bounding box.
			ufloat_g_t vBx_m = min(min(vx1,vx2),vx3);
			ufloat_g_t vBx_M = max(max(vx1,vx2),vx3);
			ufloat_g_t vBy_m = max(min(vy1,vy2),vy3);
			ufloat_g_t vBy_M = max(max(vy1,vy2),vy3);
			ufloat_g_t vBz_m = max(min(vz1,vz2),vz3);
			ufloat_g_t vBz_M = max(max(vz1,vz2),vz3);
			
			// Discretize the triangle and check if one of the points ends up inside the bin.
			// Only do the check by discretization if the bounding box intersects the current bin.
			if ( !( (vBx_m < xm && vBx_M < xm) || (vBx_m > xM && vBx_M > xM) || (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) ) || (vBz_m < zm && vBz_M < zm) || (vBz_m > zM && vBz_M > zM) )
			{
				ufloat_g_t L_l1 = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1));
				ufloat_g_t L_l2 = sqrt((vx3-vx3)*(vx3-vx2) + (vy3-vy2)*(vy3-vy2) + (vz3-vz2)*(vz3-vz2));
				ufloat_g_t L_l3 = sqrt((vx1-vx3)*(vx1-vx3) + (vy1-vy3)*(vy1-vy3) + (vz1-vz3)*(vz1-vz3));
				L_l1 = max(L_l1,L_l2);
				L_l1 = max(L_l1,L_l3);
				int Nu = 5;
				Nu = max((L_l1 / 0.25*minL0g),(ufloat_g_t)Nu);
				for (int p = 0; p < Nu; p++)
				{
					ufloat_g_t d_p = (p + 1.0 - 0.5)/Nu;
					ufloat_g_t vxp = vx2 + d_p*(vx3-vx2);
					ufloat_g_t vyp = vy2 + d_p*(vy3-vy2);
					ufloat_g_t vzp = vz2 + d_p*(vz3-vz2);
					
					for (int q = 0; q < Nu; q++)
					{
						ufloat_g_t d_q = (q + 1.0 - 0.5)/Nu;
						ufloat_g_t vxq = vx1 + d_q*(vxp-vx1);
						ufloat_g_t vyq = vy1 + d_q*(vyp-vy1);
						ufloat_g_t vzq = vz1 + d_q*(vzp-vz1);
						
						if ( (vxq > xm) && (vxq < xM) && (vyq > ym) && (vyq < yM) && (vzq > zm) && (vzq < zM) )
						{
							bin_indicators[kap + (j-j0)*n_faces_a] = kap;
							break;
						}
					}
				}
			}
}
		}
	}
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_MakeBins(int i_dev)
{
	// Assumes that coords list has already been initialized, make sure you do that first.
	n_bins = 1;
	for (int d = 0; d < N_DIM; d++)
		n_bins *= G_BIN_DENSITY;
	int G_BIN_NUM = n_bins / G_BIN_FRAC;
	ufloat_g_t Lx0g = Lx/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Ly0g = Ly/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Lz0g = Lz/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t minL0g = std::min({Lx0g,Ly0g,N_DIM==2?Lx0g:Lz0g});
	int rad = 0; //(int)(G_NEAR_WALL_DISTANCE/minL0g)+1;
	
	// Initialize the intermediate device array to store the bin indicators. Set values to -1 as default.
	gpuErrchk( cudaMalloc((void **)&c_bin_indicators[i_dev], G_BIN_NUM*n_faces_a[i_dev]*sizeof(int)) );
	Cu_ResetToValue<<<(M_BLOCK+G_BIN_NUM*n_faces_a[i_dev]-1)/M_BLOCK, M_BLOCK>>>(G_BIN_NUM*n_faces_a[i_dev], c_bin_indicators[i_dev], -1);
	
	// Just assuming max. of 2*n_faces_a, need to make this more robust. I don't want to store n_bins*n_faces of data.
	binned_face_ids_N[i_dev] = new int[n_bins];
	binned_face_ids_n[i_dev] = new int[n_bins];
	binned_face_ids[i_dev] = new int[10*n_faces_a[i_dev]];
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids[i_dev], 5*n_faces_a[i_dev]*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n[i_dev], n_bins*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N[i_dev], n_bins*sizeof(int)) );
	
	// Fill the bins, and then do stream compaction to get contiguous binned data.
	int n_pm1 = 0;
	int n_p = 0;
	cudaDeviceSynchronize();
	tic_simple("");
	std::cout << "[-] Starting binning process..." << std::endl;
	for (int j = 0; j < G_BIN_FRAC; j++)
	{
		// Fill the bins.
		int j0 = j*G_BIN_NUM;
		//std::cout << "Iter " << j << ", filling bins " << j0 << "-" << (j0+G_BIN_NUM) << std::endl;
		Cu_FillBins<ufloat_g_t,AP> <<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			j0, n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM, c_geom_f_face_X[i_dev], c_bin_indicators[i_dev],
			rad, Lx0g, Ly0g, Lz0g, minL0g, G_BIN_DENSITY
		);
		
		// Perform stream compaction.
		for (int p = 0; p < G_BIN_NUM; p++)
		{
			// Get the number of faces in this traversal.
			int j_p = j0 + p;
			n_p =  thrust::count_if(
				thrust::device, &c_bin_indicators[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], is_nonnegative()
			);
			binned_face_ids_N[i_dev][j_p] = n_pm1;
			binned_face_ids_n[i_dev][j_p] = n_p;
			//std::cout << n_p << " faces found in bin " << j_p << std::endl;
			
			// Copy the faces into the stored binned ID set.
			if (n_pm1 >= 10*n_faces_a[i_dev]-1)
				std::cout << "[-] WARNING: Insufficient memoery to store binned faces..." << std::endl;
			if (n_p > 0)
			{
				thrust::copy_if(
					thrust::device, &c_bin_indicators[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], &c_binned_face_ids[i_dev][n_pm1], is_nonnegative()
				);
			}
			
			// Update trackers.
			n_pm1 = binned_face_ids_N[i_dev][j_p] + n_p;
			//std::cout << "Cumulative count: " << n_pm1 << std::endl;
		}
		
		// Reset intermediate arrays before next run.
		Cu_ResetBins<<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM, c_bin_indicators[i_dev]
		);
	}
	cudaDeviceSynchronize();
	std::cout << "Elapsed time: " << toc_simple("",T_MS) << std::endl;
	std::cout << "[-] Finished binning process (" << n_pm1 << " faces)..." << std::endl;
	
	// Copy the face id counts to the GPU.
	gpuErrchk( cudaMemcpy(c_binned_face_ids_n[i_dev], binned_face_ids_n[i_dev], n_bins*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(c_binned_face_ids_N[i_dev], binned_face_ids_N[i_dev], n_bins*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(binned_face_ids[i_dev], c_binned_face_ids[i_dev], n_pm1*sizeof(int), cudaMemcpyDeviceToHost) );
	
	// DEBUG
// 	gpuErrchk( cudaMemcpy(binned_face_ids, c_binned_face_ids[i_dev], n_pm1*sizeof(int), cudaMemcpyDeviceToHost) );
// 	for (int p = 0; p < n_pm1; p++)
// 		std::cout << binned_face_ids[p] << " ";
// 	std::cout << std::endl;
// 	for (int p = 0; p < n_bins; p++)
// 		std::cout << binned_face_ids_N[p] << " ";
// 	std::cout << std::endl;
// 	for (int p = 0; p < n_bins; p++)
// 		std::cout << binned_face_ids_n[p] << " ";
// 	std::cout << std::endl;
	
	// Free memory in intermediate device arrays.
	gpuErrchk( cudaFree(c_bin_indicators[i_dev]) );
	
	return 0;
}
