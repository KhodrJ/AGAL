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

__global__
void Cu_FillBins(
	int j0, int n_faces, int n_faces_a, int G_BIN_NUM, double *geom_f_face_X, int *bin_indicators,
	double LxOg, double LyOg, double LzOg, double minL0g,int G_BIN_DENSITY
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_faces)
	{
		// Loop over the possible bins.
		for (int j = j0; j < j0+G_BIN_NUM; j++)
		{
			int bi = j%G_BIN_DENSITY;
			int bj = (j/G_BIN_DENSITY)%G_BIN_DENSITY;
			int bk = (j/G_BIN_DENSITY)/G_BIN_DENSITY;
			
			double xm = bi*LxOg;
			double xM = (bi+1)*LxOg;
			double ym = bj*LyOg;
			double yM = (bj+1)*LyOg;
#if (N_DIM==3)
			double zm = bk*LzOg;
			double zM = (bk+1)*LzOg;
#endif
			
			
			
#if (N_DIM==2)
			// Load the current line vertices.
			double vx1 = geom_f_face_X[kap + 0*n_faces_a];
			double vy1 = geom_f_face_X[kap + 1*n_faces_a];
			double vz1 = geom_f_face_X[kap + 2*n_faces_a];
			double vx2 = geom_f_face_X[kap + 3*n_faces_a];
			double vy2 = geom_f_face_X[kap + 4*n_faces_a];
			double vz2 = geom_f_face_X[kap + 5*n_faces_a];
			
			// Get the bounding box.
			double vBx_m = min(vx1,vx2);
			double vBx_M = max(vx1,vx2);
			double vBy_m = min(vy1,vy2);
			double vBy_M = max(vy1,vy2);
			//printf("Bounding box of edge %i ([%17.5f,%17.5f]-[%17.5f,%17.5f]) is [%17.5f,%17.5f]x[%17.5f,%17.5f]\n",kap,vx1,vy1,vx2,vy2,vBx_m,vBx_M,vBy_m,vBy_M);
			//printf("Current bin (edge %i): %i (%i,%i), [%17.5f,%17.5f]x[%17.5f,%17.5f]\n",kap, bi+G_BIN_DENSITY*bj, bi, bj, xm,xM,ym,yM);
			
			// Discretize the line and check if one of the points ends up inside the bin.
			// Only do the check by discretization if the bounding box intersects the current bin.
			if ( !( (vBx_m < xm && vBx_M < xm) || (vBx_m > xM && vBx_M > xM) || (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) ) )
			{
				double L_l = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1));
				int Nu = 10;
				Nu = max((L_l / 0.25*minL0g), (double)Nu);
				//printf("Length of current line %i is %17.15f, minL=%17.15f\n", kap, L_l, minL0g);
				for (int p = 0; p < Nu; p++)
				{
					double d_p = (p + 1.0 - 0.5)/Nu;
					double vxp = vx1 + d_p*(vx2-vx1);
					double vyp = vy1 + d_p*(vy2-vy1);
					
					if ( (vxp > xm) && (vxp < xM) && (vyp > ym) && (vyp < yM) )
					{
						// At the first inidication that a point lies in the bounding box, terminate the loop.
						bin_indicators[kap + (j-j0)*n_faces_a] = kap;
						//printf("Face %i ([%17.5f,%17.5f]-[%17.5f,%17.5f]) lies in bin %i\n", kap, vx1, vy1, vx2, vy2, bi+G_BIN_DENSITY*bj);
						break;
					}
				}
			}
		
#else
			// Load the current line vertices.
			double vx1 = geom_f_face_X[kap + 0*n_faces_a];
			double vy1 = geom_f_face_X[kap + 1*n_faces_a];
			double vz1 = geom_f_face_X[kap + 2*n_faces_a];
			double vx2 = geom_f_face_X[kap + 3*n_faces_a];
			double vy2 = geom_f_face_X[kap + 4*n_faces_a];
			double vz2 = geom_f_face_X[kap + 5*n_faces_a];
			double vx3 = geom_f_face_X[kap + 6*n_faces_a];
			double vy3 = geom_f_face_X[kap + 7*n_faces_a];
			double vz3 = geom_f_face_X[kap + 8*n_faces_a];
			
			// Get bounding box.
			double vBx_m = min(min(vx1,vx2),vx3);
			double vBx_M = max(max(vx1,vx2),vx3);
			double vBy_m = max(min(vy1,vy2),vy3);
			double vBy_M = max(max(vy1,vy2),vy3);
			double vBz_m = max(min(vz1,vz2),vz3);
			double vBz_M = max(max(vz1,vz2),vz3);
			
			// Discretize the triangle and check if one of the points ends up inside the bin.
			// Only do the check by discretization if the bounding box intersects the current bin.
			if ( !( (vBx_m < xm && vBx_M < xm) || (vBx_m > xM && vBx_M > xM) || (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) ) || (vBz_m < zm && vBz_M < zm) || (vBz_m > zM && vBz_M > zM) )
			{
				double L_l1 = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1));
				double L_l2 = sqrt((vx3-vx3)*(vx3-vx2) + (vy3-vy2)*(vy3-vy2) + (vz3-vz2)*(vz3-vz2));
				double L_l3 = sqrt((vx1-vx3)*(vx1-vx3) + (vy1-vy3)*(vy1-vy3) + (vz1-vz3)*(vz1-vz3));
				L_l1 = max(L_l1,L_l2);
				L_l1 = max(L_l1,L_l3);
				int Nu = 10;
				Nu = std::max((L_l1 / 0.25*minL0g),(double)Nu);
				for (int p = 0; p < Nu; p++)
				{
					double d_p = (p + 1.0 - 0.5)/Nu;
					double vxp = vx2 + d_p*(vx3-vx2);
					double vyp = vy2 + d_p*(vy3-vy2);
					double vzp = vz2 + d_p*(vz3-vz2);
					
					for (int q = 0; q < Nu; q++)
					{
						double d_q = (q + 1.0 - 0.5)/Nu;
						double vxq = vx1 + d_q*(vxp-vx1);
						double vyq = vy1 + d_q*(vyp-vy1);
						double vzq = vz1 + d_q*(vzp-vz1);
						
						if ( (vxq > xm) && (vxq < xM) && (vyq > ym) && (vyq < yM) && (vzq > zm) && (vzq < zM) )
						{
							bin_indicators[kap + (j-j0)*n_faces_a] = kap;
							break;
						}
					}
				}
			}
#endif
		}
	}
}

struct is_nonnegative
{
	__device__ bool operator()(const int ID)
	{
		return ID>=0;
	}
};

int Geometry::G_MakeBins(int i_dev)
{
	// Assumes that coords list has already been initialized, make sure you do that first.
	int n_bins = 1;
	for (int d = 0; d < N_DIM; d++)
		n_bins *= G_BIN_DENSITY;
	int G_BIN_NUM = n_bins / G_BIN_FRAC;
	double Lx0g = Lx/(double)G_BIN_DENSITY;
	double Ly0g = Ly/(double)G_BIN_DENSITY;
	double Lz0g = Lz/(double)G_BIN_DENSITY;
#if (N_DIM==2)
	double minL0g = std::min({Lx0g,Lx0g,Lx0g});
#else
	double minL0g = std::min({Lx0g,Lx0g,Lx0g});
#endif
	
	// Initialize the intermediate device array to store the bin indicators. Set values to -1 as default.
	bin_indicators[i_dev] = new int[G_BIN_NUM*n_faces_a[i_dev]];
	gpuErrchk( cudaMalloc((void **)&c_bin_indicators[i_dev], G_BIN_NUM*n_faces_a[i_dev]*sizeof(int)) );
	Cu_ResetToValue<<<(M_BLOCK+G_BIN_NUM*n_faces_a[i_dev]-1)/M_BLOCK, M_BLOCK>>>(G_BIN_NUM*n_faces_a[i_dev], c_bin_indicators[i_dev], -1);
	
	// NOTE: Just assuming max. of 10*n_faces_a, need to make this more robust. I don't want to store n_bins*n_faces of data.
	int *binned_face_ids_N = new int[n_bins];
	int *binned_face_ids = new int[10*n_faces_a[i_dev]];
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids[i_dev], 10*n_faces_a[i_dev]*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N[i_dev], n_bins*sizeof(int)) );
	//int *c_binned_face_ids[N_DEV];
	//int *c_binned_face_ids_N[N_DEV];
	
	int n_pm1 = 0;
	int n_p = 0;
	for (int j = 0; j < G_BIN_FRAC; j++)
	{
		// Fill the bins.
		int j0 = j*G_BIN_NUM;
		Cu_FillBins<<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			j0, n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM, c_geom_f_face_X[i_dev], c_bin_indicators[i_dev],
			Lx0g, Ly0g, Lz0g, minL0g, G_BIN_DENSITY
		);
		
		// Perform stream compaction.
		for (int p = 0; p < G_BIN_NUM; p++)
		{
			// Get the number of faces in this traversal.
			int j_p = j0 + p;
			int n_p =  thrust::count_if(
				thrust::device, &c_bin_indicators[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], is_nonnegative()
			);
			binned_face_ids_N[j_p] = n_p + n_pm1;
			std::cout << n_p << " faces found in bin " << j_p << std::endl;
			
			// Copy the faces into the stored binned ID set.
			thrust::copy_if(
				thrust::device, &c_bin_indicators[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], &c_binned_face_ids[i_dev][n_pm1], is_nonnegative()
			);
			
			// Update trackers.
			n_pm1 = n_p;
		}
		
		// Reset intermediate arrays before next run.
		Cu_ResetBins<<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM, c_bin_indicators[i_dev]
		);
	}
	
	// DEBUG
	
	
	// Free memory in intermediate device arrays.
	
	
	return 0;
}
