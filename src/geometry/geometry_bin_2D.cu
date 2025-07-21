/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "structs.h"
#include "geometry.h"





/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_ResetBins2D ]===========================================================  */
/*                                                                                    */
/*  The bin indicator array is reset for the next call to Cu_FillBins.                */
/*                                                                                    */
/**************************************************************************************/

__global__
void Cu_ResetBins2D
(
	const int n_faces,
	const int n_faces_a,
	const int G_BIN_NUM,
	int *__restrict__ bin_indicators
)
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

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_FillBins2D_V1 ]=========================================================  */
/*                                                                                    */
/*  This CUDA kernel bins the faces of the geometry according to their bounding       */
/*  boxes. The vertices of the bounding box are truncated to the nearest integer      */
/*  (after scaling by a user-defined 'bin density'), and the face is considered to    */
/*  be a part of a bin if the bin is enclosed by the bounding box. Threads are        */
/*  assigned to each face and loop over a subset of bins for consideration (if the    */
/*  total number of bins is small enough, binning can be completed in one pass).      */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_FillBins2D_V1
(
	const int j0,
	const int n_faces,
	const int n_faces_a,
	const int G_BIN_NUM,
	const ufloat_g_t *__restrict__ geom_f_face_X,
	int *__restrict__ bin_indicators,
	const ufloat_g_t dx,
	const ufloat_g_t Lx,
	const ufloat_g_t LxOg,
	const ufloat_g_t LyOg,
	const ufloat_g_t LzOg,
	const int G_BIN_DENSITY
)
{
	constexpr int N_DIM = AP->N_DIM;
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_faces)
	{
		// Loop over the possible bins.
		for (int j = j0; j < j0+G_BIN_NUM; j++)
		{
			int bj = j%G_BIN_DENSITY;
			int bk = (j/G_BIN_DENSITY);
			
			// Load the current line vertices.
			ufloat_g_t vx1 = geom_f_face_X[kap + 0*n_faces_a];
			ufloat_g_t vy1 = geom_f_face_X[kap + 1*n_faces_a];
			ufloat_g_t vz1 = geom_f_face_X[kap + 2*n_faces_a];
			ufloat_g_t vx2 = geom_f_face_X[kap + 3*n_faces_a];
			ufloat_g_t vy2 = geom_f_face_X[kap + 4*n_faces_a];
			ufloat_g_t vz2 = geom_f_face_X[kap + 5*n_faces_a];
if (N_DIM==2)
{
			// Get the bounding box.
			ufloat_g_t vBx_m = fmin(vx1,vx2);
			ufloat_g_t vBx_M = fmax(vx1,vx2);
			ufloat_g_t vBy_m = fmin(vy1,vy2);
			ufloat_g_t vBy_M = fmax(vy1,vy2);
			
// 			int bin_id_xl = (int)(vBx_m*G_BIN_DENSITY)-1;
			int bin_id_yl = (int)(vBy_m*G_BIN_DENSITY)-1;
// 			int bin_id_xL = (int)(vBx_M*G_BIN_DENSITY)+2;
			int bin_id_yL = (int)(vBy_M*G_BIN_DENSITY)+2;
			
			if (bj >= bin_id_yl && bj < bin_id_yL)
				bin_indicators[kap + (j-j0)*n_faces_a] = kap;
}
else
{
			// Load the remaining line vertices.
			ufloat_g_t vx3 = geom_f_face_X[kap + 6*n_faces_a];
			ufloat_g_t vy3 = geom_f_face_X[kap + 7*n_faces_a];
			ufloat_g_t vz3 = geom_f_face_X[kap + 8*n_faces_a];
			
			// Get bounding box.
			ufloat_g_t vBx_m = fmin(fmin(vx1,vx2),vx3);
			ufloat_g_t vBx_M = fmax(fmax(vx1,vx2),vx3);
			ufloat_g_t vBy_m = fmin(fmin(vy1,vy2),vy3);
			ufloat_g_t vBy_M = fmax(fmax(vy1,vy2),vy3);
			ufloat_g_t vBz_m = fmin(fmin(vz1,vz2),vz3);
			ufloat_g_t vBz_M = fmax(fmax(vz1,vz2),vz3);
			
			// Identify the bin indices of the lower and upper bounds.
// 			int bin_id_xl = (int)(vBx_m*G_BIN_DENSITY)-1;
			int bin_id_yl = (int)(vBy_m*G_BIN_DENSITY)-1;
			int bin_id_zl = (int)(vBz_m*G_BIN_DENSITY)-1;
// 			int bin_id_xL = (int)(vBx_M*G_BIN_DENSITY)+2;
			int bin_id_yL = (int)(vBy_M*G_BIN_DENSITY)+2;
			int bin_id_zL = (int)(vBz_M*G_BIN_DENSITY)+2;
			
			if (bj >= bin_id_yl && bj+1 < bin_id_yL && bk >= bin_id_zl && bk+1 < bin_id_zL)
				bin_indicators[kap + (j-j0)*n_faces_a] = kap;
}
		}
	}
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_FillBins2D_V2 ]=========================================================  */
/*                                                                                    */
/*  This CUDA kernel bins the faces of the geometry according to their shape. The     */
/*  face is considered to be a part of a bin if it 1) is enclosed entirely by the     */
/*  bin, 2) cuts through the bin, 3) enloses the bin entirely. Threads are assigned   */
/*  to each face and loop over a subset of bins for consideration (if the total       */
/*  number of bins is small enough, binning can be completed in one pass).            */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_FillBins2D_V2
(
	const int j0,
	const int n_faces,
	const int n_faces_a,
	const int G_BIN_NUM,
	const ufloat_g_t *__restrict__ geom_f_face_X,
	int *__restrict__ bin_indicators,
	const ufloat_g_t dx,
	const ufloat_g_t Lx,
	const ufloat_g_t LxOg,
	const ufloat_g_t LyOg,
	const ufloat_g_t LzOg,
	const int G_BIN_DENSITY
)
{
	constexpr int N_DIM = AP->N_DIM;
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	ufloat_g_t ex1 = (ufloat_g_t)0.0;
	ufloat_g_t ey1 = (ufloat_g_t)0.0;
	ufloat_g_t ez1 = (ufloat_g_t)0.0;
	ufloat_g_t tmp = (ufloat_g_t)0.0;
	bool C = false;
	
	if (kap < n_faces)
	{
		// Loop over the possible bins.
		for (int j = j0; j < j0+G_BIN_NUM; j++)
		{
			int bj = j%G_BIN_DENSITY;
			int bk = (j/G_BIN_DENSITY);
			
			ufloat_g_t xm = (ufloat_g_t)0.0;
			ufloat_g_t xM = Lx;
			ufloat_g_t ym = (bj)*LyOg - dx;
			ufloat_g_t yM = (bj+1)*LyOg + dx;
			ufloat_g_t zm;
			ufloat_g_t zM;
			if (N_DIM==3)
			{
				zm = (bk)*LzOg - dx;
				zM = (bk+1)*LzOg + dx;
			}
			
			
			// Load the current line vertices.
			ufloat_g_t vx1 = geom_f_face_X[kap + 0*n_faces_a];
			ufloat_g_t vy1 = geom_f_face_X[kap + 1*n_faces_a];
			ufloat_g_t vz1 = geom_f_face_X[kap + 2*n_faces_a];
			ufloat_g_t vx2 = geom_f_face_X[kap + 3*n_faces_a];
			ufloat_g_t vy2 = geom_f_face_X[kap + 4*n_faces_a];
			ufloat_g_t vz2 = geom_f_face_X[kap + 5*n_faces_a];
if (N_DIM==2)
{
			
			// Get the bounding box.
			ufloat_g_t vBx_m = fmin(vx1,vx2);
			ufloat_g_t vBx_M = fmax(vx1,vx2);
			ufloat_g_t vBy_m = fmin(vy1,vy2);
			ufloat_g_t vBy_M = fmax(vy1,vy2);
			
			// Only consider this calculation if the bounding box intersects the bin.
			if ( !( (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) ) )
			{
				// Check if bounding box is entirely inside current bin.
				if (vBy_m > ym && vBy_M < yM) { C = true; }
				
				// Check if at least one of the vertices is inside the bin.
				if (vy1 > ym && vy1 < yM) { C = true; }
				if (vy2 > ym && vy2 < yM) { C = true; }
				
				// Check the bottom edge of the bin.
				{
					ey1 = vy2-vy1;
					tmp = (ym-vy1)/(ey1);
					ex1 = vx1 + tmp*(vx2-vx1);
					ey1 = vy1 + tmp*(vy2-vy1);
					if (CheckInLine(tmp,ex1,xm,xM)) { C = true; }
				}
				
				// Check the top edge of the bin.
				{
					ey1 = vy2-vy1;
					tmp = (yM-vy1)/(ey1);
					ex1 = vx1 + tmp*(vx2-vx1);
					ey1 = vy1 + tmp*(vy2-vy1);
					if (CheckInLine(tmp,ex1,xm,xM)) { C = true; }
				}
				
				if (C)
					bin_indicators[kap + (j-j0)*n_faces_a] = kap;
				C = false;
			}
}
else
{
			// Load the remaining line vertices.
			ufloat_g_t vx3 = geom_f_face_X[kap + 6*n_faces_a];
			ufloat_g_t vy3 = geom_f_face_X[kap + 7*n_faces_a];
			ufloat_g_t vz3 = geom_f_face_X[kap + 8*n_faces_a];
			
			// Get bounding box.
			ufloat_g_t vBx_m = fmin(fmin(vx1,vx2),vx3);
			ufloat_g_t vBx_M = fmax(fmax(vx1,vx2),vx3);
			ufloat_g_t vBy_m = fmin(fmin(vy1,vy2),vy3);
			ufloat_g_t vBy_M = fmax(fmax(vy1,vy2),vy3);
			ufloat_g_t vBz_m = fmin(fmin(vz1,vz2),vz3);
			ufloat_g_t vBz_M = fmax(fmax(vz1,vz2),vz3);
			
			// Only consider this calculation if the bounding box intersects the bin.
			if ( !( (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) || (vBz_m < zm && vBz_M < zm) || (vBz_m > zM && vBz_M > zM) ) )
			{
				// Check if bounding box is entirely inside current bin.
				if (vBy_m > ym && vBy_M < yM && vBz_m > zm && vBz_M < zM) { C = true; }
				
				// Check if bounding box completely surrounds the bin.
				if (vBx_m < xm && vBx_M > xM && vBy_m < ym && vBy_M > yM) { C = true; }
				if (vBy_m < ym && vBy_M > yM && vBz_m < zm && vBz_M > zM) { C = true; }
				if (vBz_m < zm && vBz_M > zM && vBx_m < xm && vBx_M > xM) { C = true; }
				
				// Check if at least one of the vertices is inside the bin.
				if (vy1 > ym && vy1 < yM && vz1 > zm && vz1 < zM) { C = true; }
				if (vy2 > ym && vy2 < yM && vz2 > zm && vz2 < zM) { C = true; }
				if (vy3 > ym && vy3 < yM && vz3 > zm && vz3 < zM) { C = true; }
				
				// Check the bottom face of the bin.
				{
					ez1 = vz2-vz1;
					tmp = (zm-vz1)/(ez1);
					ex1 = vx1 + tmp*(vx2-vx1);
					ey1 = vy1 + tmp*(vy2-vy1);
					ez1 = vz1 + tmp*(vz2-vz1);
					if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { C = true; }
				}
				{
					ez1 = vz3-vz2;
					tmp = (zm-vz2)/(ez1);
					ex1 = vx2 + tmp*(vx3-vx2);
					ey1 = vy2 + tmp*(vy3-vy2);
					ez1 = vz2 + tmp*(vz3-vz2);
					if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { C = true; }
				}
				{
					ez1 = vz1-vz3;
					tmp = (zm-vz3)/(ez1);
					ex1 = vx3 + tmp*(vx1-vx3);
					ey1 = vy3 + tmp*(vy1-vy3);
					ez1 = vz3 + tmp*(vz1-vz3);
					if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { C = true; }
				}
				
				// Check the top face of the bin.
				{
					ez1 = vz2-vz1;
					tmp = (zM-vz1)/(ez1);
					ex1 = vx1 + tmp*(vx2-vx1);
					ey1 = vy1 + tmp*(vy2-vy1);
					ez1 = vz1 + tmp*(vz2-vz1);
					if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { C = true; }
				}
				{
					ez1 = vz3-vz2;
					tmp = (zM-vz2)/(ez1);
					ex1 = vx2 + tmp*(vx3-vx2);
					ey1 = vy2 + tmp*(vy3-vy2);
					ez1 = vz2 + tmp*(vz3-vz2);
					if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { C = true; }
				}
				{
					ez1 = vz1-vz3;
					tmp = (zM-vz3)/(ez1);
					ex1 = vx3 + tmp*(vx1-vx3);
					ey1 = vy3 + tmp*(vy1-vy3);
					ez1 = vz3 + tmp*(vz1-vz3);
					if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { C = true; }
				}
				
				// Check the back face of the bin.
				{
					ey1 = vy2-vy1;
					tmp = (ym-vy1)/(ey1);
					ex1 = vx1 + tmp*(vx2-vx1);
					ey1 = vy1 + tmp*(vy2-vy1);
					ez1 = vz1 + tmp*(vz2-vz1);
					if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { C = true; }
				}
				{
					ey1 = vy3-vy2;
					tmp = (ym-vy2)/(ey1);
					ex1 = vx2 + tmp*(vx3-vx2);
					ey1 = vy2 + tmp*(vy3-vy2);
					ez1 = vz2 + tmp*(vz3-vz2);
					if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { C = true; }
				}
				{
					ey1 = vy1-vy3;
					tmp = (ym-vy3)/(ey1);
					ex1 = vx3 + tmp*(vx1-vx3);
					ey1 = vy3 + tmp*(vy1-vy3);
					ez1 = vz3 + tmp*(vz1-vz3);
					if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { C = true; }
				}
				
				// Check the front face of the bin.
				{
					ey1 = vy2-vy1;
					tmp = (yM-vy1)/(ey1);
					ex1 = vx1 + tmp*(vx2-vx1);
					ey1 = vy1 + tmp*(vy2-vy1);
					ez1 = vz1 + tmp*(vz2-vz1);
					if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { C = true; }
				}
				{
					ey1 = vy3-vy2;
					tmp = (yM-vy2)/(ey1);
					ex1 = vx2 + tmp*(vx3-vx2);
					ey1 = vy2 + tmp*(vy3-vy2);
					ez1 = vz2 + tmp*(vz3-vz2);
					if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { C = true; }
				}
				{
					ey1 = vy1-vy3;
					tmp = (yM-vy3)/(ey1);
					ex1 = vx3 + tmp*(vx1-vx3);
					ey1 = vy3 + tmp*(vy1-vy3);
					ez1 = vz3 + tmp*(vz1-vz3);
					if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { C = true; }
				}
				
				if (C)
					bin_indicators[kap + (j-j0)*n_faces_a] = kap;
				C = false;
			}
}
		}
	}
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_MakeBins2D ]=============================================================  */
/*                                                                                    */
/*  Performs the '2D binning' portion of the geometry-incorporation algorithm.        */
/*  Here, faces (edges or triangles in 2D or 3D, respectively) are assigned to        */
/*  rectangularprismatic bins to be used in the cell-face linkage identification      */
/*  step.                                                                             */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_MakeBins2D(int i_dev)
{
	// Assumes that coords list has already been initialized, make sure you do that first.
	n_bins_v = 1;
	for (int d = 0; d < N_DIM-1; d++)
		n_bins_v *= G_BIN_DENSITY;
	int G_BIN_NUM = n_bins_v / G_BIN_FRAC;
	ufloat_g_t Lx0g = Lx/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Ly0g = Ly/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Lz0g = Lz/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t eps = 1e-5;
	if (std::is_same<ufloat_g_t, float>::value) eps = FLT_EPSILON;
	if (std::is_same<ufloat_g_t, double>::value) eps = DBL_EPSILON;
	
	// Initialize the intermediate device array to store the bin indicators. Set values to -1 as default.
	gpuErrchk( cudaMalloc((void **)&c_bin_indicators_v[i_dev], G_BIN_NUM*n_faces_a[i_dev]*sizeof(int)) );
	Cu_ResetToValue<<<(M_BLOCK+G_BIN_NUM*n_faces_a[i_dev]-1)/M_BLOCK, M_BLOCK>>>(G_BIN_NUM*n_faces_a[i_dev], c_bin_indicators_v[i_dev], -1);
	
	// Just assuming max. of 2*n_faces_a, need to make this more robust. I don't want to store n_bins_v*n_faces of data.
	binned_face_ids_N_v[i_dev] = new int[n_bins_v];
	binned_face_ids_n_v[i_dev] = new int[n_bins_v];
	binned_face_ids_v[i_dev] = new int[10*n_faces_a[i_dev]];
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_v[i_dev], 10*n_faces_a[i_dev]*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_v[i_dev], n_bins_v*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_v[i_dev], n_bins_v*sizeof(int)) );
	
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
		if (G_BIN_APPROACH==0)
		{
			Cu_FillBins2D_V1<ufloat_g_t,AP> <<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
				j0, n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM, c_geom_f_face_X[i_dev], c_bin_indicators_v[i_dev],
				eps, Lx, Lx0g, Ly0g, Lz0g, G_BIN_DENSITY
			);
		}
		else
		{
			Cu_FillBins2D_V2<ufloat_g_t,AP> <<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
				j0, n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM, c_geom_f_face_X[i_dev], c_bin_indicators_v[i_dev],
				eps, Lx, Lx0g, Ly0g, Lz0g, G_BIN_DENSITY
			);
		}
		
		// Perform stream compaction.
		for (int p = 0; p < G_BIN_NUM; p++)
		{
			// Get the number of faces in this traversal.
			int j_p = j0 + p;
			n_p =  thrust::count_if(
				thrust::device, &c_bin_indicators_v[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators_v[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], is_nonnegative()
			);
			binned_face_ids_N_v[i_dev][j_p] = n_pm1;
			binned_face_ids_n_v[i_dev][j_p] = n_p;
			
			// Copy the faces into the stored binned ID set.
			if (n_pm1 >= 10*n_faces_a[i_dev]-1)
				std::cout << "[-] WARNING: Insufficient memory to store binned faces..." << std::endl;
			if (n_p > 0)
			{
				thrust::copy_if(
					thrust::device, &c_bin_indicators_v[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators_v[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], &c_binned_face_ids_v[i_dev][n_pm1], is_nonnegative()
				);
			}
			
			// Update trackers.
			n_pm1 = binned_face_ids_N_v[i_dev][j_p] + n_p;
		}
		
		// Reset intermediate arrays before next run.
		Cu_ResetBins2D<<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM, c_bin_indicators_v[i_dev]
		);
	}
	cudaDeviceSynchronize();
	std::cout << "Elapsed time: " << toc_simple("",T_MS) << std::endl;
	std::cout << "[-] Finished binning process (" << n_pm1 << " faces/" << 10*n_faces_a[i_dev] << ")..." << std::endl;
	
	// Copy the face id counts to the GPU.
	gpuErrchk( cudaMemcpy(c_binned_face_ids_n_v[i_dev], binned_face_ids_n_v[i_dev], n_bins_v*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(c_binned_face_ids_N_v[i_dev], binned_face_ids_N_v[i_dev], n_bins_v*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(binned_face_ids_v[i_dev], c_binned_face_ids_v[i_dev], n_pm1*sizeof(int), cudaMemcpyDeviceToHost) );
	
	// DEBUG
	std::cout << "APPROACH: 2D" << std::endl;
	for (int p = 0; p < n_bins_v; p++)
	{
		int Nbp = binned_face_ids_N_v[i_dev][p];
		int npv = binned_face_ids_n_v[i_dev][p];
		if (npv > 0)
		{
			std::cout << "Bin #" << p << ": ";
			for (int K = 0; K < npv; K++)
				std::cout << binned_face_ids_v[i_dev][Nbp + K] << " ";
			std::cout << std::endl;
		}
	}
	
	// Free memory in intermediate device arrays.
	gpuErrchk( cudaFree(c_bin_indicators_v[i_dev]) );
	
	return 0;
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_DrawBinsAndFaces2D ]=====================================================  */
/*                                                                                    */
/*  This is a debug routine that generates a MATLAB script in which the various       */
/*  bins and faces assignments can be plotted.                                        */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_DrawBinsAndFaces2D(int i_dev)
{
	// Initial parameters. Open the output file.
	int N_DIM = AP->N_DIM;
	ufloat_g_t Lx0g = Lx/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Ly0g = Ly/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Lz0g = Lz/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t minL0g = std::min({Lx0g,Ly0g,N_DIM==2?Lx0g:Lz0g});
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	std::ofstream out = std::ofstream("debug_bins.txt");
	std:cout << "Drawing 2D bins..." << std::endl;
	
	for (int k = 0; k < G_BIN_DENSITY; k++)
	{
		for (int j = 0; j < G_BIN_DENSITY; j++)
		{
			// Identify the bin.
			int global_bin_id = j + G_BIN_DENSITY*k;
			
			// Get the number of faces in the bin.
			int n_f = binned_face_ids_n_v[i_dev][global_bin_id];
			int N_f = 0;
			if (n_f > 0)
				N_f = binned_face_ids_N_v[i_dev][global_bin_id];
			
			// If there are faces to draw, draw the bin too. Each bin gets its own unique color.
			if (n_f > 0)
			{
				c0 = (double)(std::rand() % 256) / 256.0;
				c1 = (double)(std::rand() % 256) / 256.0;
				c2 = (double)(std::rand() % 256) / 256.0;
				DebugDrawCubeInMATLAB(out, 0, Lx, j*Ly0g, (j+1)*Ly0g, k*Lz0g, (k+1)*Lz0g, c0, c1, c2);
			}
			
			for (int p = 0; p < n_f; p++)
			{
				int f_p = binned_face_ids_v[i_dev][N_f+p];
				ufloat_g_t vx1 = geom_f_face_X[i_dev][f_p + 0*n_faces_a[i_dev]];
				ufloat_g_t vy1 = geom_f_face_X[i_dev][f_p + 1*n_faces_a[i_dev]];
				ufloat_g_t vz1 = geom_f_face_X[i_dev][f_p + 2*n_faces_a[i_dev]];
				ufloat_g_t vx2 = geom_f_face_X[i_dev][f_p + 3*n_faces_a[i_dev]];
				ufloat_g_t vy2 = geom_f_face_X[i_dev][f_p + 4*n_faces_a[i_dev]];
				ufloat_g_t vz2 = geom_f_face_X[i_dev][f_p + 5*n_faces_a[i_dev]];
				ufloat_g_t vx3 = geom_f_face_X[i_dev][f_p + 6*n_faces_a[i_dev]];
				ufloat_g_t vy3 = geom_f_face_X[i_dev][f_p + 7*n_faces_a[i_dev]];
				ufloat_g_t vz3 = geom_f_face_X[i_dev][f_p + 8*n_faces_a[i_dev]];
				
				// Draw the faces in the current bin.
				DebugDrawTriangleInMATLAB(out, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, c0, c1, c2);
			}
		}
	}
	
	// Close the file.
	std::cout << "Finished drawing 2D bins..." << std::endl;
	out.close();
	
	return 0;
}
