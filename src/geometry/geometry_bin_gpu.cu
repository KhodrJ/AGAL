/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_ComputeBoundingBoxLimits2D ]============================================  */
/*                                                                                    */
/*  Faces are traversed, and the limits of their bounding boxes are computed and      */
/*  truncated to produce indices that map to the uniformly-sized bins. Each face      */
/*  may cross up to 4^(D-1) bins depending on their size and orientation. A           */
/*  duplicate of the current face index is stored in bounding_box_index_limits for    */
/*  each overlapped bin index stored in bounding_box_limits, so that the result       */
/*  after sorting by key is a set of pairs (face,bin) that accounts for bin overlap   */
/*  properly. The 2D version of this kernel does not requiring modifying the bin      */
/*  volume.                                                                           */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_g_t, int N_DIM>
__global__
void Cu_ComputeBoundingBoxLimits2D
(
	const int n_faces,
	const int n_faces_a,
	const ufloat_g_t *__restrict__ geom_f_face_X,
	int *__restrict__ bounding_box_limits,
	int *__restrict__ bounding_box_index_limits,
	const ufloat_g_t dx,
	const ufloat_g_t Lx,
	const ufloat_g_t Ly,
	const ufloat_g_t Lz,
	const int G_BIN_DENSITY
)
{
int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_faces)
	{
		if (N_DIM==2)
		{
			ufloat_g_t vx1 = geom_f_face_X[kap + 0*n_faces_a];
			ufloat_g_t vy1 = geom_f_face_X[kap + 1*n_faces_a];
			ufloat_g_t vx2 = geom_f_face_X[kap + 3*n_faces_a];
			ufloat_g_t vy2 = geom_f_face_X[kap + 4*n_faces_a];
			
			ufloat_g_t vBx_m = fmin(vx1, vx2);
			ufloat_g_t vBx_M = fmax(vx1, vx2);
			ufloat_g_t vBy_m = fmin(vy1, vy2);
			ufloat_g_t vBy_M = fmax(vy1, vy2);
			
			// C is used to determine if a face is completely outside of the bounding box.
			bool C = true;
			if (vBx_m<-dx&&vBx_M<-dx || vBx_m>Lx+dx&&vBx_M>Lx+dx)
				C = false;
			if (vBy_m<-dx&&vBy_M<-dx || vBy_m>Ly+dx&&vBy_M>Ly+dx)
				C = false;
			
			int bin_id_yl = (int)(vBy_m*G_BIN_DENSITY);
			int bin_id_yL = (int)(vBy_M*G_BIN_DENSITY);
			
			// Note: this assumes that the faces intersect, at most, eight bins.
			int counter = 0;
			for (int J = bin_id_yl; J < bin_id_yL+1; J++)
			{
				//for (int I = bin_id_xl; I < bin_id_xL+1; I++)
				//{
					if (C && counter < 4)
					{
						int global_id = J;
						bounding_box_limits[kap + counter*n_faces] = global_id;
						bounding_box_index_limits[kap + counter*n_faces] = kap;
						counter++;
					}
				//}
			}
		}
		else // N_DIM==3
		{
			ufloat_g_t vx1 = geom_f_face_X[kap + 0*n_faces_a];
			ufloat_g_t vy1 = geom_f_face_X[kap + 1*n_faces_a];
			ufloat_g_t vz1 = geom_f_face_X[kap + 2*n_faces_a];
			ufloat_g_t vx2 = geom_f_face_X[kap + 3*n_faces_a];
			ufloat_g_t vy2 = geom_f_face_X[kap + 4*n_faces_a];
			ufloat_g_t vz2 = geom_f_face_X[kap + 5*n_faces_a];
			ufloat_g_t vx3 = geom_f_face_X[kap + 6*n_faces_a];
			ufloat_g_t vy3 = geom_f_face_X[kap + 7*n_faces_a];
			ufloat_g_t vz3 = geom_f_face_X[kap + 8*n_faces_a];
			
			ufloat_g_t vBx_m = fmin(fmin(vx1, vx2), vx3);
			ufloat_g_t vBx_M = fmax(fmax(vx1, vx2), vx3);
			ufloat_g_t vBy_m = fmin(fmin(vy1, vy2), vy3);
			ufloat_g_t vBy_M = fmax(fmax(vy1, vy2), vy3);
			ufloat_g_t vBz_m = fmin(fmin(vz1, vz2), vz3);
			ufloat_g_t vBz_M = fmax(fmax(vz1, vz2), vz3);
			
			// C is used to determine if a face is completely outside of the bounding box.
			bool C = true;
			if (vBx_m<-dx&&vBx_M<-dx || vBx_m>Lx+dx&&vBx_M>Lx+dx)
				C = false;
			if (vBy_m<-dx&&vBy_M<-dx || vBy_m>Ly+dx&&vBy_M>Ly+dx)
				C = false;
			if (vBz_m<-dx&&vBz_M<-dx || vBz_m>Lz+dx&&vBz_M>Lz+dx)
				C = false;
			
			int bin_id_yl = (int)(vBy_m*G_BIN_DENSITY);
			int bin_id_zl = (int)(vBz_m*G_BIN_DENSITY);
			int bin_id_yL = (int)(vBy_M*G_BIN_DENSITY);
			int bin_id_zL = (int)(vBz_M*G_BIN_DENSITY);
			
			// Note: this assumes that the faces intersect, at most, eight bins.
			int counter = 0;
			for (int K = bin_id_zl; K < bin_id_zL+1; K++)
			{
				for (int J = bin_id_yl; J < bin_id_yL+1; J++)
				{
					//for (int I = bin_id_xl; I < bin_id_xL+1; I++)
					//{
						if (C && counter < 16)
						{
							int global_id = J + G_BIN_DENSITY*K;
							bounding_box_limits[kap + counter*n_faces] = global_id;
							bounding_box_index_limits[kap + counter*n_faces] = kap;
							counter++;
						}
					//}
				}
			}
		}
	}
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_ComputeBoundingBoxLimits3D ]============================================  */
/*                                                                                    */
/*  Faces are traversed, and the limits of their bounding boxes are computed and      */
/*  truncated to produce indices that map to the uniformly-sized bins. Each face      */
/*  may cross up to 4^D bins depending on their size and orientation. A duplicate     */
/*  of the current face index is stored in bounding_box_index_limits for each         */
/*  overlapped bin index stored in bounding_box_limits, so that the result after      */
/*  sorting by key is a set of pairs (face,bin) that accounts for bin overlap         */
/*  properly. The 3D version of this kernel requires extending the bin volume by an   */
/*  amount dx in each direction to account for cut-links that may cross bins for      */
/*  cells lying directly on a bin boundary.                                           */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_g_t, int N_DIM>
__global__
void Cu_ComputeBoundingBoxLimits3D
(
	const int n_faces,
	const int n_faces_a,
	const ufloat_g_t *__restrict__ geom_f_face_X,
	int *__restrict__ bounding_box_limits,
	int *__restrict__ bounding_box_index_limits,
	const ufloat_g_t dx,
	const ufloat_g_t Lx,
	const ufloat_g_t Ly,
	const ufloat_g_t Lz,
	const int G_BIN_DENSITY
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_faces)
	{
		if (N_DIM==2)
		{
			ufloat_g_t vx1 = geom_f_face_X[kap + 0*n_faces_a];
			ufloat_g_t vy1 = geom_f_face_X[kap + 1*n_faces_a];
			ufloat_g_t vx2 = geom_f_face_X[kap + 3*n_faces_a];
			ufloat_g_t vy2 = geom_f_face_X[kap + 4*n_faces_a];
			
			ufloat_g_t vBx_m = fmin(vx1, vx2);
			ufloat_g_t vBx_M = fmax(vx1, vx2);
			ufloat_g_t vBy_m = fmin(vy1, vy2);
			ufloat_g_t vBy_M = fmax(vy1, vy2);
			
			// C is used to determine if a face is completely outside of the bounding box.
			bool C = true;
			if (vBx_m<-dx&&vBx_M<-dx || vBx_m>Lx+dx&&vBx_M>Lx+dx)
				C = false;
			if (vBy_m<-dx&&vBy_M<-dx || vBy_m>Ly+dx&&vBy_M>Ly+dx)
				C = false;
			
			int bin_id_xl = (int)((vBx_m-dx)*G_BIN_DENSITY);
			int bin_id_yl = (int)((vBy_m-dx)*G_BIN_DENSITY);
			int bin_id_xL = (int)((vBx_M+dx)*G_BIN_DENSITY);
			int bin_id_yL = (int)((vBy_M+dx)*G_BIN_DENSITY);
			
			// Note: this assumes that the faces intersect, at most, eight bins.
			int counter = 0;
			for (int J = bin_id_yl; J < bin_id_yL+1; J++)
			{
				for (int I = bin_id_xl; I < bin_id_xL+1; I++)
				{
					if (C && counter < 16)
					{
						int global_id = I + G_BIN_DENSITY*J;
						bounding_box_limits[kap + counter*n_faces] = global_id;
						bounding_box_index_limits[kap + counter*n_faces] = kap;
						counter++;
					}
				}
			}
		}
		else // N_DIM==3
		{
			ufloat_g_t vx1 = geom_f_face_X[kap + 0*n_faces_a];
			ufloat_g_t vy1 = geom_f_face_X[kap + 1*n_faces_a];
			ufloat_g_t vz1 = geom_f_face_X[kap + 2*n_faces_a];
			ufloat_g_t vx2 = geom_f_face_X[kap + 3*n_faces_a];
			ufloat_g_t vy2 = geom_f_face_X[kap + 4*n_faces_a];
			ufloat_g_t vz2 = geom_f_face_X[kap + 5*n_faces_a];
			ufloat_g_t vx3 = geom_f_face_X[kap + 6*n_faces_a];
			ufloat_g_t vy3 = geom_f_face_X[kap + 7*n_faces_a];
			ufloat_g_t vz3 = geom_f_face_X[kap + 8*n_faces_a];
			
			ufloat_g_t vBx_m = fmin(fmin(vx1, vx2), vx3);
			ufloat_g_t vBx_M = fmax(fmax(vx1, vx2), vx3);
			ufloat_g_t vBy_m = fmin(fmin(vy1, vy2), vy3);
			ufloat_g_t vBy_M = fmax(fmax(vy1, vy2), vy3);
			ufloat_g_t vBz_m = fmin(fmin(vz1, vz2), vz3);
			ufloat_g_t vBz_M = fmax(fmax(vz1, vz2), vz3);
			
			// C is used to determine if a face is completely outside of the bounding box.
			bool C = true;
			if (vBx_m<-dx&&vBx_M<-dx || vBx_m>Lx+dx&&vBx_M>Lx+dx)
				C = false;
			if (vBy_m<-dx&&vBy_M<-dx || vBy_m>Ly+dx&&vBy_M>Ly+dx)
				C = false;
			if (vBz_m<-dx&&vBz_M<-dx || vBz_m>Lz+dx&&vBz_M>Lz+dx)
				C = false;
			
			int bin_id_xl = (int)((vBx_m-dx)*G_BIN_DENSITY);
			int bin_id_yl = (int)((vBy_m-dx)*G_BIN_DENSITY);
			int bin_id_zl = (int)((vBz_m-dx)*G_BIN_DENSITY);
			int bin_id_xL = (int)((vBx_M+dx)*G_BIN_DENSITY);
			int bin_id_yL = (int)((vBy_M+dx)*G_BIN_DENSITY);
			int bin_id_zL = (int)((vBz_M+dx)*G_BIN_DENSITY);
			
			// Note: this assumes that the faces intersect, at most, eight bins.
			int counter = 0;
			for (int K = bin_id_zl; K < bin_id_zL+1; K++)
			{
				for (int J = bin_id_yl; J < bin_id_yL+1; J++)
				{
					for (int I = bin_id_xl; I < bin_id_xL+1; I++)
					{
						if (C && counter < 64)
						{
							int global_id = I + G_BIN_DENSITY*J + G_BIN_DENSITY*G_BIN_DENSITY*K;
							bounding_box_limits[kap + counter*n_faces] = global_id;
							bounding_box_index_limits[kap + counter*n_faces] = kap;
							counter++;
						}
					}
				}
			}
		}
	}
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_MakeBinsGPU ]============================================================  */
/*                                                                                    */
/*  Performs a uniform spatial binning of geometry faces inside of the domain in      */
/*  parallel on the GPU. Faces outside of the domain are filtered out. The result     */
/*  is the allocation of memory for and filling of three sets of arrays: 1)           */
/*  c_binned_ids_v/b, a set of contiguous binned faces such that the first batch      */
/*  correspond to the faces of bin 0, the second batch corresponds to bin 1 and so    */
/*  on, 2) c_binned_ids_n_v/b, the sizes of the n_bins_v/b bins, and 3)               */
/*  c_binned_ids_N_v/b, the starting indices for the faces of each bin in             */
/*  c_binned_ids_v/b. The set of arrays with '_v' corresponds to a 2D binning which   */
/*  enables a raycast algorithm for solid-cell identification. The one with '_b'      */
/*  corresponds to the 3D binning, where the bins are extended in volume by an        */
/*  amount dx specified by the mesh resolution and which is used to restrict the      */
/*  search-space when cells are computing the lengths of cut-links across the         */
/*  geometry.                                                                         */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_MakeBinsGPU(int i_dev)
{
	// Some constants.
	ufloat_g_t Lx0g __attribute__((unused)) = Lx/(ufloat_g_t)G_BIN_DENSITY;   // Get bin lengths along axes.
	ufloat_g_t Ly0g __attribute__((unused)) = Ly/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Lz0g __attribute__((unused)) = Lz/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t eps __attribute__((unused)) = 1e-5;                            // An epsilon for the 3D binning.
	if (std::is_same<ufloat_g_t, float>::value) eps = FLT_EPSILON;
	if (std::is_same<ufloat_g_t, double>::value) eps = DBL_EPSILON;
	int use_zip = true;
	
	// Proceed only if there are actual faces loaded in the current object.
	if (v_geom_f_face_1_X.size() > 0)
	{
		// Declare and allocate std::vector<int> bin arrays, which will be updated during traversal.
		int n_limits_v = 1;
		int n_limits_b = 1;
		int n_bins_v = 1; for (int d = 0; d < N_DIM-1; d++) { n_bins_v *= G_BIN_DENSITY; n_limits_v *= 4; }
		int n_bins_b = 1; for (int d = 0; d < N_DIM; d++)   { n_bins_b *= G_BIN_DENSITY; n_limits_b *= 4; }
		int n_lim_size_v = n_limits_v*n_faces[i_dev];
		int n_lim_size_b = n_limits_b*n_faces[i_dev];
		
		// Some array declarations.
		int *c_bounding_box_limits;
		int *c_bounding_box_index_limits;
		int *c_tmp_b_i;    // Used to store unique bin Ids.
		int *c_tmp_b_ii;   // Used to gather starting-location indices.
		
		// Declare and allocate memory for the c_bounding_box_limits.
		tic_simple("");
		cudaDeviceSynchronize();
		gpuErrchk( cudaMalloc((void **)&c_bounding_box_limits, n_lim_size_b*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_bounding_box_index_limits, n_lim_size_b*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_tmp_b_i, n_bins_b*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_tmp_b_ii, n_bins_b*sizeof(int)) );
		//
		// Allocate memory for bins sizes and locations.
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_v[i_dev], n_bins_v*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_v[i_dev], n_bins_v*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_b[i_dev], n_bins_b*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_b[i_dev], n_bins_b*sizeof(int)) );
		//
		// Reset values.
		Cu_ResetToValue<<<(M_BLOCK+n_bins_b-1)/M_BLOCK, M_BLOCK>>>(n_bins_b, c_binned_face_ids_n_b[i_dev], 0);
		Cu_ResetToValue<<<(M_BLOCK+n_bins_b-1)/M_BLOCK, M_BLOCK>>>(n_bins_b, c_binned_face_ids_N_b[i_dev], 0);
		Cu_ResetToValue<<<(M_BLOCK+n_lim_size_b-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_b, c_bounding_box_limits, n_bins_b);
		Cu_ResetToValue<<<(M_BLOCK+n_lim_size_b-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_b, c_bounding_box_index_limits, -1);
		Cu_ResetToValue<<<(M_BLOCK+n_bins_b-1)/M_BLOCK, M_BLOCK>>>(n_bins_b, c_tmp_b_i, -1);
		Cu_ResetToValue<<<(M_BLOCK+n_bins_b-1)/M_BLOCK, M_BLOCK>>>(n_bins_b, c_tmp_b_ii, -1);
		cudaDeviceSynchronize();
		std::cout << "Memory allocation: "; toc_simple("",T_US,1);
		
		// Wrap raw pointers with thrust device_ptr. // TODO reword
		thrust::device_ptr<int> bbi_ptr = thrust::device_pointer_cast(c_bounding_box_index_limits);
		thrust::device_ptr<int> bb_ptr = thrust::device_pointer_cast(c_bounding_box_limits);
		thrust::device_ptr<int> bnv_ptr = thrust::device_pointer_cast(c_binned_face_ids_n_v[i_dev]);
		thrust::device_ptr<int> bNv_ptr = thrust::device_pointer_cast(c_binned_face_ids_N_v[i_dev]);
		thrust::device_ptr<int> bnb_ptr = thrust::device_pointer_cast(c_binned_face_ids_n_b[i_dev]);
		thrust::device_ptr<int> bNb_ptr = thrust::device_pointer_cast(c_binned_face_ids_N_b[i_dev]);
		thrust::device_ptr<int> c_tmp_b_i_ptr = thrust::device_pointer_cast(c_tmp_b_i);
		thrust::device_ptr<int> c_tmp_b_ii_ptr = thrust::device_pointer_cast(c_tmp_b_ii);
		cudaDeviceSynchronize();
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		// STEP 1: Traverse faces and identify the bins they should go in.
		//std::cout << "Starting GPU binning..." << std::endl;
		std::cout << "STARTING 3D NOW" << std::endl;
		tic_simple("");
		cudaDeviceSynchronize();
		Cu_ComputeBoundingBoxLimits3D<ufloat_g_t,AP->N_DIM><<<(M_BLOCK+n_faces[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			n_faces[i_dev], n_faces_a[i_dev],
			c_geom_f_face_X[i_dev], c_bounding_box_limits, c_bounding_box_index_limits,
			(ufloat_g_t)dx, (ufloat_g_t)Lx, (ufloat_g_t)Ly, (ufloat_g_t)Lz, G_BIN_DENSITY
		);
		cudaDeviceSynchronize();
		std::cout << "Computing bounding box limits"; toc_simple("",T_US,1);
		
		// STEP 2: If selected, make a zip iterator out of the bounding box limits and index arrays, and then remove all invalid bins.
		// This might speed up the sort-by-key that follows.
		if (use_zip)
		{
			tic_simple("");
			auto zipped = thrust::make_zip_iterator(thrust::make_tuple(bb_ptr, bbi_ptr));
			auto zipped_end = thrust::remove_if(thrust::device, zipped, zipped + n_lim_size_b, is_equal_to_zip(n_bins_b));
			n_lim_size_b = zipped_end - zipped;
			//n_lim_size_b = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_b, is_nonnegative_and_less_than(n_bins_b));
			cudaDeviceSynchronize();
			std::cout << "Compaction"; toc_simple("",T_US,1);
		}
		
		// STEP 3: Sort by key. // TODO reword
		tic_simple("");
		thrust::sort_by_key(thrust::device, bb_ptr, bb_ptr + n_lim_size_b, bbi_ptr);
		int n_binned_faces_b = n_lim_size_b;
		//int n_binned_faces_b = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_b, is_nonnegative_and_less_than(n_bins_b));
		cudaDeviceSynchronize();
		std::cout << "Sort by key (n_binned_faces=" << n_binned_faces_b << ")"; toc_simple("",T_US,1);
		
		// STEP 4: Reduce the keys to get the number of faces in each bin. Scatter them to c_binned_face_ids_n_v.
		tic_simple("");
		auto result = thrust::reduce_by_key(
			thrust::device, bb_ptr, bb_ptr + n_lim_size_b,
			thrust::make_constant_iterator(1),
			c_tmp_b_i_ptr,    // stores unique keys
			c_tmp_b_ii_ptr    // stores reduction
		);
		int n_unique_bins_b = result.first - c_tmp_b_i_ptr;
		//int n_unique_bins_b = thrust::count_if(thrust::device, c_tmp_b_i_ptr, c_tmp_b_i_ptr + n_bins_b, is_nonnegative_and_less_than(n_bins_b));
		cudaDeviceSynchronize();
		std::cout << "Reduction (nbins=" << n_unique_bins_b << ") by key"; toc_simple("",T_US,1);
		
		// STEP 5: Scatter the bin sizes.
		tic_simple("");
		thrust::scatter(
			thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_b,
			c_tmp_b_i_ptr,
			bnb_ptr
		);
		cudaDeviceSynchronize();
		std::cout << "Scatter (1)"; toc_simple("",T_US,1);
		
		// STEP 6: Get the difference in the bounding box limits to identify the starting location for the Ids of each individual bin.
		tic_simple("");
		thrust::adjacent_difference(thrust::device, bb_ptr, bb_ptr + n_lim_size_b, bb_ptr);
		cudaDeviceSynchronize();
		std::cout << "Adjacent difference"; toc_simple("",T_US,1);
		
		// STEP 7: Gather the indices of the starting locations.
		tic_simple("");
		auto counting_iter = thrust::counting_iterator<int>(1);
		thrust::transform(
			thrust::device, counting_iter, counting_iter + n_lim_size_b,
			bb_ptr, bb_ptr,
			replace_diff_with_indexM1()
		);
		thrust::copy_if(thrust::device, &bb_ptr[1], &bb_ptr[1] + (n_lim_size_b-1), &c_tmp_b_ii_ptr[1], is_positive());
		int fZ = 0;
		cudaMemcpy(c_tmp_b_ii, &fZ, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		std::cout << "Copy-if"; toc_simple("",T_US,1);
		
		// STEP 8: Now scatter the bin sizes and starting-location indices.
		tic_simple("");
		thrust::scatter(
			thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_b,
			c_tmp_b_i_ptr,
			bNb_ptr
		);
		cudaDeviceSynchronize();
		std::cout << "Scatter (2)"; toc_simple("",T_US,1);
		
		// Copy the indices of the binned faces.
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_b[i_dev], n_binned_faces_b*sizeof(int)) );
		cudaMemcpy(c_binned_face_ids_b[i_dev], c_bounding_box_index_limits, n_binned_faces_b*sizeof(int), cudaMemcpyDeviceToDevice);
		//
		// Copy the GPU data to the CPU for drawing.
		binned_face_ids_n_b[i_dev] = new int[n_bins_b];
		binned_face_ids_N_b[i_dev] = new int[n_bins_b];
		binned_face_ids_b[i_dev] = new int[n_binned_faces_b];
		cudaMemcpy(binned_face_ids_n_b[i_dev], c_binned_face_ids_n_b[i_dev], n_bins_b*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(binned_face_ids_N_b[i_dev], c_binned_face_ids_N_b[i_dev], n_bins_b*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(binned_face_ids_b[i_dev], c_binned_face_ids_b[i_dev], n_binned_faces_b*sizeof(int), cudaMemcpyDeviceToHost);
		
		// DEBUG
// 		for (int i = 0; i < n_bins_b; i++)
// 		{
// 			int nbins_i = binned_face_ids_n_b[i_dev][i];
// 			int Nbins_i = binned_face_ids_N_b[i_dev][i];
// 			if (nbins_i > 0)
// 			{
// 				std::cout << "Bin " << i << ": (nbins=" << nbins_i << ",Nbins=" << Nbins_i << ")" << std::endl;
// 				for (int j = 0; j < nbins_i; j++)
// 					std::cout << binned_face_ids_b[i_dev][Nbins_i+j] << " ";
// 				std::cout << std::endl;
// 			}
// 		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		// Reset values for the 2D binning part. I can reuse the temporary allocations since they are always larger than needed.
		std::cout << "STARTING 2D NOW" << std::endl;
		tic_simple("");
		cudaDeviceSynchronize();
		Cu_ResetToValue<<<(M_BLOCK+n_bins_v-1)/M_BLOCK, M_BLOCK>>>(n_bins_v, c_binned_face_ids_n_v[i_dev], 0);
		Cu_ResetToValue<<<(M_BLOCK+n_bins_v-1)/M_BLOCK, M_BLOCK>>>(n_bins_v, c_binned_face_ids_N_v[i_dev], 0);
		Cu_ResetToValue<<<(M_BLOCK+n_lim_size_v-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_v, c_bounding_box_limits, n_bins_v);
		Cu_ResetToValue<<<(M_BLOCK+n_lim_size_v-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_v, c_bounding_box_index_limits, -1);
		Cu_ResetToValue<<<(M_BLOCK+n_bins_v-1)/M_BLOCK, M_BLOCK>>>(n_bins_v, c_tmp_b_i, -1);
		Cu_ResetToValue<<<(M_BLOCK+n_bins_v-1)/M_BLOCK, M_BLOCK>>>(n_bins_v, c_tmp_b_ii, -1);
		cudaDeviceSynchronize();
		std::cout << "Memory allocation"; toc_simple("",T_US,1);
		
		// STEP 1: Traverse faces and identify the bins they should go in.
		tic_simple("");
		cudaDeviceSynchronize();
		Cu_ComputeBoundingBoxLimits2D<ufloat_g_t,AP->N_DIM><<<(M_BLOCK+n_faces[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			n_faces[i_dev], n_faces_a[i_dev],
			c_geom_f_face_X[i_dev], c_bounding_box_limits, c_bounding_box_index_limits,
			(ufloat_g_t)dx, (ufloat_g_t)Lx, (ufloat_g_t)Ly, (ufloat_g_t)Lz, G_BIN_DENSITY
		);
		cudaDeviceSynchronize();
		std::cout << "Computing bounding box limits"; toc_simple("",T_US,1);
		
		// STEP 2: If selected, make a zip iterator out of the bounding box limits and index arrays, and then remove all invalid bins.
		// This might speed up the sort-by-key that follows.
		if (use_zip)
		{
			tic_simple("");
			auto zipped = thrust::make_zip_iterator(thrust::make_tuple(bb_ptr, bbi_ptr));
			auto zipped_end = thrust::remove_if(thrust::device, zipped, zipped + n_lim_size_v, is_equal_to_zip(n_bins_v));
			n_lim_size_v = zipped_end - zipped;
			//n_lim_size_v = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_v, is_nonnegative_and_less_than(n_bins_v));
			cudaDeviceSynchronize();
			std::cout << "Compaction"; toc_simple("",T_US,1);
		}
		
		// STEP 3: Sort by key. // TODO reword
		tic_simple("");
		thrust::sort_by_key(thrust::device, bb_ptr, bb_ptr + n_lim_size_v, bbi_ptr);
		int n_binned_faces_v = n_lim_size_v;
		//int n_binned_faces_v = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_v, is_nonnegative_and_less_than(n_bins_v));
		cudaDeviceSynchronize();
		std::cout << "Sort by key (n_binned_faces=" << n_binned_faces_v << ")"; toc_simple("",T_US,1);
		
		// STEP 4: Reduce the keys to get the number of faces in each bin. Scatter them to c_binned_face_ids_n_v.
		tic_simple("");
		result = thrust::reduce_by_key(
			thrust::device, bb_ptr, bb_ptr + n_lim_size_v,
			thrust::make_constant_iterator(1),
			c_tmp_b_i_ptr,    // stores unique keys
			c_tmp_b_ii_ptr    // stores reduction
		);
		int n_unique_bins_v = result.first - c_tmp_b_i_ptr;
		//int n_unique_bins_v = thrust::count_if(thrust::device, c_tmp_b_i_ptr, c_tmp_b_i_ptr + n_bins_v, is_nonnegative_and_less_than(n_bins_v));
		cudaDeviceSynchronize();
		std::cout << "Reduction (nbins=" << n_unique_bins_v << ") by key "; toc_simple("",T_US,1);
		
		// STEP 5: Scatter the bin sizes.
		tic_simple("");
		thrust::scatter(
			thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_v,
			c_tmp_b_i_ptr,
			bnv_ptr
		);
		cudaDeviceSynchronize();
		std::cout << "Scatter (1)"; toc_simple("",T_US,1);
		
		// STEP 6: Get the difference in the bounding box limits to identify the starting location for the Ids of each individual bin.
		tic_simple("");
		thrust::adjacent_difference(thrust::device, bb_ptr, bb_ptr + n_lim_size_v, bb_ptr);
		cudaDeviceSynchronize();
		std::cout << "Adjacent difference"; toc_simple("",T_US,1);
		
		// STEP 7: Gather the indices of the starting locations.
		tic_simple("");
		counting_iter = thrust::counting_iterator<int>(1);
		thrust::transform(
			thrust::device, counting_iter, counting_iter + n_lim_size_v,
			bb_ptr, bb_ptr,
			replace_diff_with_indexM1()
		);
		thrust::copy_if(thrust::device, &bb_ptr[1], &bb_ptr[1] + (n_lim_size_v-1), &c_tmp_b_ii_ptr[1], is_positive());
		fZ = 0;
		cudaMemcpy(c_tmp_b_ii, &fZ, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		std::cout << "Copy-if"; toc_simple("",T_US,1);
		
		// STEP 8: Now scatter the bin sizes and starting-location indices.
		tic_simple("");
		thrust::scatter(
			thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_v,
			c_tmp_b_i_ptr,
			bNv_ptr
		);
		cudaDeviceSynchronize();
		std::cout << "Scatter (2)"; toc_simple("",T_US,1);
		
		// Copy the indices of the binned faces.
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_v[i_dev], n_binned_faces_v*sizeof(int)) );
		cudaMemcpy(c_binned_face_ids_v[i_dev], c_bounding_box_index_limits, n_binned_faces_v*sizeof(int), cudaMemcpyDeviceToDevice);
		//
		// Copy the GPU data to the CPU for drawing.
		binned_face_ids_n_v[i_dev] = new int[n_bins_v];
		binned_face_ids_N_v[i_dev] = new int[n_bins_v];
		binned_face_ids_v[i_dev] = new int[n_binned_faces_v];
		cudaMemcpy(binned_face_ids_n_v[i_dev], c_binned_face_ids_n_v[i_dev], n_bins_v*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(binned_face_ids_N_v[i_dev], c_binned_face_ids_N_v[i_dev], n_bins_v*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(binned_face_ids_v[i_dev], c_binned_face_ids_v[i_dev], n_binned_faces_v*sizeof(int), cudaMemcpyDeviceToHost);
		
		// DEBUG
// 		for (int i = 0; i < n_bins_v; i++)
// 		{
// 			int nbins_i = binned_face_ids_n_v[i_dev][i];
// 			int Nbins_i = binned_face_ids_N_v[i_dev][i];
// 			if (nbins_i > 0)
// 			{
// 				std::cout << "Bin " << i << ": (nbins=" << nbins_i << ",Nbins=" << Nbins_i << ")" << std::endl;
// 				for (int j = 0; j < nbins_i; j++)
// 					std::cout << binned_face_ids_v[i_dev][Nbins_i+j] << " ";
// 				std::cout << std::endl;
// 			}
// 		}
		
		
		
		
		
		// Free temporary arrays.
		cudaFree(c_bounding_box_limits);
		cudaFree(c_bounding_box_index_limits);
		cudaFree(c_tmp_b_i);
		cudaFree(c_tmp_b_ii);
	}
	
	return 0;
}
