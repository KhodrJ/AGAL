/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"


template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_FillBinned_V1
(
	const int n_ids_idev_L,
	const int *__restrict__ id_set_idev_L,
	const int n_maxcblocks,
	const ufloat_t dx_L,
	const bool hit_max,
	int *__restrict__ cells_ID_mask,
	int *__restrict__ cblock_ID_mask,
	const ufloat_t *__restrict__ cblock_f_X,
	const int *__restrict__ cblock_ID_ref,
	const int *__restrict__ cblock_ID_nbr,
	const int n_faces,
	const int n_faces_a,
	const ufloat_g_t *__restrict__ geom_f_face_X,
	const ufloat_g_t *__restrict__ geom_f_face_Xt,
	const int *__restrict__ binned_face_ids_n,
	const int *__restrict__ binned_face_ids_N,
	const int *__restrict__ binned_face_ids,
	const int G_BIN_DENSITY
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int M_TBLOCK = AP->M_TBLOCK;
	constexpr int M_CBLOCK = AP->M_CBLOCK;
	constexpr int M_LBLOCK = AP->M_LBLOCK;
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_D[M_TBLOCK];
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
	int I_kap = threadIdx.x % 4;
	int J_kap = (threadIdx.x / 4) % 4;
	int K_kap = (threadIdx.x / 4) / 4;
	int i_kap_b;
	int global_bin_id;
	int bin_id_y = 0;
	int bin_id_z __attribute__((unused)) = 0;
	ufloat_g_t vx1 = (ufloat_g_t)0.0;
	ufloat_g_t vy1 = (ufloat_g_t)0.0;
	ufloat_g_t vz1 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vx2 = (ufloat_g_t)0.0;
	ufloat_g_t vy2 = (ufloat_g_t)0.0;
	ufloat_g_t vz2 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vx3 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vy3 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vz3 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t nx = (ufloat_g_t)0.0;
	ufloat_g_t ny = (ufloat_g_t)0.0;
	ufloat_g_t nz = (ufloat_g_t)0.0;
	ufloat_g_t ex1 = (ufloat_g_t)0.0;
	ufloat_g_t ey1 = (ufloat_g_t)0.0;
	ufloat_g_t ez1 = (ufloat_g_t)0.0;
	ufloat_g_t ex2 = (ufloat_g_t)0.0;
	ufloat_g_t ey2 = (ufloat_g_t)0.0;
	ufloat_g_t ez2 = (ufloat_g_t)0.0;
	ufloat_g_t vxp = (ufloat_g_t)0.0;
	ufloat_g_t vyp = (ufloat_g_t)0.0;
	ufloat_g_t vzp = (ufloat_g_t)0.0;
	ufloat_g_t tmp = (ufloat_g_t)0.0;
	ufloat_g_t tmp2 = (ufloat_g_t)0.0;
	int intersect_counter = 0;
	bool C;
	//bool eligible __attribute__((unused)) = true;
	//bool D = false;
	
	s_ID_cblock[threadIdx.x] = -1;
	if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
	{
		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
	}
	__syncthreads();
	
	// Loop over block Ids.
	for (int k = 0; k < M_LBLOCK; k += 1)
	{
		i_kap_b = s_ID_cblock[k];

		// Latter condition is added only if n>0.
		if (i_kap_b > -1)
		{
			vxp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I_kap*dx_L + 0.5*dx_L;
			vyp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J_kap*dx_L + 0.5*dx_L;
			if (N_DIM==3)
				vzp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K_kap*dx_L + 0.5*dx_L;
			
			// For each face, check if the current cell is within the appropriate bounds. If at least
			// one condition is satisfied, exit the loop and make a note.
			bin_id_y = (int)(vyp*G_BIN_DENSITY);
			if (N_DIM==3)
				bin_id_z = (int)(vzp*G_BIN_DENSITY);
			
			// Now find the total number of intersections a ray makes in the direction with the smallest number of bins.
			global_bin_id = bin_id_y + G_BIN_DENSITY*bin_id_z;
			int n_f = binned_face_ids_n[global_bin_id];
			int N_f = 0;
			if (n_f > 0)
				N_f = binned_face_ids_N[global_bin_id];
			for (int p = 0; p < n_f; p++)
			{
				int f_p = binned_face_ids[N_f+p];
				vx1 = geom_f_face_X[f_p + 0*n_faces_a];
				vy1 = geom_f_face_X[f_p + 1*n_faces_a];
				vx2 = geom_f_face_X[f_p + 3*n_faces_a];
				vy2 = geom_f_face_X[f_p + 4*n_faces_a];
				nx = vy2-vy1;
				ny = vx1-vx2;
				
				if (N_DIM==2)
				{
					// Find the distance along a ray with direction [1,0].
					tmp = (vx1-vxp) + (vy1-vyp)*(ny/nx);
					if (tmp > 0)
					{
						tmp2 = vxp + tmp; // Stores the x-component of the intersection point.
						C = CheckPointInLine(tmp2, vyp, vx1, vy1, vx2, vy2);
						if (C)
							intersect_counter++;
					}
				}
				else
				{
					vz1 = geom_f_face_X[f_p + 2*n_faces_a];
					vz2 = geom_f_face_X[f_p + 5*n_faces_a];
					vx3 = geom_f_face_X[f_p + 6*n_faces_a];
					vy3 = geom_f_face_X[f_p + 7*n_faces_a];
					vz3 = geom_f_face_X[f_p + 8*n_faces_a];
					
					ex1 = vx2-vx1;
					ey1 = vy2-vy1;
					ez1 = vz2-vz1;
					ex2 = vx3-vx1;
					ey2 = vy3-vy1;
					ez2 = vz3-vz1;
					Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
					tmp = Tsqrt(nx*nx + ny*ny + nz*nz);
					nx /= tmp;
					ny /= tmp;
					nz /= tmp;
					
					// Find the distance along a ray with direction [1,0,0].
					tmp = (vx1-vxp) + (vy1-vyp)*(ny/nx) + (vz1-vzp)*(nz/nx);
					if (tmp > 0)
					{
						tmp2 = vxp + tmp; // Stores the x-component of the intersection point.
						C = CheckPointInTriangle(tmp2, vyp, vzp, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2);
						if (C)
							intersect_counter++;
					}
				}
			}
			
			// Now, if there are an even number of intersections, the current cell is in the solid.
			// Otherwise, it is a fluid cell.
			s_D[threadIdx.x] = 0;
			if (intersect_counter%2 == 1)
			{
				cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = -1;
				s_D[threadIdx.x] = 1;
			}
			__syncthreads();
			
			// Block reduction for sum.
			for (int s=blockDim.x/2; s>0; s>>=1)
			{
				if (threadIdx.x < s)
				{
					s_D[threadIdx.x] = s_D[threadIdx.x] + s_D[threadIdx.x + s];
				}
				__syncthreads();
			}
			
			// If at least one cell is solid, update the block mask.
			if (threadIdx.x == 0 && s_D[threadIdx.x]>0)
				cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
			
			// Reset.
			intersect_counter = 0;
		}
	}
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_FillBinned_V2
(
	const int n_ids_idev_L,
	const int *__restrict__ id_set_idev_L,
	const int n_maxcblocks,
	const ufloat_t dx_L,
	const bool hit_max,
	int *__restrict__ cells_ID_mask,
	int *__restrict__ cblock_ID_mask,
	const ufloat_t *__restrict__ cblock_f_X,
	const int *__restrict__ cblock_ID_ref,
	const int *__restrict__ cblock_ID_nbr,
	const int n_faces,
	const int n_faces_a,
	const ufloat_g_t *__restrict__ geom_f_face_X,
	const ufloat_g_t *__restrict__ geom_f_face_Xt,
	const int *__restrict__ binned_face_ids_n,
	const int *__restrict__ binned_face_ids_N,
	const int *__restrict__ binned_face_ids,
	const int G_BIN_DENSITY
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int M_TBLOCK = AP->M_TBLOCK;
	constexpr int M_CBLOCK = AP->M_CBLOCK;
	constexpr int M_LBLOCK = AP->M_LBLOCK;
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_D[M_TBLOCK];
	__shared__ int s_fI[M_TBLOCK];
	__shared__ ufloat_g_t s_fD[16];
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
	int i_kap_b;
	int global_bin_id;
	int plim = M_TBLOCK;
	ufloat_g_t vxp = (ufloat_g_t)0.0;
	ufloat_g_t vyp = (ufloat_g_t)0.0;
	ufloat_g_t vzp = (ufloat_g_t)0.0;
	ufloat_g_t vx = (ufloat_g_t)0.0;
	ufloat_g_t vy = (ufloat_g_t)0.0;
	ufloat_g_t vz __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vx2 = (ufloat_g_t)0.0;
	ufloat_g_t vy2 = (ufloat_g_t)0.0;
	ufloat_g_t vz2 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t nx = (ufloat_g_t)0.0;
	ufloat_g_t ny = (ufloat_g_t)0.0;
	ufloat_g_t nz __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t sx __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t sy __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t sz __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t tmp = (ufloat_g_t)0.0;
	ufloat_g_t tmp2 = (ufloat_g_t)0.0;
	int intersect_counter = 0;
	bool C = true;
	
	s_ID_cblock[threadIdx.x] = -1;
	s_fI[threadIdx.x] = -1;
	if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
	{
		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
	}
	__syncthreads();
	
	// Loop over block Ids.
	for (int k = 0; k < M_LBLOCK; k += 1)
	{
		i_kap_b = s_ID_cblock[k];

		// Latter condition is added only if n>0.
		if (i_kap_b > -1)
		{
			// Compute cell-center coordinates.
			vxp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + (threadIdx.x % 4)*dx_L + 0.5*dx_L;
			vyp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + ((threadIdx.x / 4) % 4)*dx_L + 0.5*dx_L;
			if (N_DIM==3)
				vzp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + ((threadIdx.x / 4) / 4)*dx_L + 0.5*dx_L;
			
			// For each face, check if the current cell is within the appropriate bounds. If at least
			// one condition is satisfied, exit the loop and make a note.
			global_bin_id = ((int)(vyp*G_BIN_DENSITY)) + G_BIN_DENSITY*((int)(vzp*G_BIN_DENSITY));
			
			// Now find the total number of intersections a ray makes in the direction with the smallest number of bins.
			//s_fI[threadIdx.x] = binned_face_ids[global_bin_id+threadIdx.x];
			int n_f = binned_face_ids_n[global_bin_id];
			int N_f = 0;
			if (n_f > 0)
			{
				N_f = binned_face_ids_N[global_bin_id];
				for (int k = 0; k < n_f/M_TBLOCK+1; k++)
				{
					// Read the next M_TBLOCK faces.
					plim = M_TBLOCK;
					if ((k+1)*M_TBLOCK >= n_f)
						plim = M_TBLOCK - ((k+1)*M_TBLOCK - n_f);
					if (threadIdx.x < plim)
						s_fI[threadIdx.x] = binned_face_ids[N_f + k*M_TBLOCK + threadIdx.x];
					__syncthreads();
					
					for (int p = 0; p < plim; p++)
					//for (int p = 0; p < n_f; p++)
					{
						//int f_p = binned_face_ids[N_f+p];
						int f_p = s_fI[p];
						
						// Load face data.
						if (threadIdx.x < 16)
							s_fD[threadIdx.x] = geom_f_face_Xt[threadIdx.x + f_p*16];
						__syncthreads();
						
						if (N_DIM==2)
						{
							// Load normal.
							nx = s_fD[9];
							ny = s_fD[10];
							
							// Load vertices 1 and 2.
							vx = s_fD[0];
							vy = s_fD[1];
							vx2 = s_fD[3];
							vy2 = s_fD[4];
							
							// Find the distance along a ray with direction [1,0].
							tmp = (vx-vxp) + (vy-vyp)*(ny/nx);
							if (tmp > 0)
							{
								tmp2 = vxp + tmp; // Stores the x-component of the intersection point.
								
								// First check if point is inside the line.
								C = -( ((vx-tmp2)*(vx2-vx))+((vy-vyp)*(vy2-vy)) ) > 0;
								
								// Second check if point is inside the line.
								C = C && ( ((vx2-tmp2)*(vx2-vx))+((vy2-vyp)*(vy2-vy)) ) > 0;
								
								if (C)
									intersect_counter++;
							}
						}
						else
						{
							// Load normal.
							nx = s_fD[9];
							ny = s_fD[10];
							nz = s_fD[11];
							
							// Load vertex 1.
							vx = s_fD[0];
							vy = s_fD[1];
							vz = s_fD[2];
							
							// Find the distance along a ray with direction [1,0,0].
							tmp = (vx-vxp) + (vy-vyp)*(ny/nx) + (vz-vzp)*(nz/nx);
							if (tmp > 0)
							{
								tmp2 = vxp + tmp; // Stores the x-component of the intersection point.
								
								// First check that point is inside the triangle.
								vx2 = s_fD[3] - vx; // vx2 stores x-comp. of edge 1: vx2-vx1.
								vy2 = s_fD[4] - vy; // I don't need real vx2 individually.
								vz2 = s_fD[5] - vz;
								sx = vy2*nz - vz2*ny;
								sy = vz2*nx - vx2*nz;
								sz = vx2*ny - vy2*nx;
								C = (vx-tmp2)*sx + (vy-vyp)*sy + (vz-vzp)*sz > 0;
								
								// Second check that point is inside the triangle.
								vx += vx2; // Recover vertex 2.
								vy += vy2;
								vz += vz2;
								vx2 = s_fD[6] - vx; // vx2 stores x-comp. of edge 2: vx3-vx2.
								vy2 = s_fD[7] - vy; // I don't need real vx2 individually.
								vz2 = s_fD[8] - vz;
								sx = vy2*nz - vz2*ny;
								sy = vz2*nx - vx2*nz;
								sz = vx2*ny - vy2*nx;
								C = C && (vx-tmp2)*sx + (vy-vyp)*sy + (vz-vzp)*sz > 0;
								
								// Second check that point is inside the triangle.
								vx += vx2; // Recover vertex 3.
								vy += vy2;
								vz += vz2;
								vx2 = s_fD[0] - vx; // vx2 stores x-comp. of edge 3: vx1-vx3.
								vy2 = s_fD[1] - vy; // I don't need real vx2 individually.
								vz2 = s_fD[2] - vz;
								sx = vy2*nz - vz2*ny;
								sy = vz2*nx - vx2*nz;
								sz = vx2*ny - vy2*nx;
								C = C && (vx-tmp2)*sx + (vy-vyp)*sy + (vz-vzp)*sz > 0;
								
								if (C)
									intersect_counter++;
							}
						}
						
						__syncthreads();
					}
				}
			}
			
			// Now, if there are an even number of intersections, the current cell is in the solid.
			// Otherwise, it is a fluid cell.
			s_D[threadIdx.x] = 0;
			if (intersect_counter%2 == 1)
			{
				cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = -1;
				s_D[threadIdx.x] = 1;
			}
			__syncthreads();
			
			// Block reduction for sum.
			for (int s=blockDim.x/2; s>0; s>>=1)
			{
				if (threadIdx.x < s)
				{
					s_D[threadIdx.x] = s_D[threadIdx.x] + s_D[threadIdx.x + s];
				}
				__syncthreads();
			}
			
			// If at least one cell is solid, update the block mask.
			if (threadIdx.x == 0 && s_D[threadIdx.x]>0)
				cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
			
			// Reset.
			intersect_counter = 0;
		}
	}
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_MarkBlocks_S1 ]=========================================================  */
/*                                                                                    */
/*  This kernel traverses the cell-blocks according to the 'secondary' mode of        */
/*  access, where threads are assigned to individual cell-blocks and access data      */
/*  from arrays arranged according to the Structure of Arrays format, in order to     */
/*  determine which cell-blocks are adjacent to entirely solid-blocks. These          */
/*  indicate a sort of boundary around the geometry at the block level (the Ids       */
/*  updated in cells_ID_mask indicates this boundary at the cell level according to   */
/*  Cu_FillBins).                                                                     */
/*                                                                                    */
/**************************************************************************************/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_Alt_S1
(
	int id_max_curr, int n_maxcblocks, bool hit_max, int L,
	int *cblock_ID_ref, int *cblock_ID_mask, int *cblock_ID_nbr, int *cblock_level
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int M_BLOCK = AP->M_BLOCK;
	constexpr int N_Q_max = AP->N_Q_max;
	__shared__ int s_ID_nbr[M_BLOCK*9];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int mask_kap;
	bool mark_solid_boundary = false;
	bool eligible = true;
	
	for (int p = 0; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	
	for (int k = 0; k < (N_DIM==2?1:3); k++)
	{
		// Load mask.
		mask_kap = cblock_ID_mask[kap];
		
		// First, read neighbor Ids and place in shared memory. Arrange for contiguity.
		if (kap < id_max_curr && cblock_level[kap] == L && mask_kap == V_BLOCKMASK_SOLID)
		{
			for (int p = 0; p < 9; p++)
				s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
		}
		__syncthreads();
		
		// Replace neighbor Ids with their respective marks.
		for (int p = 0; p < 9; p++)
		{
			int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
			if (i_p > -1 && cblock_ID_mask[i_p] == V_BLOCKMASK_REGULAR)
				s_ID_nbr[threadIdx.x + p*M_BLOCK] = 1;
			else
				s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
		}
		__syncthreads();
		
		// Run again and check if any of the masks indicated adjacency to regular blocks.
		if (kap < id_max_curr)
		{
			for (int p = 0; p < 9; p++)
			{
				if (s_ID_nbr[p + threadIdx.x*9] == 1)
					mark_solid_boundary = true;
			}
		}
		
		if (N_DIM==3)
		{
			__syncthreads();
			
			for (int p = 0; p < 9; p++)
				s_ID_nbr[p + threadIdx.x*9] = -1;
			__syncthreads();
		}
	}
	
	// If near at least one regular block, this block is on the boundary of the solid.
	if (kap < id_max_curr && mark_solid_boundary && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
	{
		cblock_ID_mask[kap] = V_BLOCKMASK_SOLIDB;
		// Only refine if not on the finest grid level.
		if (!hit_max)
		{
			for (int p = 0; p < N_Q_max; p++)
			{
				if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
					eligible = false;
			}
			
			if (eligible)
				cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
		}
		
		//if (mask_kap == V_BLOCKMASK_SOLIDB)
		//	cblock_ID_mask[kap] = V_BLOCKMASK_INDETERMINATE;
	}
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_MarkBlocks_GetMasks ]===================================================  */
/*                                                                                    */
/*  This kernel identifies the boundary cells (contained within the interior of the   */
/*  domain) that are adjacent to solid cells. The cell masks of each block are        */
/*  placed in shared memory along with a one-cell surrounding halo. Since cells       */
/*  will consider all possible directions, placement in shared memory prevents        */
/*  searching through the same data over and over again from global memory.           */
/*                                                                                    */
/**************************************************************************************/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_GetMasks_V2
(
	const int n_ids_idev_L,
	const int *__restrict__ id_set_idev_L,
	const int n_maxcblocks,
	int *__restrict__ cells_ID_mask,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_nbr,
	const int *__restrict__ cblock_ID_nbr_child,
	int *__restrict__ cblock_ID_onb,
	int *tmp_1
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int N_Q_max = AP->N_Q_max;
	constexpr int M_TBLOCK = AP->M_TBLOCK;
	constexpr int M_CBLOCK = AP->M_CBLOCK;
	constexpr int M_LBLOCK = AP->M_LBLOCK;
	constexpr int M_HBLOCK = AP->M_HBLOCK;
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_D[M_TBLOCK];
	__shared__ int s_ID_nbr[27];
	__shared__ int s_ID_mask[M_HBLOCK];
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
	int i_kap_b = -1;
	int I = threadIdx.x % 4;
	int J = (threadIdx.x / 4) % 4;
	int K = 0;
	if (N_DIM==3)
		K = (threadIdx.x / 4) / 4;
	bool near_a_solid_cell = false;
	
	s_ID_cblock[threadIdx.x] = -1;
	if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
	{
		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
	}
	for (int k = 0; k < M_HBLOCK/M_TBLOCK+1; k++)
	{
		if (k*M_TBLOCK + threadIdx.x < M_HBLOCK)
			s_ID_mask[k*M_TBLOCK + threadIdx.x] = N_SKIPID;
	}
	__syncthreads();
	
	// Loop over block Ids.
	for (int k = 0; k < M_LBLOCK; k += 1)
	{
		i_kap_b = s_ID_cblock[k];

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && cblock_ID_nbr_child[i_kap_b] < 0 && (cblock_ID_mask[i_kap_b]<0 && cblock_ID_mask[i_kap_b] != V_BLOCKMASK_SOLID))
		{
			// Load neighbor-block indices into shared memory.
			if (threadIdx.x==0)
			{
				//#pragma unroll
				for (int p = 0; p < N_Q_max; p++)
				{
					s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
					//if (i_kap_b == 25099)
					//	printf("C%i=[%i %i %i]\n", p, V_CONN_ID[p + 0*27], V_CONN_ID[p + 1*27], V_CONN_ID[p + 2*27]);
				}
			}
			__syncthreads();
			
			// Retrieve cell masks from the current block and from one cell-layer around it from neighboring blocks.
			for (int p = 0; p < N_Q_max; p++)
			{
				// nbr_kap_b is the index of the neighboring block w.r.t the current cell.
				// nbr_kap_c is the index of the cell in that neighboring block.
				// nbr_kap_h is the index of the halo to store that value.
				
				// First, increment indices along pth direction. Store the resulting halo index.
				int Ip = I + V_CONN_ID[p + 0*27];
				int Jp = J + V_CONN_ID[p + 1*27];
				int Kp = 0;
				if (N_DIM==3)
					Kp = K + V_CONN_ID[p + 2*27];
				int nbr_kap_h = (Ip+1) + 6*(Jp+1);
				if (N_DIM==3)
					nbr_kap_h += 36*(Kp+1);
				
				// Then, identify the appropriate neighbor block to store the retrieved cell masks.
				int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				if (N_DIM==3)
					Kp = (4 + (Kp % 4)) % 4;
				int nbr_kap_c = Ip + 4*Jp + 16*Kp;
				
				// Write cell mask to the halo.
				bool changed = (Ip != I+V_CONN_ID[p + 0*27] || V_CONN_ID[p + 0*27]==0) && (Jp != J+V_CONN_ID[p + 1*27] || V_CONN_ID[p + 1*27]==0) && (Kp != K+V_CONN_ID[p + 2*27] || V_CONN_ID[p + 2*27]==0);
				if (changed && nbr_kap_b > -1)
					s_ID_mask[nbr_kap_h] = cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c];
			}
			int curr_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
			s_ID_mask[(I+1)+6*(J+1)+36*(K+1)] = curr_mask;
			__syncthreads();
			
			// Now go through the shared memory array and check if the current cells are adjacent to any solid cells.
			for (int p = 0; p < N_Q_max; p++)
			{
				// First, increment indices along pth direction. Store the resulting halo index.
				int Ip = I + V_CONN_ID[p + 0*27];
				int Jp = J + V_CONN_ID[p + 1*27];
				int Kp = 0;
				if (N_DIM==3)
					Kp = K + V_CONN_ID[p + 2*27];
				int nbr_kap_h = (Ip+1) + 6*(Jp+1);
				if (N_DIM==3)
					nbr_kap_h += 36*(Kp+1);
				
				// Now, check the neighboring cell mask for all cells using values stored in shared memory.
				if (s_ID_mask[nbr_kap_h] == V_CELLMASK_SOLID)
					near_a_solid_cell = true;
			}
			
			
			
			// [DEPRECATED]
			// Each cell checks the mask of its neighboring cell, if it exists.
			// If at least one is a solid cell, then mark this as 'on the boundary'.
			/*
			int curr_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
			for (int p = 0; p < N_Q_max; p++)
			{
				int Ip = I + V_CONN_ID[p + 0*27];
				int Jp = J + V_CONN_ID[p + 1*27];
				int Kp = 0;
				if (N_DIM==3)
					Kp = K + V_CONN_ID[p + 2*27];
				
				int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				if (N_DIM==3)
					Kp = (4 + (Kp % 4)) % 4;
				int nbr_kap_c = Ip + 4*Jp + 16*Kp;
				
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == V_CELLMASK_SOLID)
					near_a_solid_cell = true;
			}
			*/
			
			
			
			s_D[threadIdx.x] = 0;
			if (near_a_solid_cell && curr_mask != V_CELLMASK_SOLID)
			{
				cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = V_CELLMASK_BOUNDARY;
				s_D[threadIdx.x] = 1;
			}
			__syncthreads();
			
			// Block reduction for sum.
			for (int s=blockDim.x/2; s>0; s>>=1)
			{
				if (threadIdx.x < s)
				{
					s_D[threadIdx.x] = s_D[threadIdx.x] + s_D[threadIdx.x + s];
				}
				__syncthreads();
			}
			
			if (threadIdx.x == 0 && s_D[threadIdx.x] > 0)
			{
				tmp_1[i_kap_b] = s_D[threadIdx.x];
				cblock_ID_onb[i_kap_b] = 1;
			}
		}
	}
}

/*
template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_GetMasks
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks,
	int *cells_ID_mask, int *cblock_ID_mask, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_onb,
	int *tmp_1
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int N_Q_max __attribute__((unused)) = AP->N_Q_max;
	constexpr int M_TBLOCK = AP->M_TBLOCK;
	constexpr int M_CBLOCK = AP->M_CBLOCK;
	constexpr int M_LBLOCK = AP->M_LBLOCK;
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_D[M_TBLOCK];
	__shared__ int s_ID_nbr[27];
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
	int i_kap_b = -1;
	int nbr_kap_b = -1;
	int nbr_kap_c = -1;
	int I = threadIdx.x % 4;
	int Ip = I;
	int J = (threadIdx.x / 4) % 4;
	int Jp = J;
	int K = (threadIdx.x / 4) / 4;
	int Kp = K;
	bool near_a_solid_cell = false;
	
	s_ID_cblock[threadIdx.x] = -1;
	if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
	{
		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
	}
	__syncthreads();
	
	// Loop over block Ids.
	for (int k = 0; k < M_LBLOCK; k += 1)
	{
		i_kap_b = s_ID_cblock[k];

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && cblock_ID_nbr_child[i_kap_b] < 0 && (cblock_ID_mask[i_kap_b]<0 && cblock_ID_mask[i_kap_b] != V_BLOCKMASK_SOLID))
		{
			// Each cell checks the mask of its neighboring cell, if it exists.
			// If at least one is a solid cell, then mark this as 'on the boundary'.
			if (N_DIM == 2)
			{
				if (threadIdx.x==0)
				{
					s_ID_nbr[1] = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
					s_ID_nbr[2] = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
					s_ID_nbr[3] = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
					s_ID_nbr[4] = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
					s_ID_nbr[5] = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
					s_ID_nbr[6] = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
					s_ID_nbr[7] = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
					s_ID_nbr[8] = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
				}
				__syncthreads();
				
				//
				// p == 1
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 2
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip>=0)and(Ip<4)and(Jp==4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-] Correct the neighbor cell indices.
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 3
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 3.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 4
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
					nbr_kap_b = s_ID_nbr[4];
				// [-] Correct the neighbor cell indices.
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 5
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 2.
				if ( (Ip>=0)and(Ip<4)and(Jp==4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 5.
				if ( (Ip==4)and(Jp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 6
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip>=0)and(Ip<4)and(Jp==4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 3.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 6.
				if ( (Ip==-1)and(Jp==4) )
					nbr_kap_b = s_ID_nbr[6];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 7
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 3.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 7.
				if ( (Ip==-1)and(Jp==-1) )
					nbr_kap_b = s_ID_nbr[7];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 8
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 8.
				if ( (Ip==4)and(Jp==-1) )
					nbr_kap_b = s_ID_nbr[8];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
			}
			if (N_DIM == 3)
			{
				if (threadIdx.x == 0)
				{
					s_ID_nbr[1] = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
					s_ID_nbr[2] = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
					s_ID_nbr[3] = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
					s_ID_nbr[4] = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
					s_ID_nbr[5] = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
					s_ID_nbr[6] = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
					s_ID_nbr[7] = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
					s_ID_nbr[8] = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
					s_ID_nbr[9] = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
					s_ID_nbr[10] = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
					s_ID_nbr[11] = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
					s_ID_nbr[12] = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
					s_ID_nbr[13] = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
					s_ID_nbr[14] = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
					s_ID_nbr[15] = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
					s_ID_nbr[16] = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
					s_ID_nbr[17] = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
					s_ID_nbr[18] = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
					s_ID_nbr[19] = cblock_ID_nbr[i_kap_b + 19*n_maxcblocks];
					s_ID_nbr[20] = cblock_ID_nbr[i_kap_b + 20*n_maxcblocks];
					s_ID_nbr[21] = cblock_ID_nbr[i_kap_b + 21*n_maxcblocks];
					s_ID_nbr[22] = cblock_ID_nbr[i_kap_b + 22*n_maxcblocks];
					s_ID_nbr[23] = cblock_ID_nbr[i_kap_b + 23*n_maxcblocks];
					s_ID_nbr[24] = cblock_ID_nbr[i_kap_b + 24*n_maxcblocks];
					s_ID_nbr[25] = cblock_ID_nbr[i_kap_b + 25*n_maxcblocks];
					s_ID_nbr[26] = cblock_ID_nbr[i_kap_b + 26*n_maxcblocks];
				}
				__syncthreads();
				
				//
				// p == 1
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + 0;
				Kp = K + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 2
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + 0;
				Kp = K + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 3
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + 1;
				Kp = K + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-] Correct the neighbor cell indices.
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 4
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + -1;
				Kp = K + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-] Correct the neighbor cell indices.
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 5
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + 0;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-] Correct the neighbor cell indices.
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 6
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + 0;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-] Correct the neighbor cell indices.
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 7
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + 1;
				Kp = K + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 7.
				if ( (Ip==4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[7];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 8
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + -1;
				Kp = K + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 8.
				if ( (Ip==-1)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[8];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 9
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + 0;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-][-] Consider nbr 9.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[9];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 10
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + 0;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-][-] Consider nbr 10.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[10];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 11
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + 1;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-][-] Consider nbr 11.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[11];
				// [-] Correct the neighbor cell indices.
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 12
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + -1;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-][-] Consider nbr 12.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[12];
				// [-] Correct the neighbor cell indices.
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 13
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + -1;
				Kp = K + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 13.
				if ( (Ip==4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[13];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 14
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + 1;
				Kp = K + 0;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 14.
				if ( (Ip==-1)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[14];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 15
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + 0;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-][-] Consider nbr 15.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[15];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 16
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + 0;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-][-] Consider nbr 16.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[16];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 17
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + 1;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-][-] Consider nbr 17.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[17];
				// [-] Correct the neighbor cell indices.
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 18
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 0;
				Jp = J + -1;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-][-] Consider nbr 18.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[18];
				// [-] Correct the neighbor cell indices.
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 19
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + 1;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-][-] Consider nbr 7.
				if ( (Ip==4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[7];
				// [-][-] Consider nbr 9.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[9];
				// [-][-] Consider nbr 11.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[11];
				// [-][-] Consider nbr 19.
				if ( (Ip==4)and(Jp==4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[19];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 20
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + -1;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-][-] Consider nbr 8.
				if ( (Ip==-1)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[8];
				// [-][-] Consider nbr 10.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[10];
				// [-][-] Consider nbr 12.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[12];
				// [-][-] Consider nbr 20.
				if ( (Ip==-1)and(Jp==-1)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[20];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 21
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + 1;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-][-] Consider nbr 7.
				if ( (Ip==4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[7];
				// [-][-] Consider nbr 15.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[15];
				// [-][-] Consider nbr 17.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[17];
				// [-][-] Consider nbr 21.
				if ( (Ip==4)and(Jp==4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[21];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 22
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + -1;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-][-] Consider nbr 8.
				if ( (Ip==-1)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[8];
				// [-][-] Consider nbr 16.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[16];
				// [-][-] Consider nbr 18.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[18];
				// [-][-] Consider nbr 22.
				if ( (Ip==-1)and(Jp==-1)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[22];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 23
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + -1;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-][-] Consider nbr 9.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[9];
				// [-][-] Consider nbr 13.
				if ( (Ip==4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[13];
				// [-][-] Consider nbr 18.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[18];
				// [-][-] Consider nbr 23.
				if ( (Ip==4)and(Jp==-1)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[23];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 24
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + 1;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-][-] Consider nbr 10.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[10];
				// [-][-] Consider nbr 14.
				if ( (Ip==-1)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[14];
				// [-][-] Consider nbr 17.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[17];
				// [-][-] Consider nbr 24.
				if ( (Ip==-1)and(Jp==4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[24];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 25
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + -1;
				Jp = J + 1;
				Kp = K + 1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 2.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[2];
				// [-][-] Consider nbr 3.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[3];
				// [-][-] Consider nbr 5.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[5];
				// [-][-] Consider nbr 11.
				if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[11];
				// [-][-] Consider nbr 14.
				if ( (Ip==-1)and(Jp==4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[14];
				// [-][-] Consider nbr 16.
				if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[16];
				// [-][-] Consider nbr 25.
				if ( (Ip==-1)and(Jp==4)and(Kp==4) )
					nbr_kap_b = s_ID_nbr[25];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
				//
				// p == 26
				//
				
				// [-] Compute neighbor cell indices.
				Ip = I + 1;
				Jp = J + -1;
				Kp = K + -1;
				// [-] Assign the correct neighbor cell-block ID.
				nbr_kap_b = i_kap_b;
				// [-][-] Consider nbr 1.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[1];
				// [-][-] Consider nbr 4.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[4];
				// [-][-] Consider nbr 6.
				if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[6];
				// [-][-] Consider nbr 12.
				if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[12];
				// [-][-] Consider nbr 13.
				if ( (Ip==4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
					nbr_kap_b = s_ID_nbr[13];
				// [-][-] Consider nbr 15.
				if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[15];
				// [-][-] Consider nbr 26.
				if ( (Ip==4)and(Jp==-1)and(Kp==-1) )
					nbr_kap_b = s_ID_nbr[26];
				// [-] Correct the neighbor cell indices.
				Ip = (4 + (Ip % 4)) % 4;
				Jp = (4 + (Jp % 4)) % 4;
				Kp = (4 + (Kp % 4)) % 4;
				nbr_kap_c = Ip + 4*Jp + 16*Kp;
				// [-] Check the mask of the cell, it is exists.
				if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
					near_a_solid_cell = true;
			}
			
			s_D[threadIdx.x] = 0;
			if (near_a_solid_cell && cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] != -1)
			{
				cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = -2;
				s_D[threadIdx.x] = 1;
			}
			__syncthreads();
			
			// Block reduction for sum.
			for (int s=blockDim.x/2; s>0; s>>=1)
			{
				if (threadIdx.x < s)
				{
					s_D[threadIdx.x] = s_D[threadIdx.x] + s_D[threadIdx.x + s];
				}
				__syncthreads();
			}
			
			if (threadIdx.x == 0 && s_D[threadIdx.x] > 0)
			{
				tmp_1[i_kap_b] = s_D[threadIdx.x];
				cblock_ID_onb[i_kap_b] = 1;
			}
		}
	}
}
*/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_Alt_S2
(
	int id_max_curr, int n_maxcblocks, bool hit_max, int L,
	int *cblock_ID_ref, int *cblock_ID_mask, int *cblock_ID_nbr, int *cblock_level
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int M_BLOCK = AP->M_BLOCK;
	constexpr int N_Q_max = AP->N_Q_max;
	__shared__ int s_ID_nbr[M_BLOCK*9];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	bool mark_for_refinement = false;
	bool eligible = true;
	
	for (int p = 0; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	
	for (int k = 0; k < (N_DIM==2?1:3); k++)
	{
		// First, read neighbor Ids and place in shared memory. Arrange for contiguity.
		if (kap < id_max_curr && cblock_level[kap] == L)
		{
			for (int p = 0; p < 9; p++)
				s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
		}
		__syncthreads();
		
		// Replace neighbor Ids with their respective marks.
		for (int p = 0; p < 9; p++)
		{
			int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
			if (i_p > -1 && cblock_ID_mask[i_p] == V_BLOCKMASK_SOLIDB)
				s_ID_nbr[threadIdx.x + p*M_BLOCK] = 1;
			else
				s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
		}
		__syncthreads();
		
		// Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
		if (kap < id_max_curr)
		{
			for (int p = 0; p < 9; p++)
			{
				if (s_ID_nbr[p + threadIdx.x*9] == 1)
					mark_for_refinement = true;
			}
		}
		
		if (N_DIM==3)
		{
			__syncthreads();
			
			for (int p = 0; p < 9; p++)
				s_ID_nbr[p + threadIdx.x*9] = -1;
			__syncthreads();
		}
	}
	
	// If at least one neighbor was a boundary-interface block, then mark intermediate.
	// Make sure to refine only eligible blocks (should be currently unrefined, 2:1 balanced afterwards).
	if (kap < id_max_curr && mark_for_refinement && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
	{
		if (!hit_max)
		{
			for (int p = 0; p < N_Q_max; p++)
			{
				if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
					eligible = false;
			}
			
			if (eligible)
				cblock_ID_ref[kap] = V_REF_ID_INDETERMINATE;
		}
		
		if (cblock_ID_mask[kap] == V_BLOCKMASK_REGULAR)
			cblock_ID_mask[kap] = V_BLOCKMASK_SOLIDA;
	}
}

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_Alt_S3
(
	int id_max_curr, int n_maxcblocks, int L,
	int *cblock_ID_ref, int *cblock_ID_mask, int *cblock_ID_nbr, int *cblock_level
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int M_BLOCK = AP->M_BLOCK;
	constexpr int N_Q_max = AP->N_Q_max;
	__shared__ int s_ID_nbr[M_BLOCK*9];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	bool mark_for_refinement = false;
	bool eligible = true;
	
	for (int p = 0; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	
	for (int k = 0; k < (N_DIM==2?1:3); k++)
	{
		// First, read neighbor Ids and place in shared memory. Arrange for contiguity.
		if (kap < id_max_curr && cblock_ID_mask[kap] > -1 && cblock_level[kap] == L)
		{
			for (int p = 0; p < 9; p++)
				s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
		}
		__syncthreads();
		
		// Replace neighbor Ids with their respective marks.
		for (int p = 0; p < 9; p++)
		{
			int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
			if (i_p > -1 && cblock_ID_ref[i_p] == V_REF_ID_INDETERMINATE)
				s_ID_nbr[threadIdx.x + p*M_BLOCK] = 1;
			else
				s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
		}
		__syncthreads();
		
		// Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
		if (kap < id_max_curr)
		{
			for (int p = 0; p < 9; p++)
			{
				if (s_ID_nbr[p + threadIdx.x*9] == 1)
					mark_for_refinement = true;
			}
		}
		
		if (N_DIM==3)
		{
			__syncthreads();
			
			for (int p = 0; p < 9; p++)
				s_ID_nbr[p + threadIdx.x*9] = -1;
			__syncthreads();
		}
	}
	
	// If at least one neighbor was a boundary-interface block, then mark intermediate.
	// Make sure to refine only eligible blocks (should be currently unrefined, 2:1 balanced afterwards).
	if (kap < id_max_curr && mark_for_refinement && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
	{
		for (int p = 0; p < N_Q_max; p++)
		{
			if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
				eligible = false;
		}
		
		if (eligible)
			cblock_ID_ref[kap] = V_REF_ID_INDETERMINATE;
	}
}

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_Alt_S4
(
	int id_max_curr, int *cblock_ID_ref
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr && cblock_ID_ref[kap] == V_REF_ID_INDETERMINATE)
		cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
	
}





template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_FillBinned_S1(int i_dev, int L)
{
	if (n_ids[i_dev][L] > 0)
	{
		ufloat_g_t R = geometry->G_NEAR_WALL_DISTANCE/pow(2.0,(ufloat_g_t)L);
		int Nprop = (int)(R/(ufloat_g_t)(4.0*sqrt(2.0)*dxf_vec[L])) + 1;
		
		bool hit_max = L==MAX_LEVELS-1;
		//int Nprop = pow(2.0,(double)L)+2;
		std::cout << "Level " << L << ": " << Nprop << " propagations.\n";
		
		Cu_FillBinned_V1<ufloat_t,ufloat_g_t,AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
			n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], hit_max,
			c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev],
			geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->c_geom_f_face_X[i_dev], geometry->c_geom_f_face_Xt[i_dev],
			geometry->c_binned_face_ids_n_v[i_dev], geometry->c_binned_face_ids_N_v[i_dev], geometry->c_binned_face_ids_v[i_dev], geometry->G_BIN_DENSITY
		);
		
		Cu_MarkBlocks_Alt_S1<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
			id_max[i_dev][L], n_maxcblocks, hit_max, L,
			c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
		);
		Cu_MarkBlocks_Alt_S2<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
			id_max[i_dev][L], n_maxcblocks, hit_max, L,
			c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
		);
		for (int j = 0; j < Nprop; j++)
		{
			Cu_MarkBlocks_Alt_S3<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
				id_max[i_dev][L], n_maxcblocks, L,
				c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
			);
		}
		Cu_MarkBlocks_Alt_S4<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
			id_max[i_dev][L], c_cblock_ID_ref[i_dev]
		);
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_FillBinned_S2(int i_dev, int L)
{
	if (n_ids[i_dev][L] > 0)
	{
		// Reset one of the intermediate arrays in preparation for copying.
		if (L == 0)
			Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_tmp_1[i_dev], 0);
		
		// Update solid-adjacent cell masks and indicate adjacency of blocks to the geometry boundary.
		Cu_MarkBlocks_GetMasks_V2<AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
			n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
			c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev], c_cblock_ID_onb[i_dev],
			c_tmp_1[i_dev]
		);
		cudaDeviceSynchronize();
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_FillBinned_S2A(int i_dev)
{
	// Compute the number of solid-adjacent cells, and the number of blocks these cells occupy.
	n_solida = thrust::reduce(
		thrust::device, c_tmp_1_dptr[i_dev], c_tmp_1_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], 0
	);
	n_solidb = thrust::count_if(
		thrust::device, c_tmp_1_dptr[i_dev], c_tmp_1_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], is_positive()
	);
	n_solidb = ((n_solidb + 32) / 32) * 32;
	n_maxcells_b = n_solidb*M_CBLOCK;
	std::cout << "Counted " << n_solida << " cells adjacent to the solid boundary (" << (double)n_solida / (double)n_maxcells << ", in " << n_solidb << " blocks or " << n_maxcells_b << " cells)..." << std::endl;
	
	if (n_solidb > 0)
	{
		// Allocate memory for the solid cell linkage data.
		cells_ID_mask_b[i_dev] = new int[n_maxcells_b*N_Q_max]{1};
		cells_f_X_b[i_dev] = new ufloat_g_t[n_maxcells_b*N_Q_max]{0};
		cblock_ID_onb_solid[i_dev] = new int[n_maxcblocks]{-1};
		gpuErrchk( cudaMalloc((void **)&c_cells_ID_mask_b[i_dev], n_maxcells_b*N_Q_max*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_cells_f_X_b[i_dev], n_maxcells_b*N_Q_max*sizeof(ufloat_g_t)) );
		gpuErrchk( cudaMalloc((void **)&c_cblock_ID_onb_solid[i_dev], n_maxcblocks*sizeof(int)) );
// 		gpuErrchk( cudaMalloc((void **)&c_cblock_f_Ff_solid[i_dev], n_solidb*6*sizeof(double)) ); [DEPRECATED]
		gpuErrchk( cudaMalloc((void **)&c_cblock_ID_face[i_dev], n_solidb*N_Q_max*sizeof(int)) );
		
		// Reset some arrays. Make a device pointer to the new cblock_ID_onb_solid array.
		Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_cblock_ID_onb_solid[i_dev], -1);
// 		Cu_ResetToValue<<<(M_BLOCK+(n_solidb*6)-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_solidb*6, c_cblock_f_Ff_solid[i_dev], 0.0); [DEPRECATED]
		Cu_ResetToValue<<<(M_BLOCK+(n_solidb*N_Q_max)-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_solidb*N_Q_max, c_cblock_ID_face[i_dev], -1);
		thrust::device_ptr<int> *c_cblock_ID_onb_solid_dptr = new thrust::device_ptr<int>[N_DEV];
		c_cblock_ID_onb_solid_dptr[i_dev] = thrust::device_pointer_cast(c_cblock_ID_onb_solid[i_dev]);
// 		c_cblock_f_Ff_solid_dptr[i_dev] = thrust::device_pointer_cast(c_cblock_f_Ff_solid[i_dev]); [DEPRECATED]
		
		// Now create the map from block Ids in their usual order to the correct region in the linkage data arrays.
		// NOTE: Make sure c_tmp_1 still has the number of solid-adjacent cells for each block.
		thrust::copy_if(
			thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], c_tmp_1_dptr[i_dev], c_tmp_2_dptr[i_dev], is_positive()
		);
		thrust::scatter(
			thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + n_solidb, c_tmp_2_dptr[i_dev], c_cblock_ID_onb_solid_dptr[i_dev]
		);
		
		cudaMemGetInfo(&free_t, &total_t);
		std::cout << "[-] After allocations:\n";
		std::cout << "    Free: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_IdentifyFaces(int i_dev, int L)
{
	if (n_solidb > 0)
	{
		if (L == N_LEVEL_START)
		{
			
		}
		
		solver->S_IdentifyFaces(0,L);
		
		if (L == MAX_LEVELS_WALL-1)
		{
			// Copy the new solid-cell linkage data to CPU arrays. For now, these don't change since geometry is stationary.
			
		}
	}
	
	return 0;
}
