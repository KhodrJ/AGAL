/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"
#include "solver_lbm.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, int VS>
__global__
void Cu_IdentifyFaces
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, int n_maxcells_b, int n_maxblocks_b, ufloat_t dx_L,
	int *id_set_idev_L, int *cells_ID_mask, int *cells_ID_mask_b, ufloat_g_t *cells_f_X_b,
	ufloat_t *cblock_f_X, int *cblock_ID_nbr, int *cblock_ID_mask, int *cblock_ID_onb_solid, int *cblock_ID_face,
	int n_faces, int n_faces_a, ufloat_g_t *geom_f_face_X,
	int *binned_face_ids_n, int *binned_face_ids_N, int *binned_face_ids, int G_BIN_DENSITY
)
{
	//constexpr int N_Q_max = AP->N_Q_max;
	constexpr int N_DIM = AP->N_DIM;
	constexpr int M_TBLOCK = AP->M_TBLOCK;
	constexpr int M_CBLOCK = AP->M_CBLOCK;
	constexpr int M_LBLOCK = AP->M_LBLOCK;
	__shared__ int s_ID_cblock[M_TBLOCK];
	//__shared__ int s_D[M_TBLOCK];
	//__shared__ ufloat_g_t s_face[12*27];
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
	int i_kap_b = -1;
	int valid_block = -1;
	int global_bin_id;
	int bin_id_x = 0;
	int bin_id_y = 0;
	int bin_id_z __attribute__((unused)) = 0;
	int f_p = -1;
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
	ufloat_g_t tmpx = (ufloat_g_t)0.0;
	ufloat_g_t tmpy = (ufloat_g_t)0.0;
	int b_id_p = -8;
	ufloat_g_t dist_p = (ufloat_g_t)(-1.0);
	ufloat_g_t tmpz __attribute__((unused)) = (ufloat_g_t)0.0;
	
	s_ID_cblock[threadIdx.x] = -1;
	if ((threadIdx.x<M_LBLOCK)and(kap<n_ids_idev_L))
	{
		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
	}
	__syncthreads();
	
	// Loop over block Ids.
	for (int k = 0; k < M_LBLOCK; k += 1)
	{
		i_kap_b = s_ID_cblock[k];

		// Needed for condition.
		if (i_kap_b>-1)
		{
			valid_block = cblock_ID_onb_solid[i_kap_b];
		}

		// Proceed only if the current cell-block is on a fluid-solid boundary.
		if ((i_kap_b>-1)&&(valid_block>-1))
		{
			// Threads calculate and store cell coordinates.
			vxp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + (threadIdx.x%4)*dx_L + (ufloat_g_t)(0.5)*dx_L;
			vyp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + ((threadIdx.x/4)%4)*dx_L + (ufloat_g_t)(0.5)*dx_L;
			if (N_DIM==3)
				vzp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + ((threadIdx.x/4)/4)*dx_L + (ufloat_g_t)(0.5)*dx_L;
			
			// For each face, check if the current cell is within the appropriate bounds. If at least
			// one condition is satisfied, exit the loop and make a note.
			bin_id_x = (int)(vxp*G_BIN_DENSITY);
			bin_id_y = (int)(vyp*G_BIN_DENSITY);
			if (N_DIM==3)
				bin_id_z = (int)(vzp*G_BIN_DENSITY);
			
			// Identify the correct bin, and start processing faces.
			global_bin_id = bin_id_x + G_BIN_DENSITY*bin_id_y + G_BIN_DENSITY*G_BIN_DENSITY*bin_id_z;
			int n_f = binned_face_ids_n[global_bin_id];
			int N_f = 0;
			if (n_f > 0)
				N_f = binned_face_ids_N[global_bin_id];
			
			/*
			for (int p = 0; p < n_f; p++)
			{
				//f_p = binned_face_ids[N_f+p];
				//f_p = cblock_ID_face[valid_block + p*n_maxblocks_b];
				//if (f_p >= 0)
				{
					vx1 = geom_f_face_X[f_p + 0*n_faces_a];
					vy1 = geom_f_face_X[f_p + 1*n_faces_a];
					vz1 = geom_f_face_X[f_p + 2*n_faces_a];
					vx2 = geom_f_face_X[f_p + 3*n_faces_a];
					vy2 = geom_f_face_X[f_p + 4*n_faces_a];
					vz2 = geom_f_face_X[f_p + 5*n_faces_a];
					vx3 = geom_f_face_X[f_p + 6*n_faces_a];
					vy3 = geom_f_face_X[f_p + 7*n_faces_a];
					vz3 = geom_f_face_X[f_p + 8*n_faces_a];
					
					// Compute the normal vector of the current face.
					if (N_DIM == 2)
					{
						nx = vy2-vy1;
						ny = vx1-vx2;
					}
					else
					{
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
					}
					
					s_face[0 + p*12] = vx1;
					s_face[1 + p*12] = vy1;
					s_face[2 + p*12] = vz1;
					s_face[3 + p*12] = vx2;
					s_face[4 + p*12] = vy2;
					s_face[5 + p*12] = vz2;
					s_face[6 + p*12] = vx3;
					s_face[7 + p*12] = vy3;
					s_face[8 + p*12] = vz3;
					s_face[9 + p*12] = nx;
					s_face[10 + p*12] = ny;
					s_face[11 + p*12] = nz;
					n_f++;
				}
			}
			*/
			
			//for (int p = 0; p < n_f; p++)
			{
				// Loop over particle velocity vectors and check for intersections.
				if (N_DIM == 2)
				{
					//
					// p = 1
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						nx = vy2-vy1; ny = vx1-vx2;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / ((nx));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpx = (vxp+tmp);
						if (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcells_b] = dist_p;
					
					//
					// p = 2
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						nx = vy2-vy1; ny = vx1-vx2;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / ((ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpx = (vxp);
						if (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 2*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 2*n_maxcells_b] = dist_p;
					
					//
					// p = 3
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						nx = vy2-vy1; ny = vx1-vx2;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / (-(nx));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcells_b] = dist_p;
					
					//
					// p = 4
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						nx = vy2-vy1; ny = vx1-vx2;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / (-(ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpx = (vxp);
						if (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 4*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 4*n_maxcells_b] = dist_p;
					
					//
					// p = 5
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						nx = vy2-vy1; ny = vx1-vx2;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / ((nx+ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpx = (vxp+tmp);
						if (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 5*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 5*n_maxcells_b] = dist_p;
					
					//
					// p = 6
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						nx = vy2-vy1; ny = vx1-vx2;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / ((ny)-(nx));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 6*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 6*n_maxcells_b] = dist_p;
					
					//
					// p = 7
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						nx = vy2-vy1; ny = vx1-vx2;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / (-(nx+ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 7*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 7*n_maxcells_b] = dist_p;
					
					//
					// p = 8
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						nx = vy2-vy1; ny = vx1-vx2;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / ((nx)-(ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpx = (vxp+tmp);
						if (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 8*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 8*n_maxcells_b] = dist_p;
				}
				if (N_DIM == 3)
				{
					//
					// p = 1
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpz = (vzp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcells_b] = dist_p;
					
					
					//
					// p = 2
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nx));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpz = (vzp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 2*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 2*n_maxcells_b] = dist_p;
					
					
					//
					// p = 3
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp);
						tmpx = (vxp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcells_b] = dist_p;
					
					
					//
					// p = 4
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp);
						tmpx = (vxp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 4*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 4*n_maxcells_b] = dist_p;
					
					
					//
					// p = 5
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpz = (vzp+tmp);
						tmpx = (vxp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 5*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 5*n_maxcells_b] = dist_p;
					
					
					//
					// p = 6
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 6*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 6*n_maxcells_b] = dist_p;
					
					
					//
					// p = 7
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 7*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 7*n_maxcells_b] = dist_p;
					
					
					//
					// p = 8
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nx+ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 8*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 8*n_maxcells_b] = dist_p;
					
					
					//
					// p = 9
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpz = (vzp+tmp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 9*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 9*n_maxcells_b] = dist_p;
					
					
					//
					// p = 10
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nx+nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 10*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 10*n_maxcells_b] = dist_p;
					
					
					//
					// p = 11
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny+nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp+tmp);
						tmpx = (vxp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 11*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 11*n_maxcells_b] = dist_p;
					
					
					//
					// p = 12
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(ny+nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 12*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 12*n_maxcells_b] = dist_p;
					
					
					//
					// p = 13
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx)-(ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 13*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 13*n_maxcells_b] = dist_p;
					
					
					//
					// p = 14
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny)-(nx));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 14*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 14*n_maxcells_b] = dist_p;
					
					
					//
					// p = 15
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx)-(nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 15*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 15*n_maxcells_b] = dist_p;
					
					
					//
					// p = 16
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nz)-(nx));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp);
						tmpz = (vzp+tmp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 16*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 16*n_maxcells_b] = dist_p;
					
					
					//
					// p = 17
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny)-(nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 17*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 17*n_maxcells_b] = dist_p;
					
					
					//
					// p = 18
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nz)-(ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp+tmp);
						tmpx = (vxp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 18*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 18*n_maxcells_b] = dist_p;
					
					
					//
					// p = 19
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+ny+nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp+tmp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 19*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 19*n_maxcells_b] = dist_p;
					
					
					//
					// p = 20
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nx+ny+nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 20*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 20*n_maxcells_b] = dist_p;
					
					
					//
					// p = 21
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+ny)-(nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 21*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 21*n_maxcells_b] = dist_p;
					
					
					//
					// p = 22
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nz)-(nx+ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp+tmp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 22*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 22*n_maxcells_b] = dist_p;
					
					
					//
					// p = 23
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+nz)-(ny));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp+tmp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 23*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 23*n_maxcells_b] = dist_p;
					
					
					//
					// p = 24
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny)-(nx+nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 24*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 24*n_maxcells_b] = dist_p;
					
					
					//
					// p = 25
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny+nz)-(nx));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp+tmp);
						tmpz = (vzp+tmp);
						tmpx = (vxp)-(tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 25*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 25*n_maxcells_b] = dist_p;
					
					
					//
					// p = 26
					//
					
					b_id_p = -8;
					dist_p = -1;
					for (int p = 0; p < n_f; p += 1)
					{
						f_p = binned_face_ids[N_f+p];
						vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
						vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
						vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
						ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
						ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
						Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
						tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
						tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx)-(ny+nz));
						if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
						{
						tmpy = (vyp)-(tmp);
						tmpz = (vzp)-(tmp);
						tmpx = (vxp+tmp);
						if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
						{
							if (tmp < dist_p || dist_p < 0)
							dist_p = tmp;
						}
						}
					}
					cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 26*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
					cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 26*n_maxcells_b] = dist_p;
				}
			}
		}
	}
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_IdentifyFaces(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_IdentifyFaces<ufloat_t,ufloat_g_t,AP,LP->VS> <<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, mesh->n_solidb, dxf_vec[L],
			&mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_ID_mask_b[i_dev], mesh->c_cells_f_X_b[i_dev],
			mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->c_cblock_ID_face[i_dev],
			mesh->geometry->n_faces[i_dev], mesh->geometry->n_faces_a[i_dev], mesh->geometry->c_geom_f_face_X[i_dev],
			mesh->geometry->c_binned_face_ids_n_b[i_dev], mesh->geometry->c_binned_face_ids_N_b[i_dev], mesh->geometry->c_binned_face_ids_b[i_dev], mesh->geometry->G_BIN_DENSITY
		);
	}
	
	return 0;
}
