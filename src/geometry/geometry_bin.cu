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
/*  ===[ Cu_ResetBins ]=============================================================  */
/*                                                                                    */
/*  The bin indicator array is reset for the next call to Cu_FillBins.                */
/*                                                                                    */
/**************************************************************************************/

__global__
void Cu_ResetBins
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

template <typename ufloat_g_t, int N_DIM>
__device__ inline
bool IncludeInBinGPU
(
	const ufloat_g_t xm,
	const ufloat_g_t xM,
	const ufloat_g_t ym,
	const ufloat_g_t yM,
	const ufloat_g_t zm,
	const ufloat_g_t zM,
	const ufloat_g_t vBx_m,
	const ufloat_g_t vBx_M,
	const ufloat_g_t vBy_m,
	const ufloat_g_t vBy_M,
	const ufloat_g_t vBz_m,
	const ufloat_g_t vBz_M,
	const ufloat_g_t vx1,
	const ufloat_g_t vy1,
	const ufloat_g_t vz1,
	const ufloat_g_t vx2,
	const ufloat_g_t vy2,
	const ufloat_g_t vz2,
	const ufloat_g_t vx3,
	const ufloat_g_t vy3,
	const ufloat_g_t vz3
)
{
	ufloat_g_t tmp = (ufloat_g_t)(0.0);
	ufloat_g_t ex1 = (ufloat_g_t)(0.0);
	ufloat_g_t ey1 = (ufloat_g_t)(0.0);
	
if (N_DIM==2)
{
	// Only consider this calculation if the bounding box intersects the bin.
	if ( !( (vBx_m < xm && vBx_M < xm) || (vBx_m > xM && vBx_M > xM) || (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) ) )
	{
		// Check if bounding box is entirely inside current bin.
		if (vBx_m > xm && vBx_M < xM && vBy_m > ym && vBy_M < yM) { return true; }
		
		// Check if at least one of the vertices is inside the bin.
		if (vx1 > xm && vx1 < xM) { return true; }
		if (vx2 > xm && vx2 < xM) { return true; }
		if (vy1 > ym && vy1 < yM) { return true; }
		if (vy2 > ym && vy2 < yM) { return true; }
		
		// Check the bottom edge of the bin.
		{
			ey1 = vy2-vy1;
			tmp = (ym-vy1)/(ey1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			if (CheckInLine(tmp,ex1,xm,xM)) { return true; }
		}
		
		// Check the top edge of the bin.
		{
			ey1 = vy2-vy1;
			tmp = (yM-vy1)/(ey1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			if (CheckInLine(tmp,ex1,xm,xM)) { return true; }
		}
		
		// Check the left edge of the bin.
		{
			ex1 = vx2-vx1;
			tmp = (xm-vx1)/(ex1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			if (CheckInLine(tmp,ey1,ym,yM)) { return true; }
		}
		
		// Check the right edge of the bin.
		{
			ex1 = vx2-vx1;
			tmp = (xM-vx1)/(ex1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			if (CheckInLine(tmp,ey1,ym,yM)) { return true; }
		}
	}
}
else
{
	ufloat_g_t ez1 = (ufloat_g_t)(0.0);
	
	if ( !( (vBx_m < xm && vBx_M < xm) || (vBx_m > xM && vBx_M > xM) || (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) || (vBz_m < zm && vBz_M < zm) || (vBz_m > zM && vBz_M > zM) ) )
	{
		// Check if bounding box is entirely inside current bin.
		if (vBx_m > xm && vBx_M < xM && vBy_m > ym && vBy_M < yM && vBz_m > zm && vBz_M < zM) { return true; }
		
		// Check if bounding box completely surrounds the bin.
		if (vBx_m < xm && vBx_M > xM && vBy_m < ym && vBy_M > yM) { return true; }
		if (vBy_m < ym && vBy_M > yM && vBz_m < zm && vBz_M > zM) { return true; }
		if (vBz_m < zm && vBz_M > zM && vBx_m < xm && vBx_M > xM) { return true; }
		
		// Check if at least one of the vertices is inside the bin.
		if (vx1 > xm && vx1 < xM && vy1 > ym && vy1 < yM && vz1 > zm && vz1 < zM) { return true; }
		if (vx2 > xm && vx2 < xM && vy2 > ym && vy2 < yM && vz2 > zm && vz2 < zM) { return true; }
		if (vx3 > xm && vx3 < xM && vy3 > ym && vy3 < yM && vz3 > zm && vz3 < zM) { return true; }
		
		// Check the bottom face of the bin.
		{
			ez1 = vz2-vz1;
			tmp = (zm-vz1)/(ez1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		{
			ez1 = vz3-vz2;
			tmp = (zm-vz2)/(ez1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		{
			ez1 = vz1-vz3;
			tmp = (zm-vz3)/(ez1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		
		// Check the top face of the bin.
		{
			ez1 = vz2-vz1;
			tmp = (zM-vz1)/(ez1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		{
			ez1 = vz3-vz2;
			tmp = (zM-vz2)/(ez1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		{
			ez1 = vz1-vz3;
			tmp = (zM-vz3)/(ez1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		
		// Check the back face of the bin.
		{
			ey1 = vy2-vy1;
			tmp = (ym-vy1)/(ey1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		{
			ey1 = vy3-vy2;
			tmp = (ym-vy2)/(ey1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		{
			ey1 = vy1-vy3;
			tmp = (ym-vy3)/(ey1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		
		// Check the front face of the bin.
		{
			ey1 = vy2-vy1;
			tmp = (yM-vy1)/(ey1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		{
			ey1 = vy3-vy2;
			tmp = (yM-vy2)/(ey1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		{
			ey1 = vy1-vy3;
			tmp = (yM-vy3)/(ey1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		
		// Check the left face of the bin.
		{
			ex1 = vx2-vx1;
			tmp = (xm-vx1)/(ex1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		{
			ex1 = vx3-vx2;
			tmp = (xm-vx2)/(ex1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		{
			ex1 = vx1-vx3;
			tmp = (xm-vx3)/(ex1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		
		// Check the right face of the bin.
		{
			ex1 = vx2-vx1;
			tmp = (xM-vx1)/(ex1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		{
			ex1 = vx3-vx2;
			tmp = (xM-vx2)/(ex1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		{
			ex1 = vx1-vx3;
			tmp = (xM-vx3)/(ex1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
	}
}
	
	return false;
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_FillBins_V1 ]===========================================================  */
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
void Cu_FillBins_V1
(
	const int j0v,
	const int j0b,
	const int n_faces,
	const int n_faces_a,
	const int G_BIN_NUM_V,
	const int G_BIN_NUM_B,
	const ufloat_g_t *__restrict__ geom_f_face_X,
	int *__restrict__ bin_indicators_2D,
	int *__restrict__ bin_indicators_3D,
	const ufloat_g_t eps,
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
		
		int bin_id_xl = (int)(vBx_m*G_BIN_DENSITY)-1;
		int bin_id_yl = (int)(vBy_m*G_BIN_DENSITY)-1;
		int bin_id_xL = (int)(vBx_M*G_BIN_DENSITY)+2;
		int bin_id_yL = (int)(vBy_M*G_BIN_DENSITY)+2;
		
		// Loop over the possible bins.
		for (int j = j0v; j < j0v+G_BIN_NUM_V; j++)
		{
			int bj = j%G_BIN_DENSITY;
			
			if (bj >= bin_id_yl && bj < bin_id_yL)
			bin_indicators_2D[kap + (j-j0v)*n_faces_a] = kap;
		}
		for (int j = j0b; j < j0b+G_BIN_NUM_B; j++)
		{
			int bi = j%G_BIN_DENSITY;
			int bj = (j/G_BIN_DENSITY)%G_BIN_DENSITY;
			
			if (bi >= bin_id_xl && bi < bin_id_xL && bj >= bin_id_yl && bj < bin_id_yL)
			bin_indicators_3D[kap + (j-j0b)*n_faces_a] = kap;
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
		
		// Identify the bin indices of the lower and upper bounds.
		int bin_id_xl = (int)(vBx_m*G_BIN_DENSITY)-1;
		int bin_id_yl = (int)(vBy_m*G_BIN_DENSITY)-1;
		int bin_id_zl = (int)(vBz_m*G_BIN_DENSITY)-1;
		int bin_id_xL = (int)(vBx_M*G_BIN_DENSITY)+2;
		int bin_id_yL = (int)(vBy_M*G_BIN_DENSITY)+2;
		int bin_id_zL = (int)(vBz_M*G_BIN_DENSITY)+2;
		
		for (int j = j0v; j < j0v+G_BIN_NUM_V; j++)
		{
			int bj = j%G_BIN_DENSITY;
			int bk = (j/G_BIN_DENSITY)%G_BIN_DENSITY;
		
			if (bj >= bin_id_yl && bj+1 < bin_id_yL && bk >= bin_id_zl && bk+1 < bin_id_zL)
			bin_indicators_2D[kap + (j-j0v)*n_faces_a] = kap;
		}
		for (int j = j0b; j < j0b+G_BIN_NUM_B; j++)
		{
			int bi = j%G_BIN_DENSITY;
			int bj = (j/G_BIN_DENSITY)%G_BIN_DENSITY;
			int bk = (j/G_BIN_DENSITY)/G_BIN_DENSITY;
			
			if (bi >= bin_id_xl && bi+1 < bin_id_xL && bj >= bin_id_yl && bj+1 < bin_id_yL && bk >= bin_id_zl && bk+1 < bin_id_zL)
			bin_indicators_3D[kap + (j-j0b)*n_faces_a] = kap;
}
		}
	}
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_FillBins_V2 ]===========================================================  */
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
void Cu_FillBins_V2
(
	const int j0v,
	const int j0b,
	const int n_faces,
	const int n_faces_a,
	const int G_BIN_NUM_V,
	const int G_BIN_NUM_B,
	const ufloat_g_t *__restrict__ geom_f_face_X,
	int *__restrict__ bin_indicators_2D,
	int *__restrict__ bin_indicators_3D,
	const ufloat_g_t eps,
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
		
		// Loop over the possible bins.
		for (int j = j0v; j < j0v+G_BIN_NUM_V; j++)
		{
			int bj = j%G_BIN_DENSITY;
			ufloat_g_t ym = (bj)*LyOg;
			ufloat_g_t yM = (bj+1)*LyOg;
			
			// Only consider this calculation if the bounding box intersects the bin.
			if (IncludeInBinGPU<ufloat_g_t,2>(0-eps,Lx+eps,ym-eps,yM+eps,0,0,vBx_m,vBx_M,vBy_m,vBy_M,0,0,vx1,vy1,0,vx2,vy2,0,0,0,0))
				bin_indicators_2D[kap + (j-j0v)*n_faces_a] = kap;
		}
		for (int j = j0b; j < j0b+G_BIN_NUM_B; j++)
		{
			int bi = j%G_BIN_DENSITY;
			int bj = (j/G_BIN_DENSITY)%G_BIN_DENSITY;
			ufloat_g_t xm = (bi)*LxOg;
			ufloat_g_t xM = (bi+1)*LxOg;
			ufloat_g_t ym = (bj)*LyOg;
			ufloat_g_t yM = (bj+1)*LyOg;
			
			// Only consider this calculation if the bounding box intersects the bin.
			if (IncludeInBinGPU<ufloat_g_t,2>(xm-dx,xM+dx,ym-dx,yM+dx,0,0,vBx_m,vBx_M,vBy_m,vBy_M,0,0,vx1,vy1,0,vx2,vy2,0,0,0,0))
				bin_indicators_3D[kap + (j-j0b)*n_faces_a] = kap;
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
		
		// Loop over the possible bins.
		for (int j = j0v; j < j0v+G_BIN_NUM_V; j++)
		{
			int bj = j%G_BIN_DENSITY;
			int bk = (j/G_BIN_DENSITY)%G_BIN_DENSITY;
			ufloat_g_t ym = (bj)*LyOg;
			ufloat_g_t yM = (bj+1)*LyOg;
			ufloat_g_t zm = (bk)*LzOg;
			ufloat_g_t zM = (bk+1)*LzOg;
			
			// Only consider this calculation if the bounding box intersects the bin.
			if (IncludeInBinGPU<ufloat_g_t,3>(-eps,Lx+eps,ym-eps,yM+eps,zm-eps,zM+eps,vBx_m,vBx_M,vBy_m,vBy_M,vBz_m,vBz_M,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3))
				bin_indicators_2D[kap + (j-j0v)*n_faces_a] = kap;
		}
		for (int j = j0b; j < j0b+G_BIN_NUM_B; j++)
		{
			int bi = j%G_BIN_DENSITY;
			int bj = (j/G_BIN_DENSITY)%G_BIN_DENSITY;
			int bk = (j/G_BIN_DENSITY)/G_BIN_DENSITY;
			ufloat_g_t xm = (bi)*LxOg;
			ufloat_g_t xM = (bi+1)*LxOg;
			ufloat_g_t ym = (bj)*LyOg;
			ufloat_g_t yM = (bj+1)*LyOg;
			ufloat_g_t zm = (bk)*LzOg;
			ufloat_g_t zM = (bk+1)*LzOg;
			
			// Only consider this calculation if the bounding box intersects the bin.
			if (IncludeInBinGPU<ufloat_g_t,3>(xm-dx,xM+dx,ym-dx,yM+dx,zm-dx,zM+dx,vBx_m,vBx_M,vBy_m,vBy_M,vBz_m,vBz_M,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3))
				bin_indicators_3D[kap + (j-j0b)*n_faces_a] = kap;
		}
}
	}
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_MakeBins ]===============================================================  */
/*                                                                                    */
/*  Performs the '3D binning' portion of the geometry-incorporation algorithm.        */
/*  Here, faces (edges or triangles in 2D or 3D, respectively) are assigned to        */
/*  square/cubic bins to be used in the cell-face linkage identification step.        */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_MakeBins(int i_dev)
{
	// Assumes that coords list has already been initialized, make sure you do that first.
	n_bins_v = 1; for (int d = 0; d < N_DIM-1; d++) n_bins_v *= G_BIN_DENSITY;
	n_bins_b = 1; for (int d = 0; d < N_DIM; d++)   n_bins_b *= G_BIN_DENSITY;
	int G_BIN_NUM_V = n_bins_v / G_BIN_FRAC;
	int G_BIN_NUM_B = n_bins_b / G_BIN_FRAC;
	ufloat_g_t Lx0g = Lx/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Ly0g = Ly/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Lz0g = Lz/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t eps = 1e-5;
	if (std::is_same<ufloat_g_t, float>::value) eps = FLT_EPSILON;
	if (std::is_same<ufloat_g_t, double>::value) eps = DBL_EPSILON;
	
	// Initialize the intermediate device array to store the bin indicators. Set values to -1 as default.
	gpuErrchk( cudaMalloc((void **)&c_bin_indicators_v[i_dev], G_BIN_NUM_V*n_faces_a[i_dev]*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_bin_indicators_b[i_dev], G_BIN_NUM_B*n_faces_a[i_dev]*sizeof(int)) );
	Cu_ResetToValue<<<(M_BLOCK+G_BIN_NUM_V*n_faces_a[i_dev]-1)/M_BLOCK, M_BLOCK>>>(G_BIN_NUM_V*n_faces_a[i_dev], c_bin_indicators_v[i_dev], -1);
	Cu_ResetToValue<<<(M_BLOCK+G_BIN_NUM_B*n_faces_a[i_dev]-1)/M_BLOCK, M_BLOCK>>>(G_BIN_NUM_B*n_faces_a[i_dev], c_bin_indicators_b[i_dev], -1);
	
	// Just assuming max. of 2*n_faces_a, need to make this more robust. I don't want to store n_bins_b*n_faces of data.
	// 2D
	binned_face_ids_N_v[i_dev] = new int[n_bins_v];
	binned_face_ids_n_v[i_dev] = new int[n_bins_v];
	binned_face_ids_v[i_dev] = new int[G_BIN_OVERLAP*n_faces_a[i_dev]];
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_v[i_dev], G_BIN_OVERLAP*n_faces_a[i_dev]*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_v[i_dev], n_bins_v*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_v[i_dev], n_bins_v*sizeof(int)) );
	// 3D
	binned_face_ids_N_b[i_dev] = new int[n_bins_b];
	binned_face_ids_n_b[i_dev] = new int[n_bins_b];
	binned_face_ids_b[i_dev] = new int[G_BIN_OVERLAP*n_faces_a[i_dev]];
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_b[i_dev], G_BIN_OVERLAP*n_faces_a[i_dev]*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_b[i_dev], n_bins_b*sizeof(int)) );
	gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_b[i_dev], n_bins_b*sizeof(int)) );
	
	// Fill the bins, and then do stream compaction to get contiguous binned data.
	int n_pm1v = 0;
	int n_pm1b = 0;
	int n_pv = 0;
	int n_pb = 0;
	cudaDeviceSynchronize();
	tic_simple("");
	std::cout << "[-] Starting binning process..." << std::endl;
	for (int j = 0; j < G_BIN_FRAC; j++)
	{
		// Fill the bins.
		int j0v = j*G_BIN_NUM_V;
		int j0b = j*G_BIN_NUM_B;
		if (G_BIN_APPROACH==0)
		{
			Cu_FillBins_V1<ufloat_g_t,AP> <<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
				j0v, j0b, n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM_V, G_BIN_NUM_B, c_geom_f_face_X[i_dev], c_bin_indicators_v[i_dev], c_bin_indicators_b[i_dev],
				eps, dx, Lx, Lx0g, Ly0g, Lz0g, G_BIN_DENSITY
			);
		}
		else
		{
			Cu_FillBins_V2<ufloat_g_t,AP> <<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
				j0v, j0b, n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM_V, G_BIN_NUM_B, c_geom_f_face_X[i_dev], c_bin_indicators_v[i_dev], c_bin_indicators_b[i_dev],
				eps, dx, Lx, Lx0g, Ly0g, Lz0g, G_BIN_DENSITY
			);
		}
		
		// Perform stream compaction for the 2D bin array.
		for (int p = 0; p < G_BIN_NUM_V; p++)
		{
			// Get the number of faces in this traversal.
			int j_pv = j0v + p;
			n_pv =  thrust::count_if(
				thrust::device, &c_bin_indicators_v[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators_v[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], is_nonnegative()
			);
			binned_face_ids_N_v[i_dev][j_pv] = n_pm1v;
			binned_face_ids_n_v[i_dev][j_pv] = n_pv;
			
			// Copy the faces into the stored binned ID set.
			if (n_pm1v >= G_BIN_OVERLAP*n_faces_a[i_dev]-1)
				std::cout << "[-] WARNING: Insufficient memory to store binned faces..." << std::endl;
			if (n_pv > 0)
			{
				thrust::copy_if(
					thrust::device, &c_bin_indicators_v[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators_v[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], &c_binned_face_ids_v[i_dev][n_pm1v], is_nonnegative()
				);
			}
			
			// Update trackers.
			n_pm1v = binned_face_ids_N_v[i_dev][j_pv] + n_pv;
		}
		
		// Perform stream compaction for the 3D bin array.
		for (int p = 0; p < G_BIN_NUM_B; p++)
		{
			// Get the number of faces in this traversal.
			int j_pb = j0b + p;
			n_pb =  thrust::count_if(
				thrust::device, &c_bin_indicators_b[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators_b[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], is_nonnegative()
			);
			binned_face_ids_N_b[i_dev][j_pb] = n_pm1b;
			binned_face_ids_n_b[i_dev][j_pb] = n_pb;
			
			// Copy the faces into the stored binned ID set.
			if (n_pm1b >= G_BIN_OVERLAP*n_faces_a[i_dev]-1)
				std::cout << "[-] WARNING: Insufficient memory to store binned faces..." << std::endl;
			if (n_pb > 0)
			{
				thrust::copy_if(
					thrust::device, &c_bin_indicators_b[i_dev][p*n_faces_a[i_dev]], &c_bin_indicators_b[i_dev][p*n_faces_a[i_dev]] + n_faces[i_dev], &c_binned_face_ids_b[i_dev][n_pm1b], is_nonnegative()
				);
			}
			
			// Update trackers.
			n_pm1b = binned_face_ids_N_b[i_dev][j_pb] + n_pb;
		}
		
		// Reset intermediate arrays before next run.
		Cu_ResetBins<<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM_V, c_bin_indicators_v[i_dev]
		);
		Cu_ResetBins<<<(M_BLOCK+n_faces_a[i_dev]-1)/M_BLOCK,M_BLOCK>>>(
			n_faces[i_dev], n_faces_a[i_dev], G_BIN_NUM_B, c_bin_indicators_b[i_dev]
		);
	}
	cudaDeviceSynchronize();
	std::cout << "Elapsed time: " << toc_simple("",T_MS) << std::endl;
	std::cout << "[-] Finished binning process (2D) (" << n_pm1v << " faces/" << G_BIN_OVERLAP*n_faces_a[i_dev] << ")..." << std::endl;
	std::cout << "[-] Finished binning process (3D) (" << n_pm1b << " faces/" << G_BIN_OVERLAP*n_faces_a[i_dev] << ")..." << std::endl;
	
	// Copy the face id counts to the GPU.
	// 2D
	gpuErrchk( cudaMemcpy(c_binned_face_ids_n_v[i_dev], binned_face_ids_n_v[i_dev], n_bins_v*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(c_binned_face_ids_N_v[i_dev], binned_face_ids_N_v[i_dev], n_bins_v*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(binned_face_ids_v[i_dev], c_binned_face_ids_v[i_dev], n_pm1v*sizeof(int), cudaMemcpyDeviceToHost) );
	// 3D
	gpuErrchk( cudaMemcpy(c_binned_face_ids_n_b[i_dev], binned_face_ids_n_b[i_dev], n_bins_b*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(c_binned_face_ids_N_b[i_dev], binned_face_ids_N_b[i_dev], n_bins_b*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(binned_face_ids_b[i_dev], c_binned_face_ids_b[i_dev], n_pm1b*sizeof(int), cudaMemcpyDeviceToHost) );
	
	// DEBUG
// 	std::cout << "APPROACH: 3D" << std::endl;
// 	for (int p = 0; p < n_bins_b; p++)
// 	{
// 		int Nbp = binned_face_ids_N_b[i_dev][p];
// 		int npb = binned_face_ids_n_b[i_dev][p];
// 		if (npb > 0)
// 		{
// 			std::cout << "Bin #" << p << ": ";
// 			for (int K = 0; K < npb; K++)
// 				std::cout << binned_face_ids_b[i_dev][Nbp + K] << " ";
// 			std::cout << std::endl;
// 		}
// 	}
	
	// Free memory in intermediate device arrays.
	gpuErrchk( cudaFree(c_bin_indicators_v[i_dev]) );
	gpuErrchk( cudaFree(c_bin_indicators_b[i_dev]) );
	
	return 0;
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_DrawBinsAndFaces ]=======================================================  */
/*                                                                                    */
/*  This is a debug routine that generates a MATLAB script in which the various       */
/*  bins and faces assignments can be plotted.                                        */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_DrawBinsAndFaces(int i_dev)
{
	// Initial parameters. Open the output file.
	int N_DIM = AP->N_DIM;
	ufloat_g_t Lx0g = Lx/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Ly0g = Ly/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Lz0g = Lz/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t minL0g = std::min({Lx0g,Ly0g,(N_DIM==2?Lx0g:Lz0g)});
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	std::ofstream out = std::ofstream("debug_bins_3D.txt");
	std:cout << "Drawing 3D bins..." << std::endl;
	
	int counter = 1;
	for (int k = 0; k < G_BIN_DENSITY; k++)
	{
		for (int j = 0; j < G_BIN_DENSITY; j++)
		{
			for (int i = 0; i < G_BIN_DENSITY; i++)
			{
				// Identify the bin.
				int global_bin_id = i + G_BIN_DENSITY*j + G_BIN_DENSITY*G_BIN_DENSITY*k;
				
				// Get the number of faces in the bin.
				int n_f = binned_face_ids_n_b[i_dev][global_bin_id];
				int N_f = 0;
				if (n_f > 0)
				{
					N_f = binned_face_ids_N_b[i_dev][global_bin_id];
					out << "% Bin #" << counter << std::endl;
					out << "subplot(4,2," << counter << ")" << std::endl;
					counter++;
				}
				
				// If there are faces to draw, draw the bin too. Each bin gets its own unique color.
				if (n_f > 0)
				{
					c0 = (double)(std::rand() % 256) / 256.0;
					c1 = (double)(std::rand() % 256) / 256.0;
					c2 = (double)(std::rand() % 256) / 256.0;
					DebugDrawCubeInMATLAB(out, i*Lx0g, (i+1)*Lx0g, j*Ly0g, (j+1)*Ly0g, k*Lz0g, (k+1)*Lz0g, c0, c1, c2);
				}
				
				for (int p = 0; p < n_f; p++)
				{
					int f_p = binned_face_ids_b[i_dev][N_f+p];
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
	}
	
	// Close the file.
	std::cout << "Finished drawing 3D bins..." << std::endl;
	out.close();
	
	return 0;
}
