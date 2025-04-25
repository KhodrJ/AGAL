/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

// template <class T>
// __device__
// void Cu_CrossProduct(T a1, T a2, T a3, T b1, T b2, T b3, T &s1, T &s2, T &s3)
// {
// 	s1 = a2*b3 - a3*b2;
// 	s2 = a3*b1 - a1*b3;
// 	s3 = a1*b2 - a2*b1;
// }

template <typename ufloat_t, const ArgsPack *AP>
__global__
void Cu_ComputeRefCriteria_NearWall_Cases
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dxb_L, int L,
	int *cblock_ID_ref, int *cblock_ID_onb, int *cblock_ID_nbr, ufloat_t *cblock_f_X
)
{
	constexpr int N_DIM = AP->N_DIM;
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		// Evaluate only if current cell-block is not refined already.
		if (cblock_ID_ref[i_kap] == V_REF_ID_UNREFINED)
		{
			// Get the coordinates of the block.
			ufloat_t x_k_plus __attribute__((unused)) = cblock_f_X[i_kap + 0*n_maxcblocks] + (ufloat_t)(0.5)*dxb_L;
			ufloat_t y_k_plus __attribute__((unused)) = cblock_f_X[i_kap + 1*n_maxcblocks] + (ufloat_t)(0.5)*dxb_L;
			ufloat_t z_k_plus __attribute__((unused)) = cblock_f_X[i_kap + 2*n_maxcblocks] + (ufloat_t)(0.5)*dxb_L;
			
			
			
// #if (N_CASE==0 || N_CASE==2)
			// Loop over cavity walls and identify the closest one.
			// If this closest wall is within a certain threshhold, mark for refinement.
			ufloat_t dist_min __attribute__((unused)) = (ufloat_t)(1.0);
			ufloat_t dist_tmp __attribute__((unused)) = (ufloat_t)(1.0);
				// xM
			//dist_min = x_k_plus - (ufloat_t)(0.0);
				// xP
			//dist_tmp = (ufloat_t)(1.0) - x_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
if (N_DIM==2)
{
				// yM
			//dist_tmp = y_k_plus - (ufloat_t)(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yP
			//dist_tmp = (ufloat_t)(1.0) - y_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
}
else
{
				// zM
			//dist_tmp = z_k_plus - (ufloat_t)(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// zP
			//dist_tmp = (ufloat_t)(1.0) - z_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
}
			
			// Evaluate criterion based on dist_min.
			//    + '(cblock_ID_onb[i_kap] == 1)' only refines near boundary.
			//    + 'dist_min <= (ufloat_t)(d_spec)/( (ufloat_t)(1<<L) )' refined by specified distance d_spec.
			if (cblock_ID_nbr[i_kap + 1*n_maxcblocks] == -2)
			//if (cblock_ID_onb[i_kap] == 1)
			//if ( dist_min <= (ufloat_t)(0.2)/( (ufloat_t)(1<<L) ) )
			//if (dist_min < dxb_L)
				cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
// #endif
			
			
			
#if (N_CASE==1)	
if (N_DIM==2)
{
			ufloat_t D = (ufloat_t)(1.0)/(ufloat_t)(32.0);
			ufloat_t rad = (ufloat_t)(2.5)*D/( (ufloat_t)(1<<L) ); // Old value was 1.5
			//if (x_k_plus >= (ufloat_t)(0.3125)-(ufloat_t)(0.5)*D - rad   &&   x_k_plus <= (ufloat_t)(0.3125)+(ufloat_t)(0.5)*D + rad   &&   y_k_plus >= (ufloat_t)(0.5)*(ufloat_t)(1.0)-(ufloat_t)(0.5)*D - rad   &&   y_k_plus <= (ufloat_t)(0.5)*(ufloat_t)(1.0)+(ufloat_t)(0.5)*D + rad)
			if (x_k_plus >= (ufloat_t)(0.3125)-(ufloat_t)(0.5)*D - rad  &&   y_k_plus >= (ufloat_t)(0.5)*(ufloat_t)(1.0)-(ufloat_t)(0.5)*D - rad   &&   y_k_plus <= (ufloat_t)(0.5)*(ufloat_t)(1.0)+(ufloat_t)(0.5)*D + rad)
				cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
			
			
			
			ufloat_t dist_min = (ufloat_t)(1.0);
			ufloat_t dist_tmp = (ufloat_t)(1.0);
				// xM
			//dist_min = x_k_plus - (ufloat_t)(0.0);
				// xP
			dist_tmp = (ufloat_t)(1.0) - x_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yM
			//dist_tmp = y_k_plus - (ufloat_t)(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yP
			//dist_tmp = L_fy - y_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
			
			// Evaluate criterion based on dist_min. (old value 0.05).
			//if ( dist_min <= (ufloat_t)(0.05)/( (ufloat_t)(1<<L) )  ||  cblock_ID_onb[i_kap]==1 )
			if (cblock_ID_onb[i_kap] == 1)
				cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
}
#endif	
			
			
			
			
			// DEBUG
			//if (L == 0 && x_k_plus > (ufloat_t)(0.3) && x_k_plus < (ufloat_t)(0.7) && y_k_plus > (ufloat_t)(0.3) && y_k_plus < (ufloat_t)(0.7) && z_k_plus > (ufloat_t)(0.3) && z_k_plus < (ufloat_t)(0.8))
			//	cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
			//if (L == 0 && x_k_plus > (ufloat_t)(0.3) && x_k_plus < (ufloat_t)(0.7) && y_k_plus > (ufloat_t)(0.3) && y_k_plus <= (ufloat_t)(0.85))
			//	cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
		}
	}
}

template <typename ufloat_t>
__global__
void Cu_ComputeRefCriteria_Uniform
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dxb_L, int L,
	int *cblock_ID_ref
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		// Evaluate only if current cell-block is not refined already.
		if (cblock_ID_ref[i_kap] == V_REF_ID_UNREFINED)
			cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
	}
}



template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_ComputeRefCriteria(int i_dev, int L, int var)
{
	if (var == V_MESH_REF_NW_CASES) // Near-wall distance criterion.
	{
		if (n_ids[i_dev][L] > 0)
		{
			Cu_ComputeRefCriteria_NearWall_Cases<ufloat_t,AP> <<<(M_BLOCK+n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L]*(4*Nqx), L,
				c_cblock_ID_ref[i_dev], c_cblock_ID_onb[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_f_X[i_dev]
			);
		}
	}
	if (var == V_MESH_REF_NW_GEOMETRY) // Complex geometry.
	{
		ufloat_g_t R = geometry->G_NEAR_WALL_DISTANCE/pow(2.0,(ufloat_g_t)L);
		ufloat_g_t R2 __attribute__((unused)) = R*R;
		if (n_ids[i_dev][L] > 0)
		{
// 			Cu_ComputeRefCriteria_NearWall_Geometry_Naive<<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
// 				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], L,
// 				c_cells_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_onb[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
// 				geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->c_geom_f_face_X[i_dev], R, R2
// 			);
			//if (geometry->G_BIN_DENSITY==1)
			//	M_ComputeRefCriteria_Geometry_Naive(i_dev, L);
			//else
				M_ComputeRefCriteria_Geometry_Binned(i_dev, L);
			
			
// 			Cu_ComputeRefCriteria_NearWall_Geometry_Binned<<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
// 				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], L,
// 				c_cells_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_onb[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
// 				geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->c_geom_f_face_X[i_dev], R, R2
// 			);
		}
	}
	if (var == V_MESH_REF_UNIFORM) // Refine the whole level.
	{
		if (n_ids[i_dev][L] > 0)
		{
			Cu_ComputeRefCriteria_Uniform<ufloat_t> <<<(M_BLOCK+n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L]*4, L,
				c_cblock_ID_ref[i_dev]
			);
		}
	}
	if (var == V_MESH_REF_SOLUTION) // Refine the whole level.
	{	
		if (n_ids[i_dev][L] > 0)
			solver->S_ComputeRefCriteria(i_dev, L, var);
	}
	
	return 0;
}
