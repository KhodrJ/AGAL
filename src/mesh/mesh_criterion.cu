/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

template <typename ufloat_t, const ArgsPack *AP>
__global__
void Cu_ComputeRefCriteria_NearWall_Cases
(
    int n_ids_idev_L, int *id_set_idev_L, long int n_maxcblocks, ufloat_t dxb_L, int L,
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
            // Get the bounding box of the block.
            ufloat_t x_k __attribute__((unused)) = cblock_f_X[i_kap + 0*n_maxcblocks] + (ufloat_t)1e-5;
            ufloat_t y_k __attribute__((unused)) = cblock_f_X[i_kap + 1*n_maxcblocks] + (ufloat_t)1e-5;
            ufloat_t z_k __attribute__((unused)) = cblock_f_X[i_kap + 2*n_maxcblocks] + (ufloat_t)1e-5;
            ufloat_t x_kp __attribute__((unused)) = cblock_f_X[i_kap + 0*n_maxcblocks] + dxb_L - (ufloat_t)1e-5;
            ufloat_t y_kp __attribute__((unused)) = cblock_f_X[i_kap + 1*n_maxcblocks] + dxb_L - (ufloat_t)1e-5;
            ufloat_t z_kp __attribute__((unused)) = cblock_f_X[i_kap + 2*n_maxcblocks] + dxb_L - (ufloat_t)1e-5;
            
            
            // Identify if in custom region, if it is specified.
            int onb = cblock_ID_onb[i_kap];
            int nbr_1 = cblock_ID_nbr[i_kap + 1*n_maxcblocks];
            int nbr_2 = cblock_ID_nbr[i_kap + 2*n_maxcblocks];
            int nbr_3 = cblock_ID_nbr[i_kap + 3*n_maxcblocks];
            int nbr_4 = cblock_ID_nbr[i_kap + 4*n_maxcblocks];
            int nbr_5 = N_SKIPID;
            int nbr_6 = N_SKIPID;
            if (N_DIM==3)
            {
                nbr_5 = cblock_ID_nbr[i_kap + 5*n_maxcblocks];
                nbr_6 = cblock_ID_nbr[i_kap + 6*n_maxcblocks];
            }
            bool C = Cu_RefineRegion<ufloat_t>(L,x_k,y_k,z_k,x_kp,y_kp,z_kp,onb,nbr_1,nbr_2,nbr_3,nbr_4,nbr_5,nbr_6);
            
            
            // Evaluate criterion based on dist_min.
            //    + '(cblock_ID_onb[i_kap] == 1)' only refines near boundary.
            //    + 'dist_min <= (ufloat_t)(d_spec)/( (ufloat_t)(1<<L) )' refined by specified distance d_spec.
            //if (cblock_ID_nbr[i_kap + 1*n_maxcblocks] == -2 || C)
            //if (cblock_ID_onb[i_kap] == 1)
            //if ( dist_min <= (ufloat_t)(0.2)/( (ufloat_t)(1<<L) ) )
            //if (dist_min < dxb_L)
            
            if (C)
                cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
        }
    }
}

template <typename ufloat_t>
__global__
void Cu_ComputeRefCriteria_Uniform
(
    int n_ids_idev_L, int *id_set_idev_L, long int n_maxcblocks, ufloat_t dxb_L, int L,
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
