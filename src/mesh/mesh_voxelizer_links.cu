/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

__device__ int correction_counter_d = 0;

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_LinkLengthComputation ]=================================================  */
/*                                                                                    */
/*  This kernel computes the lengths of links that cross through the geometry via     */
/*  ray cast and point-in-triangle testing of intersection points. The number of      */
/*  directions considered can be restricted to the six canonical directions if a      */
/*  smaller stencil is being used in the vicinity of the geometry to reduce memory    */
/*  consumption and total execution time.                                             */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_LinkLengthComputation
(
    const int n_ids_idev_L,
    const long int n_maxcblocks,
    const int n_maxcells_b,
    const ufloat_g_t dx_L,
    const int *__restrict__ id_set_idev_L,
    int *__restrict__ cells_ID_mask_b,
    ufloat_g_t *__restrict__ cells_f_X_b,
    int *__restrict__ cells_ID_mask,
    const ufloat_g_t *__restrict__ cblock_f_X,
    int *__restrict__ cblock_ID_onb_solid,
    const long int n_faces,
    const long int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_Xt,
    const int *__restrict__ binned_face_ids_n,
    const int *__restrict__ binned_face_ids_N,
    const int *__restrict__ binned_face_ids,
    const int n_bin_density,
    const int NVDP=16
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    ufloat_g_t dQ[N_Q_max];
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // Loop over block Ids.
    if (i_kap_b > -1)
    {
        int valid_block = cblock_ID_onb_solid[i_kap_b];

        // Proceed only if the current cell-block is on a fluid-solid boundary.
        if (valid_block > -1)
        {
            // Threads calculate and store cell coordinates.
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            int K = 0;
            if (N_DIM==3)
                K = (threadIdx.x / 4) / 4;
            vec3<ufloat_g_t> vp
            (
                    cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + 0.5*dx_L,
                    cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + 0.5*dx_L,
                    (N_DIM==2) ? (ufloat_g_t)0.0 : cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + 0.5*dx_L
            );
            
            // For each face, check if the current cell is within the appropriate bounds. If at least
            // one condition is satisfied, exit the loop and make a note.
            int bin_id_x = (int)(vp.x*n_bin_density);
            int bin_id_y = (int)(vp.y*n_bin_density);
            int bin_id_z = 0;
            if (N_DIM==3)
                bin_id_z = (int)(vp.z*n_bin_density);
            
            // Identify the correct bin, and start processing faces.
            int global_bin_id = bin_id_x + n_bin_density*bin_id_y + n_bin_density*n_bin_density*bin_id_z;
            int n_f = binned_face_ids_n[global_bin_id];
            int N_f = 0;
            if (n_f > 0)
            {
                N_f = binned_face_ids_N[global_bin_id];
                
                // Initialize face-cell links.
                for (int q = 0; q < N_Q_max; q++)
                    dQ[q] = (ufloat_g_t)(-1.0);
                //int b_id_p = -8;
                
                for (int p = 0; p < n_f; p++)
                {
                    int f_p = binned_face_ids[N_f+p];
                    vec3<ufloat_g_t> v1, v2, v3;
                    LoadFaceData<ufloat_g_t,FaceArrangement::AoS>(f_p, geom_f_face_Xt, NVDP, n_faces_a, v1, v2, v3);
                    vec3<ufloat_g_t> n = FaceNormalUnit<ufloat_g_t,N_DIM>(v1,v2,v3);
                    
                    // DEBUG
                    //if (threadIdx.x==0)
                    //   DebugDraw2DLineSegmentInMATLAB_DEV(v1,v2,'m','-','o');
                    
                    if (N_DIM==2)
                    {
                        // Loop over all directions, update links for this face if applicable.
                        for (int q = 1; q < N_Q_max; q++)
                        {
                            vec3<ufloat_g_t> ray(
                                static_cast<ufloat_g_t>(V_CONN_ID[q+0*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[q+1*27]),
                                static_cast<ufloat_g_t>(0.0)
                            );
                            ufloat_g_t d = DotV(ray,n);
                            if (d != static_cast<ufloat_g_t>(0.0))
                            {
                                d = DotV(v1-vp,n) / d;
                                if (d > static_cast<ufloat_g_t>(0.0) && d < dx_L + EPS<ufloat_g_t>())
                                {
                                    vec3<ufloat_g_t> vi = vp + ray*d;
                                    
                                    if (CheckPointInLineExtended(vi,v1,v2))
                                    {
                                        ufloat_g_t dist_q = dQ[q];
                                        if (d < dist_q || dist_q < 0)
                                            dQ[q] = d / dx_L;
                                    }
                                }
                            }
                        }
                    }
                    else // N_DIM==3
                    {
                        for (int q = 1; q < N_Q_max; q++)
                        {
                            vec3<ufloat_g_t> ray(
                                static_cast<ufloat_g_t>(V_CONN_ID[q+0*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[q+1*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[q+2*27])
                            );
                            ufloat_g_t d = DotV(ray,n);
                            if (d != static_cast<ufloat_g_t>(0.0))
                            {
                                d = DotV(v1-vp,n) / d;
                                if (d > static_cast<ufloat_g_t>(0.0) && d < dx_L + EPS<ufloat_g_t>())
                                {
                                    vec3<ufloat_g_t> vi = vp + ray*d;
                                    if (CheckPointInTriangleExtended(vi,v1,v2,v3,n))
                                    {
                                        ufloat_g_t dist_q = dQ[q];
                                        if (d < dist_q || dist_q < 0)
                                            dQ[q] = d / dx_L;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Write face-cell links to global memory.
                if (cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] == V_CELLMASK_BOUNDARY)
                {
                    for (int q = 1; q < N_Q_max; q++)
                    {
                        cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + q*n_maxcells_b] = -8; // This will be updated to read actual boundary Ids later.
                        cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + q*n_maxcells_b] = dQ[q];
                        //if (dQ[q] > 0)
                        //    printf("%17.5f\n", dQ[q]/dx_L);
                        
                        // DEBUG
                        vec3<ufloat_g_t> ray(
                            static_cast<ufloat_g_t>(V_CONN_ID[q+0*27]),
                            static_cast<ufloat_g_t>(V_CONN_ID[q+1*27]),
                            static_cast<ufloat_g_t>(V_CONN_ID[q+2*27])
                        );
                        if (dQ[q] > 0)
                        {
                            //DebugDraw2DLineSegmentInMATLAB_DEV(vp,vp+ray*dQ[q]*dx_L,'b');
                            //DebugDraw2DPointInMATLAB_DEV(vp+ray*dQ[q]*dx_L,'r','*');
                        }
                    }
                }
                
                // DEBUG
                //DebugDraw2DSquareInMATLAB_DEV(vp-(0.5*dx_L),vp+(0.5*dx_L),'k');
                //if (cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] == V_CELLMASK_BOUNDARY)
                //   DebugDraw2DPointInMATLAB_DEV(vp,'k','o');
                //if (cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] == V_CELLMASK_SOLID)
                //   DebugDraw2DPointInMATLAB_DEV(vp,'k','.');
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_LinkLengthValidation
(
    const int n_ids_idev_L,
    const long int n_maxcblocks,
    const int n_maxcells_b,
    const ufloat_g_t dx_L,
    const int *__restrict__ id_set_idev_L,
    ufloat_g_t *__restrict__ cells_f_X_b,
    int *__restrict__ cells_ID_mask,
    ufloat_g_t *__restrict__ cblock_f_X,
    int *__restrict__ cblock_ID_nbr,
    int *__restrict__ cblock_ID_onb_solid
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ int s_ID_nbr[27];
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // Loop over block Ids.
    if (i_kap_b > -1)
    {
        int valid_block = cblock_ID_onb_solid[i_kap_b];
        
        // Latter condition is added only if n>0.
        if (valid_block > -1)
        {
            // Threads calculate and store cell coordinates.
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            int K = 0;
            if (N_DIM==3)
                K = (threadIdx.x / 4) / 4;
            vec3<ufloat_g_t> vp
            (
                    cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + 0.5*dx_L,
                    cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + 0.5*dx_L,
                    (N_DIM==2) ? (ufloat_g_t)0.0 : cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + 0.5*dx_L
            );
            
            // Load neighbor-block indices into shared memory.
            if (threadIdx.x==0)
            {
                for (int p = 0; p < N_Q_max; p++)
                    s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
            }
            __syncthreads();
            
            // Loop over (half of) directions and check between neighbors.
            for (int p = 1; p < N_Q_max; p++)
            {
                if ( (N_DIM==2 && (p==1||(p+1)%3==0))   ||   (N_DIM==3 && (p==26||((p-1)%2==0&&p<25))) )
                {
                    // Get neighbor cell index.
                    // nbr_kap_b is the index of the neighboring block w.r.t the current cell.
                    // nbr_kap_c is the index of the cell in that neighboring block.
                    // nbr_kap_h is the index of the halo to store that value.
                    int nbr_kap_b = N_SKIPID;
                    int nbr_kap_c = N_SKIPID;
                    Cu_GetNbrIndices<N_DIM>(p,&nbr_kap_b,&nbr_kap_c,I,J,K,s_ID_nbr);
                    if (nbr_kap_b > -1)
                        nbr_kap_b = cblock_ID_onb_solid[nbr_kap_b];
                
                    if (nbr_kap_b > -1)
                    {
                        // Get dQ_p and dQ_pb.
                        ufloat_g_t dQ_p = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + p*n_maxcells_b];
                        ufloat_g_t dQ_pb = cells_f_X_b[nbr_kap_b*M_CBLOCK + nbr_kap_c + V_CONN_ID_PB[p]*n_maxcells_b];
                    
                        // Get cell mask in current cell and its neighbor.
                        int mask_p = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
                        int mask_pb = cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c];
                        
                        // Report (DEBUG).
                        /*
                        if ((dQ_p > 0 && mask_pb == V_CELLMASK_BOUNDARY && dQ_pb > 0))
                        {
                            vec3<ufloat_g_t> ray(
                                static_cast<ufloat_g_t>(V_CONN_ID[p+0*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[p+1*27]),
                                static_cast<ufloat_g_t>(0.0)
                            );
                            DebugDraw2DLineSegmentInMATLAB_DEV(vp,vp+ray*(dQ_p*0.75F*dx_L),'r','*','-');
                            //printf("Detected link between boundary and non-solid...\n");
                        }
                        if ((dQ_pb > 0 && mask_p == V_CELLMASK_BOUNDARY && dQ_p > 0))
                        {
                            vec3<ufloat_g_t> ray(
                                static_cast<ufloat_g_t>(V_CONN_ID[p+0*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[p+1*27]),
                                static_cast<ufloat_g_t>(0.0)
                            );
                            DebugDraw2DLineSegmentInMATLAB_DEV(vp,vp+ray*(dQ_p*0.75F*dx_L),'g','*','-');
                            //printf("Detected link between boundary and non-solid...\n");
                        }
                        if ((mask_p == V_CELLMASK_BOUNDARY && dQ_p < 0 && mask_pb == V_CELLMASK_SOLID))
                        {
                            vec3<ufloat_g_t> ray(
                                static_cast<ufloat_g_t>(V_CONN_ID[p+0*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[p+1*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[p+2*27])
                            );
                            DebugDraw2DLineSegmentInMATLAB_DEV(vp,vp+ray*(dQ_p*0.75F*dx_L),'b','*','-');
                            //printf("Detected missing link between solid and fluid...\n");
                        }
                        if ((mask_pb == V_CELLMASK_BOUNDARY && dQ_pb < 0 && mask_p == V_CELLMASK_SOLID))
                        {
                            vec3<ufloat_g_t> ray(
                                static_cast<ufloat_g_t>(V_CONN_ID[p+0*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[p+1*27]),
                                static_cast<ufloat_g_t>(V_CONN_ID[p+2*27])
                            );
                            DebugDraw2DLineSegmentInMATLAB_DEV(vp,vp+ray*(dQ_p*0.75F*dx_L),'c','*','-');
                            //printf("Detected missing link between solid and fluid...\n");
                        }
                        */
                        
                        // Registers for corrections.
                        ufloat_g_t dQ_p_c = static_cast<ufloat_g_t>(-1.0);
                        ufloat_g_t dQ_pb_c = static_cast<ufloat_g_t>(-1.0);
                        
                        // Implement corrections.
                        if (dQ_p > 0 && mask_pb == V_CELLMASK_BOUNDARY && dQ_pb > 0)
                        {
                            dQ_p_c = static_cast<ufloat_g_t>(-1.0);
                            atomicAdd(&correction_counter_d, 1);
                        }
                        if (dQ_pb > 0 && mask_p == V_CELLMASK_BOUNDARY && dQ_p > 0)
                        {
                            dQ_pb_c = static_cast<ufloat_g_t>(-1.0);
                            atomicAdd(&correction_counter_d, 1);
                        }
                        if (mask_p == V_CELLMASK_BOUNDARY && dQ_p < 0 && mask_pb == V_CELLMASK_SOLID)
                        {
                            dQ_p_c = static_cast<ufloat_g_t>(0.5);
                            atomicAdd(&correction_counter_d, 1);
                        }
                        if (mask_pb == V_CELLMASK_BOUNDARY && dQ_pb < 0 && mask_p == V_CELLMASK_SOLID)
                        {
                            dQ_pb_c = static_cast<ufloat_g_t>(0.5);
                            atomicAdd(&correction_counter_d, 1);
                        }
                        
                        // Write corrections.
                        if (dQ_p_c > 0)
                            cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + p*n_maxcells_b] = dQ_p_c;
                        if (dQ_pb_c > 0)
                            cells_f_X_b[nbr_kap_b*M_CBLOCK + nbr_kap_c + V_CONN_ID_PB[p]*n_maxcells_b] = dQ_pb_c;
                    }
                }
            }
        }
    }
}
