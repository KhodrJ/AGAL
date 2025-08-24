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
void Cu_IdentifyFaces_V2
(
    const int n_ids_idev_L,
    const long int n_maxcells,
    const int n_maxcblocks,
    const int n_maxcells_b,
    const int n_maxblocks_b,
    const ufloat_t dx_L,
    const int *__restrict__ id_set_idev_L,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cells_ID_mask_b,
    ufloat_g_t *__restrict__ cells_f_X_b,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_ID_mask,
    int *__restrict__ cblock_ID_onb_solid,
    int *__restrict__ cblock_ID_face,
    const int n_faces,
    const int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    const int *__restrict__ binned_face_ids_n,
    const int *__restrict__ binned_face_ids_N,
    const int *__restrict__ binned_face_ids,
    const int n_bin_density
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    ufloat_g_t dQ[N_Q_max];
    
    s_ID_cblock[threadIdx.x] = -1;
    if ((threadIdx.x<M_LBLOCK)and(kap<n_ids_idev_L))
    {
        s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
    }
    __syncthreads();
    
    // Loop over block Ids.
    for (int k = 0; k < M_LBLOCK; k += 1)
    {
        int i_kap_b = s_ID_cblock[k];
        int valid_block = -1;

        // Needed for condition.
        if (i_kap_b>-1)
        {
            valid_block = cblock_ID_onb_solid[i_kap_b];
        }
        
        // Proceed only if the current cell-block is on a fluid-solid boundary.
        if ((i_kap_b>-1)&&(valid_block>-1))
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
                    vec3<ufloat_g_t> v1
                    (
                        geom_f_face_X[f_p + 0*n_faces_a],
                        geom_f_face_X[f_p + 1*n_faces_a],
                        geom_f_face_X[f_p + 2*n_faces_a]
                    );
                    vec3<ufloat_g_t> v2
                    (
                        geom_f_face_X[f_p + 3*n_faces_a],
                        geom_f_face_X[f_p + 4*n_faces_a],
                        geom_f_face_X[f_p + 5*n_faces_a]
                    );
                    vec3<ufloat_g_t> v3
                    (
                        geom_f_face_X[f_p + 6*n_faces_a],
                        geom_f_face_X[f_p + 7*n_faces_a],
                        geom_f_face_X[f_p + 8*n_faces_a]
                    );
                    vec3<ufloat_g_t> n = FaceNormalUnit<ufloat_g_t,N_DIM>(v1,v2,v3);
                    
                    if (N_DIM==2)
                    {
                        // Loop over all directions, update links for this face if applicable.
                        for (int q = 1; q < N_Q_max; q++)
                        {
                            ufloat_g_t tmp = ((v1.x-vp.x)*n.x + (v1.y-vp.y)*n.y) / ((ufloat_g_t)V_CONN_ID[q+0*27]*n.x + (ufloat_g_t)V_CONN_ID[q+1*27]*n.y);
                            if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
                            {
                                ufloat_g_t tmpy = vp.y + V_CONN_ID[q+1*27]*tmp;
                                ufloat_g_t tmpx = vp.x + V_CONN_ID[q+0*27]*tmp;
                                
                                if (CheckPointInLine(tmpx, tmpy, v1.x, v1.y, v2.x, v2.y))
                                {
                                    ufloat_g_t dist_q = dQ[q];
                                    if (tmp < dist_q || dist_q < 0)
                                        dQ[q] = tmp;
                                }
                            }
                        }
                    }
                    
                    if (N_DIM==3)
                    {
                        for (int q = 1; q < N_Q_max; q++)
                        {
                            ufloat_g_t tmp = ((v1.x-vp.x)*n.x + (v1.y-vp.y)*n.y + (v1.z-vp.z)*n.z) / ((ufloat_g_t)V_CONN_ID[q+0*27]*n.x + (ufloat_g_t)V_CONN_ID[q+1*27]*n.y + (ufloat_g_t)V_CONN_ID[q+2*27]*n.z);
                            if (tmp > (ufloat_g_t)0.0 && tmp < dx_L + (ufloat_g_t)1e-5)
                            {
                                vec3<ufloat_g_t> tmpX
                                (
                                    vp.x + V_CONN_ID[q+0*27]*tmp,
                                    vp.y + V_CONN_ID[q+1*27]*tmp,
                                    vp.z + V_CONN_ID[q+2*27]*tmp
                                );
                                if (CheckPointInTriangleA(tmpX, v1, v2, v3, n))
                                {
                                    ufloat_g_t dist_q = dQ[q];
                                    if (tmp < dist_q || dist_q < 0)
                                        dQ[q] = tmp;
                                }
                            }
                        }
                    }
                }
                
                // Write face-cell links to global memory.
                for (int q = 1; q < N_Q_max; q++)
                {
                    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + q*n_maxcells_b] = -8; // This will be updated to read actual boundary Ids later.
                    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + q*n_maxcells_b] = dQ[q];
                    //if (dQ[q] > 0)
                    //    printf("%17.5f\n", dQ[q]/dx_L);
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
        Cu_IdentifyFaces_V2<ufloat_t,ufloat_g_t,AP,LP->VS> <<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(
            mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, mesh->n_solidb, dxf_vec[L],
            &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_ID_mask_b[i_dev], mesh->c_cells_f_X_b[i_dev],
            mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->c_cblock_ID_face[i_dev],
            mesh->geometry->n_faces, mesh->geometry->n_faces_a, mesh->geometry->c_geom_f_face_X,
            mesh->geometry->bins->c_binned_face_ids_n_3D[0], mesh->geometry->bins->c_binned_face_ids_N_3D[0], mesh->geometry->bins->c_binned_face_ids_3D[0], mesh->geometry->bins->n_bin_density[0]
        );
    }
    
    return 0;
}
