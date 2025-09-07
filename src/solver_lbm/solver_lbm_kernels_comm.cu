/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, int interp_type>
__global__
void Cu_Interpolate_Linear
(
    const int n_ids_idev_L,
    const long int n_maxcells,
    const int n_maxcblocks,
    const ufloat_t tau_ratio,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    ufloat_t *__restrict__ cells_f_F,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr_child,
    const int *__restrict__ cblock_ID_mask
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int N_Q = LP->N_Q;
    __shared__ int s_F[M_TBLOCK];

    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        int i_kap_bc = cblock_ID_nbr_child[i_kap_b];
        int block_on_boundary = cblock_ID_mask[i_kap_b];
        
        // Check if:
        // 1) the block is on the interface (and interpolating on interfaces)...
        // 2) OR interpolating after mesh adaptation and marked for refinement.
        bool b = true;
        {
            bool b1 = (interp_type==V_INTERP_INTERFACE) && (block_on_boundary==V_BLOCKMASK_INTERFACE);
            bool b2 = (interp_type==V_INTERP_ADDED) && (cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE);
            b = b1 || b2;
        }
        
        if (b)
        {
            // Compute cell indices.
            ufloat_t Ix = static_cast<ufloat_t>(-0.25) + static_cast<ufloat_t>(0.5)*( threadIdx.x % 4 );
            ufloat_t Iy = static_cast<ufloat_t>(-0.25) + static_cast<ufloat_t>(0.5)*( (threadIdx.x / 4) % 4 );
            ufloat_t Iz = static_cast<ufloat_t>(0.0);
            if (N_DIM==3)
                Iz = static_cast<ufloat_t>(-0.25) + static_cast<ufloat_t>(0.5)*( (threadIdx.x / 4) / 4 );
            
            // Load macroscopic properties.
            ufloat_t rho = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+0)*n_maxcells];
            ufloat_t u = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+1)*n_maxcells];
            ufloat_t v = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+2)*n_maxcells];
            ufloat_t w = static_cast<ufloat_t>(0.0);
            if (N_DIM==3)
                w = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+3)*n_maxcells];
            ufloat_t udotu = u*u + v*v + w*w;
            
            // Loop over DDFs and interpolate.
            for (int p = 0; p < N_Q; p++)
            {
                // Load DDF.
                ufloat_t f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells];
                
                // Compute equilibrium distribution and place rescaled DDF in shared memory.
                ufloat_t cdotu = (
                                    static_cast<ufloat_t>(V_CONN_ID[p+0*27])*u + 
                                    static_cast<ufloat_t>(V_CONN_ID[p+1*27])*v + 
                                    static_cast<ufloat_t>(V_CONN_ID[p+2*27])*w
                );
                ufloat_t feq_p = LBMw[p]*rho*(
                                    static_cast<ufloat_t>(1.0) + 
                                    static_cast<ufloat_t>(3.0)*cdotu + 
                                    static_cast<ufloat_t>(4.5)*cdotu*cdotu - 
                                    static_cast<ufloat_t>(1.5)*udotu
                );
                s_F[threadIdx.x] = feq_p + (f_p - feq_p)*tau_ratio;
                __syncthreads();
                
                // Loop over child blocks and fill child data.
                if (N_DIM==2)
                {
                    for (int cj = 0; cj < 2; cj++)
                    for (int ci = 0; ci < 2; ci++)
                    {
                        int c = ci + 2*cj;       // Child index.
                        int s = 2*ci + 2*4*cj;   // Shift in shared memory array.
                        if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+c)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type == 1))
                        {
                            cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = 
                                s_F[0+s] + 
                                (s_F[1+s]-s_F[0+s])*Ix + 
                                (s_F[4+s]-s_F[0+s])*Iy + 
                                (s_F[5+s]-s_F[4+s]-s_F[1+s]+s_F[0+s])*Ix*Iy;
                        }
                    }
                }
                else
                {
                    for (int ck = 0; ck < 2; ck++)
                    for (int cj = 0; cj < 2; cj++)
                    for (int ci = 0; ci < 2; ci++)
                    {
                        int c = ci + 2*cj + 4*ck;           // Child index.
                        int s = 2*ci + 2*4*cj + 2*4*4*ck;   // Shift in shared memory array.
                        if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+c)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type == 1))
                        {
                            cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = 
                                s_F[0+s] + 
                                (s_F[1+s] - s_F[0+s])*Ix + 
                                (s_F[4+s] - s_F[0+s])*Iy + 
                                (s_F[16+s] - s_F[0+s])*Iz + 
                                (s_F[5+s] - s_F[1+s] - s_F[4+s] + s_F[0+s])*Ix*Iy + 
                                (s_F[17+s] - s_F[1+s] - s_F[16+s] + s_F[0+s])*Ix*Iz + 
                                (s_F[20+s] - s_F[4+s] - s_F[16+s] + s_F[0+s])*Iy*Iz + 
                                (s_F[21+s] + s_F[1+s] + s_F[4+s] + s_F[16+s] - s_F[5+s] - s_F[17+s] - s_F[20+s] - s_F[0+s])*Ix*Iy*Iz;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Interpolate_Linear(int i_dev, int L)
{
if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_INTERFACE)
    {
        Cu_Interpolate_Linear<ufloat_t,ufloat_g_t,AP,V_INTERP_INTERFACE><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }
    if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_ADDED)
    {
        Cu_Interpolate_Linear<ufloat_t,ufloat_g_t,AP,V_INTERP_ADDED><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }

    return 0;
}


































