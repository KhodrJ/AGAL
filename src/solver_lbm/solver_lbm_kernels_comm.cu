/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"
#include "util_interp.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP, int interp_type>
__global__
void Cu_Interpolate_Linear
(
    const int n_ids_idev_L,
    const long int n_maxcells,
    const long int n_maxcblocks,
    const ufloat_t tau_ratio,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    ufloat_t *__restrict__ cells_f_F,
    const int *__restrict__ cblock_ID_nbr_child,
    const int *__restrict__ cblock_ID_mask
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int N_Q = LP->N_Q;
    __shared__ ufloat_t s_F[M_TBLOCK];

    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Get the current block's mask and first child Id.
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
                ufloat_t feq_p = static_cast<ufloat_t>(LBMw[p])*rho*(
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
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Interpolate_Linear(int i_dev, int L, int var)
{
    S_RefreshVariables(i_dev, L);
    
    if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_INTERFACE)
    {
        Cu_Interpolate_Linear<ufloat_t,ufloat_g_t,AP,LP,V_INTERP_INTERFACE><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }
    if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_ADDED)
    {
        Cu_Interpolate_Linear<ufloat_t,ufloat_g_t,AP,LP,V_INTERP_ADDED><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }

    return 0;
}


















template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP, int ave_type>
__global__
void Cu_Average
(
    const int n_ids_idev_L,
    const long int n_maxcells,
    const long int n_maxcblocks,
    const ufloat_t tau_ratio,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    ufloat_t *__restrict__ cells_f_F,
    const int *__restrict__ cblock_ID_nbr_child,
    const int *__restrict__ cblock_ID_mask
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int N_Q = LP->N_Q;
    __shared__ ufloat_t s_Fc[M_TBLOCK];
    __shared__ int s_ID_mask_child[M_TBLOCK];

    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Get the current block's mask and first child Id.
        int i_kap_bc = cblock_ID_nbr_child[i_kap_b];
        int block_on_boundary = cblock_ID_mask[i_kap_b];
        
        // Check if:
        // 1) This block is children...
        // 2) AND (averaging over the whole grid OR if the block is on the refinement interface).
        bool b = true;
        {
            bool b1 = i_kap_bc>-1;
            bool b2 = (ave_type==V_AVERAGE_GRID || block_on_boundary==V_BLOCKMASK_INTERFACE);
            b = b1 && b2;
        }
        
        if (b)
        {
            // Compute cell indices, and a helper index for mapping children to parents.
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            int K = 0;
            int child0_IJK = 2*(I%2) + 4*(2*(J%2));
            if (N_DIM==3)
            {
                K = (threadIdx.x / 4) / 4;
                child0_IJK += 4*4*(2*(K%2));
            }
            
            // Traverse children and begin averaging.
            if (N_DIM==2)
            {
                for (int cj = 0; cj < 2; cj++)
                for (int ci = 0; ci < 2; ci++)
                {
                    int c = ci + 2*cj;
                    
                    // Average rescaled fi to parent if applicable.
                    // TODO: Get rid of magic numbers here.
                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+c)*M_CBLOCK + threadIdx.x];
                    if ((ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2))
                        s_ID_mask_child[threadIdx.x] = 1;
                    __syncthreads();
                    
                    // Load macroscopic properties.
                    ufloat_t rho = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + (N_Q+0)*n_maxcells];
                    ufloat_t u = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + (N_Q+1)*n_maxcells];
                    ufloat_t v = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + (N_Q+2)*n_maxcells];
                    ufloat_t udotu = u*u + v*v;
                    
                    for (int p = 0; p < N_Q; p++)
                    {
                        // Load DDF.
                        ufloat_t f_p = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells];
                        
                        // Compute equilibrium distribution and place rescaled DDF in shared memory.
                        ufloat_t cdotu = (
                                            static_cast<ufloat_t>(V_CONN_ID[p+0*27])*u + 
                                            static_cast<ufloat_t>(V_CONN_ID[p+1*27])*v
                        );
                        ufloat_t feq_p = static_cast<ufloat_t>(LBMw[p])*rho*(
                                            static_cast<ufloat_t>(1.0) + 
                                            static_cast<ufloat_t>(3.0)*cdotu + 
                                            static_cast<ufloat_t>(4.5)*cdotu*cdotu - 
                                            static_cast<ufloat_t>(1.5)*udotu
                        );
                        s_Fc[threadIdx.x] = feq_p + (f_p - feq_p)*tau_ratio;
                        __syncthreads();
                        
                        // If these child cells participate in averaging, compute average and store in parent cells in this quadrant.
                        if ((s_ID_mask_child[child0_IJK] == 1) && (I >= 2*ci) && (I < 2+2*ci) && (J >= 2*cj) && (J < 2+2*cj))
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = static_cast<ufloat_t>(0.25)*(
                                s_Fc[(child0_IJK + 0 + 4*0)] + 
                                s_Fc[(child0_IJK + 1 + 4*0)] + 
                                s_Fc[(child0_IJK + 0 + 4*1)] + 
                                s_Fc[(child0_IJK + 1 + 4*1)]
                            );
                        }
                        __syncthreads();
                    }
                }
            }
            else // N_DIM==3
            {
                for (int ck = 0; ck < 2; ck++)
                for (int cj = 0; cj < 2; cj++)
                for (int ci = 0; ci < 2; ci++)
                {
                    int c = ci + 2*cj + 4*ck;
                    
                    // Average rescaled fi to parent if applicable.
                    // TODO: Get rid of magic numbers here.
                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+c)*M_CBLOCK + threadIdx.x];
                    if ((ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2))
                        s_ID_mask_child[threadIdx.x] = 1;
                    __syncthreads();
                    
                    // Load macroscopic properties.
                    ufloat_t rho = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + (N_Q+0)*n_maxcells];
                    ufloat_t u = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + (N_Q+1)*n_maxcells];
                    ufloat_t v = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + (N_Q+2)*n_maxcells];
                    ufloat_t w = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + (N_Q+3)*n_maxcells];
                    ufloat_t udotu = u*u + v*v + w*w;
                    
                    #pragma unroll 1
                    for (int p = 0; p < N_Q; p++)
                    {
                        // Load DDF.
                        ufloat_t f_p = cells_f_F[(i_kap_bc+c)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells];
                        
                        // Compute equilibrium distribution and place rescaled DDF in shared memory.
                        ufloat_t cdotu = (
                                            static_cast<ufloat_t>(V_CONN_ID[p+0*27])*u + 
                                            static_cast<ufloat_t>(V_CONN_ID[p+1*27])*v + 
                                            static_cast<ufloat_t>(V_CONN_ID[p+2*27])*w
                        );
                        ufloat_t feq_p = static_cast<ufloat_t>(LBMw[p])*rho*(
                                            static_cast<ufloat_t>(1.0) + 
                                            static_cast<ufloat_t>(3.0)*cdotu + 
                                            static_cast<ufloat_t>(4.5)*cdotu*cdotu - 
                                            static_cast<ufloat_t>(1.5)*udotu
                        );
                        s_Fc[threadIdx.x] = feq_p + (f_p - feq_p)*tau_ratio;
                        __syncthreads();
                        
                        // If these child cells participate in averaging, compute average and store in parent cells in this quadrant.
                        if ((s_ID_mask_child[child0_IJK] == 1) && (I >= 2*ci) && (I < 2+2*ci) && (J >= 2*cj) && (J < 2+2*cj) && (K >= 2*ck) && (K < 2+2*ck))
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = static_cast<ufloat_t>(0.125)*(
                                s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + 
                                s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + 
                                s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + 
                                s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + 
                                s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + 
                                s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + 
                                s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + 
                                s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)]
                            );
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }
}



template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Average(int i_dev, int L, int var)
{
    S_RefreshVariables(i_dev, L+1);
    
    if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_INTERFACE)
    {
        Cu_Average<ufloat_t,ufloat_g_t,AP,LP,V_AVERAGE_INTERFACE><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_vec[L]/tau_vec[L+1], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }
    if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_BLOCK)
    {
        Cu_Average<ufloat_t,ufloat_g_t,AP,LP,V_AVERAGE_BLOCK><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_vec[L]/tau_vec[L+1], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }
    if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_GRID)
    {
        Cu_Average<ufloat_t,ufloat_g_t,AP,LP,V_AVERAGE_GRID><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_vec[L]/tau_vec[L+1], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }

    return 0;
}



































template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP,  const LBMPack *LP, int interp_type>
__global__
void Cu_Interpolate_Cubic
(
    const int n_ids_idev_L,
    const long int n_maxcells,
    const long int n_maxcblocks,
    const ufloat_t tau_ratio,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    ufloat_t *__restrict__ cells_f_F,
    const int *__restrict__ cblock_ID_nbr_child,
    const int *__restrict__ cblock_ID_mask
)
{
    constexpr ufloat_t M_ONE_TWELFTH = static_cast<ufloat_t>(-0.083333333333333);
    constexpr ufloat_t ONE_SIXTH = static_cast<ufloat_t>(0.166666666666667);
    constexpr ufloat_t TWO_THIRDS = static_cast<ufloat_t>(0.666666666666667);
    constexpr ufloat_t ZERO_THIRDS = static_cast<ufloat_t>(0.0);
    
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int N_Q = LP->N_Q;
    __shared__ ufloat_t s_F[M_TBLOCK];

    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Get the current block's mask and first child Id.
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
            // Compute cell spatial coordinates.
            ufloat_t x = M_ONE_TWELFTH + ONE_SIXTH*(threadIdx.x % 4);
            ufloat_t y = M_ONE_TWELFTH + ONE_SIXTH*((threadIdx.x / 4) % 4);
            ufloat_t z = static_cast<ufloat_t>(0.0);
            if (N_DIM==3)
                z = M_ONE_TWELFTH + ONE_SIXTH*((threadIdx.x / 4) / 4);
            
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
                ufloat_t feq_p = static_cast<ufloat_t>(LBMw[p])*rho*(
                                    static_cast<ufloat_t>(1.0) + 
                                    static_cast<ufloat_t>(3.0)*cdotu + 
                                    static_cast<ufloat_t>(4.5)*cdotu*cdotu - 
                                    static_cast<ufloat_t>(1.5)*udotu
                );
                s_F[threadIdx.x] = feq_p + (f_p - feq_p)*tau_ratio;
                __syncthreads();
                
                if (N_DIM==2)
                {
                    // Declare and initialize intermediate variables.
                    ufloat_t S_res_0 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_1 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_2 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_3 = static_cast<ufloat_t>(0.0);
                    
                    for (int mj = 0; mj < 4; mj++)
                    {
                        // Declare and initialize intermediate variables.
                        ufloat_t S_beta_0 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_beta_1 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_beta_2 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_beta_3 = static_cast<ufloat_t>(0.0);
                        
                        for (int mi = 0; mi < 4; mi++)
                        {
                            // Adding contribution of alpha_{3-mi,3-mj}.
                            //ufloat_t alpha_ij = static_cast<ufloat_t>(0.0);
                            //for (int q = 0; q < 16; q++)
                            //    alpha_ij += V_INTERP[q+16*( (3-mi)+4*(3-mj) )]*s_F[q];
                            ufloat_t alpha_ij = Cu_Interp2D( (3-mi)+4*(3-mj), s_F );
                            
                            // Update intermediate values.
                            S_beta_0 = alpha_ij + (x+ZERO_THIRDS)*S_beta_0;
                            S_beta_1 = alpha_ij + (x+TWO_THIRDS)*S_beta_1;
                            S_beta_2 = alpha_ij + (x+ZERO_THIRDS)*S_beta_2;
                            S_beta_3 = alpha_ij + (x+TWO_THIRDS)*S_beta_3;
                        }
                        
                        // Update intermediate values.
                        S_res_0 = S_beta_0 + (y+ZERO_THIRDS)*S_res_0;
                        S_res_1 = S_beta_1 + (y+ZERO_THIRDS)*S_res_1;
                        S_res_2 = S_beta_2 + (y+TWO_THIRDS)*S_res_2;
                        S_res_3 = S_beta_3 + (y+TWO_THIRDS)*S_res_3;
                    }
                    
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_0;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_1;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_2;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_3;
                    __syncthreads();
                }
                else // N_DIM==3
                {
                    // Declare and initialize intermediate variables.
                    ufloat_t S_res_0 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_1 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_2 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_3 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_4 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_5 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_6 = static_cast<ufloat_t>(0.0);
                    ufloat_t S_res_7 = static_cast<ufloat_t>(0.0);
                    
                    for (int mk = 0; mk < 4; mk++)
                    {
                        // Declare and initialize intermediate variables.
                        ufloat_t S_gamma_0 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_gamma_1 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_gamma_2 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_gamma_3 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_gamma_4 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_gamma_5 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_gamma_6 = static_cast<ufloat_t>(0.0);
                        ufloat_t S_gamma_7 = static_cast<ufloat_t>(0.0);
                        
                        for (int mj = 0; mj < 4; mj++)
                        {
                            // Declare and initialize intermediate variables.
                            ufloat_t S_beta_0 = static_cast<ufloat_t>(0.0);
                            ufloat_t S_beta_1 = static_cast<ufloat_t>(0.0);
                            ufloat_t S_beta_2 = static_cast<ufloat_t>(0.0);
                            ufloat_t S_beta_3 = static_cast<ufloat_t>(0.0);
                            ufloat_t S_beta_4 = static_cast<ufloat_t>(0.0);
                            ufloat_t S_beta_5 = static_cast<ufloat_t>(0.0);
                            ufloat_t S_beta_6 = static_cast<ufloat_t>(0.0);
                            ufloat_t S_beta_7 = static_cast<ufloat_t>(0.0);
                            
                            for (int mi = 0; mi < 4; mi++)
                            {
                                // Adding contribution of alpha_{3-mi,3-mj,3-mk}.
                                //ufloat_t alpha_ijk = static_cast<ufloat_t>(0.0);
                                //for (int q = 0; q < 64; q++)
                                //    alpha_ijk += V_INTERP[q+64*( (3-mi)+4*(3-mj)+16*(3-mk) )]*s_F[q];
                                ufloat_t alpha_ijk = Cu_Interp3D( (3-mi)+4*(3-mj)+16*(3-mk), s_F );
                                
                                // Update intermediate values.
                                S_beta_0 = alpha_ijk + (x+ZERO_THIRDS)*S_beta_0;
                                S_beta_1 = alpha_ijk + (x+TWO_THIRDS)*S_beta_1;
                                S_beta_2 = alpha_ijk + (x+ZERO_THIRDS)*S_beta_2;
                                S_beta_3 = alpha_ijk + (x+TWO_THIRDS)*S_beta_3;
                                S_beta_4 = alpha_ijk + (x+ZERO_THIRDS)*S_beta_4;
                                S_beta_5 = alpha_ijk + (x+TWO_THIRDS)*S_beta_5;
                                S_beta_6 = alpha_ijk + (x+ZERO_THIRDS)*S_beta_6;
                                S_beta_7 = alpha_ijk + (x+TWO_THIRDS)*S_beta_7;
                            }
                            
                            // Update intermediate values.
                            S_gamma_0 = S_beta_0 + (y+ZERO_THIRDS)*S_gamma_0;
                            S_gamma_1 = S_beta_1 + (y+ZERO_THIRDS)*S_gamma_1;
                            S_gamma_2 = S_beta_2 + (y+TWO_THIRDS)*S_gamma_2;
                            S_gamma_3 = S_beta_3 + (y+TWO_THIRDS)*S_gamma_3;
                            S_gamma_4 = S_beta_4 + (y+ZERO_THIRDS)*S_gamma_4;
                            S_gamma_5 = S_beta_5 + (y+ZERO_THIRDS)*S_gamma_5;
                            S_gamma_6 = S_beta_6 + (y+TWO_THIRDS)*S_gamma_6;
                            S_gamma_7 = S_beta_7 + (y+TWO_THIRDS)*S_gamma_7;
                        }
                        
                        // Update intermediate values.
                        S_res_0 = S_gamma_0 + (z+ZERO_THIRDS)*S_res_0;
                        S_res_1 = S_gamma_1 + (z+ZERO_THIRDS)*S_res_1;
                        S_res_2 = S_gamma_2 + (z+ZERO_THIRDS)*S_res_2;
                        S_res_3 = S_gamma_3 + (z+ZERO_THIRDS)*S_res_3;
                        S_res_4 = S_gamma_4 + (z+TWO_THIRDS)*S_res_4;
                        S_res_5 = S_gamma_5 + (z+TWO_THIRDS)*S_res_5;
                        S_res_6 = S_gamma_6 + (z+TWO_THIRDS)*S_res_6;
                        S_res_7 = S_gamma_7 + (z+TWO_THIRDS)*S_res_7;
                    }
                    
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_0;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_1;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_2;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_3;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_4;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_5;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_6;
                    if ((interp_type==V_INTERP_INTERFACE && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==V_CELLMASK_GHOST) || (interp_type==V_INTERP_ADDED))
                        cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = S_res_7;
                    __syncthreads();
                }
            }
        }
    }
}




template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Interpolate_Cubic(int i_dev, int L, int var)
{
    S_RefreshVariables(i_dev, L);
    
    if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_INTERFACE)
    {
        Cu_Interpolate_Cubic<ufloat_t,ufloat_g_t,AP,LP,V_INTERP_INTERFACE><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }
    if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_ADDED)
    {
        Cu_Interpolate_Cubic<ufloat_t,ufloat_g_t,AP,LP,V_INTERP_ADDED><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
    }
    
    return 0;
}

