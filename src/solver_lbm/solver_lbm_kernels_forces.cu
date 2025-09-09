/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:20 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "custom.h"
#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP, int post_step>
__global__
void Cu_ComputeForcesCV
(
    const int is_root,
    const int n_ids_idev_L,
    const long int n_maxcells,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    const ufloat_t dv_L,
    const ufloat_t otau_0,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    const ufloat_t *__restrict__ cells_f_F,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr_child,
    ufloat_t *__restrict__ cblock_f_Ff,
    const ufloat_t cv_xm,
    const ufloat_t cv_xM,
    const ufloat_t cv_ym,
    const ufloat_t cv_yM,
    const ufloat_t cv_zm,
    const ufloat_t cv_zM
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q = LP->N_Q;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ ufloat_t s_Fp[M_TBLOCK*N_DIM];
    __shared__ ufloat_t s_Fm[M_TBLOCK*N_DIM];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
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
        
        // Latter condition is added only if n>0.
        if (i_kap_b>-1)
        {
            // Compute cell coordinates and retrieve macroscopic properties.
            int valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            int i_kap_bc = cblock_ID_nbr_child[i_kap_b];
            
            // Initialize the shared memory arrays if this block has boundary cells.
            for (int d = 0; d < N_DIM; d++)
            {
                s_Fp[threadIdx.x + d*M_TBLOCK] = 0;
                s_Fm[threadIdx.x + d*M_TBLOCK] = 0;
            }
            __syncthreads();
            
            
            // Load the cell coordinates. Check if the current cell participates.
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            ufloat_t x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
            ufloat_t y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
            ufloat_t z = (ufloat_t)0.0;
            if (N_DIM==3)
            {
                int K = (threadIdx.x / 4) / 4;
                z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
            }
            bool participatesV;
            if (N_DIM==2) participatesV = CheckPointInRegion2D(x,y,cv_xm,cv_xM,cv_ym,cv_yM) && valid_mask != V_CELLMASK_SOLID;
            if (N_DIM==3) participatesV = CheckPointInRegion3D(x,y,z,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM) && valid_mask != V_CELLMASK_SOLID;
            
            // Initialize macroscopic properties.
            ufloat_t rho = (ufloat_t)(0.0);
            ufloat_t rhoup = (ufloat_t)(0.0);
            ufloat_t rhoum = (ufloat_t)(0.0);
            ufloat_t rhovp = (ufloat_t)(0.0);
            ufloat_t rhovm = (ufloat_t)(0.0);
            ufloat_t rhowp = (ufloat_t)(0.0);
            ufloat_t rhowm = (ufloat_t)(0.0);
            ufloat_t f_p[N_Q];
            
            
            // Load the DDFs. Compute the momentum in all cells.
            #pragma unroll
            for (int p = 0; p < N_Q; p++)
            {
                ufloat_t f_pi = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells];
                rho += f_pi;
                if (V_CONN_ID[p+0*27] == 1)  rhoup += f_pi;
                if (V_CONN_ID[p+0*27] == -1) rhoum += f_pi;
                if (V_CONN_ID[p+1*27] == 1)  rhovp += f_pi;
                if (V_CONN_ID[p+1*27] == -1) rhovm += f_pi;
                if (V_CONN_ID[p+2*27] == 1)  rhowp += f_pi;
                if (V_CONN_ID[p+2*27] == -1) rhowm += f_pi;
                f_p[p] = f_pi;
            }
            rhoup = rhoup - rhoum;
            rhovp = rhovp - rhovm;
            rhowp = rhowp - rhowm;
            
            
            
            // Add this cells momentum density if it is inside the control volume.
            if (participatesV && post_step==0 && i_kap_bc<0)
            {
                if (rhoup > 0) s_Fp[threadIdx.x + 0*M_TBLOCK] += rhoup;
                if (rhoup < 0) s_Fm[threadIdx.x + 0*M_TBLOCK] += rhoup;
                if (rhoup > 0) s_Fp[threadIdx.x + 1*M_TBLOCK] += rhovp;
                if (rhoup < 0) s_Fm[threadIdx.x + 1*M_TBLOCK] += rhovp;
                if (rhoup > 0) s_Fp[threadIdx.x + 2*M_TBLOCK] += rhowp;
                if (rhowp < 0) s_Fm[threadIdx.x + 2*M_TBLOCK] += rhowp;
            }
            if (participatesV && post_step==1 && i_kap_bc<0)
            {
                if (rhoup > 0) s_Fm[threadIdx.x + 0*M_TBLOCK] += rhoup;
                if (rhoup < 0) s_Fp[threadIdx.x + 0*M_TBLOCK] += rhoup;
                if (rhoup > 0) s_Fm[threadIdx.x + 1*M_TBLOCK] += rhovp;
                if (rhoup < 0) s_Fp[threadIdx.x + 1*M_TBLOCK] += rhovp;
                if (rhoup > 0) s_Fm[threadIdx.x + 2*M_TBLOCK] += rhowp;
                if (rhowp < 0) s_Fp[threadIdx.x + 2*M_TBLOCK] += rhowp;
            }
            
            
            // Add this DDF's contribution if it leaves/enters the control surface. Perform collision if leaving.
            ufloat_t omeg = otau_0;
            ufloat_t omegp = (ufloat_t)(1.0) - omeg;
            if (post_step==0)
            {
                rhoup = rhoup / rho;
                rhovp = rhovp / rho;
                rhowp = rhowp / rho;
            }
            //#pragma unroll
            for (int p = 0; p < N_Q; p++)
            {
                // Load it first.
                ufloat_t f_pi = f_p[p];
                
                // Perform collision, if leaving.
                if (post_step==0)
                {
                    ufloat_t udotu = rhoup*rhoup + rhovp*rhovp + rhowp*rhowp;
                    ufloat_t cdotu = (ufloat_t)V_CONN_ID[p+0*27]*rhoup + (ufloat_t)V_CONN_ID[p+1*27]*rhovp + (ufloat_t)V_CONN_ID[p+2*27]*rhowp;
                    f_pi = f_pi*omegp + ( (ufloat_t)LBMw[p]*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                }
                
                // Now add contributions.
                if (post_step==0 && is_root)
                {
                    bool participatesS;
                    if (N_DIM==2) participatesS = !CheckPointInRegion2D(x+V_CONN_ID[p+0*27]*dx_L,y+V_CONN_ID[p+1*27]*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                    if (N_DIM==3) participatesS = !CheckPointInRegion3D(x+V_CONN_ID[p+0*27]*dx_L,y+V_CONN_ID[p+1*27]*dx_L,z+V_CONN_ID[p+2*27]*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                    if (participatesS && participatesV)
                    {
                        for (int d = 0; d < N_DIM; d++)
                        {
                            if (V_CONN_ID[p+d*27] == 1)  s_Fm[threadIdx.x+d*M_TBLOCK] += f_pi;
                            if (V_CONN_ID[p+d*27] == -1) s_Fp[threadIdx.x+d*M_TBLOCK] += f_pi;
                        }
                    }
                }
                if (post_step==1 && is_root)
                {
                    bool participatesS;
                    if (N_DIM==2) participatesS = !CheckPointInRegion2D(x+V_CONN_ID[LBMpb[p]+0*27]*dx_L,y+V_CONN_ID[LBMpb[p]+1*27]*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                    if (N_DIM==3) participatesS = !CheckPointInRegion3D(x+V_CONN_ID[LBMpb[p]+0*27]*dx_L,y+V_CONN_ID[LBMpb[p]+1*27]*dx_L,z+V_CONN_ID[LBMpb[p]+2*27]*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                    if (participatesS && participatesV)
                    {
                        for (int d = 0; d < N_DIM; d++)
                        {
                            if (V_CONN_ID[p+d*27] == 1)  s_Fp[threadIdx.x+d*M_TBLOCK] += f_pi;
                            if (V_CONN_ID[p+d*27] == -1) s_Fm[threadIdx.x+d*M_TBLOCK] += f_pi;
                        }
                    }
                }
            }
            
            
            // Reductions for the sums of force contributions in this cell-block.
            __syncthreads();
            for (int s=blockDim.x/2; s>0; s>>=1)
            {
                if (threadIdx.x < s)
                {
                    for (int d = 0; d < N_DIM; d++)
                    {
                        s_Fp[threadIdx.x+d*M_TBLOCK] = s_Fp[threadIdx.x+d*M_TBLOCK] + s_Fp[threadIdx.x+s+d*M_TBLOCK];
                        s_Fm[threadIdx.x+d*M_TBLOCK] = s_Fm[threadIdx.x+d*M_TBLOCK] + s_Fm[threadIdx.x+s+d*M_TBLOCK];
                    }
                }
                __syncthreads();
            }
            // Store the sums of contributions in global memory; this will be reduced further later.
            if (threadIdx.x == 0)
            {
                if (post_step == 0)
                {
                    for (int d = 0; d < N_DIM; d++)
                    {
                        cblock_f_Ff[i_kap_b + (2*d+0)*n_maxcblocks] = s_Fp[0+d*M_TBLOCK]*dv_L;
                        cblock_f_Ff[i_kap_b + (2*d+1)*n_maxcblocks] = s_Fm[0+d*M_TBLOCK]*dv_L;
                    }
                }
                else
                {
                    for (int d = 0; d < N_DIM; d++)
                    {
                        cblock_f_Ff[i_kap_b + (2*d+0)*n_maxcblocks] += s_Fp[0+d*M_TBLOCK]*dv_L;
                        cblock_f_Ff[i_kap_b + (2*d+1)*n_maxcblocks] += s_Fm[0+d*M_TBLOCK]*dv_L;
                    }
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeForcesCV(int i_dev, int L, int var)
{
    if (mesh->n_ids[i_dev][L]>0 && var==0)
    {
        Cu_ComputeForcesCV<ufloat_t,ufloat_g_t,AP,LP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>
        (
            L==N_LEVEL_START,
            mesh->n_ids[i_dev][L],
            n_maxcells,
            n_maxcblocks,
            dxf_vec[L],
            dvf_vec[L],
            dxf_vec[N_LEVEL_START]/tau_vec[N_LEVEL_START],
            &mesh->c_id_set[i_dev][L*n_maxcblocks],
            mesh->c_cells_ID_mask[i_dev],
            mesh->c_cells_f_F[i_dev],
            mesh->c_cblock_f_X[i_dev],
            mesh->c_cblock_ID_nbr_child[i_dev],
            mesh->c_cblock_f_Ff[i_dev],
            S_FORCEVOLUME_Xm,
            S_FORCEVOLUME_XM,
            S_FORCEVOLUME_Ym,
            S_FORCEVOLUME_YM,
            S_FORCEVOLUME_Zm,
            S_FORCEVOLUME_ZM
        );
    }
    if (mesh->n_ids[i_dev][L]>0 && var==1)
    {
        Cu_ComputeForcesCV<ufloat_t,ufloat_g_t,AP,LP,1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>
        (
            L==N_LEVEL_START,
            mesh->n_ids[i_dev][L],
            n_maxcells,
            n_maxcblocks,
            dxf_vec[L],
            dvf_vec[L],
            dxf_vec[N_LEVEL_START]/tau_vec[N_LEVEL_START],
            &mesh->c_id_set[i_dev][L*n_maxcblocks],
            mesh->c_cells_ID_mask[i_dev],
            mesh->c_cells_f_F[i_dev],
            mesh->c_cblock_f_X[i_dev],
            mesh->c_cblock_ID_nbr_child[i_dev],
            mesh->c_cblock_f_Ff[i_dev],
            S_FORCEVOLUME_Xm,
            S_FORCEVOLUME_XM,
            S_FORCEVOLUME_Ym,
            S_FORCEVOLUME_YM,
            S_FORCEVOLUME_Zm,
            S_FORCEVOLUME_ZM
        );
    }

    return 0;
}


















template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP, int post_step>
__global__
void Cu_ComputeForcesMEA
(
    const int n_ids_idev_L,
    const long int n_maxcells,
    const int n_maxcblocks,
    const int n_maxcells_b,
    const int n_maxblocks_b,
    const ufloat_t dx_L,
    const ufloat_t dv_L,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    const ufloat_t *__restrict__ cells_f_F,
    const ufloat_g_t *__restrict__ cells_f_X_b,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_onb,
    const int *__restrict__ cblock_ID_onb_solid,
    ufloat_t *__restrict__ cblock_f_Ff,
    const bool geometry_init,
    const int order
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int N_Q = LP->N_Q;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_nbr[N_Q_max];
    __shared__ ufloat_t s_Fp[M_TBLOCK*N_DIM];
    __shared__ ufloat_t s_Fm[M_TBLOCK*N_DIM];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = 0;
    if (N_DIM==3)
        K = (threadIdx.x / 4) / 4;
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
        
        // Load data for conditions on cell-blocks.
        if (i_kap_b>-1)
        {
            valid_block=cblock_ID_onb[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((valid_block==1)))
        {
            // Compute cell coordinates and retrieve macroscopic properties.
            int valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            int block_mask = -1;
            if (geometry_init)
                block_mask = cblock_ID_onb_solid[i_kap_b];
            
            // Initialize the shared memory arrays if this block has boundary cells.
            if (n_maxblocks_b > 0 && block_mask > -1)
            {
                for (int d = 0; d < N_DIM; d++)
                {
                    s_Fp[threadIdx.x+d*M_TBLOCK] = 0;
                    s_Fm[threadIdx.x+d*M_TBLOCK] = 0;
                }
            }
            
            // Get the indices of neighboring blocks.
            if (threadIdx.x == 0)
            {
                for (int p = 0; p < N_Q_max; p++)
                    s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
            }
            __syncthreads();
            
            
            #pragma unroll
            for (int p = 1; p < N_Q; p++)
            {
                if ( (N_DIM==2 && (p==1||(p+1)%3==0))   ||   (N_DIM==3 && (p==26||((p-1)%2==0&&p<25))) )
                {
                    // Retrieve the DDF. Use correct order of access this time.
                    ufloat_t f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells];
                    ufloat_t f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells];
                    
                    
                    // Find the right neighboring DDFs if the second-order Ginzburg and d'Humieres calculation is being used.
                    if (valid_mask == V_CELLMASK_BOUNDARY)
                    {
                        // Check if DDF p is directed towards the solid object.
                        ufloat_g_t dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + p*n_maxcells_b];
                        if (dist_p > 0)
                        {
                            // Compute incremented local indices.
                            int Ip = I + V_CONN_ID[LBMpb[p]+0*27];
                            int Jp = J + V_CONN_ID[LBMpb[p]+1*27];
                            int Kp = 0;
                            if (N_DIM==3)
                                    Kp = K + V_CONN_ID[LBMpb[p]+2*27];
                            
                            // Assign the correct neighbor cell-block ID.
                            int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                            int nbr_kap_c = Cu_NbrCellId<N_DIM>(Ip,Jp,Kp);
                            
                            // Get DDF from 'behind' and normalize link.
                            ufloat_t f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + p*n_maxcells];
                            dist_p /= dx_L;
                            
                            // Add force contributions.
                            if (order == 1)
                            {
                                for (int d = 0; d < N_DIM; d++)
                                {
                                    if (V_CONN_ID[p+d*27] == 1)  s_Fp[threadIdx.x+d*M_TBLOCK] += f_p;
                                    if (V_CONN_ID[p+d*27] == -1) s_Fm[threadIdx.x+d*M_TBLOCK] += f_p;
                                }
                            }
                            else
                            {
                                for (int d = 0; d < N_DIM; d++)
                                {
                                    if (V_CONN_ID[p+d*27] == 1)  s_Fp[threadIdx.x+d*M_TBLOCK] += (0.5+(ufloat_t)dist_p)*f_p + (0.5-(ufloat_t)dist_p)*f_m;
                                    if (V_CONN_ID[p+d*27] == -1) s_Fm[threadIdx.x+d*M_TBLOCK] += (0.5+(ufloat_t)dist_p)*f_p + (0.5-(ufloat_t)dist_p)*f_m;
                                }
                            }
                        }
 
                        // Check if DDF pb is directed towards the solid object.
                        dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells_b];
                        if (dist_p > 0)
                        {
                            // Compute incremented local indices.
                            int Ip = I + V_CONN_ID[p+0*27];
                            int Jp = J + V_CONN_ID[p+1*27];
                            int Kp = 0;
                            if (N_DIM==3)
                                    Kp = K + V_CONN_ID[p+2*27];
                            
                            // Assign the correct neighbor cell-block ID.
                            int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                            int nbr_kap_c = Cu_NbrCellId<N_DIM>(Ip,Jp,Kp);
                            
                            ufloat_t f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + LBMpb[p]*n_maxcells];
                            dist_p /= dx_L;
                            
                            // Add force contributions.
                            if (order == 1)
                            {
                                for (int d = 0; d < N_DIM; d++)
                                {
                                    if (V_CONN_ID[p+d*27] == 1)  s_Fm[threadIdx.x+d*M_TBLOCK] += f_q;
                                    if (V_CONN_ID[p+d*27] == -1) s_Fp[threadIdx.x+d*M_TBLOCK] += f_q;
                                }
                            }
                            else
                            {
                                for (int d = 0; d < N_DIM; d++)
                                {
                                    if (V_CONN_ID[p+d*27] == 1)  s_Fm[threadIdx.x+d*M_TBLOCK] += (0.5+(ufloat_t)dist_p)*f_q + (0.5-(ufloat_t)dist_p)*f_m;
                                    if (V_CONN_ID[p+d*27] == -1) s_Fp[threadIdx.x+d*M_TBLOCK] += (0.5+(ufloat_t)dist_p)*f_q + (0.5-(ufloat_t)dist_p)*f_m;
                                }
                            }
                        }
                    }
                }
            }
            
            
            // Reductions for the sums of force contributions in this cell-block.
            if (n_maxblocks_b > 0 && block_mask > -1)
            {
                __syncthreads();
                for (int s=blockDim.x/2; s>0; s>>=1)
                {
                    if (threadIdx.x < s)
                    {
                        for (int d = 0; d < N_DIM; d++)
                        {
                            s_Fp[threadIdx.x+d*M_TBLOCK] = s_Fp[threadIdx.x+d*M_TBLOCK] + s_Fp[threadIdx.x+s+d*M_TBLOCK];
                            s_Fm[threadIdx.x+d*M_TBLOCK] = s_Fm[threadIdx.x+d*M_TBLOCK] + s_Fm[threadIdx.x+s+d*M_TBLOCK];
                        }
                    }
                    __syncthreads();
                }
                // Store the sums of contributions in global memory; this will be reduced further later.
                if (threadIdx.x == 0)
                {
                    if (post_step == 0)
                    {
                        for (int d = 0; d < N_DIM; d++)
                        {
                            cblock_f_Ff[i_kap_b + (2*d+0)*n_maxcblocks] = s_Fp[0+d*M_TBLOCK]*dv_L;
                            cblock_f_Ff[i_kap_b + (2*d+1)*n_maxcblocks] = s_Fm[0+d*M_TBLOCK]*dv_L;
                        }
                    }
                    else
                    {
                        for (int d = 0; d < N_DIM; d++)
                        {
                            cblock_f_Ff[i_kap_b + (2*d+0)*n_maxcblocks] += s_Fp[0+d*M_TBLOCK]*dv_L;
                            cblock_f_Ff[i_kap_b + (2*d+1)*n_maxcblocks] += s_Fm[0+d*M_TBLOCK]*dv_L;
                        }
                    }
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeForcesMEA(int i_dev, int L, int var)
{
    if (mesh->n_ids[i_dev][L]>0 && var==0)
    {
        Cu_ComputeForcesMEA<ufloat_t,ufloat_g_t,AP,LP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>
        (
            mesh->n_ids[i_dev][L],
            n_maxcells,
            n_maxcblocks,
            mesh->n_maxcells_b,
            mesh->n_solidb,
            dxf_vec[L],
            dvf_vec[L],
            &mesh->c_id_set[i_dev][L*n_maxcblocks],
            mesh->c_cells_ID_mask[i_dev],
            mesh->c_cells_f_F[i_dev],
            mesh->c_cells_f_X_b[i_dev],
            mesh->c_cblock_f_X[i_dev],
            mesh->c_cblock_ID_nbr[i_dev],
            mesh->c_cblock_ID_mask[i_dev],
            mesh->c_cblock_ID_onb[i_dev],
            mesh->c_cblock_ID_onb_solid[i_dev],
            mesh->c_cblock_f_Ff[i_dev],
            mesh->geometry_init,
            S_FORCE_ORDER
        );
    }
    if (mesh->n_ids[i_dev][L]>0 && var==1)
    {
        Cu_ComputeForcesMEA<ufloat_t,ufloat_g_t,AP,LP,1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>
        (
            mesh->n_ids[i_dev][L],
            n_maxcells,
            n_maxcblocks,
            mesh->n_maxcells_b,
            mesh->n_solidb,
            dxf_vec[L],
            dvf_vec[L],
            &mesh->c_id_set[i_dev][L*n_maxcblocks],
            mesh->c_cells_ID_mask[i_dev],
            mesh->c_cells_f_F[i_dev],
            mesh->c_cells_f_X_b[i_dev],
            mesh->c_cblock_f_X[i_dev],
            mesh->c_cblock_ID_nbr[i_dev],
            mesh->c_cblock_ID_mask[i_dev],
            mesh->c_cblock_ID_onb[i_dev],
            mesh->c_cblock_ID_onb_solid[i_dev],
            mesh->c_cblock_f_Ff[i_dev],
            mesh->geometry_init,
            S_FORCE_ORDER
        );
    }

    return 0;
}






























template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP, int post_step>
__global__
void Cu_ComputePressureOnWall
(
    const int n_ids_idev_L,
    const long int n_maxcells,
    const int n_maxcblocks,
    const int n_maxcells_b,
    const int n_maxblocks_b,
    const ufloat_t dx_L,
    const ufloat_t dv_L,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    const ufloat_t *__restrict__ cells_f_F,
    const ufloat_g_t *__restrict__ cells_f_X_b,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_onb,
    const int *__restrict__ cblock_ID_onb_solid,
    const bool geometry_init,
    const int order
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int M_HBLOCK = AP->M_HBLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int N_Q = LP->N_Q;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_nbr[N_Q_max];
    __shared__ ufloat_t s_F_p[M_HBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = 0;
    if (N_DIM==3)
        K = (threadIdx.x / 4) / 4;
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
        
        // Load data for conditions on cell-blocks.
        if (i_kap_b>-1)
        {
            valid_block=cblock_ID_onb[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((valid_block==1)))
        {
            // Compute cell coordinates and retrieve macroscopic properties.
            int valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            int block_mask = -1;
            if (geometry_init)
                block_mask = cblock_ID_onb_solid[i_kap_b];
            
            // Get the indices of neighboring blocks.
            if (threadIdx.x == 0)
            {
                for (int p = 0; p < N_Q_max; p++)
                    s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
            }
            __syncthreads();
            
            
            // Retrieve cell masks from the current block and from one cell-layer around it from neighboring blocks.
            for (int p = 1; p < N_Q_max; p++)
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
                            s_F_p[nbr_kap_h] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + (N_Q+0)*n_maxcells];
            }
            ufloat_t rho = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            s_F_p[(I+1)+6*(J+1)+(N_DIM-2)*36*(K+1)] = rho;
            __syncthreads();
            
            
            // Compute cell coordinates.
            ufloat_t x __attribute__((unused)) = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + (ufloat_t)0.5*dx_L;
            ufloat_t y __attribute__((unused)) = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + (ufloat_t)0.5*dx_L;
            ufloat_t z __attribute__((unused)) = (ufloat_t)0.0;
            if (N_DIM==3)
                z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + (ufloat_t)0.5*dx_L;
            
            
            #pragma unroll
            for (int p = 1; p < N_Q; p++)
            {
                // Find the right neighboring DDFs if the second-order Ginzburg and d'Humieres calculation is being used.
                if (valid_mask == V_CELLMASK_BOUNDARY)
                {
                    // Check if DDF p is directed towards the solid object.
                    ufloat_g_t dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + p*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        // Compute incremented local indices.
                        int Ip = I + V_CONN_ID[LBMpb[p]+0*27];
                        int Jp = J + V_CONN_ID[LBMpb[p]+1*27];
                        int Kp = 0;
                        if (N_DIM==3)
                                Kp = K + V_CONN_ID[LBMpb[p]+2*27];
                        int nbr_kap_h = (Ip+1) + 6*(Jp+1);
                        if (N_DIM==3)
                            nbr_kap_h += 36*(Kp+1);
                        
                        // Assign the correct neighbor cell-block ID.
                        int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                        
                        // Get DDF from 'behind' and normalize link. Print out the answer.
                        ufloat_t rho_m = s_F_p[nbr_kap_h];
                        dist_p /= dx_L;
                        
                        
                    }

                    // Check if DDF pb is directed towards the solid object.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        // Compute incremented local indices.
                        int Ip = I + V_CONN_ID[p+0*27];
                        int Jp = J + V_CONN_ID[p+1*27];
                        int Kp = 0;
                        if (N_DIM==3)
                                Kp = K + V_CONN_ID[p+2*27];
                        int nbr_kap_h = (Ip+1) + 6*(Jp+1);
                        if (N_DIM==3)
                            nbr_kap_h += 36*(Kp+1);
                        
                        // Assign the correct neighbor cell-block ID.
                        int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                        int nbr_kap_c = Cu_NbrCellId<N_DIM>(Ip,Jp,Kp);
                        
                        ufloat_t f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + LBMpb[p]*n_maxcells];
                        dist_p /= dx_L;
                    }
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputePressureOnWall(int i_dev, int L, int var)
{
    if (mesh->n_ids[i_dev][L]>0)
    {
        Cu_ComputePressureOnWall<ufloat_t,ufloat_g_t,AP,LP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>
        (
            mesh->n_ids[i_dev][L],
            n_maxcells,
            n_maxcblocks,
            mesh->n_maxcells_b,
            mesh->n_solidb,
            dxf_vec[L],
            dvf_vec[L],
            &mesh->c_id_set[i_dev][L*n_maxcblocks],
            mesh->c_cells_ID_mask[i_dev],
            mesh->c_cells_f_F[i_dev],
            mesh->c_cells_f_X_b[i_dev],
            mesh->c_cblock_f_X[i_dev],
            mesh->c_cblock_ID_nbr[i_dev],
            mesh->c_cblock_ID_mask[i_dev],
            mesh->c_cblock_ID_onb[i_dev],
            mesh->c_cblock_ID_onb_solid[i_dev],
            mesh->geometry_init,
            S_FORCE_ORDER
        );
    }

    return 0;
}
