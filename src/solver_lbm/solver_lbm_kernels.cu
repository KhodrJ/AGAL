/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Mon Jul 14 23:00:24 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_Collision
(
	const int n_ids_idev_L,
	const long int n_maxcells,
	const int n_maxcblocks,
	const ufloat_t dx_L,
	const ufloat_t tau_L,
	const int *__restrict__ id_set_idev_L,
	const int *__restrict__ cells_ID_mask,
	ufloat_t *__restrict__ cells_f_F,
	const ufloat_t *__restrict__ cblock_f_X,
	const int *__restrict__ cblock_ID_nbr,
	const int *__restrict__ cblock_ID_nbr_child,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_onb
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q = LP->VS;
    __shared__ int s_ID_cblock[M_TBLOCK];
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
        int i_kap_bc = -1;
        int valid_block = -1;
        
        // Load data for conditions on cell-blocks.
        if (i_kap_b>-1)
        {
            i_kap_bc=cblock_ID_nbr_child[i_kap_b];
            valid_block=cblock_ID_mask[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(valid_block>-3)))
        {
            int valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            ufloat_t rho = (ufloat_t)0.0;
            ufloat_t u = (ufloat_t)0.0;
            ufloat_t v = (ufloat_t)0.0;
            ufloat_t w = (ufloat_t)0.0;
            #pragma unroll
            for (int p = 0; p < N_Q; p++)
            {
                ufloat_t f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells];
                rho += f_p;
                u += V_CONN_ID[p+0*27]*f_p;
                v += V_CONN_ID[p+1*27]*f_p;
                w += V_CONN_ID[p+2*27]*f_p;
            }
            ufloat_t udotu = u*u + v*v + w*w;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+0)*n_maxcells] = rho;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+1)*n_maxcells] = u;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+2)*n_maxcells] = v;
            if (N_DIM==3)
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+3)*n_maxcells] = w;
            
            ufloat_t omeg = dx_L / tau_L;
            ufloat_t omegp = (ufloat_t)(1.0) - omeg;
            #pragma unroll
            for (int p = 0; p < N_Q; p++)
            {
                ufloat_t cdotu = V_CONN_ID[p+0*27]*u + V_CONN_ID[p+1*27]*v + V_CONN_ID[p+2*27]*w;
                ufloat_t f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells]*omegp + ( LBMwp[p]*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                if (valid_mask != -1)
                    cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = f_p;
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Cu_Collision<ufloat_t,ufloat_g_t,AP,LP>::S_Collision(int i_dev, int L)
{
    if (mesh->n_ids[i_dev][L] > 0)
    {
        Cu_Collision_New_D2Q9<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(
            mesh->n_ids[i_dev][L], n_maxcells,
            n_maxcblocks,
            dxf_vec[L],
            tau_vec[L],
            &mesh->c_id_set[i_dev][L*n_maxcblocks],
            mesh->c_cells_ID_mask[i_dev],
            mesh->c_cells_f_F[i_dev],
            mesh->c_cblock_f_X[i_dev],
            mesh->c_cblock_ID_nbr[i_dev],
            mesh->c_cblock_ID_nbr_child[i_dev],
            mesh->c_cblock_ID_mask[i_dev],
            mesh->c_cblock_ID_onb[i_dev]
        );
    }

    return 0;
}













/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:18 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_Stream_Original_D2Q9
(
	const int n_ids_idev_L,
	const long int n_maxcells,
	const int n_maxcblocks,
	const int n_maxcells_b,
	const int *__restrict__ id_set_idev_L,
	const int *__restrict__ cells_ID_mask,
	ufloat_t *__restrict__ cells_f_F,
	const ufloat_g_t *__restrict__ cells_f_X_b,
	const int *__restrict__ cblock_ID_nbr,
	const int *__restrict__ cblock_ID_nbr_child,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_onb_solid,
	const bool geometry_init
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q = LP->VS;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_nbr[27];
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
        int i_kap_bc = -1;
        int valid_block = -1;
        
        // Load data for conditions on cell-blocks.
        if (i_kap_b>-1)
        {
            i_kap_bc=cblock_ID_nbr_child[i_kap_b];
            valid_block=cblock_ID_mask[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(valid_block>-3)))
        {
            // Get masks for possible boundary nodes.
            int valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            if (geometry_init && valid_mask == -2)
                valid_block = cblock_ID_onb_solid[i_kap_b];
         
            // Load neighbor-block indices into shared memory.
            if (threadIdx.x==0)
            {
                for (int p = 0; p < N_Q_max; p++)
                {
                    if (V_CONN_ID[p+0*27] >= 0 && V_CONN_ID[p+1*27] >= 0 && V_CONN_ID[p+2*27] >= 0)
                        s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
                }
            }
            __syncthreads();
            
            // Loop over DDFs and perform streaming step.
            #pragma unroll
            for (int p = 1; p < N_Q; p++)
            {
                if ( (N_DIM==2 && (p==1||(p+1)%3==0))   ||   (N_DIM==3 && (p==26||((p-1)%2==0&&p<25))) )
                {
                    // Load current DDF. Get the face-cell link, if applicable.
                    ufloat_t f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells];
                    ufloat_g_t dQ = (ufloat_g_t)(-1.0);
                    if (valid_mask == V_CELLMASK_BOUNDARY)
                        dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + p*n_maxcells_b];
                    
                    // Compute incremented local indices.
                    int Ip = I + V_CONN_ID[p + 0*27];
                    int Jp = J + V_CONN_ID[p + 1*27];
                    int Kp = 0;
                    if (N_DIM==3)
                            Kp = K + V_CONN_ID[p + 2*27];
                    
                    // Assign the correct neighbor cell-block ID.
                    int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                    Ip = (4 + (Ip % 4)) % 4;
                    Jp = (4 + (Jp % 4)) % 4;
                    if (N_DIM==3)
                            Kp = (4 + (Kp % 4)) % 4;
                    int nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    
                    // Retrieve neighboring DDFs, if applicable.
                    ufloat_t f_pb = (ufloat_t)(-1.0);
                    if ( nbr_kap_b>=0 )
                        f_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + LBMpb[p]*n_maxcells];
                    
                    // Exchange, if applicable.
                    if ( valid_mask != -1 && f_pb>=0 && dQ < 0)
                    {
                        cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + LBMpb[p]*n_maxcells] = f_p;
                        cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = f_pb;
                    }
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Stream_Original_D2Q9(int i_dev, int L)
{
    if (mesh->n_ids[i_dev][L] > 0)
    {
        Cu_Stream_Original_D2Q9<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(
            mesh->n_ids[i_dev][L],
            n_maxcells,
            n_maxcblocks,
            mesh->n_maxcells_b,
            &mesh->c_id_set[i_dev][L*n_maxcblocks],
            mesh->c_cells_ID_mask[i_dev],
            mesh->c_cells_f_F[i_dev],
            mesh->c_cells_f_X_b[i_dev],
            mesh->c_cblock_ID_nbr[i_dev],
            mesh->c_cblock_ID_nbr_child[i_dev],
            mesh->c_cblock_ID_mask[i_dev],
            mesh->c_cblock_ID_onb_solid[i_dev],
            mesh->geometry_init
        );
    }

    return 0;
}






















/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_ImposeBC_D3Q19
(
	const int n_ids_idev_L,
	const long int n_maxcells,
	const int n_maxcblocks,
	const int n_maxcells_b,
	const int n_maxblocks_b,
	const ufloat_t dx_L,
	const ufloat_t dx_L_g,
	const ufloat_t tau_L,
	const int *__restrict__ id_set_idev_L,
	const int *__restrict__ cells_ID_mask,
	ufloat_t *__restrict__ cells_f_F,
	const ufloat_g_t *__restrict__ cells_f_X_b,
	const ufloat_t *__restrict__ cblock_f_X,
	const int *__restrict__ cblock_ID_nbr,
	const int *__restrict__ cblock_ID_nbr_child,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_onb,
	const int *__restrict__ cblock_ID_onb_solid,
	const bool geometry_init,
	const int force_type,
	const int bc_type
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int N_Q = LP->VS;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_nbr[N_Q_max];
    __shared__ ufloat_t s_u[3*M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = (threadIdx.x / 4) / 4;
    ufloat_g_t dQ = (ufloat_g_t)(0.0);
    ufloat_t cdotu = (ufloat_t)(0.0);
    ufloat_t udotu __attribute__((unused)) = (ufloat_t)(0.0);
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
            ufloat_t x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
            ufloat_t y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
            ufloat_t z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
            
            // Get macroscopic properties and place them in shared memory.
            ufloat_t rho = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+0)*n_maxcells];
            ufloat_t u = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+1)*n_maxcells];
            ufloat_t v = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+2)*n_maxcells];
            ufloat_t w;
            if (N_DIM==3)
                w = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+3)*n_maxcells];
            s_u[threadIdx.x + 0*M_TBLOCK] = u;
            s_u[threadIdx.x + 1*M_TBLOCK] = v;
            s_u[threadIdx.x + 2*M_TBLOCK] = w;
            
            // Retrieve indices of neighboring blocks.
            if (threadIdx.x==0)
            {
                for (int p = 0; p < N_Q_max; p++)
                    s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
            }
            __syncthreads()
            
            // Loop over pairs of DDFs and impose BC.
            #pragma unroll
            for (int p = 1; p < N_Q; p++)
            {
                if ( (N_DIM==2 && (p==1||(p+1)%3==0))   ||   (N_DIM==3 && (p==26||((p-1)%2==0&&p<25))) )
                {
                    // Retrieve the DDF. Use correct order of access this time.
                    ufloat_t f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells];
                    ufloat_t f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells];
                    ufloat_t f_m = -1;
                    
                    // Compute incremented local indices.
                    int Ip = I + V_CONN_ID[p + 0*27];
                    int Jp = J + V_CONN_ID[p + 1*27];
                    int Kp = 0;
                    if (N_DIM==3)
                            Kp = K + V_CONN_ID[p + 2*27];
                    
                    //
                    // BC of p.
                    //
                    
                    // Identify the correct neighboring block and cell.
                    int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                    Ip = (4 + (Ip % 4)) % 4;
                    Jp = (4 + (Jp % 4)) % 4;
                    if (N_DIM==3)
                            Kp = (4 + (Kp % 4)) % 4;
                    int nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    
                    // Impose default BC.
                    if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                        Cu_ImposeBC(nbr_kap_b
                            
                            int nbr_id, ufloat_t f, ufloat_t rho, ufloat_t u, ufloat_t v, ufloat_t w, ufloat_t x, ufloat_t y, ufloat_t z,
	ufloat_t wp, ufloat_t cxp, ufloat_t cyp, ufloat_t czp
                }
            }
        }
    }
}


































//             Compute cell coordinates and retrieve macroscopic properties.
//             valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
//             if (geometry_init)
//                 block_mask = cblock_ID_onb_solid[i_kap_b];
//             x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
//             y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
//             z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
//             rho = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
//             u = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells]; ub = u;
//             v = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells]; vb = v;
//             w = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells]; wb = w;
//             s_u[threadIdx.x + 0*M_TBLOCK] = u;
//             s_u[threadIdx.x + 1*M_TBLOCK] = v;
//             s_u[threadIdx.x + 2*M_TBLOCK] = w;
//             __syncthreads();
//             if (valid_block == 1 && threadIdx.x == 0)
//             {
//                 s_ID_nbr[1] = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
//                 s_ID_nbr[2] = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
//                 s_ID_nbr[3] = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
//                 s_ID_nbr[4] = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
//                 s_ID_nbr[5] = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
//                 s_ID_nbr[6] = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
//                 s_ID_nbr[7] = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
//                 s_ID_nbr[8] = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
//                 s_ID_nbr[9] = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
//                 s_ID_nbr[10] = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
//                 s_ID_nbr[11] = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
//                 s_ID_nbr[12] = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
//                 s_ID_nbr[13] = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
//                 s_ID_nbr[14] = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
//                 s_ID_nbr[15] = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
//                 s_ID_nbr[16] = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
//                 s_ID_nbr[17] = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
//                 s_ID_nbr[18] = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
//                 s_ID_nbr[19] = cblock_ID_nbr[i_kap_b + 19*n_maxcblocks];
//                 s_ID_nbr[20] = cblock_ID_nbr[i_kap_b + 20*n_maxcblocks];
//                 s_ID_nbr[21] = cblock_ID_nbr[i_kap_b + 21*n_maxcblocks];
//                 s_ID_nbr[22] = cblock_ID_nbr[i_kap_b + 22*n_maxcblocks];
//                 s_ID_nbr[23] = cblock_ID_nbr[i_kap_b + 23*n_maxcblocks];
//                 s_ID_nbr[24] = cblock_ID_nbr[i_kap_b + 24*n_maxcblocks];
//                 s_ID_nbr[25] = cblock_ID_nbr[i_kap_b + 25*n_maxcblocks];
//                 s_ID_nbr[26] = cblock_ID_nbr[i_kap_b + 26*n_maxcblocks];
//             }
//             __syncthreads();
            
            
            //
            // p = 1
            //
            
            // Retrieve the DDF. Use correct order of access this time.
//             f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
//             f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 1;
            Jp = J + 0;
            Kp = K + 0;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[1];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/18.0), (ufloat_t)(1), (ufloat_t)(0), (ufloat_t)(0), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + -1;
            Jp = J + 0;
            Kp = K + 0;
            // Consider nbr 2.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[2];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/18.0), (ufloat_t)(-1), (ufloat_t)(0), (ufloat_t)(0), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 1 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 1*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + -1;
                    Jp = J + 0;
                    Kp = K + 0;
                    // Consider nbr 2.
                    if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[2];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 1*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 2 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 2*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 1;
                    Jp = J + 0;
                    Kp = K + 0;
                    // Consider nbr 1.
                    if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[1];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 2*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_q;
            }
            
            //
            // p = 3
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + 1;
            Kp = K + 0;
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[3];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/18.0), (ufloat_t)(0), (ufloat_t)(1), (ufloat_t)(0), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + -1;
            Kp = K + 0;
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[4];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/18.0), (ufloat_t)(0), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 3 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 3*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 0;
                    Jp = J + -1;
                    Kp = K + 0;
                    // Consider nbr 4.
                    if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[4];
                    // Get the fluid node behind this boundary node.
                    Jp = (4 + (Jp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 3*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 4 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 4*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 0;
                    Jp = J + 1;
                    Kp = K + 0;
                    // Consider nbr 3.
                    if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[3];
                    // Get the fluid node behind this boundary node.
                    Jp = (4 + (Jp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 4*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_q;
            }
            
            //
            // p = 5
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + 0;
            Kp = K + 1;
            // Consider nbr 5.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[5];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/18.0), (ufloat_t)(0), (ufloat_t)(0), (ufloat_t)(1), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + 0;
            Kp = K + -1;
            // Consider nbr 6.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[6];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/18.0), (ufloat_t)(0), (ufloat_t)(0), (ufloat_t)(-1), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 5 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 5*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 0;
                    Jp = J + 0;
                    Kp = K + -1;
                    // Consider nbr 6.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[6];
                    // Get the fluid node behind this boundary node.
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 5*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 6 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 6*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 0;
                    Jp = J + 0;
                    Kp = K + 1;
                    // Consider nbr 5.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[5];
                    // Get the fluid node behind this boundary node.
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 6*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_q;
            }
            
            //
            // p = 7
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 1;
            Jp = J + 1;
            Kp = K + 0;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[1];
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[3];
            // Consider nbr 7.
            if ( (Ip==4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[7];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(1), (ufloat_t)(1), (ufloat_t)(0), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + -1;
            Jp = J + -1;
            Kp = K + 0;
            // Consider nbr 2.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[2];
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[4];
            // Consider nbr 8.
            if ( (Ip==-1)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[8];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(-1), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 7 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 7*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + -1;
                    Jp = J + -1;
                    Kp = K + 0;
                    // Consider nbr 2.
                    if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[2];
                    // Consider nbr 4.
                    if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[4];
                    // Consider nbr 8.
                    if ( (Ip==-1)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[8];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    Jp = (4 + (Jp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 7*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 8 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 8*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 1;
                    Jp = J + 1;
                    Kp = K + 0;
                    // Consider nbr 1.
                    if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[1];
                    // Consider nbr 3.
                    if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[3];
                    // Consider nbr 7.
                    if ( (Ip==4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[7];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    Jp = (4 + (Jp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 8*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_q;
            }
            
            //
            // p = 9
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 1;
            Jp = J + 0;
            Kp = K + 1;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[1];
            // Consider nbr 5.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[5];
            // Consider nbr 9.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[9];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(1), (ufloat_t)(0), (ufloat_t)(1), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + -1;
            Jp = J + 0;
            Kp = K + -1;
            // Consider nbr 2.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[2];
            // Consider nbr 6.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[6];
            // Consider nbr 10.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[10];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(-1), (ufloat_t)(0), (ufloat_t)(-1), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 9 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 9*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + -1;
                    Jp = J + 0;
                    Kp = K + -1;
                    // Consider nbr 2.
                    if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[2];
                    // Consider nbr 6.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[6];
                    // Consider nbr 10.
                    if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[10];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 9*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 10 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 10*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 1;
                    Jp = J + 0;
                    Kp = K + 1;
                    // Consider nbr 1.
                    if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[1];
                    // Consider nbr 5.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[5];
                    // Consider nbr 9.
                    if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[9];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 10*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_q;
            }
            
            //
            // p = 11
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + 1;
            Kp = K + 1;
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[3];
            // Consider nbr 5.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[5];
            // Consider nbr 11.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[11];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(0), (ufloat_t)(1), (ufloat_t)(1), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + -1;
            Kp = K + -1;
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[4];
            // Consider nbr 6.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[6];
            // Consider nbr 12.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[12];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(0), (ufloat_t)(-1), (ufloat_t)(-1), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 11 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 11*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 0;
                    Jp = J + -1;
                    Kp = K + -1;
                    // Consider nbr 4.
                    if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[4];
                    // Consider nbr 6.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[6];
                    // Consider nbr 12.
                    if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[12];
                    // Get the fluid node behind this boundary node.
                    Jp = (4 + (Jp % 4)) % 4;
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 11*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 12 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 12*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 0;
                    Jp = J + 1;
                    Kp = K + 1;
                    // Consider nbr 3.
                    if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[3];
                    // Consider nbr 5.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[5];
                    // Consider nbr 11.
                    if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[11];
                    // Get the fluid node behind this boundary node.
                    Jp = (4 + (Jp % 4)) % 4;
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 12*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_q;
            }
            
            //
            // p = 13
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 1;
            Jp = J + -1;
            Kp = K + 0;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[1];
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[4];
            // Consider nbr 13.
            if ( (Ip==4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[13];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(1), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + -1;
            Jp = J + 1;
            Kp = K + 0;
            // Consider nbr 2.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[2];
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[3];
            // Consider nbr 14.
            if ( (Ip==-1)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[14];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(-1), (ufloat_t)(1), (ufloat_t)(0), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 13 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 13*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + -1;
                    Jp = J + 1;
                    Kp = K + 0;
                    // Consider nbr 2.
                    if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[2];
                    // Consider nbr 3.
                    if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[3];
                    // Consider nbr 14.
                    if ( (Ip==-1)and(Jp==4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[14];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    Jp = (4 + (Jp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 13*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 14 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 14*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 1;
                    Jp = J + -1;
                    Kp = K + 0;
                    // Consider nbr 1.
                    if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[1];
                    // Consider nbr 4.
                    if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[4];
                    // Consider nbr 13.
                    if ( (Ip==4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[13];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    Jp = (4 + (Jp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 14*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_q;
            }
            
            //
            // p = 15
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 1;
            Jp = J + 0;
            Kp = K + -1;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[1];
            // Consider nbr 6.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[6];
            // Consider nbr 15.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[15];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(1), (ufloat_t)(0), (ufloat_t)(-1), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + -1;
            Jp = J + 0;
            Kp = K + 1;
            // Consider nbr 2.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[2];
            // Consider nbr 5.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[5];
            // Consider nbr 16.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[16];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(-1), (ufloat_t)(0), (ufloat_t)(1), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 15 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 15*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + -1;
                    Jp = J + 0;
                    Kp = K + 1;
                    // Consider nbr 2.
                    if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[2];
                    // Consider nbr 5.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[5];
                    // Consider nbr 16.
                    if ( (Ip==-1)and(Jp>=0)and(Jp<4)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[16];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 15*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 16 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 16*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 1;
                    Jp = J + 0;
                    Kp = K + -1;
                    // Consider nbr 1.
                    if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[1];
                    // Consider nbr 6.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[6];
                    // Consider nbr 15.
                    if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[15];
                    // Get the fluid node behind this boundary node.
                    Ip = (4 + (Ip % 4)) % 4;
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 16*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_q;
            }
            
            //
            // p = 17
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + 1;
            Kp = K + -1;
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[3];
            // Consider nbr 6.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[6];
            // Consider nbr 17.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==-1) )
                nbr_kap_b = s_ID_nbr[17];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(0), (ufloat_t)(1), (ufloat_t)(-1), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + -1;
            Kp = K + 1;
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = s_ID_nbr[4];
            // Consider nbr 5.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[5];
            // Consider nbr 18.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==4) )
                nbr_kap_b = s_ID_nbr[18];
            ub = u;
            if (nbr_kap_b == -2 && I == 3)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+16*K+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I == 3)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+16*K+1*M_TBLOCK]);
            wb = w;
            if (nbr_kap_b == -2 && I == 3)
                wb = w + (ufloat_t)(0.5)*(w - s_u[(I-1)+4*J+16*K+2*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(0), (ufloat_t)(-1), (ufloat_t)(1), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Check if DDF 17 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 17*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (pb).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 0;
                    Jp = J + -1;
                    Kp = K + 1;
                    // Consider nbr 4.
                    if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[4];
                    // Consider nbr 5.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[5];
                    // Consider nbr 18.
                    if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp==4) )
                        nbr_kap_b = s_ID_nbr[18];
                    // Get the fluid node behind this boundary node.
                    Jp = (4 + (Jp % 4)) % 4;
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 17*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                    }
                }
                // Check if DDF 18 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                if (bc_type==2)
                {
                    dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 18*n_maxcells_b] / dx_L_g;
                    // Pick the right neighbor block for this cell (p).
                    nbr_kap_b = i_kap_b;
                    Ip = I + 0;
                    Jp = J + 1;
                    Kp = K + -1;
                    // Consider nbr 3.
                    if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                        nbr_kap_b = s_ID_nbr[3];
                    // Consider nbr 6.
                    if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[6];
                    // Consider nbr 17.
                    if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==-1) )
                        nbr_kap_b = s_ID_nbr[17];
                    // Get the fluid node behind this boundary node.
                    Jp = (4 + (Jp % 4)) % 4;
                    Kp = (4 + (Kp % 4)) % 4;
                    nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 18*n_maxcells];
                    // ULI.
                    if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    {
                        f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                    }
                    // DLI.
                    if (dQ >= (ufloat_g_t)(0.5))
                    {
                        f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_q;
            }
            
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ImposeBC_D3Q19(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_ImposeBC_D3Q19<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, mesh->n_solidb, dxf_vec[L], (ufloat_g_t)dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_X_b[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->geometry_init, S_FORCE_TYPE, S_BC_TYPE);
	}

	return 0;
}



