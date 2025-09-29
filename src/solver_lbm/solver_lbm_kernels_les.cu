/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "custom.h"
#include "solver_lbm.h"
#include "mesh.h"

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
T SubgridScale_Smagorinsky
(
    const T dx,
    const T &ux,
    const T &uy,
    const T &uz,
    const T &vx,
    const T &vy,
    const T &vz,
    const T &wx,
    const T &wy,
    const T &wz
)
{
    constexpr T ZERO = static_cast<T>(0.0);
    constexpr T HALF = static_cast<T>(0.5);
    constexpr T TWO = static_cast<T>(2.0);
    constexpr T CS2 = static_cast<T>(0.1*0.1);
    
    T nu_SGS = static_cast<T>(0.0);

    if (N_DIM==2)
    {
        T S_xx = ux;
        T S_xy = HALF*(uy + vx);
        //T S_yx = S_xy;
        T S_yy = vy;
        
        T F = 
            S_xx*S_xx + 
            TWO*S_xy*S_xy + 
            S_yy*S_yy;
        nu_SGS = CS2*dx*dx*sqrt(TWO*F);
    }
    else // N_DIM==3
    {
        T S_xx = ux;
        T S_xy = HALF*(uy + vx);
        T S_xz = HALF*(uz + wx);
        //T S_yx = S_xy;
        T S_yy = vy;
        T S_yz = HALF*(vz + wy);
        //T S_zx = S_xz;
        //T S_zy = S_yz;
        T S_zz = wz;
        
        T F = 
            S_xx*S_xx + 
            TWO*S_xy*S_xy + 
            TWO*S_xz*S_xz + 
            S_yy*S_yy + 
            TWO*S_yz*S_yz + 
            S_zz*S_zz;
        nu_SGS = CS2*dx*dx*sqrt(TWO*F);
    }
    
    return nu_SGS;
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
T SubgridScale_WALE
(
    const T dx,
    const T &ux,
    const T &uy,
    const T &uz,
    const T &vx,
    const T &vy,
    const T &vz,
    const T &wx,
    const T &wy,
    const T &wz
)
{
    constexpr T ZERO = static_cast<T>(0.0);
    constexpr T ATHIRD = static_cast<T>(1.0/3.0);
    constexpr T HALF = static_cast<T>(0.5);
    constexpr T TWO = static_cast<T>(2.0);
    constexpr T Cw2 = static_cast<T>(0.325*0.325);
    
    T nu_SGS = static_cast<T>(0.0);
    
    if (N_DIM==2)
    {
        // Strain-rate tensor.
        T S_xx = ux;
        T S_xy = HALF*(uy + vx);
        T S_yy = vy;
        
        // Subgrid-scale operator.
        //
        T gxx = ux*ux + uy*vx + uz*wx;
        T gxy = ux*uy + uy*vy + uz*wy;
        //
        T gyx = vx*ux + vy*vx + vz*wx;
        T gyy = vx*uy + vy*vy + vz*wy;
        //
        T gkk = gxx + gyy;
        //
        T Sd_xx = gxx - ATHIRD*gkk;
        T Sd_xy = HALF*(gxy + gyx);
        //T Sd_yx = Sd_xy;
        T Sd_yy = gyy - ATHIRD*gkk;
        
        // Final computation.
        T F1 = 
            Sd_xx*Sd_xx + 
            TWO*Sd_xy*Sd_xy + 
            Sd_yy*Sd_yy;
        T F2 = 
            S_xx*S_xx + 
            TWO*S_xy*S_xy + 
            S_yy*S_yy;
        nu_SGS = 
            Cw2*dx*dx*(F1*Tsqrt(F1)) / (F2*F2*Tsqrt(F2) + F1*Tsqrt(Tsqrt(F1)));
    }
    else // N_DIM==3
    {
        // Strain-rate tensor.
        T S_xx = ux;
        T S_xy = HALF*(uy + vx);
        T S_xz = HALF*(uz + wx);
        T S_yy = vy;
        //T S_yx = S_xy;
        T S_yz = HALF*(vz + wy);
        //T S_zx = S_xz;
        //T S_zy = S_yz;
        T S_zz = wz;
        
        // Subgrid-scale operator.
        //
        T gxx = ux*ux + uy*vx + uz*wx;
        T gxy = ux*uy + uy*vy + uz*wy;
        T gxz = ux*uz + uy*vz + uz*wz;
        //
        T gyx = vx*ux + vy*vx + vz*wx;
        T gyy = vx*uy + vy*vy + vz*wy;
        T gyz = vx*uz + vy*vz + vz*wz;
        //
        T gzx = wx*ux + wy*vx + wz*wx;
        T gzy = wx*uy + wy*vy + wz*wy;
        T gzz = wx*uz + wy*vz + wz*wz;
        //
        T gkk = gxx + gyy + gzz;
        //
        T Sd_xx = gxx - ATHIRD*gkk;
        T Sd_xy = static_cast<T>(0.5)*(gxy + gyx);
        T Sd_xz = static_cast<T>(0.5)*(gxz + gzx);
        //T Sd_yx = Sd_xy;
        T Sd_yy = gyy - ATHIRD*gkk;
        T Sd_yz = static_cast<T>(0.5)*(gyz + gzy);
        //T Sd_zx = Sd_xz;
        //T Sd_zy = Sd_yz;
        T Sd_zz = gzz - ATHIRD*gkk;
        
        // Final computation.
        T F1 = 
            Sd_xx*Sd_xx + 
            TWO*Sd_xy*Sd_xy + 
            TWO*Sd_xz*Sd_xz + 
            Sd_yy*Sd_yy + 
            TWO*Sd_yz*Sd_yz + 
            Sd_zz*Sd_zz;
        T F2 = 
            S_xx*S_xx + 
            TWO*S_xy*S_xy + 
            TWO*S_xz*S_xz + 
            S_yy*S_yy + 
            TWO*S_yz*S_yz + 
            S_zz*S_zz;
        nu_SGS = Cw2*dx*dx*(F1*Tsqrt(F1)) / (F2*F2*Tsqrt(F2) + F1*Tsqrt(Tsqrt(F1)));
    }
    
    return nu_SGS;
}

template <typename T, int N_DIM, int model=1>
__host__ __device__ __forceinline__
T SubgridScaleModel
(
    const T dx,
    const T &ux,
    const T &uy,
    const T &uz,
    const T &vx,
    const T &vy,
    const T &vz,
    const T &wx,
    const T &wy,
    const T &wz
)
{
    if (model == 0)
        return SubgridScale_Smagorinsky<T,N_DIM>(dx,ux,uy,uz,vx,vy,vz,wx,wy,wz);
    else
        return SubgridScale_WALE<T,N_DIM>(dx,ux,uy,uz,vx,vy,vz,wx,wy,wz);
}









template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_ComputeEddyViscosity
(
    const int n_ids_idev_L,
    const long int n_maxcells,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    const int *__restrict__ id_set_idev_L,
    ufloat_t *__restrict__ cells_f_F,
    const int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_ID_nbr_child
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_HBLOCK = AP->M_HBLOCK;
    constexpr int N_Q = LP->N_Q;
    __shared__ ufloat_t s_u[M_TBLOCK];
    __shared__ ufloat_t s_v[M_TBLOCK];
    __shared__ ufloat_t s_w[M_TBLOCK];
    __shared__ int s_ID_nbr[N_Q_max];
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        int i_kap_bc = cblock_ID_nbr_child[i_kap_b];
        int valid_block = cblock_ID_mask[i_kap_b];
        
        if (i_kap_bc < 0 || BlockNotSolid(valid_block))
        {
            // Load neighbor-block indices into shared memory.
            if (threadIdx.x==0)
            {
                for (int p = 0; p < N_Q_max; p++)
                    s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
            }
            __syncthreads();
            
            // Compute local cell indices.
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            int K = 0;
            if (N_DIM==3)
                K = (threadIdx.x / 4) / 4;
            
            // Macroscopic properties.
            ufloat_t u = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+1)*n_maxcells];
            ufloat_t v = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+2)*n_maxcells];
            ufloat_t w = static_cast<ufloat_t>(0.0);
            if (N_DIM==3)
                w = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+3)*n_maxcells];
            
            // Load current block data to shared memory.
            s_u[threadIdx.x] = u;
            s_v[threadIdx.x] = v;
            if (N_DIM==3)
                s_w[threadIdx.x] = w;
            
            // If any of the current cells are near the geometry, we need to get neighbor data.
            if (BlockNotNearSolid(cblock_ID_mask[i_kap_b]))
            {
                for (int p = 1; p < N_DIM+1; p++)
                {
                    Cu_FillHaloWithNbrs<ufloat_t,N_DIM>(
                        p, s_u, cells_f_F,
                        I,J,K,
                        s_ID_nbr,
                        M_CBLOCK, (N_Q+1)*n_maxcells
                    );
                    Cu_FillHaloWithNbrs<ufloat_t,N_DIM>(
                        p, s_v, cells_f_F,
                        I,J,K,
                        s_ID_nbr,
                        M_CBLOCK, (N_Q+2)*n_maxcells
                    );
                    if (N_DIM==3)
                    {
                        Cu_FillHaloWithNbrs<ufloat_t,N_DIM>(
                            p, s_w, cells_f_F,
                            I,J,K,
                            s_ID_nbr,
                            M_CBLOCK, (N_Q+3)*n_maxcells
                        );
                    }
                }
            }
            else // Otherwise, just extrapolate from the interior of this block.
            {
                __syncthreads();
                ExtrapolateToHalo<ufloat_t,N_DIM>(s_u,I,J,K);
                ExtrapolateToHalo<ufloat_t,N_DIM>(s_v,I,J,K);
                if (N_DIM==3)
                    ExtrapolateToHalo<ufloat_t,N_DIM>(s_w,I,J,K);
            }
            __syncthreads();
            
            // Compute velocity gradient tensor.
            ufloat_t ux = FinDiff_D1_Central_Stencil(
                dx_L,
                s_u[Cu_Halo<N_DIM>(I-1,J,K)],
                s_u[Cu_Halo<N_DIM>(I,J,K)],
                s_u[Cu_Halo<N_DIM>(I+1,J,K)]
            );
            ufloat_t uy = FinDiff_D1_Central_Stencil(
                dx_L,
                s_u[Cu_Halo<N_DIM>(I,J-1,K)],
                s_u[Cu_Halo<N_DIM>(I,J,K)],
                s_u[Cu_Halo<N_DIM>(I,J+1,K)]
            );
            ufloat_t uz = FinDiff_D1_Central_Stencil(
                dx_L,
                s_u[Cu_Halo<N_DIM>(I,J,K-1)],
                s_u[Cu_Halo<N_DIM>(I,J,K)],
                s_u[Cu_Halo<N_DIM>(I,J,K+1)]
            );
            
            ufloat_t vx = FinDiff_D1_Central_Stencil(
                dx_L,
                s_v[Cu_Halo<N_DIM>(I-1,J,K)],
                s_v[Cu_Halo<N_DIM>(I,J,K)],
                s_v[Cu_Halo<N_DIM>(I+1,J,K)]
            );
            ufloat_t vy = FinDiff_D1_Central_Stencil(
                dx_L,
                s_v[Cu_Halo<N_DIM>(I,J-1,K)],
                s_v[Cu_Halo<N_DIM>(I,J,K)],
                s_v[Cu_Halo<N_DIM>(I,J+1,K)]
            );
            ufloat_t vz = FinDiff_D1_Central_Stencil(
                dx_L,
                s_v[Cu_Halo<N_DIM>(I,J,K-1)],
                s_v[Cu_Halo<N_DIM>(I,J,K)],
                s_v[Cu_Halo<N_DIM>(I,J,K+1)]
            );
            
            ufloat_t wx = FinDiff_D1_Central_Stencil(
                dx_L,
                s_v[Cu_Halo<N_DIM>(I-1,J,K)],
                s_v[Cu_Halo<N_DIM>(I,J,K)],
                s_v[Cu_Halo<N_DIM>(I+1,J,K)]
            );
            ufloat_t wy = FinDiff_D1_Central_Stencil(
                dx_L,
                s_v[Cu_Halo<N_DIM>(I,J-1,K)],
                s_v[Cu_Halo<N_DIM>(I,J,K)],
                s_v[Cu_Halo<N_DIM>(I,J+1,K)]
            );
            ufloat_t wz = FinDiff_D1_Central_Stencil(
                dx_L,
                s_v[Cu_Halo<N_DIM>(I,J,K-1)],
                s_v[Cu_Halo<N_DIM>(I,J,K)],
                s_v[Cu_Halo<N_DIM>(I,J,K+1)]
            );
            
            // Store macroscopic properties.
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+4)*n_maxcells] = SubgridScaleModel<ufloat_t,N_DIM,1>(
                dx_L,
                ux,uy,uz,
                vx,vy,vz,
                wx,wy,wz
            );
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeEddyViscosity(int i_dev, int L)
{
    if (mesh->n_ids[i_dev][L] > 0)
    {
        Cu_ComputeEddyViscosity<ufloat_t,ufloat_g_t,AP,LP><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>
        (
            mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L],
            &mesh->c_id_set[i_dev][L*n_maxcblocks],
            mesh->c_cells_f_F[i_dev],
            mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev]
        );
    }

    return 0;
}
