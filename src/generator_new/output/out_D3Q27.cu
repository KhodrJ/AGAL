/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Mon Mar 24 07:01:00 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, int ave_type>
__global__
void Cu_Average_Original_D3Q27(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t tau_ratio,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,int *cblock_ID_nbr,int *cblock_ID_nbr_child,int *cblock_ID_mask,int *cblock_ID_onb)
{
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_mask_child[M_TBLOCK];
    __shared__ ufloat_t s_Fc[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = (threadIdx.x / 4) / 4;
    int i_Q = -1;
    int i_Qc = -1;
    int i_Qcp = -1;
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int child0_IJK = 2*((threadIdx.x % 4)%2) + 4*(2*(((threadIdx.x / 4) % 4)%2)) + 4*4*(2*(((threadIdx.x / 4) / 4)%2));
    int block_on_boundary = -1;
    ufloat_t f_0 = (ufloat_t)(0.0);
    ufloat_t f_1 = (ufloat_t)(0.0);
    ufloat_t f_2 = (ufloat_t)(0.0);
    ufloat_t f_3 = (ufloat_t)(0.0);
    ufloat_t f_4 = (ufloat_t)(0.0);
    ufloat_t f_5 = (ufloat_t)(0.0);
    ufloat_t f_6 = (ufloat_t)(0.0);
    ufloat_t f_7 = (ufloat_t)(0.0);
    ufloat_t f_8 = (ufloat_t)(0.0);
    ufloat_t f_9 = (ufloat_t)(0.0);
    ufloat_t f_10 = (ufloat_t)(0.0);
    ufloat_t f_11 = (ufloat_t)(0.0);
    ufloat_t f_12 = (ufloat_t)(0.0);
    ufloat_t f_13 = (ufloat_t)(0.0);
    ufloat_t f_14 = (ufloat_t)(0.0);
    ufloat_t f_15 = (ufloat_t)(0.0);
    ufloat_t f_16 = (ufloat_t)(0.0);
    ufloat_t f_17 = (ufloat_t)(0.0);
    ufloat_t f_18 = (ufloat_t)(0.0);
    ufloat_t f_19 = (ufloat_t)(0.0);
    ufloat_t f_20 = (ufloat_t)(0.0);
    ufloat_t f_21 = (ufloat_t)(0.0);
    ufloat_t f_22 = (ufloat_t)(0.0);
    ufloat_t f_23 = (ufloat_t)(0.0);
    ufloat_t f_24 = (ufloat_t)(0.0);
    ufloat_t f_25 = (ufloat_t)(0.0);
    ufloat_t f_26 = (ufloat_t)(0.0);
    ufloat_t tmp_i = (ufloat_t)(0.0);
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t u = (ufloat_t)(0.0);
    ufloat_t v = (ufloat_t)(0.0);
    ufloat_t w = (ufloat_t)(0.0);
    ufloat_t cdotu = (ufloat_t)(0.0);
    ufloat_t udotu = (ufloat_t)(0.0);
    s_ID_cblock[threadIdx.x] = -1;
    if ((threadIdx.x<M_LBLOCK)and(kap<n_ids_idev_L))
    {
        s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
    }
    __syncthreads();
    
    // Loop over block Ids.
    for (int k = 0; k < M_LBLOCK; k += 1)
    {
        i_kap_b = s_ID_cblock[k];
        
        // This part is included if n>0 only.
        if (i_kap_b>-1)
        {
            i_kap_bc=cblock_ID_nbr_child[i_kap_b];
            block_on_boundary=cblock_ID_mask[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((i_kap_bc>-1)and((ave_type==2)or(block_on_boundary==1))))
        {
            for (int k_q = 0; k_q < Nqx/2; k_q += 1)
            {
                for (int i_q = 0; i_q < Nqx/2; i_q += 1)
                {
                    for (int j_q = 0; j_q < Nqx/2; j_q += 1)
                    {
                        //
                        // Child block 0.
                        //
                        //
                        i_Q = (i_q+0*Nqx/2) + Nqx*(j_q+0*Nqx/2) + Nqx*Nqx*(k_q+0*Nqx/2);
                        i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
                        for (int xc_k = 0; xc_k < 2; xc_k += 1)
                        {
                            for (int xc_i = 0; xc_i < 2; xc_i += 1)
                            {
                                for (int xc_j = 0; xc_j < 2; xc_j += 1)
                                {
                                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                                    
                                    // Load DDFs and compute macroscopic properties.
                                    f_0 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                                    f_1 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                                    f_2 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                                    f_3 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                                    f_4 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                                    f_5 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                                    f_6 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                                    f_7 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                                    f_8 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                                    f_9 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 10*n_maxcells];
                                    f_10 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 9*n_maxcells];
                                    f_11 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 12*n_maxcells];
                                    f_12 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 11*n_maxcells];
                                    f_13 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 14*n_maxcells];
                                    f_14 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 13*n_maxcells];
                                    f_15 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 16*n_maxcells];
                                    f_16 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 15*n_maxcells];
                                    f_17 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 18*n_maxcells];
                                    f_18 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 17*n_maxcells];
                                    f_19 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 20*n_maxcells];
                                    f_20 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 19*n_maxcells];
                                    f_21 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 22*n_maxcells];
                                    f_22 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 21*n_maxcells];
                                    f_23 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 24*n_maxcells];
                                    f_24 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 23*n_maxcells];
                                    f_25 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 26*n_maxcells];
                                    f_26 = cells_f_F[(i_kap_bc+gI(0+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 25*n_maxcells];
                                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
                                    u = (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26))-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) / rho;
                                    v = (((Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>)) / rho;
                                    w = (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
                                    udotu = u*u + v*v + w*w;
                                    
                                    // Average rescaled fi to parent if applicable.
                                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                                    if (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                                    {
                                        s_ID_mask_child[threadIdx.x] = 1;
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 0
                                    cdotu = (ufloat_t)(0.0);
                                    tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 1
                                    cdotu = (u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 2
                                    cdotu = -(u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 3
                                    cdotu = (v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 4
                                    cdotu = -(v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 5
                                    cdotu = (w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 6
                                    cdotu = -(w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 7
                                    cdotu = (u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 8
                                    cdotu = -(u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 9
                                    cdotu = (u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 10*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 10
                                    cdotu = -(u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 9*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 11
                                    cdotu = (v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 12*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 12
                                    cdotu = -(v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 11*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 13
                                    cdotu = (u)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 14*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 14
                                    cdotu = (v)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 13*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 15
                                    cdotu = (u)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 16*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 16
                                    cdotu = (w)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 15*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 17
                                    cdotu = (v)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 18*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 18
                                    cdotu = (w)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 17*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 19
                                    cdotu = (u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 20*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 20
                                    cdotu = -(u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 19*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 21
                                    cdotu = (u+v)-(w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 22*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 22
                                    cdotu = (w)-(u+v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 21*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 23
                                    cdotu = (u+w)-(v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 24*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 24
                                    cdotu = (v)-(u+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 23*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 25
                                    cdotu = (v+w)-(u);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 26*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 26
                                    cdotu = (u)-(v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 25*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                }
                            }
                        }
                        //
                        // Child block 1.
                        //
                        //
                        i_Q = (i_q+1*Nqx/2) + Nqx*(j_q+0*Nqx/2) + Nqx*Nqx*(k_q+0*Nqx/2);
                        i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
                        for (int xc_k = 0; xc_k < 2; xc_k += 1)
                        {
                            for (int xc_i = 0; xc_i < 2; xc_i += 1)
                            {
                                for (int xc_j = 0; xc_j < 2; xc_j += 1)
                                {
                                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                                    
                                    // Load DDFs and compute macroscopic properties.
                                    f_0 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                                    f_1 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                                    f_2 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                                    f_3 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                                    f_4 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                                    f_5 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                                    f_6 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                                    f_7 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                                    f_8 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                                    f_9 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 10*n_maxcells];
                                    f_10 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 9*n_maxcells];
                                    f_11 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 12*n_maxcells];
                                    f_12 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 11*n_maxcells];
                                    f_13 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 14*n_maxcells];
                                    f_14 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 13*n_maxcells];
                                    f_15 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 16*n_maxcells];
                                    f_16 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 15*n_maxcells];
                                    f_17 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 18*n_maxcells];
                                    f_18 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 17*n_maxcells];
                                    f_19 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 20*n_maxcells];
                                    f_20 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 19*n_maxcells];
                                    f_21 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 22*n_maxcells];
                                    f_22 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 21*n_maxcells];
                                    f_23 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 24*n_maxcells];
                                    f_24 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 23*n_maxcells];
                                    f_25 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 26*n_maxcells];
                                    f_26 = cells_f_F[(i_kap_bc+gI(1+2*0+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 25*n_maxcells];
                                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
                                    u = (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26))-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) / rho;
                                    v = (((Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>)) / rho;
                                    w = (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
                                    udotu = u*u + v*v + w*w;
                                    
                                    // Average rescaled fi to parent if applicable.
                                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                                    if (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                                    {
                                        s_ID_mask_child[threadIdx.x] = 1;
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 0
                                    cdotu = (ufloat_t)(0.0);
                                    tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 1
                                    cdotu = (u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 2
                                    cdotu = -(u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 3
                                    cdotu = (v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 4
                                    cdotu = -(v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 5
                                    cdotu = (w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 6
                                    cdotu = -(w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 7
                                    cdotu = (u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 8
                                    cdotu = -(u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 9
                                    cdotu = (u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 10*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 10
                                    cdotu = -(u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 9*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 11
                                    cdotu = (v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 12*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 12
                                    cdotu = -(v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 11*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 13
                                    cdotu = (u)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 14*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 14
                                    cdotu = (v)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 13*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 15
                                    cdotu = (u)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 16*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 16
                                    cdotu = (w)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 15*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 17
                                    cdotu = (v)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 18*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 18
                                    cdotu = (w)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 17*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 19
                                    cdotu = (u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 20*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 20
                                    cdotu = -(u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 19*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 21
                                    cdotu = (u+v)-(w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 22*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 22
                                    cdotu = (w)-(u+v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 21*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 23
                                    cdotu = (u+w)-(v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 24*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 24
                                    cdotu = (v)-(u+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 23*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 25
                                    cdotu = (v+w)-(u);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 26*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 26
                                    cdotu = (u)-(v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 25*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                }
                            }
                        }
                        //
                        // Child block 2.
                        //
                        //
                        i_Q = (i_q+0*Nqx/2) + Nqx*(j_q+1*Nqx/2) + Nqx*Nqx*(k_q+0*Nqx/2);
                        i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
                        for (int xc_k = 0; xc_k < 2; xc_k += 1)
                        {
                            for (int xc_i = 0; xc_i < 2; xc_i += 1)
                            {
                                for (int xc_j = 0; xc_j < 2; xc_j += 1)
                                {
                                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                                    
                                    // Load DDFs and compute macroscopic properties.
                                    f_0 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                                    f_1 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                                    f_2 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                                    f_3 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                                    f_4 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                                    f_5 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                                    f_6 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                                    f_7 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                                    f_8 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                                    f_9 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 10*n_maxcells];
                                    f_10 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 9*n_maxcells];
                                    f_11 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 12*n_maxcells];
                                    f_12 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 11*n_maxcells];
                                    f_13 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 14*n_maxcells];
                                    f_14 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 13*n_maxcells];
                                    f_15 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 16*n_maxcells];
                                    f_16 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 15*n_maxcells];
                                    f_17 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 18*n_maxcells];
                                    f_18 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 17*n_maxcells];
                                    f_19 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 20*n_maxcells];
                                    f_20 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 19*n_maxcells];
                                    f_21 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 22*n_maxcells];
                                    f_22 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 21*n_maxcells];
                                    f_23 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 24*n_maxcells];
                                    f_24 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 23*n_maxcells];
                                    f_25 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 26*n_maxcells];
                                    f_26 = cells_f_F[(i_kap_bc+gI(0+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 25*n_maxcells];
                                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
                                    u = (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26))-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) / rho;
                                    v = (((Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>)) / rho;
                                    w = (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
                                    udotu = u*u + v*v + w*w;
                                    
                                    // Average rescaled fi to parent if applicable.
                                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                                    if (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                                    {
                                        s_ID_mask_child[threadIdx.x] = 1;
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 0
                                    cdotu = (ufloat_t)(0.0);
                                    tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 1
                                    cdotu = (u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 2
                                    cdotu = -(u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 3
                                    cdotu = (v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 4
                                    cdotu = -(v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 5
                                    cdotu = (w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 6
                                    cdotu = -(w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 7
                                    cdotu = (u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 8
                                    cdotu = -(u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 9
                                    cdotu = (u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 10*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 10
                                    cdotu = -(u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 9*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 11
                                    cdotu = (v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 12*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 12
                                    cdotu = -(v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 11*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 13
                                    cdotu = (u)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 14*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 14
                                    cdotu = (v)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 13*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 15
                                    cdotu = (u)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 16*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 16
                                    cdotu = (w)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 15*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 17
                                    cdotu = (v)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 18*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 18
                                    cdotu = (w)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 17*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 19
                                    cdotu = (u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 20*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 20
                                    cdotu = -(u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 19*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 21
                                    cdotu = (u+v)-(w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 22*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 22
                                    cdotu = (w)-(u+v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 21*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 23
                                    cdotu = (u+w)-(v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 24*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 24
                                    cdotu = (v)-(u+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 23*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 25
                                    cdotu = (v+w)-(u);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 26*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 26
                                    cdotu = (u)-(v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 25*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                }
                            }
                        }
                        //
                        // Child block 3.
                        //
                        //
                        i_Q = (i_q+1*Nqx/2) + Nqx*(j_q+1*Nqx/2) + Nqx*Nqx*(k_q+0*Nqx/2);
                        i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
                        for (int xc_k = 0; xc_k < 2; xc_k += 1)
                        {
                            for (int xc_i = 0; xc_i < 2; xc_i += 1)
                            {
                                for (int xc_j = 0; xc_j < 2; xc_j += 1)
                                {
                                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                                    
                                    // Load DDFs and compute macroscopic properties.
                                    f_0 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                                    f_1 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                                    f_2 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                                    f_3 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                                    f_4 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                                    f_5 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                                    f_6 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                                    f_7 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                                    f_8 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                                    f_9 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 10*n_maxcells];
                                    f_10 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 9*n_maxcells];
                                    f_11 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 12*n_maxcells];
                                    f_12 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 11*n_maxcells];
                                    f_13 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 14*n_maxcells];
                                    f_14 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 13*n_maxcells];
                                    f_15 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 16*n_maxcells];
                                    f_16 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 15*n_maxcells];
                                    f_17 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 18*n_maxcells];
                                    f_18 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 17*n_maxcells];
                                    f_19 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 20*n_maxcells];
                                    f_20 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 19*n_maxcells];
                                    f_21 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 22*n_maxcells];
                                    f_22 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 21*n_maxcells];
                                    f_23 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 24*n_maxcells];
                                    f_24 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 23*n_maxcells];
                                    f_25 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 26*n_maxcells];
                                    f_26 = cells_f_F[(i_kap_bc+gI(1+2*1+4*0))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 25*n_maxcells];
                                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
                                    u = (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26))-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) / rho;
                                    v = (((Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>)) / rho;
                                    w = (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
                                    udotu = u*u + v*v + w*w;
                                    
                                    // Average rescaled fi to parent if applicable.
                                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                                    if (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                                    {
                                        s_ID_mask_child[threadIdx.x] = 1;
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 0
                                    cdotu = (ufloat_t)(0.0);
                                    tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 1
                                    cdotu = (u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 2
                                    cdotu = -(u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 3
                                    cdotu = (v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 4
                                    cdotu = -(v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 5
                                    cdotu = (w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 6
                                    cdotu = -(w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 7
                                    cdotu = (u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 8
                                    cdotu = -(u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 9
                                    cdotu = (u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 10*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 10
                                    cdotu = -(u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 9*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 11
                                    cdotu = (v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 12*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 12
                                    cdotu = -(v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 11*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 13
                                    cdotu = (u)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 14*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 14
                                    cdotu = (v)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 13*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 15
                                    cdotu = (u)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 16*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 16
                                    cdotu = (w)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 15*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 17
                                    cdotu = (v)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 18*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 18
                                    cdotu = (w)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 17*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 19
                                    cdotu = (u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 20*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 20
                                    cdotu = -(u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 19*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 21
                                    cdotu = (u+v)-(w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 22*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 22
                                    cdotu = (w)-(u+v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 21*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 23
                                    cdotu = (u+w)-(v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 24*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 24
                                    cdotu = (v)-(u+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 23*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 25
                                    cdotu = (v+w)-(u);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 26*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 26
                                    cdotu = (u)-(v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 25*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                }
                            }
                        }
                        //
                        // Child block 4.
                        //
                        //
                        i_Q = (i_q+0*Nqx/2) + Nqx*(j_q+0*Nqx/2) + Nqx*Nqx*(k_q+1*Nqx/2);
                        i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
                        for (int xc_k = 0; xc_k < 2; xc_k += 1)
                        {
                            for (int xc_i = 0; xc_i < 2; xc_i += 1)
                            {
                                for (int xc_j = 0; xc_j < 2; xc_j += 1)
                                {
                                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                                    
                                    // Load DDFs and compute macroscopic properties.
                                    f_0 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                                    f_1 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                                    f_2 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                                    f_3 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                                    f_4 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                                    f_5 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                                    f_6 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                                    f_7 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                                    f_8 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                                    f_9 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 10*n_maxcells];
                                    f_10 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 9*n_maxcells];
                                    f_11 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 12*n_maxcells];
                                    f_12 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 11*n_maxcells];
                                    f_13 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 14*n_maxcells];
                                    f_14 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 13*n_maxcells];
                                    f_15 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 16*n_maxcells];
                                    f_16 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 15*n_maxcells];
                                    f_17 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 18*n_maxcells];
                                    f_18 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 17*n_maxcells];
                                    f_19 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 20*n_maxcells];
                                    f_20 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 19*n_maxcells];
                                    f_21 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 22*n_maxcells];
                                    f_22 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 21*n_maxcells];
                                    f_23 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 24*n_maxcells];
                                    f_24 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 23*n_maxcells];
                                    f_25 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 26*n_maxcells];
                                    f_26 = cells_f_F[(i_kap_bc+gI(0+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 25*n_maxcells];
                                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
                                    u = (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26))-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) / rho;
                                    v = (((Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>)) / rho;
                                    w = (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
                                    udotu = u*u + v*v + w*w;
                                    
                                    // Average rescaled fi to parent if applicable.
                                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                                    if (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                                    {
                                        s_ID_mask_child[threadIdx.x] = 1;
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 0
                                    cdotu = (ufloat_t)(0.0);
                                    tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 1
                                    cdotu = (u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 2
                                    cdotu = -(u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 3
                                    cdotu = (v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 4
                                    cdotu = -(v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 5
                                    cdotu = (w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 6
                                    cdotu = -(w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 7
                                    cdotu = (u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 8
                                    cdotu = -(u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 9
                                    cdotu = (u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 10*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 10
                                    cdotu = -(u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 9*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 11
                                    cdotu = (v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 12*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 12
                                    cdotu = -(v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 11*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 13
                                    cdotu = (u)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 14*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 14
                                    cdotu = (v)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 13*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 15
                                    cdotu = (u)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 16*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 16
                                    cdotu = (w)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 15*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 17
                                    cdotu = (v)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 18*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 18
                                    cdotu = (w)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 17*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 19
                                    cdotu = (u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 20*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 20
                                    cdotu = -(u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 19*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 21
                                    cdotu = (u+v)-(w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 22*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 22
                                    cdotu = (w)-(u+v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 21*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 23
                                    cdotu = (u+w)-(v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 24*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 24
                                    cdotu = (v)-(u+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 23*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 25
                                    cdotu = (v+w)-(u);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 26*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 26
                                    cdotu = (u)-(v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 25*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                }
                            }
                        }
                        //
                        // Child block 5.
                        //
                        //
                        i_Q = (i_q+1*Nqx/2) + Nqx*(j_q+0*Nqx/2) + Nqx*Nqx*(k_q+1*Nqx/2);
                        i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
                        for (int xc_k = 0; xc_k < 2; xc_k += 1)
                        {
                            for (int xc_i = 0; xc_i < 2; xc_i += 1)
                            {
                                for (int xc_j = 0; xc_j < 2; xc_j += 1)
                                {
                                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                                    
                                    // Load DDFs and compute macroscopic properties.
                                    f_0 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                                    f_1 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                                    f_2 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                                    f_3 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                                    f_4 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                                    f_5 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                                    f_6 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                                    f_7 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                                    f_8 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                                    f_9 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 10*n_maxcells];
                                    f_10 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 9*n_maxcells];
                                    f_11 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 12*n_maxcells];
                                    f_12 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 11*n_maxcells];
                                    f_13 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 14*n_maxcells];
                                    f_14 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 13*n_maxcells];
                                    f_15 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 16*n_maxcells];
                                    f_16 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 15*n_maxcells];
                                    f_17 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 18*n_maxcells];
                                    f_18 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 17*n_maxcells];
                                    f_19 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 20*n_maxcells];
                                    f_20 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 19*n_maxcells];
                                    f_21 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 22*n_maxcells];
                                    f_22 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 21*n_maxcells];
                                    f_23 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 24*n_maxcells];
                                    f_24 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 23*n_maxcells];
                                    f_25 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 26*n_maxcells];
                                    f_26 = cells_f_F[(i_kap_bc+gI(1+2*0+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 25*n_maxcells];
                                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
                                    u = (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26))-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) / rho;
                                    v = (((Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>)) / rho;
                                    w = (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
                                    udotu = u*u + v*v + w*w;
                                    
                                    // Average rescaled fi to parent if applicable.
                                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                                    if (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                                    {
                                        s_ID_mask_child[threadIdx.x] = 1;
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 0
                                    cdotu = (ufloat_t)(0.0);
                                    tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 1
                                    cdotu = (u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 2
                                    cdotu = -(u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 3
                                    cdotu = (v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 4
                                    cdotu = -(v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 5
                                    cdotu = (w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 6
                                    cdotu = -(w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 7
                                    cdotu = (u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 8
                                    cdotu = -(u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 9
                                    cdotu = (u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 10*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 10
                                    cdotu = -(u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 9*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 11
                                    cdotu = (v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 12*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 12
                                    cdotu = -(v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 11*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 13
                                    cdotu = (u)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 14*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 14
                                    cdotu = (v)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 13*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 15
                                    cdotu = (u)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 16*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 16
                                    cdotu = (w)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 15*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 17
                                    cdotu = (v)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 18*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 18
                                    cdotu = (w)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 17*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 19
                                    cdotu = (u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 20*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 20
                                    cdotu = -(u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 19*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 21
                                    cdotu = (u+v)-(w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 22*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 22
                                    cdotu = (w)-(u+v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 21*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 23
                                    cdotu = (u+w)-(v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 24*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 24
                                    cdotu = (v)-(u+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 23*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 25
                                    cdotu = (v+w)-(u);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 26*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 26
                                    cdotu = (u)-(v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 25*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                }
                            }
                        }
                        //
                        // Child block 6.
                        //
                        //
                        i_Q = (i_q+0*Nqx/2) + Nqx*(j_q+1*Nqx/2) + Nqx*Nqx*(k_q+1*Nqx/2);
                        i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
                        for (int xc_k = 0; xc_k < 2; xc_k += 1)
                        {
                            for (int xc_i = 0; xc_i < 2; xc_i += 1)
                            {
                                for (int xc_j = 0; xc_j < 2; xc_j += 1)
                                {
                                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                                    
                                    // Load DDFs and compute macroscopic properties.
                                    f_0 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                                    f_1 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                                    f_2 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                                    f_3 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                                    f_4 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                                    f_5 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                                    f_6 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                                    f_7 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                                    f_8 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                                    f_9 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 10*n_maxcells];
                                    f_10 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 9*n_maxcells];
                                    f_11 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 12*n_maxcells];
                                    f_12 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 11*n_maxcells];
                                    f_13 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 14*n_maxcells];
                                    f_14 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 13*n_maxcells];
                                    f_15 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 16*n_maxcells];
                                    f_16 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 15*n_maxcells];
                                    f_17 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 18*n_maxcells];
                                    f_18 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 17*n_maxcells];
                                    f_19 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 20*n_maxcells];
                                    f_20 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 19*n_maxcells];
                                    f_21 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 22*n_maxcells];
                                    f_22 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 21*n_maxcells];
                                    f_23 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 24*n_maxcells];
                                    f_24 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 23*n_maxcells];
                                    f_25 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 26*n_maxcells];
                                    f_26 = cells_f_F[(i_kap_bc+gI(0+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 25*n_maxcells];
                                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
                                    u = (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26))-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) / rho;
                                    v = (((Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>)) / rho;
                                    w = (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
                                    udotu = u*u + v*v + w*w;
                                    
                                    // Average rescaled fi to parent if applicable.
                                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                                    if (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                                    {
                                        s_ID_mask_child[threadIdx.x] = 1;
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 0
                                    cdotu = (ufloat_t)(0.0);
                                    tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 1
                                    cdotu = (u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 2
                                    cdotu = -(u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 3
                                    cdotu = (v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 4
                                    cdotu = -(v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 5
                                    cdotu = (w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 6
                                    cdotu = -(w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 7
                                    cdotu = (u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 8
                                    cdotu = -(u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 9
                                    cdotu = (u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 10*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 10
                                    cdotu = -(u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 9*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 11
                                    cdotu = (v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 12*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 12
                                    cdotu = -(v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 11*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 13
                                    cdotu = (u)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 14*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 14
                                    cdotu = (v)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 13*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 15
                                    cdotu = (u)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 16*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 16
                                    cdotu = (w)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 15*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 17
                                    cdotu = (v)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 18*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 18
                                    cdotu = (w)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 17*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 19
                                    cdotu = (u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 20*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 20
                                    cdotu = -(u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 19*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 21
                                    cdotu = (u+v)-(w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 22*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 22
                                    cdotu = (w)-(u+v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 21*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 23
                                    cdotu = (u+w)-(v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 24*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 24
                                    cdotu = (v)-(u+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 23*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 25
                                    cdotu = (v+w)-(u);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 26*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 26
                                    cdotu = (u)-(v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 25*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                }
                            }
                        }
                        //
                        // Child block 7.
                        //
                        //
                        i_Q = (i_q+1*Nqx/2) + Nqx*(j_q+1*Nqx/2) + Nqx*Nqx*(k_q+1*Nqx/2);
                        i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
                        for (int xc_k = 0; xc_k < 2; xc_k += 1)
                        {
                            for (int xc_i = 0; xc_i < 2; xc_i += 1)
                            {
                                for (int xc_j = 0; xc_j < 2; xc_j += 1)
                                {
                                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                                    
                                    // Load DDFs and compute macroscopic properties.
                                    f_0 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                                    f_1 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                                    f_2 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                                    f_3 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                                    f_4 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                                    f_5 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                                    f_6 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                                    f_7 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                                    f_8 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                                    f_9 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 10*n_maxcells];
                                    f_10 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 9*n_maxcells];
                                    f_11 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 12*n_maxcells];
                                    f_12 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 11*n_maxcells];
                                    f_13 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 14*n_maxcells];
                                    f_14 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 13*n_maxcells];
                                    f_15 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 16*n_maxcells];
                                    f_16 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 15*n_maxcells];
                                    f_17 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 18*n_maxcells];
                                    f_18 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 17*n_maxcells];
                                    f_19 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 20*n_maxcells];
                                    f_20 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 19*n_maxcells];
                                    f_21 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 22*n_maxcells];
                                    f_22 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 21*n_maxcells];
                                    f_23 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 24*n_maxcells];
                                    f_24 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 23*n_maxcells];
                                    f_25 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 26*n_maxcells];
                                    f_26 = cells_f_F[(i_kap_bc+gI(1+2*1+4*1))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + 25*n_maxcells];
                                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
                                    u = (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26))-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) / rho;
                                    v = (((Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>+(Lc1(<p>))*f_<p>)) / rho;
                                    w = (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
                                    udotu = u*u + v*v + w*w;
                                    
                                    // Average rescaled fi to parent if applicable.
                                    s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                                    if (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                                    {
                                        s_ID_mask_child[threadIdx.x] = 1;
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 0
                                    cdotu = (ufloat_t)(0.0);
                                    tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 1
                                    cdotu = (u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 2
                                    cdotu = -(u);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 3
                                    cdotu = (v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 4
                                    cdotu = -(v);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 5
                                    cdotu = (w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 6
                                    cdotu = -(w);
                                    tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 7
                                    cdotu = (u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 8
                                    cdotu = -(u+v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 9
                                    cdotu = (u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 10*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 10
                                    cdotu = -(u+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 9*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 11
                                    cdotu = (v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 12*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 12
                                    cdotu = -(v+w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 11*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 13
                                    cdotu = (u)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 14*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 14
                                    cdotu = (v)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 13*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 15
                                    cdotu = (u)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 16*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 16
                                    cdotu = (w)-(u);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 15*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 17
                                    cdotu = (v)-(w);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 18*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 18
                                    cdotu = (w)-(v);
                                    tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 17*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 19
                                    cdotu = (u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 20*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 20
                                    cdotu = -(u+v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 19*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 21
                                    cdotu = (u+v)-(w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 22*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 22
                                    cdotu = (w)-(u+v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 21*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 23
                                    cdotu = (u+w)-(v);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 24*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 24
                                    cdotu = (v)-(u+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 23*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 25
                                    cdotu = (v+w)-(u);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 26*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                    
                                    //\t p = 26
                                    cdotu = (u)-(v+w);
                                    tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                                    s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
                                    __syncthreads();
                                    if (s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k)
                                    {
                                        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 25*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                                    }
                                    __syncthreads();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Average_Original_D3Q27(int i_dev, int L, int var)
{
	if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_INTERFACE)
	{
		Cu_Average_Original_D3Q27<ufloat_t,ufloat_g_t,AP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_vec[L]/tau_vec[L+1], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_BLOCK)
	{
		Cu_Average_Original_D3Q27<ufloat_t,ufloat_g_t,AP,1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_vec[L]/tau_vec[L+1], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_GRID)
	{
		Cu_Average_Original_D3Q27<ufloat_t,ufloat_g_t,AP,2><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_vec[L]/tau_vec[L+1], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

