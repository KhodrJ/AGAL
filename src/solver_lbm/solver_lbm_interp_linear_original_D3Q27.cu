/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Tue Mar 25 04:22:58 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, int interp_type>
__global__
void Cu_Interpolate_Linear_Original_D3Q27(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t tau_ratio,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,int *cblock_ID_nbr,int *cblock_ID_nbr_child,int *cblock_ID_mask)
{
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ ufloat_t s_F[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = (threadIdx.x / 4) / 4;
    int i_kap_b = -1;
    int i_kap_bc = -1;
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
        if ((i_kap_b>-1)&&(((interp_type==0)and(block_on_boundary==1))or((interp_type==1)and(cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE))))
        {
            // Load DDFs and compute macroscopic properties.
            f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
            f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
            f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
            f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
            f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
            f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
            f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
            f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
            f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
            f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
            f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
            f_19 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells];
            f_20 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells];
            f_21 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells];
            f_22 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells];
            f_23 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells];
            f_24 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells];
            f_25 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells];
            f_26 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells];
            rho = f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26;
            u = ((f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26)-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25)) / rho;
            v = ((f_3+f_7+f_11+f_14+f_17+f_19+f_21+f_24+f_25)-(f_4+f_8+f_12+f_13+f_18+f_20+f_22+f_23+f_26)) / rho;
            w = ((f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26)) / rho;
            udotu = u*u + v*v + w*w;
            
            // Interpolate rescaled fi to children if applicable.
            //
            // DDF 0.
            //
            cdotu = (ufloat_t)(0.0);
            tmp_i = (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_0 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 1.
            //
            cdotu = (u);
            tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_1 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 2.
            //
            cdotu = -(u);
            tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_2 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 3.
            //
            cdotu = (v);
            tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_3 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 4.
            //
            cdotu = -(v);
            tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_4 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 5.
            //
            cdotu = (w);
            tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_5 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 6.
            //
            cdotu = -(w);
            tmp_i = (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_6 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 7.
            //
            cdotu = (u+v);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_7 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 8.
            //
            cdotu = -(u+v);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_8 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 9.
            //
            cdotu = (u+w);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_9 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 10.
            //
            cdotu = -(u+w);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_10 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 11.
            //
            cdotu = (v+w);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_11 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 12.
            //
            cdotu = -(v+w);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_12 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 13.
            //
            cdotu = (u)-(v);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_13 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 14.
            //
            cdotu = (v)-(u);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_14 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 15.
            //
            cdotu = (u)-(w);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_15 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 16.
            //
            cdotu = (w)-(u);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_16 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 17.
            //
            cdotu = (v)-(w);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_17 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 18.
            //
            cdotu = (w)-(v);
            tmp_i = (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_18 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 19.
            //
            cdotu = (u+v+w);
            tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_19 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 20*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 20.
            //
            cdotu = -(u+v+w);
            tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_20 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 19*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 21.
            //
            cdotu = (u+v)-(w);
            tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_21 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 22*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 22.
            //
            cdotu = (w)-(u+v);
            tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_22 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 21*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 23.
            //
            cdotu = (u+w)-(v);
            tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_23 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 24*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 24.
            //
            cdotu = (v)-(u+w);
            tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_24 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 23*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 25.
            //
            cdotu = (v+w)-(u);
            tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_25 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 26*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 26.
            //
            cdotu = (u)-(v+w);
            tmp_i = (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_26 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[0] + (s_F[1] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4] - s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[16] - s_F[0])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[5] - s_F[1] - s_F[4] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[17] - s_F[1] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[20] - s_F[4] - s_F[16] + s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[2] + (s_F[3] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6] - s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[18] - s_F[2])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[7] - s_F[3] - s_F[6] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[19] - s_F[3] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[22] - s_F[6] - s_F[18] + s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[8] + (s_F[9] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12] - s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[24] - s_F[8])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[13] - s_F[9] - s_F[12] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[25] - s_F[9] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[28] - s_F[12] - s_F[24] + s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[10] + (s_F[11] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14] - s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[26] - s_F[10])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[15] - s_F[11] - s_F[14] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[27] - s_F[11] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[30] - s_F[14] - s_F[26] + s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 4.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[32] + (s_F[33] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[36] - s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[48] - s_F[32])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[37] - s_F[33] - s_F[36] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[49] - s_F[33] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[52] - s_F[36] - s_F[48] + s_F[32])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 5.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[34] + (s_F[35] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[38] - s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[50] - s_F[34])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[39] - s_F[35] - s_F[38] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[51] - s_F[35] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[54] - s_F[38] - s_F[50] + s_F[34])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 6.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[40] + (s_F[41] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[44] - s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[56] - s_F[40])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[45] - s_F[41] - s_F[44] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[57] - s_F[41] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[60] - s_F[44] - s_F[56] + s_F[40])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            //\tChild 7.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 25*n_maxcells] = s_F[42] + (s_F[43] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[46] - s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[58] - s_F[42])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[47] - s_F[43] - s_F[46] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[59] - s_F[43] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[62] - s_F[46] - s_F[58] + s_F[42])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
            }
            __syncthreads();
            
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Interpolate_Linear_Original_D3Q27(int i_dev, int L, int var)
{
	if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_INTERFACE)
	{
		Cu_Interpolate_Linear_Original_D3Q27<ufloat_t,ufloat_g_t,AP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
	}
	if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_ADDED)
	{
		Cu_Interpolate_Linear_Original_D3Q27<ufloat_t,ufloat_g_t,AP,1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
	}

	return 0;
}

