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
void Cu_Interpolate_Linear_Original_D2Q9(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t tau_ratio,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,int *cblock_ID_nbr,int *cblock_ID_nbr_child,int *cblock_ID_mask)
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
    ufloat_t tmp_i = (ufloat_t)(0.0);
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t u = (ufloat_t)(0.0);
    ufloat_t v = (ufloat_t)(0.0);
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
            f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            rho = f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8;
            u = ((f_1+f_5+f_8)-(f_3+f_6+f_7)) / rho;
            v = ((f_2+f_5+f_6)-(f_4+f_7+f_8)) / rho;
            udotu = u*u + v*v;
            
            // Interpolate rescaled fi to children if applicable.
            //
            // DDF 0.
            //
            cdotu = (ufloat_t)(0.0);
            tmp_i = (ufloat_t)(0.444444444444444)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_0 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 1.
            //
            cdotu = (u);
            tmp_i = (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_1 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 2.
            //
            cdotu = (v);
            tmp_i = (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_2 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 3.
            //
            cdotu = -(u);
            tmp_i = (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_3 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 4.
            //
            cdotu = -(v);
            tmp_i = (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_4 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 5.
            //
            cdotu = (u+v);
            tmp_i = (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_5 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 6.
            //
            cdotu = (v)-(u);
            tmp_i = (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_6 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 7.
            //
            cdotu = -(u+v);
            tmp_i = (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_7 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
            //
            // DDF 8.
            //
            cdotu = (u)-(v);
            tmp_i = (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
            s_F[threadIdx.x] = tmp_i + (f_8 - tmp_i)*(tau_ratio);
            __syncthreads();
            //\tChild 0.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[0] + (s_F[1]-s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[4]-s_F[0])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[5]-s_F[4]-s_F[1]+s_F[0])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 1.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[2] + (s_F[3]-s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[6]-s_F[2])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[7]-s_F[6]-s_F[3]+s_F[2])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 2.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[8] + (s_F[9]-s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[12]-s_F[8])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[13]-s_F[12]-s_F[9]+s_F[8])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            //\tChild 3.
            if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            {
                cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = s_F[10] + (s_F[11]-s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + (s_F[14]-s_F[10])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + (s_F[15]-s_F[14]-s_F[11]+s_F[10])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
            }
            __syncthreads();
            
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Interpolate_Linear_Original_D2Q9(int i_dev, int L, int var)
{
	if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_INTERFACE)
	{
		Cu_Interpolate_Linear_Original_D2Q9<ufloat_t,ufloat_g_t,AP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
	}
	if (mesh->n_ids[i_dev][L]>0 && var==V_INTERP_ADDED)
	{
		Cu_Interpolate_Linear_Original_D2Q9<ufloat_t,ufloat_g_t,AP,1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, (tau_vec[L+1]/tau_vec[L]), &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
	}

	return 0;
}

