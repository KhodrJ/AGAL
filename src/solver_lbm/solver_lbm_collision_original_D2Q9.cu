/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Thu Apr 24 00:19:00 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Collision_Original_D2Q9(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t dx_L,ufloat_t tau_L,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,ufloat_t *cblock_f_X,int *cblock_ID_nbr,int *cblock_ID_nbr_child,int *cblock_ID_mask,int *cblock_ID_onb)
{
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    ufloat_t x __attribute__((unused)) = (ufloat_t)(0.0);
    ufloat_t y __attribute__((unused)) = (ufloat_t)(0.0);
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int i_Q = -1;
    int nbr_kap_b = -1;
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
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t u = (ufloat_t)(0.0);
    ufloat_t v = (ufloat_t)(0.0);
    ufloat_t cdotu = (ufloat_t)(0.0);
    ufloat_t udotu = (ufloat_t)(0.0);
    ufloat_t omeg = dx_L / tau_L;
    ufloat_t omegp = (ufloat_t)(1.0) - omeg;
    int nbr_id_1 = (ufloat_t)(0.0);
    int nbr_id_2 = (ufloat_t)(0.0);
    int nbr_id_3 = (ufloat_t)(0.0);
    int nbr_id_4 = (ufloat_t)(0.0);
    int nbr_id_5 = (ufloat_t)(0.0);
    int nbr_id_6 = (ufloat_t)(0.0);
    int nbr_id_7 = (ufloat_t)(0.0);
    int nbr_id_8 = (ufloat_t)(0.0);
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
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(block_on_boundary>-2)))
        {
            block_on_boundary = cblock_ID_onb[i_kap_b];
            if (block_on_boundary == 1)
            {
                nbr_id_1 = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
                nbr_id_2 = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
                nbr_id_3 = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
                nbr_id_4 = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
                nbr_id_5 = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
                nbr_id_6 = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
                nbr_id_7 = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
                nbr_id_8 = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
            }
            for (int i_q = 0; i_q < Nqx; i_q += 1)
            {
                for (int j_q = 0; j_q < Nqx; j_q += 1)
                {
                    // Retrieve DDFs and compute macroscopic properties.
                    i_Q = i_q + Nqx*j_q;
                    x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I) + i_q*4*dx_L;
                    y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J) + j_q*4*dx_L;
                    f_0 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells];
                    f_1 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                    f_2 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                    f_3 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                    f_4 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                    f_5 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                    f_6 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                    f_7 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                    f_8 = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                    rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8);
                    u = ((f_1+f_5+f_8)-(f_3+f_6+f_7)) / rho;
                    v = ((f_2+f_5+f_6)-(f_4+f_7+f_8)) / rho;
                    udotu = u*u + v*v;
                    
                    // Eddy viscosity calculation.
                    
                    // Collision step.
                    cdotu = (ufloat_t)(0.0);
                    f_0 = f_0*omegp + ( (ufloat_t)(0.444444444444444)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    cdotu = (u);
                    f_1 = f_1*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    cdotu = (v);
                    f_2 = f_2*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    cdotu = -(u);
                    f_3 = f_3*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    cdotu = -(v);
                    f_4 = f_4*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    cdotu = (u+v);
                    f_5 = f_5*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    cdotu = (v)-(u);
                    f_6 = f_6*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    cdotu = -(u+v);
                    f_7 = f_7*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    cdotu = (u)-(v);
                    f_8 = f_8*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                    
                    // Impose boundary conditions.
                    if (block_on_boundary == 1)
                    {
                        // nbr 1
                        if (((i_q==Nqx-1)))
                        {
                            nbr_kap_b = nbr_id_1;
                            if ((I+4*i_q==4*Nqx-1))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_1 = f_1 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (u);
                                    f_1 = -f_1 + (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_1 = f_1 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_1 = f_1 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                            if ((I+4*i_q==4*Nqx-1)and(J+4*j_q<4*Nqx-1))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (u+v);
                                    f_5 = -f_5 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                            if ((I+4*i_q==4*Nqx-1)and(J+4*j_q>0))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (u)-(v);
                                    f_8 = -f_8 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                        }
                        
                        // nbr 2
                        if (((j_q==Nqx-1)))
                        {
                            nbr_kap_b = nbr_id_2;
                            if ((J+4*j_q==4*Nqx-1))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.000000000000000);
                                    f_2 = f_2 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (v);
                                    f_2 = -f_2 + (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.000000000000000);
                                    f_2 = f_2 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.000000000000000);
                                    f_2 = f_2 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                            if ((I+4*i_q<4*Nqx-1)and(J+4*j_q==4*Nqx-1))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (u+v);
                                    f_5 = -f_5 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                            if ((I+4*i_q>0)and(J+4*j_q==4*Nqx-1))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (v)-(u);
                                    f_6 = -f_6 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                        }
                        
                        // nbr 3
                        if (((i_q==0)))
                        {
                            nbr_kap_b = nbr_id_3;
                            if ((I+4*i_q==0))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_3 = f_3 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = -(u);
                                    f_3 = -f_3 + (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_3 = f_3 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_3 = f_3 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                            if ((I+4*i_q==0)and(J+4*j_q<4*Nqx-1))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (v)-(u);
                                    f_6 = -f_6 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                            if ((I+4*i_q==0)and(J+4*j_q>0))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = -(u+v);
                                    f_7 = -f_7 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                        }
                        
                        // nbr 4
                        if (((j_q==0)))
                        {
                            nbr_kap_b = nbr_id_4;
                            if ((J+4*j_q==0))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.000000000000000);
                                    f_4 = f_4 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = -(v);
                                    f_4 = -f_4 + (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.000000000000000);
                                    f_4 = f_4 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.000000000000000);
                                    f_4 = f_4 - (ufloat_t)(2.0)*(ufloat_t)(0.111111111111111)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                            if ((I+4*i_q>0)and(J+4*j_q==0))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = -(u+v);
                                    f_7 = -f_7 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                            if ((I+4*i_q<4*Nqx-1)and(J+4*j_q==0))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (u)-(v);
                                    f_8 = -f_8 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                        }
                        
                        // nbr 5
                        if (((i_q==Nqx-1))and((j_q==Nqx-1)))
                        {
                            nbr_kap_b = nbr_id_5;
                            if ((I+4*i_q==4*Nqx-1)and(J+4*j_q==4*Nqx-1))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (u+v);
                                    f_5 = -f_5 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_5 = f_5 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                        }
                        
                        // nbr 6
                        if (((i_q==0))and((j_q==Nqx-1)))
                        {
                            nbr_kap_b = nbr_id_6;
                            if ((I+4*i_q==0)and(J+4*j_q==4*Nqx-1))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (v)-(u);
                                    f_6 = -f_6 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_6 = f_6 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                        }
                        
                        // nbr 7
                        if (((i_q==0))and((j_q==0)))
                        {
                            nbr_kap_b = nbr_id_7;
                            if ((I+4*i_q==0)and(J+4*j_q==0))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = -(u+v);
                                    f_7 = -f_7 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(-0.050000000000000);
                                    f_7 = f_7 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                        }
                        
                        // nbr 8
                        if (((i_q==Nqx-1))and((j_q==0)))
                        {
                            nbr_kap_b = nbr_id_8;
                            if ((I+4*i_q==4*Nqx-1)and(J+4*j_q==0))
                            {
                                if (nbr_kap_b == -1)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -2)
                                {
                                    cdotu = (u)-(v);
                                    f_8 = -f_8 + (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                }
                                if (nbr_kap_b == -3)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                                if (nbr_kap_b == -4)
                                {
                                    cdotu = (ufloat_t)(0.050000000000000);
                                    f_8 = f_8 - (ufloat_t)(2.0)*(ufloat_t)(0.027777777777778)*(ufloat_t)(3.0)*cdotu;
                                }
                            }
                        }
                        
                    }
                    
                    // Write fi* to global memory.
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = f_1;
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = f_2;
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = f_3;
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = f_4;
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = f_5;
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = f_6;
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = f_7;
                    cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = f_8;
                    __syncthreads();
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Collision_Original_D2Q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collision_Original_D2Q9<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

