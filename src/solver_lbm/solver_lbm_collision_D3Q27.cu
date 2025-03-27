/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Mar 23 06:25:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

template <typename ufloat_t, const ArgsPack *AP>
__global__
void Cu_Collide_D3Q27(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t tau_L,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,int *cblock_ID_nbr,int *cblock_ID_nbr_child,int *cblock_ID_mask,int *cblock_ID_onb)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = (threadIdx.x / 4) / 4;
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int nbr_kap_c = -1;
    int block_on_boundary = -1;
    int nbr_1 __attribute__((unused)) = -1;
    int nbr_2 __attribute__((unused)) = -1;
    int nbr_3 __attribute__((unused)) = -1;
    int nbr_4 __attribute__((unused)) = -1;
    int nbr_5 __attribute__((unused)) = -1;
    int nbr_6 __attribute__((unused)) = -1;
    int nbr_7 __attribute__((unused)) = -1;
    int nbr_8 __attribute__((unused)) = -1;
    int nbr_9 __attribute__((unused)) = -1;
    int nbr_10 __attribute__((unused)) = -1;
    int nbr_11 __attribute__((unused)) = -1;
    int nbr_12 __attribute__((unused)) = -1;
    int nbr_13 __attribute__((unused)) = -1;
    int nbr_14 __attribute__((unused)) = -1;
    int nbr_15 __attribute__((unused)) = -1;
    int nbr_16 __attribute__((unused)) = -1;
    int nbr_17 __attribute__((unused)) = -1;
    int nbr_18 __attribute__((unused)) = -1;
    int nbr_19 __attribute__((unused)) = -1;
    int nbr_20 __attribute__((unused)) = -1;
    int nbr_21 __attribute__((unused)) = -1;
    int nbr_22 __attribute__((unused)) = -1;
    int nbr_23 __attribute__((unused)) = -1;
    int nbr_24 __attribute__((unused)) = -1;
    int nbr_25 __attribute__((unused)) = -1;
    int nbr_26 __attribute__((unused)) = -1;
    f_0 = (ufloat_t)0.0;
    f_1 = (ufloat_t)0.0;
    f_2 = (ufloat_t)0.0;
    f_3 = (ufloat_t)0.0;
    f_4 = (ufloat_t)0.0;
    f_5 = (ufloat_t)0.0;
    f_6 = (ufloat_t)0.0;
    f_7 = (ufloat_t)0.0;
    f_8 = (ufloat_t)0.0;
    f_9 = (ufloat_t)0.0;
    f_10 = (ufloat_t)0.0;
    f_11 = (ufloat_t)0.0;
    f_12 = (ufloat_t)0.0;
    f_13 = (ufloat_t)0.0;
    f_14 = (ufloat_t)0.0;
    f_15 = (ufloat_t)0.0;
    f_16 = (ufloat_t)0.0;
    f_17 = (ufloat_t)0.0;
    f_18 = (ufloat_t)0.0;
    f_19 = (ufloat_t)0.0;
    f_20 = (ufloat_t)0.0;
    f_21 = (ufloat_t)0.0;
    f_22 = (ufloat_t)0.0;
    f_23 = (ufloat_t)0.0;
    f_24 = (ufloat_t)0.0;
    f_25 = (ufloat_t)0.0;
    f_26 = (ufloat_t)0.0;
    ufloat_t cdotu = (ufloat_t)0.0;
    ufloat_t udotu = (ufloat_t)0.0;
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
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(block_on_boundary>=-2) # TODO))
        {
            // Retrieve DDFs in alternating index. Compute macroscopic properties.
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
            u = ((N_Pf(0)*f_0+N_Pf(1)*f_1+N_Pf(-1)*f_2+N_Pf(0)*f_3+N_Pf(0)*f_4+N_Pf(0)*f_5+N_Pf(0)*f_6+N_Pf(1)*f_7+N_Pf(-1)*f_8+N_Pf(1)*f_9+N_Pf(-1)*f_10+N_Pf(0)*f_11+N_Pf(0)*f_12+N_Pf(1)*f_13+N_Pf(-1)*f_14+N_Pf(1)*f_15+N_Pf(-1)*f_16+N_Pf(0)*f_17+N_Pf(0)*f_18+N_Pf(1)*f_19+N_Pf(-1)*f_20+N_Pf(1)*f_21+N_Pf(-1)*f_22+N_Pf(1)*f_23+N_Pf(-1)*f_24+N_Pf(-1)*f_25+N_Pf(1)*f_26)) / rho_kap;
            v = ((N_Pf(0)*f_0+N_Pf(0)*f_1+N_Pf(0)*f_2+N_Pf(1)*f_3+N_Pf(-1)*f_4+N_Pf(0)*f_5+N_Pf(0)*f_6+N_Pf(1)*f_7+N_Pf(-1)*f_8+N_Pf(0)*f_9+N_Pf(0)*f_10+N_Pf(1)*f_11+N_Pf(-1)*f_12+N_Pf(-1)*f_13+N_Pf(1)*f_14+N_Pf(0)*f_15+N_Pf(0)*f_16+N_Pf(1)*f_17+N_Pf(-1)*f_18+N_Pf(1)*f_19+N_Pf(-1)*f_20+N_Pf(1)*f_21+N_Pf(-1)*f_22+N_Pf(-1)*f_23+N_Pf(1)*f_24+N_Pf(1)*f_25+N_Pf(-1)*f_26)) / rho_kap;
            w = ((N_Pf(0)*f_0+N_Pf(0)*f_1+N_Pf(0)*f_2+N_Pf(0)*f_3+N_Pf(0)*f_4+N_Pf(1)*f_5+N_Pf(-1)*f_6+N_Pf(0)*f_7+N_Pf(0)*f_8+N_Pf(1)*f_9+N_Pf(-1)*f_10+N_Pf(1)*f_11+N_Pf(-1)*f_12+N_Pf(0)*f_13+N_Pf(0)*f_14+N_Pf(-1)*f_15+N_Pf(1)*f_16+N_Pf(-1)*f_17+N_Pf(1)*f_18+N_Pf(1)*f_19+N_Pf(-1)*f_20+N_Pf(-1)*f_21+N_Pf(1)*f_22+N_Pf(1)*f_23+N_Pf(-1)*f_24+N_Pf(1)*f_25+N_Pf(-1)*f_26)) / rho_kap;
            udotu = u*u + v*v + w*w;
            
            // Apply the turbulence model.
            
            // Perform collision.
            cdotu = (ufloat_t)(0.0);
            f_0 = omegap*f_0 + omega*rho*(ufloat_t)0.296296296296296*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u);
            f_1 = omegap*f_1 + omega*rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(u);
            f_2 = omegap*f_2 + omega*rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (v);
            f_3 = omegap*f_3 + omega*rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(v);
            f_4 = omegap*f_4 + omega*rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (w);
            f_5 = omegap*f_5 + omega*rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(w);
            f_6 = omegap*f_6 + omega*rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u+v);
            f_7 = omegap*f_7 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(u+v);
            f_8 = omegap*f_8 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u+w);
            f_9 = omegap*f_9 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(u+w);
            f_10 = omegap*f_10 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (v+w);
            f_11 = omegap*f_11 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(v+w);
            f_12 = omegap*f_12 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u)-(v);
            f_13 = omegap*f_13 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (v)-(u);
            f_14 = omegap*f_14 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u)-(w);
            f_15 = omegap*f_15 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (w)-(u);
            f_16 = omegap*f_16 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (v)-(w);
            f_17 = omegap*f_17 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (w)-(v);
            f_18 = omegap*f_18 + omega*rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u+v+w);
            f_19 = omegap*f_19 + omega*rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(u+v+w);
            f_20 = omegap*f_20 + omega*rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u+v)-(w);
            f_21 = omegap*f_21 + omega*rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (w)-(u+v);
            f_22 = omegap*f_22 + omega*rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u+w)-(v);
            f_23 = omegap*f_23 + omega*rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (v)-(u+w);
            f_24 = omegap*f_24 + omega*rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (v+w)-(u);
            f_25 = omegap*f_25 + omega*rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u)-(v+w);
            f_26 = omegap*f_26 + omega*rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            
            // Apply no-slip or anti-bounce back boundary conditions. Free slip is implemented elsewhere.
            
            block_on_boundary = cblock_ID_onb[i_kap_b];
            if (block_on_boundary)
            {
                nbr_1 = cblock_ID_nbr[i_kap + 1*n_maxcblocks];
                nbr_2 = cblock_ID_nbr[i_kap + 2*n_maxcblocks];
                nbr_3 = cblock_ID_nbr[i_kap + 3*n_maxcblocks];
                nbr_4 = cblock_ID_nbr[i_kap + 4*n_maxcblocks];
                nbr_5 = cblock_ID_nbr[i_kap + 5*n_maxcblocks];
                nbr_6 = cblock_ID_nbr[i_kap + 6*n_maxcblocks];
                nbr_7 = cblock_ID_nbr[i_kap + 7*n_maxcblocks];
                nbr_8 = cblock_ID_nbr[i_kap + 8*n_maxcblocks];
                nbr_9 = cblock_ID_nbr[i_kap + 9*n_maxcblocks];
                nbr_10 = cblock_ID_nbr[i_kap + 10*n_maxcblocks];
                nbr_11 = cblock_ID_nbr[i_kap + 11*n_maxcblocks];
                nbr_12 = cblock_ID_nbr[i_kap + 12*n_maxcblocks];
                nbr_13 = cblock_ID_nbr[i_kap + 13*n_maxcblocks];
                nbr_14 = cblock_ID_nbr[i_kap + 14*n_maxcblocks];
                nbr_15 = cblock_ID_nbr[i_kap + 15*n_maxcblocks];
                nbr_16 = cblock_ID_nbr[i_kap + 16*n_maxcblocks];
                nbr_17 = cblock_ID_nbr[i_kap + 17*n_maxcblocks];
                nbr_18 = cblock_ID_nbr[i_kap + 18*n_maxcblocks];
                nbr_19 = cblock_ID_nbr[i_kap + 19*n_maxcblocks];
                nbr_20 = cblock_ID_nbr[i_kap + 20*n_maxcblocks];
                nbr_21 = cblock_ID_nbr[i_kap + 21*n_maxcblocks];
                nbr_22 = cblock_ID_nbr[i_kap + 22*n_maxcblocks];
                nbr_23 = cblock_ID_nbr[i_kap + 23*n_maxcblocks];
                nbr_24 = cblock_ID_nbr[i_kap + 24*n_maxcblocks];
                nbr_25 = cblock_ID_nbr[i_kap + 25*n_maxcblocks];
                nbr_26 = cblock_ID_nbr[i_kap + 26*n_maxcblocks];
                //
                // nbr 1
                //
                // Consider DDF 1
                // Consider DDF 7
                // Consider DDF 9
                // Consider DDF 13
                // Consider DDF 15
                // Consider DDF 19
                // Consider DDF 21
                // Consider DDF 23
                // Consider DDF 26
                //
                // nbr 2
                //
                // Consider DDF 2
                // Consider DDF 8
                // Consider DDF 10
                // Consider DDF 14
                // Consider DDF 16
                // Consider DDF 20
                // Consider DDF 22
                // Consider DDF 24
                // Consider DDF 25
                //
                // nbr 3
                //
                // Consider DDF 3
                // Consider DDF 7
                // Consider DDF 11
                // Consider DDF 14
                // Consider DDF 17
                // Consider DDF 19
                // Consider DDF 21
                // Consider DDF 24
                // Consider DDF 25
                //
                // nbr 4
                //
                // Consider DDF 4
                // Consider DDF 8
                // Consider DDF 12
                // Consider DDF 13
                // Consider DDF 18
                // Consider DDF 20
                // Consider DDF 22
                // Consider DDF 23
                // Consider DDF 26
                //
                // nbr 5
                //
                // Consider DDF 5
                // Consider DDF 9
                // Consider DDF 11
                // Consider DDF 16
                // Consider DDF 18
                // Consider DDF 19
                // Consider DDF 22
                // Consider DDF 23
                // Consider DDF 25
                //
                // nbr 6
                //
                // Consider DDF 6
                // Consider DDF 10
                // Consider DDF 12
                // Consider DDF 15
                // Consider DDF 17
                // Consider DDF 20
                // Consider DDF 21
                // Consider DDF 24
                // Consider DDF 26
                //
                // nbr 7
                //
                // Consider DDF 7
                // Consider DDF 19
                // Consider DDF 21
                //
                // nbr 8
                //
                // Consider DDF 8
                // Consider DDF 20
                // Consider DDF 22
                //
                // nbr 9
                //
                // Consider DDF 9
                // Consider DDF 19
                // Consider DDF 23
                //
                // nbr 10
                //
                // Consider DDF 10
                // Consider DDF 20
                // Consider DDF 24
                //
                // nbr 11
                //
                // Consider DDF 11
                // Consider DDF 19
                // Consider DDF 25
                //
                // nbr 12
                //
                // Consider DDF 12
                // Consider DDF 20
                // Consider DDF 26
                //
                // nbr 13
                //
                // Consider DDF 13
                // Consider DDF 23
                // Consider DDF 26
                //
                // nbr 14
                //
                // Consider DDF 14
                // Consider DDF 24
                // Consider DDF 25
                //
                // nbr 15
                //
                // Consider DDF 15
                // Consider DDF 21
                // Consider DDF 26
                //
                // nbr 16
                //
                // Consider DDF 16
                // Consider DDF 22
                // Consider DDF 25
                //
                // nbr 17
                //
                // Consider DDF 17
                // Consider DDF 21
                // Consider DDF 24
                //
                // nbr 18
                //
                // Consider DDF 18
                // Consider DDF 22
                // Consider DDF 23
                //
                // nbr 19
                //
                // Consider DDF 19
                //
                // nbr 20
                //
                // Consider DDF 20
                //
                // nbr 21
                //
                // Consider DDF 21
                //
                // nbr 22
                //
                // Consider DDF 22
                //
                // nbr 23
                //
                // Consider DDF 23
                //
                // nbr 24
                //
                // Consider DDF 24
                //
                // nbr 25
                //
                // Consider DDF 25
                //
                // nbr 26
                //
                // Consider DDF 26
            }
            
            // Write DDFs in proper index.
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_9;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_10;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_11;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_12;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_13;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_14;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_15;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_16;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_17;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_18;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 19*n_maxcells] = f_19;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 20*n_maxcells] = f_20;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 21*n_maxcells] = f_21;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 22*n_maxcells] = f_22;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 23*n_maxcells] = f_23;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 24*n_maxcells] = f_24;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 25*n_maxcells] = f_25;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 26*n_maxcells] = f_26;
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Collide_D3Q27(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collide_D3Q27<ufloat_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

