/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Tue Jul 22 17:13:20 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, int post_step>
__global__
void Cu_ComputeForces_CV_D3Q27
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
	const int *__restrict__ cblock_ID_nbr,
	const int *__restrict__ cblock_ID_nbr_child,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_onb,
	ufloat_t *__restrict__ cblock_f_Ff,
	const bool geometry_init,
	const int order,
	const ufloat_t cv_xm,
	const ufloat_t cv_xM,
	const ufloat_t cv_ym,
	const ufloat_t cv_yM,
	const ufloat_t cv_zm,
	const ufloat_t cv_zM
)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ ufloat_t s_Fpx[M_TBLOCK];
    __shared__ ufloat_t s_Fmx[M_TBLOCK];
    __shared__ ufloat_t s_Fpy[M_TBLOCK];
    __shared__ ufloat_t s_Fmy[M_TBLOCK];
    __shared__ ufloat_t s_Fpz[M_TBLOCK];
    __shared__ ufloat_t s_Fmz[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = (threadIdx.x / 4) / 4;
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int valid_mask = -10;
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
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t rhou = (ufloat_t)(0.0);
    ufloat_t rhov = (ufloat_t)(0.0);
    ufloat_t rhow = (ufloat_t)(0.0);
    ufloat_t x = (ufloat_t)(0.0);
    ufloat_t y = (ufloat_t)(0.0);
    ufloat_t z = (ufloat_t)(0.0);
    bool participatesV = false;
    bool participatesS = false;
    ufloat_t omeg = otau_0;
    ufloat_t omegp = (ufloat_t)(1.0) - omeg;
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
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1))
        {
            // Compute cell coordinates and retrieve macroscopic properties.
            valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            i_kap_bc = cblock_ID_nbr_child[i_kap_b];
            s_Fpx[threadIdx.x] = 0;
            s_Fmx[threadIdx.x] = 0;
            s_Fpy[threadIdx.x] = 0;
            s_Fmy[threadIdx.x] = 0;
            s_Fpz[threadIdx.x] = 0;
            s_Fmz[threadIdx.x] = 0;
            __syncthreads();
            
            // Load the cell coordinates. Check if the current cell participates.
            x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
            y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
            z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
            participatesV = CheckPointInRegion3D(x,y,z,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM) && valid_mask != -1;
            
            // Load the DDFs. Compute the momentum in all cells.
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
            rho = ( (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26) );
            rhou = ( (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26)-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) );
            rhov = ( (f_3+f_7+f_11+f_14+f_17+f_19+f_21+f_24+f_25)-(f_4+f_8+f_12+f_13+f_18+f_20+f_22+f_23+f_26) );
            rhow = ( (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26) );
            
            // Add force contributions in the volume.
            if (participatesV && post_step==0 && i_kap_bc<0)
            {
                if (rhou > 0)
                {
                    s_Fpx[threadIdx.x] += rhou;
                }
                if (rhou < 0)
                {
                    s_Fmx[threadIdx.x] += rhou;
                }
                if (rhov > 0)
                {
                    s_Fpy[threadIdx.x] += rhov;
                }
                if (rhov < 0)
                {
                    s_Fmy[threadIdx.x] += rhov;
                }
                if (rhow > 0)
                {
                    s_Fpz[threadIdx.x] += rhow;
                }
                if (rhow < 0)
                {
                    s_Fmz[threadIdx.x] += rhow;
                }
            }
            if (participatesV && post_step==1 && i_kap_bc<0)
            {
                if (rhou > 0)
                {
                    s_Fmx[threadIdx.x] += rhou;
                }
                if (rhou < 0)
                {
                    s_Fpx[threadIdx.x] += rhou;
                }
                if (rhov > 0)
                {
                    s_Fmy[threadIdx.x] += rhov;
                }
                if (rhov < 0)
                {
                    s_Fpy[threadIdx.x] += rhov;
                }
                if (rhow > 0)
                {
                    s_Fmz[threadIdx.x] += rhow;
                }
                if (rhow < 0)
                {
                    s_Fpz[threadIdx.x] += rhow;
                }
            }
            
            // Perform collisions right here, but don't store.
            if (post_step==0)
            {
                rhou /= rho;
                rhov /= rho;
                rhow /= rho;
                udotu = (rhou*rhou+rhov*rhov+rhow*rhow);
                cdotu = (rhou);
                f_1 = f_1*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhou);
                f_2 = f_2*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhov);
                f_3 = f_3*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhov);
                f_4 = f_4*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhow);
                f_5 = f_5*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhow);
                f_6 = f_6*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou+rhov);
                f_7 = f_7*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhou+rhov);
                f_8 = f_8*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou+rhow);
                f_9 = f_9*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhou+rhow);
                f_10 = f_10*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhov+rhow);
                f_11 = f_11*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhov+rhow);
                f_12 = f_12*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou)-(rhov);
                f_13 = f_13*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhov)-(rhou);
                f_14 = f_14*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou)-(rhow);
                f_15 = f_15*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhow)-(rhou);
                f_16 = f_16*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhov)-(rhow);
                f_17 = f_17*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhow)-(rhov);
                f_18 = f_18*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou+rhov+rhow);
                f_19 = f_19*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhou+rhov+rhow);
                f_20 = f_20*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou+rhov)-(rhow);
                f_21 = f_21*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhow)-(rhou+rhov);
                f_22 = f_22*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou+rhow)-(rhov);
                f_23 = f_23*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhov)-(rhou+rhow);
                f_24 = f_24*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhov+rhow)-(rhou);
                f_25 = f_25*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou)-(rhov+rhow);
                f_26 = f_26*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            }
            
            // Now check if DDFs are leaving the CV.
            if (post_step==0 && is_root)
            {
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+0*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_1;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+0*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_2;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmy[threadIdx.x] += f_3;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+-1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpy[threadIdx.x] += f_4;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+0*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmz[threadIdx.x] += f_5;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+0*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpz[threadIdx.x] += f_6;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_7;
                    s_Fmy[threadIdx.x] += f_7;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+-1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_8;
                    s_Fpy[threadIdx.x] += f_8;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+0*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_9;
                    s_Fmz[threadIdx.x] += f_9;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+0*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_10;
                    s_Fpz[threadIdx.x] += f_10;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmy[threadIdx.x] += f_11;
                    s_Fmz[threadIdx.x] += f_11;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+-1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpy[threadIdx.x] += f_12;
                    s_Fpz[threadIdx.x] += f_12;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+-1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_13;
                    s_Fpy[threadIdx.x] += f_13;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_14;
                    s_Fmy[threadIdx.x] += f_14;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+0*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_15;
                    s_Fpz[threadIdx.x] += f_15;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+0*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_16;
                    s_Fmz[threadIdx.x] += f_16;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmy[threadIdx.x] += f_17;
                    s_Fpz[threadIdx.x] += f_17;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+-1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpy[threadIdx.x] += f_18;
                    s_Fmz[threadIdx.x] += f_18;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_19;
                    s_Fmy[threadIdx.x] += f_19;
                    s_Fmz[threadIdx.x] += f_19;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+-1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_20;
                    s_Fpy[threadIdx.x] += f_20;
                    s_Fpz[threadIdx.x] += f_20;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_21;
                    s_Fmy[threadIdx.x] += f_21;
                    s_Fpz[threadIdx.x] += f_21;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+-1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_22;
                    s_Fpy[threadIdx.x] += f_22;
                    s_Fmz[threadIdx.x] += f_22;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+-1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_23;
                    s_Fpy[threadIdx.x] += f_23;
                    s_Fmz[threadIdx.x] += f_23;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_24;
                    s_Fmy[threadIdx.x] += f_24;
                    s_Fpz[threadIdx.x] += f_24;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_25;
                    s_Fmy[threadIdx.x] += f_25;
                    s_Fmz[threadIdx.x] += f_25;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+-1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_26;
                    s_Fpy[threadIdx.x] += f_26;
                    s_Fpz[threadIdx.x] += f_26;
                }
            }
            
            // Now check if DDFs are entering the CV.
            if (post_step==1 && is_root)
            {
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+0*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_1;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+0*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_2;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+-1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpy[threadIdx.x] += f_3;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmy[threadIdx.x] += f_4;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+0*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpz[threadIdx.x] += f_5;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+0*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmz[threadIdx.x] += f_6;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+-1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_7;
                    s_Fpy[threadIdx.x] += f_7;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_8;
                    s_Fmy[threadIdx.x] += f_8;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+0*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_9;
                    s_Fpz[threadIdx.x] += f_9;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+0*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_10;
                    s_Fmz[threadIdx.x] += f_10;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+-1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpy[threadIdx.x] += f_11;
                    s_Fpz[threadIdx.x] += f_11;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmy[threadIdx.x] += f_12;
                    s_Fmz[threadIdx.x] += f_12;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_13;
                    s_Fmy[threadIdx.x] += f_13;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+-1*dx_L,z+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_14;
                    s_Fpy[threadIdx.x] += f_14;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+0*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_15;
                    s_Fmz[threadIdx.x] += f_15;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+0*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_16;
                    s_Fpz[threadIdx.x] += f_16;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+-1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpy[threadIdx.x] += f_17;
                    s_Fmz[threadIdx.x] += f_17;
                }
                participatesS = !CheckPointInRegion3D(x+0*dx_L,y+1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmy[threadIdx.x] += f_18;
                    s_Fpz[threadIdx.x] += f_18;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+-1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_19;
                    s_Fpy[threadIdx.x] += f_19;
                    s_Fpz[threadIdx.x] += f_19;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_20;
                    s_Fmy[threadIdx.x] += f_20;
                    s_Fmz[threadIdx.x] += f_20;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+-1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_21;
                    s_Fpy[threadIdx.x] += f_21;
                    s_Fmz[threadIdx.x] += f_21;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_22;
                    s_Fmy[threadIdx.x] += f_22;
                    s_Fpz[threadIdx.x] += f_22;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_23;
                    s_Fmy[threadIdx.x] += f_23;
                    s_Fpz[threadIdx.x] += f_23;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+-1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_24;
                    s_Fpy[threadIdx.x] += f_24;
                    s_Fmz[threadIdx.x] += f_24;
                }
                participatesS = !CheckPointInRegion3D(x+1*dx_L,y+-1*dx_L,z+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fmx[threadIdx.x] += f_25;
                    s_Fpy[threadIdx.x] += f_25;
                    s_Fpz[threadIdx.x] += f_25;
                }
                participatesS = !CheckPointInRegion3D(x+-1*dx_L,y+1*dx_L,z+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
                if (participatesS && participatesV)
                {
                    s_Fpx[threadIdx.x] += f_26;
                    s_Fmy[threadIdx.x] += f_26;
                    s_Fmz[threadIdx.x] += f_26;
                }
            }
            
            // Reductions for the sums of force contributions in this cell-block.
            __syncthreads();
            for (int s=blockDim.x/2; s>0; s>>=1)
            {
            if (threadIdx.x < s)
            {
                s_Fpx[threadIdx.x] = s_Fpx[threadIdx.x] + s_Fpx[threadIdx.x + s];
                s_Fmx[threadIdx.x] = s_Fmx[threadIdx.x] + s_Fmx[threadIdx.x + s];
                s_Fpy[threadIdx.x] = s_Fpy[threadIdx.x] + s_Fpy[threadIdx.x + s];
                s_Fmy[threadIdx.x] = s_Fmy[threadIdx.x] + s_Fmy[threadIdx.x + s];
                s_Fpz[threadIdx.x] = s_Fpz[threadIdx.x] + s_Fpz[threadIdx.x + s];
                s_Fmz[threadIdx.x] = s_Fmz[threadIdx.x] + s_Fmz[threadIdx.x + s];
            }
            __syncthreads();
            }
            // Store the sums of contributions in global memory; this will be reduced further later.
            if (threadIdx.x == 0)
            {
                if (post_step == 0)
                {
                    cblock_f_Ff[i_kap_b + 0*n_maxcblocks] = s_Fpx[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 1*n_maxcblocks] = s_Fmx[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 2*n_maxcblocks] = s_Fpy[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 3*n_maxcblocks] = s_Fmy[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 4*n_maxcblocks] = s_Fpz[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 5*n_maxcblocks] = s_Fmz[0]*dv_L;
                }
                else
                {
                    cblock_f_Ff[i_kap_b + 0*n_maxcblocks] += s_Fpx[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 1*n_maxcblocks] += s_Fmx[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 2*n_maxcblocks] += s_Fpy[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 3*n_maxcblocks] += s_Fmy[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 4*n_maxcblocks] += s_Fpz[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 5*n_maxcblocks] += s_Fmz[0]*dv_L;
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeForces_CV_D3Q27(int i_dev, int L, int var)
{
	if (mesh->n_ids[i_dev][L]>0 && var==0)
	{
		Cu_ComputeForces_CV_D3Q27<ufloat_t,ufloat_g_t,AP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(L==N_LEVEL_START, mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], dvf_vec[L], dxf_vec[N_LEVEL_START]/tau_vec[N_LEVEL_START], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_f_Ff[i_dev], mesh->geometry_init, S_FORCE_TYPE, S_FORCEVOLUME_Xm, S_FORCEVOLUME_XM, S_FORCEVOLUME_Ym, S_FORCEVOLUME_YM, S_FORCEVOLUME_Zm, S_FORCEVOLUME_ZM);
	}
	if (mesh->n_ids[i_dev][L]>0 && var==1)
	{
		Cu_ComputeForces_CV_D3Q27<ufloat_t,ufloat_g_t,AP,1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(L==N_LEVEL_START, mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], dvf_vec[L], dxf_vec[N_LEVEL_START]/tau_vec[N_LEVEL_START], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_f_Ff[i_dev], mesh->geometry_init, S_FORCE_TYPE, S_FORCEVOLUME_Xm, S_FORCEVOLUME_XM, S_FORCEVOLUME_Ym, S_FORCEVOLUME_YM, S_FORCEVOLUME_Zm, S_FORCEVOLUME_ZM);
	}

	return 0;
}

