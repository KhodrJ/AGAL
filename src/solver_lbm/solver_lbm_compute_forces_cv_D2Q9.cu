/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Mon Jul 14 23:00:26 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, int post_step>
__global__
void Cu_ComputeForces_CV_D2Q9(int is_root,int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t dx_L,ufloat_t dv_L,ufloat_t otau_0,int *__restrict__ id_set_idev_L,int *__restrict__ cells_ID_mask,ufloat_t *__restrict__ cells_f_F,ufloat_t *__restrict__ cells_f_F_aux,ufloat_t *__restrict__ cblock_f_X,int *__restrict__ cblock_ID_nbr,int *__restrict__ cblock_ID_nbr_child,int *__restrict__ cblock_ID_mask,int *__restrict__ cblock_ID_onb,ufloat_t *__restrict__ cblock_f_Ff,bool geometry_init,int order,ufloat_t cv_xm,ufloat_t cv_xM,ufloat_t cv_ym,ufloat_t cv_yM,ufloat_t cv_zm,ufloat_t cv_zM)
{
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
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
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int valid_block = -1;
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
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t rhou = (ufloat_t)(0.0);
    ufloat_t rhov = (ufloat_t)(0.0);
    ufloat_t x = (ufloat_t)(0.0);
    ufloat_t y = (ufloat_t)(0.0);
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
            participatesV = CheckPointInRegion2D(x,y,cv_xm,cv_xM,cv_ym,cv_yM) && valid_mask != -1;
            
            // Load the DDFs. Compute the momentum in all cells.
            f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
            f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            rho = ( (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8) );
            rhou = ( (f_1+f_5+f_8)-(f_3+f_6+f_7) );
            rhov = ( (f_2+f_5+f_6)-(f_4+f_7+f_8) );
            
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
            }
            
            // Perform collisions right here, but don't store.
            if (post_step==0)
            {
                rhou /= rho;
                rhov /= rho;
                udotu = (rhou*rhou+rhov*rhov);
                cdotu = (rhou);
                f_1 = f_1*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhov);
                f_2 = f_2*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhou);
                f_3 = f_3*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhov);
                f_4 = f_4*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou+rhov);
                f_5 = f_5*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhov)-(rhou);
                f_6 = f_6*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = -(rhou+rhov);
                f_7 = f_7*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
                cdotu = (rhou)-(rhov);
                f_8 = f_8*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            }
            
            // Now check if DDFs are leaving the CV.
            if (post_step==0)
            {
                participatesS = !CheckPointInRegion2D(x+1*dx_L,y+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fmx[threadIdx.x] += f_1;
                }
                participatesS = !CheckPointInRegion2D(x+0*dx_L,y+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fmy[threadIdx.x] += f_2;
                }
                participatesS = !CheckPointInRegion2D(x+-1*dx_L,y+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fpx[threadIdx.x] += f_3;
                }
                participatesS = !CheckPointInRegion2D(x+0*dx_L,y+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fpy[threadIdx.x] += f_4;
                }
                participatesS = !CheckPointInRegion2D(x+1*dx_L,y+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fmx[threadIdx.x] += f_5;
                    s_Fmy[threadIdx.x] += f_5;
                }
                participatesS = !CheckPointInRegion2D(x+-1*dx_L,y+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fpx[threadIdx.x] += f_6;
                    s_Fmy[threadIdx.x] += f_6;
                }
                participatesS = !CheckPointInRegion2D(x+-1*dx_L,y+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fpx[threadIdx.x] += f_7;
                    s_Fpy[threadIdx.x] += f_7;
                }
                participatesS = !CheckPointInRegion2D(x+1*dx_L,y+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fmx[threadIdx.x] += f_8;
                    s_Fpy[threadIdx.x] += f_8;
                }
            }
            
            // Now check if DDFs are entering the CV.
            if (post_step==1)
            {
                participatesS = !CheckPointInRegion2D(x+-1*dx_L,y+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fpx[threadIdx.x] += f_1;
                }
                participatesS = !CheckPointInRegion2D(x+0*dx_L,y+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fpy[threadIdx.x] += f_2;
                }
                participatesS = !CheckPointInRegion2D(x+1*dx_L,y+0*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fmx[threadIdx.x] += f_3;
                }
                participatesS = !CheckPointInRegion2D(x+0*dx_L,y+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fmy[threadIdx.x] += f_4;
                }
                participatesS = !CheckPointInRegion2D(x+-1*dx_L,y+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fpx[threadIdx.x] += f_5;
                    s_Fpy[threadIdx.x] += f_5;
                }
                participatesS = !CheckPointInRegion2D(x+1*dx_L,y+-1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fmx[threadIdx.x] += f_6;
                    s_Fpy[threadIdx.x] += f_6;
                }
                participatesS = !CheckPointInRegion2D(x+1*dx_L,y+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fmx[threadIdx.x] += f_7;
                    s_Fmy[threadIdx.x] += f_7;
                }
                participatesS = !CheckPointInRegion2D(x+-1*dx_L,y+1*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
                if (participatesS && participatesV && is_root)
                {
                    s_Fpx[threadIdx.x] += f_8;
                    s_Fmy[threadIdx.x] += f_8;
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
                }
                else
                {
                    cblock_f_Ff[i_kap_b + 0*n_maxcblocks] += s_Fpx[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 1*n_maxcblocks] += s_Fmx[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 2*n_maxcblocks] += s_Fpy[0]*dv_L;
                    cblock_f_Ff[i_kap_b + 3*n_maxcblocks] += s_Fmy[0]*dv_L;
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeForces_CV_D2Q9(int i_dev, int L, int var)
{
	if (mesh->n_ids[i_dev][L]>0 && var==0)
	{
		Cu_ComputeForces_CV_D2Q9<ufloat_t,ufloat_g_t,AP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(L==N_LEVEL_START, mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], dvf_vec[L], dxf_vec[N_LEVEL_START]/tau_vec[N_LEVEL_START], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_f_Ff[i_dev], mesh->geometry_init, S_FORCE_TYPE, S_FORCEVOLUME_Xm, S_FORCEVOLUME_XM, S_FORCEVOLUME_Ym, S_FORCEVOLUME_YM, S_FORCEVOLUME_Zm, S_FORCEVOLUME_ZM);
	}
	if (mesh->n_ids[i_dev][L]>0 && var==1)
	{
		Cu_ComputeForces_CV_D2Q9<ufloat_t,ufloat_g_t,AP,1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(L==N_LEVEL_START, mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], dvf_vec[L], dxf_vec[N_LEVEL_START]/tau_vec[N_LEVEL_START], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_f_Ff[i_dev], mesh->geometry_init, S_FORCE_TYPE, S_FORCEVOLUME_Xm, S_FORCEVOLUME_XM, S_FORCEVOLUME_Ym, S_FORCEVOLUME_YM, S_FORCEVOLUME_Zm, S_FORCEVOLUME_ZM);
	}

	return 0;
}

