/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Tue Jul  8 00:01:41 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "custom.h"
#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_SetInitialConditions_D3Q27(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,ufloat_t *cblock_f_X,int *cblock_ID_mask,int *cblock_ID_onb,ufloat_t dx_L)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int i_kap_b = -1;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = (threadIdx.x / 4) / 4;
    ufloat_t f_0 = (ufloat_t)(-1.0);
    ufloat_t f_1 = (ufloat_t)(-1.0);
    ufloat_t f_2 = (ufloat_t)(-1.0);
    ufloat_t f_3 = (ufloat_t)(-1.0);
    ufloat_t f_4 = (ufloat_t)(-1.0);
    ufloat_t f_5 = (ufloat_t)(-1.0);
    ufloat_t f_6 = (ufloat_t)(-1.0);
    ufloat_t f_7 = (ufloat_t)(-1.0);
    ufloat_t f_8 = (ufloat_t)(-1.0);
    ufloat_t f_9 = (ufloat_t)(-1.0);
    ufloat_t f_10 = (ufloat_t)(-1.0);
    ufloat_t f_11 = (ufloat_t)(-1.0);
    ufloat_t f_12 = (ufloat_t)(-1.0);
    ufloat_t f_13 = (ufloat_t)(-1.0);
    ufloat_t f_14 = (ufloat_t)(-1.0);
    ufloat_t f_15 = (ufloat_t)(-1.0);
    ufloat_t f_16 = (ufloat_t)(-1.0);
    ufloat_t f_17 = (ufloat_t)(-1.0);
    ufloat_t f_18 = (ufloat_t)(-1.0);
    ufloat_t f_19 = (ufloat_t)(-1.0);
    ufloat_t f_20 = (ufloat_t)(-1.0);
    ufloat_t f_21 = (ufloat_t)(-1.0);
    ufloat_t f_22 = (ufloat_t)(-1.0);
    ufloat_t f_23 = (ufloat_t)(-1.0);
    ufloat_t f_24 = (ufloat_t)(-1.0);
    ufloat_t f_25 = (ufloat_t)(-1.0);
    ufloat_t f_26 = (ufloat_t)(-1.0);
    ufloat_t cdotu = (ufloat_t)0.0;
    ufloat_t udotu = (ufloat_t)0.0;
    ufloat_t rho = (ufloat_t)1.0;
    ufloat_t u = (ufloat_t)0.0;
    ufloat_t v = (ufloat_t)0.0;
    ufloat_t x = (ufloat_t)0.0;
    ufloat_t y = (ufloat_t)0.0;
    ufloat_t w = (ufloat_t)0.0;
    ufloat_t z = (ufloat_t)0.0;
    
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
            //<Compute macroscopic properties.
            f_0 = (ufloat_t)(-1.0);
            f_1 = (ufloat_t)(-1.0);
            f_2 = (ufloat_t)(-1.0);
            f_3 = (ufloat_t)(-1.0);
            f_4 = (ufloat_t)(-1.0);
            f_5 = (ufloat_t)(-1.0);
            f_6 = (ufloat_t)(-1.0);
            f_7 = (ufloat_t)(-1.0);
            f_8 = (ufloat_t)(-1.0);
            f_9 = (ufloat_t)(-1.0);
            f_10 = (ufloat_t)(-1.0);
            f_11 = (ufloat_t)(-1.0);
            f_12 = (ufloat_t)(-1.0);
            f_13 = (ufloat_t)(-1.0);
            f_14 = (ufloat_t)(-1.0);
            f_15 = (ufloat_t)(-1.0);
            f_16 = (ufloat_t)(-1.0);
            f_17 = (ufloat_t)(-1.0);
            f_18 = (ufloat_t)(-1.0);
            f_19 = (ufloat_t)(-1.0);
            f_20 = (ufloat_t)(-1.0);
            f_21 = (ufloat_t)(-1.0);
            f_22 = (ufloat_t)(-1.0);
            f_23 = (ufloat_t)(-1.0);
            f_24 = (ufloat_t)(-1.0);
            f_25 = (ufloat_t)(-1.0);
            f_26 = (ufloat_t)(-1.0);
            if ( cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] != -1 )
            {
                x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + (ufloat_t)0.5*dx_L;
                y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + (ufloat_t)0.5*dx_L;
                z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + (ufloat_t)0.5*dx_L;
                Cu_ComputeIC<ufloat_t>(rho, u, v, w, x, y, z);
                udotu = u*u + v*v + w*w;
                cdotu = (ufloat_t)(0.0);
                f_0 = rho*(ufloat_t)0.296296296296296*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u);
                f_1 = rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = -(u);
                f_2 = rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (v);
                f_3 = rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = -(v);
                f_4 = rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (w);
                f_5 = rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = -(w);
                f_6 = rho*(ufloat_t)0.074074074074074*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u+v);
                f_7 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = -(u+v);
                f_8 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u+w);
                f_9 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = -(u+w);
                f_10 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (v+w);
                f_11 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = -(v+w);
                f_12 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u)-(v);
                f_13 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (v)-(u);
                f_14 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u)-(w);
                f_15 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (w)-(u);
                f_16 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (v)-(w);
                f_17 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (w)-(v);
                f_18 = rho*(ufloat_t)0.018518518518519*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u+v+w);
                f_19 = rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = -(u+v+w);
                f_20 = rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u+v)-(w);
                f_21 = rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (w)-(u+v);
                f_22 = rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u+w)-(v);
                f_23 = rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (v)-(u+w);
                f_24 = rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (v+w)-(u);
                f_25 = rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                cdotu = (u)-(v+w);
                f_26 = rho*(ufloat_t)0.004629629629630*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            }
            
            // Write DDFs in proper index.
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = f_19;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = f_20;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = f_21;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = f_22;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = f_23;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = f_24;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = f_25;
            cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = f_26;
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_SetInitialConditions_D3Q27(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_SetInitialConditions_D3Q27<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->dxf_vec[L]);
	}

	return 0;
}

