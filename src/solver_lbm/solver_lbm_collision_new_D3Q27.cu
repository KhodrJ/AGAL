/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Tue Jul  8 00:01:39 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Collision_New_D3Q27(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t dx_L,ufloat_t tau_L,int *__restrict__ id_set_idev_L,int *__restrict__ cells_ID_mask,ufloat_t *__restrict__ cells_f_F,ufloat_t *__restrict__ cells_f_F_aux,ufloat_t *__restrict__ cblock_f_X,int *__restrict__ cblock_ID_nbr,int *__restrict__ cblock_ID_nbr_child,int *__restrict__ cblock_ID_mask,int *__restrict__ cblock_ID_onb)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int valid_block = -1;
    int valid_mask = -1;
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
    ufloat_t cdotu = (ufloat_t)(0.0);
    ufloat_t udotu = (ufloat_t)(0.0);
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t u = (ufloat_t)(0.0);
    ufloat_t v = (ufloat_t)(0.0);
    ufloat_t w = (ufloat_t)(0.0);
    ufloat_t omeg = dx_L / tau_L;
    ufloat_t omegp = (ufloat_t)(1.0) - omeg;
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
        
        // Load data for conditions on cell-blocks.
        if (i_kap_b>-1)
        {
            i_kap_bc=cblock_ID_nbr_child[i_kap_b];
            valid_block=cblock_ID_mask[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(valid_block>-3)))
        {
            // o====================================================================================
            // | BATCH #<K>
            // o====================================================================================
            
            // Retrieve DDFs one by one and compute macroscopic properties.
            // I'm using up to store the positive contributions and um to store the negative ones. Same for v and w.
            valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            rho = (ufloat_t)(0.0);
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
            rho = (f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9+f_10+f_11+f_12+f_13+f_14+f_15+f_16+f_17+f_18+f_19+f_20+f_21+f_22+f_23+f_24+f_25+f_26);
            u = ( (f_1+f_7+f_9+f_13+f_15+f_19+f_21+f_23+f_26)-(f_2+f_8+f_10+f_14+f_16+f_20+f_22+f_24+f_25) )/rho;
            v = ( (f_3+f_7+f_11+f_14+f_17+f_19+f_21+f_24+f_25)-(f_4+f_8+f_12+f_13+f_18+f_20+f_22+f_23+f_26) )/rho;
            w = ( (f_5+f_9+f_11+f_16+f_18+f_19+f_22+f_23+f_25)-(f_6+f_10+f_12+f_15+f_17+f_20+f_21+f_24+f_26) )/rho;
            udotu = u*u + v*v + w*w;
            cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = rho;
            cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = u;
            cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = v;
            cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = w;
            // Collision step.
            cdotu = (ufloat_t)(0.0);
            f_0 = f_0*omegp + ( (ufloat_t)(0.296296296296296)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u);
            f_1 = f_1*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u);
            f_2 = f_2*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (v);
            f_3 = f_3*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(v);
            f_4 = f_4*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (w);
            f_5 = f_5*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(w);
            f_6 = f_6*omegp + ( (ufloat_t)(0.074074074074074)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u+v);
            f_7 = f_7*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u+v);
            f_8 = f_8*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u+w);
            f_9 = f_9*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u+w);
            f_10 = f_10*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (v+w);
            f_11 = f_11*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(v+w);
            f_12 = f_12*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u)-(v);
            f_13 = f_13*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (v)-(u);
            f_14 = f_14*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u)-(w);
            f_15 = f_15*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (w)-(u);
            f_16 = f_16*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (v)-(w);
            f_17 = f_17*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (w)-(v);
            f_18 = f_18*omegp + ( (ufloat_t)(0.018518518518519)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u+v+w);
            f_19 = f_19*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u+v+w);
            f_20 = f_20*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u+v)-(w);
            f_21 = f_21*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (w)-(u+v);
            f_22 = f_22*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u+w)-(v);
            f_23 = f_23*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (v)-(u+w);
            f_24 = f_24*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (v+w)-(u);
            f_25 = f_25*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (u)-(v+w);
            f_26 = f_26*omegp + ( (ufloat_t)(0.004629629629630)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_9;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_10;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_11;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_12;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_13;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_14;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_15;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_16;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_17;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_18;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = f_19;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = f_20;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = f_21;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = f_22;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = f_23;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = f_24;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = f_25;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = f_26;
            }
            
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Collision_New_D3Q27(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collision_New_D3Q27<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

