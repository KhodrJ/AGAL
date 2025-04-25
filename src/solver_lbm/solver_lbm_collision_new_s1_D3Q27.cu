/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Thu Apr 24 11:53:13 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Collision_New_S1_D3Q27(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t dx_L,ufloat_t tau_L,int *__restrict__ id_set_idev_L,int *__restrict__ cells_ID_mask,ufloat_t *__restrict__ cells_f_F,ufloat_t *__restrict__ cells_f_F_aux,ufloat_t *__restrict__ cblock_f_X,int *__restrict__ cblock_ID_nbr,int *__restrict__ cblock_ID_nbr_child,int *__restrict__ cblock_ID_mask,int *__restrict__ cblock_ID_onb)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int valid_block = -1;
    ufloat_t f_p = (ufloat_t)(0.0);
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t up = (ufloat_t)(0.0);
    ufloat_t um = (ufloat_t)(0.0);
    ufloat_t vp = (ufloat_t)(0.0);
    ufloat_t vm = (ufloat_t)(0.0);
    ufloat_t wp = (ufloat_t)(0.0);
    ufloat_t wm = (ufloat_t)(0.0);
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
            valid_block=cblock_ID_mask[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(valid_block>-3)))
        {
            // Retrieve DDFs one by one and compute macroscopic properties.
            // I'm using up to store the positive contributions and um to store the negative ones. Same for v and w.
            rho = (ufloat_t)(0.0);
            up = (ufloat_t)(0.0);
            um = (ufloat_t)(0.0);
            vp = (ufloat_t)(0.0);
            vm = (ufloat_t)(0.0);
            wp = (ufloat_t)(0.0);
            wm = (ufloat_t)(0.0);
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
            rho += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            rho += f_p;
            up += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            rho += f_p;
            um += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            rho += f_p;
            vp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            rho += f_p;
            vm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            rho += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            rho += f_p;
            wm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            rho += f_p;
            up += f_p;
            vp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            rho += f_p;
            um += f_p;
            vm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
            rho += f_p;
            up += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
            rho += f_p;
            um += f_p;
            wm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
            rho += f_p;
            vp += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
            rho += f_p;
            vm += f_p;
            wm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
            rho += f_p;
            up += f_p;
            vm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
            rho += f_p;
            um += f_p;
            vp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
            rho += f_p;
            up += f_p;
            wm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
            rho += f_p;
            um += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
            rho += f_p;
            vp += f_p;
            wm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
            rho += f_p;
            vm += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells];
            rho += f_p;
            up += f_p;
            vp += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells];
            rho += f_p;
            um += f_p;
            vm += f_p;
            wm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells];
            rho += f_p;
            up += f_p;
            vp += f_p;
            wm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells];
            rho += f_p;
            um += f_p;
            vm += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells];
            rho += f_p;
            up += f_p;
            vm += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells];
            rho += f_p;
            um += f_p;
            vp += f_p;
            wm += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells];
            rho += f_p;
            um += f_p;
            vp += f_p;
            wp += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells];
            rho += f_p;
            up += f_p;
            vm += f_p;
            wm += f_p;
            cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = rho;
            cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = (up-um)/rho;
            cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = (vp-vm)/rho;
            cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = (wp-wm)/rho;
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Collision_New_S1_D3Q27(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collision_New_S1_D3Q27<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

