/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Wed Jul  2 16:17:14 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Collision_New_S2_D3Q19(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t dx_L,ufloat_t tau_L,int *__restrict__ id_set_idev_L,int *__restrict__ cells_ID_mask,ufloat_t *__restrict__ cells_f_F,ufloat_t *__restrict__ cells_f_F_aux,ufloat_t *__restrict__ cblock_f_X,int *__restrict__ cblock_ID_nbr,int *__restrict__ cblock_ID_nbr_child,int *__restrict__ cblock_ID_mask,int *__restrict__ cblock_ID_onb)
{
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int Ip = I;
    int J = (threadIdx.x / 4) % 4;
    int Jp = J;
    int K = (threadIdx.x / 4) / 4;
    int Kp = K;
    ufloat_t x __attribute__((unused)) = (ufloat_t)(0.0);
    ufloat_t y __attribute__((unused)) = (ufloat_t)(0.0);
    ufloat_t z __attribute__((unused)) = (ufloat_t)(0.0);
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int nbr_kap_b = -1;
    int valid_block = -1;
    int valid_mask = -1;
    ufloat_t f_p = (ufloat_t)(0.0);
    ufloat_t f_q = (ufloat_t)(0.0);
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t u = (ufloat_t)(0.0);
    ufloat_t ub = (ufloat_t)(0.0);
    ufloat_t v = (ufloat_t)(0.0);
    ufloat_t vb = (ufloat_t)(0.0);
    ufloat_t w = (ufloat_t)(0.0);
    ufloat_t wb = (ufloat_t)(0.0);
    ufloat_t cdotu = (ufloat_t)(0.0);
    ufloat_t udotu = (ufloat_t)(0.0);
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
            // Compute cell coordinates and retrieve macroscopic properties.
            valid_block = cblock_ID_onb[i_kap_b];
            valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            rho = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
            u = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            v = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            w = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            udotu = u*u + v*v + w*w;
            __syncthreads();
            
            //
            //
            //
            // First round of DDF loads to compute the macroscopic properties. Discard the DDFs to avoid large memory footprint for now.
            // (Need to test if this really speeds anything up, convenient to have it all one kernel for now though).
            //
            //
            //
            // Compute the velocity gradient.
            //
            //
            //
            // Retrieve DDFs one by one again, but perform collision now and apply boundary conditions.
            //
            //
            //
            
            //
            // p = 0
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
            
            // Collision step.
            cdotu = (ufloat_t)(0.0);
            f_p = f_p*omegp + ( (ufloat_t)(0.333333333333333)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_p;
            }
            
            //
            // p = 1
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            
            // Collision step.
            cdotu = (u);
            f_p = f_p*omegp + ( (ufloat_t)(0.055555555555556)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u);
            f_q = f_q*omegp + ( (ufloat_t)(0.055555555555556)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_q;
            }
            
            //
            // p = 3
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            
            // Collision step.
            cdotu = (v);
            f_p = f_p*omegp + ( (ufloat_t)(0.055555555555556)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(v);
            f_q = f_q*omegp + ( (ufloat_t)(0.055555555555556)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_q;
            }
            
            //
            // p = 5
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            
            // Collision step.
            cdotu = (w);
            f_p = f_p*omegp + ( (ufloat_t)(0.055555555555556)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(w);
            f_q = f_q*omegp + ( (ufloat_t)(0.055555555555556)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_q;
            }
            
            //
            // p = 7
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            
            // Collision step.
            cdotu = (u+v);
            f_p = f_p*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u+v);
            f_q = f_q*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_q;
            }
            
            //
            // p = 9
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
            
            // Collision step.
            cdotu = (u+w);
            f_p = f_p*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u+w);
            f_q = f_q*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_q;
            }
            
            //
            // p = 11
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
            
            // Collision step.
            cdotu = (v+w);
            f_p = f_p*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(v+w);
            f_q = f_q*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_q;
            }
            
            //
            // p = 13
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
            
            // Collision step.
            cdotu = (u)-(v);
            f_p = f_p*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (v)-(u);
            f_q = f_q*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_q;
            }
            
            //
            // p = 15
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
            
            // Collision step.
            cdotu = (u)-(w);
            f_p = f_p*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (w)-(u);
            f_q = f_q*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_q;
            }
            
            //
            // p = 17
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
            
            // Collision step.
            cdotu = (v)-(w);
            f_p = f_p*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (w)-(v);
            f_q = f_q*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_q;
            }
            
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Collision_New_S2_D3Q19(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collision_New_S2_D3Q19<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

