/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Thu Apr 24 00:19:09 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Collision_Original_D2Q9(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t dx_L,ufloat_t tau_L,int *__restrict__ id_set_idev_L,int *__restrict__ cells_ID_mask,ufloat_t *__restrict__ cells_f_F,ufloat_t *__restrict__ cells_f_F_aux,ufloat_t *__restrict__ cblock_f_X,int *__restrict__ cblock_ID_nbr,int *__restrict__ cblock_ID_nbr_child,int *__restrict__ cblock_ID_mask,int *__restrict__ cblock_ID_onb)
{
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ ufloat_t s_f_u[M_TBLOCK*(N_DIM+1)];
    __shared__ int s_ID_nbr[N_Q_max];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int Ip = I;
    int J = (threadIdx.x / 4) % 4;
    int Jp = J;
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
        
        // This part is included if n>0 only.
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
            x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
            y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
            if (valid_block == 1 && threadIdx.x == 0)
            {
                s_ID_nbr[1] = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
                s_ID_nbr[2] = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
                s_ID_nbr[3] = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
                s_ID_nbr[4] = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
                s_ID_nbr[5] = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
                s_ID_nbr[6] = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
                s_ID_nbr[7] = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
                s_ID_nbr[8] = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
            }
            __syncthreads();
            
            //
            //
            //
            // First round of DDF loads to compute the macroscopic properties. Discard the DDFs to avoid large memory footprint for now.
            // (Need to test if this really speeds anything up, convenient to have it all one kernel for now though).
            //
            //
            //
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
            rho += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            rho += f_p;
            u += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            rho += f_p;
            v += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            rho += f_p;
            ub += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            rho += f_p;
            vb += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            rho += f_p;
            u += f_p;
            v += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            rho += f_p;
            ub += f_p;
            v += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            rho += f_p;
            ub += f_p;
            vb += f_p;
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            rho += f_p;
            u += f_p;
            vb += f_p;
            u = (u-ub)/rho;
            v = (v-vb)/rho;
            udotu = u*u + v*v + w*w;
            ub = u;
            vb = v;
            wb = w;
            
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
            f_p = f_p*omegp + ( (ufloat_t)(0.444444444444444)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Impose boundary conditions.
            // Do this only if on the boundary.
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_p;
            }
            
            //
            // p = 1
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            
            // Collision step.
            cdotu = (u);
            f_p = f_p*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u);
            f_q = f_q*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Impose boundary conditions.
            // Do this only if on the boundary.
            if (valid_block==1)
            {
                // Pick the right neighbor block for this cell (p).
                nbr_kap_b = i_kap_b;
                Ip = I + 1;
                Jp = J + 0;
                // Consider nbr 1.
                if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
                if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/9.0), (ufloat_t)(1), (ufloat_t)(0), (ufloat_t)(0), cdotu);
                
                // Pick the right neighbor block for this cell (pb).
                nbr_kap_b = i_kap_b;
                Ip = I + -1;
                Jp = J + 0;
                // Consider nbr 3.
                if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
                if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/9.0), (ufloat_t)(-1), (ufloat_t)(0), (ufloat_t)(0), cdotu);
            }
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_q;
            }
            
            //
            // p = 2
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            
            // Collision step.
            cdotu = (v);
            f_p = f_p*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(v);
            f_q = f_q*omegp + ( (ufloat_t)(0.111111111111111)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Impose boundary conditions.
            // Do this only if on the boundary.
            if (valid_block==1)
            {
                // Pick the right neighbor block for this cell (p).
                nbr_kap_b = i_kap_b;
                Ip = I + 0;
                Jp = J + 1;
                // Consider nbr 2.
                if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
                if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/9.0), (ufloat_t)(0), (ufloat_t)(1), (ufloat_t)(0), cdotu);
                
                // Pick the right neighbor block for this cell (pb).
                nbr_kap_b = i_kap_b;
                Ip = I + 0;
                Jp = J + -1;
                // Consider nbr 4.
                if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
                if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/9.0), (ufloat_t)(0), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
            }
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_q;
            }
            
            //
            // p = 5
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            
            // Collision step.
            cdotu = (u+v);
            f_p = f_p*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = -(u+v);
            f_q = f_q*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Impose boundary conditions.
            // Do this only if on the boundary.
            if (valid_block==1)
            {
                // Pick the right neighbor block for this cell (p).
                nbr_kap_b = i_kap_b;
                Ip = I + 1;
                Jp = J + 1;
                // Consider nbr 1.
                if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
                // Consider nbr 2.
                if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
                // Consider nbr 5.
                if ( (Ip==4)and(Jp==4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
                if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(1), (ufloat_t)(1), (ufloat_t)(0), cdotu);
                
                // Pick the right neighbor block for this cell (pb).
                nbr_kap_b = i_kap_b;
                Ip = I + -1;
                Jp = J + -1;
                // Consider nbr 3.
                if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
                // Consider nbr 4.
                if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
                // Consider nbr 7.
                if ( (Ip==-1)and(Jp==-1) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
                if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(-1), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
            }
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_q;
            }
            
            //
            // p = 8
            //
            
            // Retrieve the DDF.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            
            // Collision step.
            cdotu = (u)-(v);
            f_p = f_p*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            cdotu = (v)-(u);
            f_q = f_q*omegp + ( (ufloat_t)(0.027777777777778)*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
            
            // Impose boundary conditions.
            // Do this only if on the boundary.
            if (valid_block==1)
            {
                // Pick the right neighbor block for this cell (p).
                nbr_kap_b = i_kap_b;
                Ip = I + 1;
                Jp = J + -1;
                // Consider nbr 1.
                if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
                // Consider nbr 4.
                if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
                // Consider nbr 8.
                if ( (Ip==4)and(Jp==-1) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
                if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(1), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
                
                // Pick the right neighbor block for this cell (pb).
                nbr_kap_b = i_kap_b;
                Ip = I + -1;
                Jp = J + 1;
                // Consider nbr 2.
                if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
                // Consider nbr 3.
                if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
                // Consider nbr 6.
                if ( (Ip==-1)and(Jp==4) )
                    nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
                if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(-1), (ufloat_t)(1), (ufloat_t)(0), cdotu);
            }
            
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_q;
            }
            
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Collision_Original_D2Q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collision_Original_D2Q9<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

