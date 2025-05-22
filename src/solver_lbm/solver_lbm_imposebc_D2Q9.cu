/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Mon May 19 17:22:25 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_ImposeBC_D2Q9(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,int n_maxcells_b,int n_maxblocks_b,ufloat_t dx_L,ufloat_t dx_L_g,ufloat_t tau_L,int *__restrict__ id_set_idev_L,int *__restrict__ cells_ID_mask,ufloat_t *__restrict__ cells_f_F,ufloat_g_t *__restrict__ cells_f_X_b,ufloat_t *__restrict__ cells_f_F_aux,ufloat_t *__restrict__ cblock_f_X,int *__restrict__ cblock_ID_nbr,int *__restrict__ cblock_ID_nbr_child,int *__restrict__ cblock_ID_mask,int *__restrict__ cblock_ID_onb,int *__restrict__ cblock_ID_onb_solid,double *__restrict__ cblock_f_Ff_solid,bool geometry_init,int force_type)
{
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_nbr[N_Q_max];
    __shared__ ufloat_t s_u[3*M_TBLOCK];
    __shared__ double s_Fpx[M_TBLOCK];
    __shared__ double s_Fmx[M_TBLOCK];
    __shared__ double s_Fpy[M_TBLOCK];
    __shared__ double s_Fmy[M_TBLOCK];
    __shared__ double s_Fpz[M_TBLOCK];
    __shared__ double s_Fmz[M_TBLOCK];
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
    int nbr_kap_c = -1;
    int valid_block = -1;
    int block_mask = -1;
    int valid_mask = -1;
    ufloat_t f_p = (ufloat_t)(0.0);
    ufloat_t f_p_p = (ufloat_t)(0.0);
    ufloat_t f_q = (ufloat_t)(0.0);
    ufloat_t f_q_p = (ufloat_t)(0.0);
    ufloat_t f_m = (ufloat_t)(0.0);
    ufloat_g_t dQ = (ufloat_g_t)(0.0);
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t u = (ufloat_t)(0.0);
    ufloat_t v = (ufloat_t)(0.0);
    ufloat_t w = (ufloat_t)(0.0);
    ufloat_t ub = (ufloat_t)(0.0);
    ufloat_t vb = (ufloat_t)(0.0);
    ufloat_t wb = (ufloat_t)(0.0);
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
            valid_block=cblock_ID_onb[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((valid_block==1)))
        {
            // Compute cell coordinates and retrieve macroscopic properties.
            valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            if (geometry_init)
                block_mask = cblock_ID_onb_solid[i_kap_b];
            if (n_maxblocks_b > 0 && force_type > 0 && block_mask > -1)
            {
                s_Fpx[threadIdx.x] = 0;
                s_Fmx[threadIdx.x] = 0;
                s_Fpy[threadIdx.x] = 0;
                s_Fmy[threadIdx.x] = 0;
                s_Fpz[threadIdx.x] = 0;
                s_Fmz[threadIdx.x] = 0;
            }
            x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
            y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
            rho = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
            u = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells]; ub = u;
            v = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells]; vb = v;
            w = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells]; wb = w;
            s_u[threadIdx.x + 0*M_TBLOCK] = u;
            s_u[threadIdx.x + 1*M_TBLOCK] = v;
            s_u[threadIdx.x + 2*M_TBLOCK] = w;
            __syncthreads();
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
            // p = 1
            //
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 1;
            Jp = J + 0;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                nbr_kap_b = s_ID_nbr[1];
            ub = u;
            if (nbr_kap_b == -2 && I > 0)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I > 0)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+1*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/9.0), (ufloat_t)(1), (ufloat_t)(0), (ufloat_t)(0), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + -1;
            Jp = J + 0;
            // Consider nbr 3.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                nbr_kap_b = s_ID_nbr[3];
            ub = u;
            if (nbr_kap_b == -2 && I > 0)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I > 0)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+1*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/9.0), (ufloat_t)(-1), (ufloat_t)(0), (ufloat_t)(0), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Store old values of DDFs p and q.
                f_p_p = f_p;
                f_q_p = f_q;
                // Check if DDF 1 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 1*n_maxcells_b] / dx_L_g;
                // Pick the right neighbor block for this cell (pb).
                nbr_kap_b = i_kap_b;
                Ip = I + -1;
                Jp = J + 0;
                // Consider nbr 3.
                if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = s_ID_nbr[3];
                // Get the fluid node behind this boundary node.
                Ip = (4 + (Ip % 4)) % 4;
                nbr_kap_c = Ip + 4*Jp;
                f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 1*n_maxcells];
                // ULI.
                if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                // DLI.
                if (dQ >= (ufloat_g_t)(0.5) && dQ < (ufloat_g_t)(1.0))
                    f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                if (force_type == 1)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fpx[threadIdx.x] += (f_p+f_p_p);
                    }
                }
                if (force_type == 2)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fpx[threadIdx.x] += (0.5+dQ)*(f_p+f_p_p)+(0.5-dQ)*(f_m+f_q_p);
                    }
                }
                // Check if DDF 3 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 3*n_maxcells_b] / dx_L_g;
                // Pick the right neighbor block for this cell (p).
                nbr_kap_b = i_kap_b;
                Ip = I + 1;
                Jp = J + 0;
                // Consider nbr 1.
                if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = s_ID_nbr[1];
                // Get the fluid node behind this boundary node.
                Ip = (4 + (Ip % 4)) % 4;
                nbr_kap_c = Ip + 4*Jp;
                f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 3*n_maxcells];
                // ULI.
                if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                // DLI.
                if (dQ >= (ufloat_g_t)(0.5) && dQ < (ufloat_g_t)(1.0))
                    f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                if (force_type == 1)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fmx[threadIdx.x] += (f_q+f_q_p);
                    }
                }
                if (force_type == 2)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fmx[threadIdx.x] += (0.5+dQ)*(f_q+f_q_p) + (0.5-dQ)*(f_m+f_p_p);
                    }
                }
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
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + 1;
            // Consider nbr 2.
            if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                nbr_kap_b = s_ID_nbr[2];
            ub = u;
            if (nbr_kap_b == -2 && I > 0)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I > 0)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+1*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/9.0), (ufloat_t)(0), (ufloat_t)(1), (ufloat_t)(0), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + 0;
            Jp = J + -1;
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                nbr_kap_b = s_ID_nbr[4];
            ub = u;
            if (nbr_kap_b == -2 && I > 0)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I > 0)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+1*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/9.0), (ufloat_t)(0), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Store old values of DDFs p and q.
                f_p_p = f_p;
                f_q_p = f_q;
                // Check if DDF 2 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 2*n_maxcells_b] / dx_L_g;
                // Pick the right neighbor block for this cell (pb).
                nbr_kap_b = i_kap_b;
                Ip = I + 0;
                Jp = J + -1;
                // Consider nbr 4.
                if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                    nbr_kap_b = s_ID_nbr[4];
                // Get the fluid node behind this boundary node.
                Jp = (4 + (Jp % 4)) % 4;
                nbr_kap_c = Ip + 4*Jp;
                f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 2*n_maxcells];
                // ULI.
                if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                // DLI.
                if (dQ >= (ufloat_g_t)(0.5) && dQ < (ufloat_g_t)(1.0))
                    f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                if (force_type == 1)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fpy[threadIdx.x] += (f_p+f_p_p);
                    }
                }
                if (force_type == 2)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fpy[threadIdx.x] += (0.5+dQ)*(f_p+f_p_p)+(0.5-dQ)*(f_m+f_q_p);
                    }
                }
                // Check if DDF 4 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 4*n_maxcells_b] / dx_L_g;
                // Pick the right neighbor block for this cell (p).
                nbr_kap_b = i_kap_b;
                Ip = I + 0;
                Jp = J + 1;
                // Consider nbr 2.
                if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                    nbr_kap_b = s_ID_nbr[2];
                // Get the fluid node behind this boundary node.
                Jp = (4 + (Jp % 4)) % 4;
                nbr_kap_c = Ip + 4*Jp;
                f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 4*n_maxcells];
                // ULI.
                if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                // DLI.
                if (dQ >= (ufloat_g_t)(0.5) && dQ < (ufloat_g_t)(1.0))
                    f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                if (force_type == 1)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fmy[threadIdx.x] += (f_q+f_q_p);
                    }
                }
                if (force_type == 2)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fmy[threadIdx.x] += (0.5+dQ)*(f_q+f_q_p) + (0.5-dQ)*(f_m+f_p_p);
                    }
                }
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
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 1;
            Jp = J + 1;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                nbr_kap_b = s_ID_nbr[1];
            // Consider nbr 2.
            if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                nbr_kap_b = s_ID_nbr[2];
            // Consider nbr 5.
            if ( (Ip==4)and(Jp==4) )
                nbr_kap_b = s_ID_nbr[5];
            ub = u;
            if (nbr_kap_b == -2 && I > 0)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I > 0)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+1*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(1), (ufloat_t)(1), (ufloat_t)(0), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + -1;
            Jp = J + -1;
            // Consider nbr 3.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                nbr_kap_b = s_ID_nbr[3];
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                nbr_kap_b = s_ID_nbr[4];
            // Consider nbr 7.
            if ( (Ip==-1)and(Jp==-1) )
                nbr_kap_b = s_ID_nbr[7];
            ub = u;
            if (nbr_kap_b == -2 && I > 0)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I > 0)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+1*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(-1), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Store old values of DDFs p and q.
                f_p_p = f_p;
                f_q_p = f_q;
                // Check if DDF 5 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 5*n_maxcells_b] / dx_L_g;
                // Pick the right neighbor block for this cell (pb).
                nbr_kap_b = i_kap_b;
                Ip = I + -1;
                Jp = J + -1;
                // Consider nbr 3.
                if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = s_ID_nbr[3];
                // Consider nbr 4.
                if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                    nbr_kap_b = s_ID_nbr[4];
                // Consider nbr 7.
                if ( (Ip==-1)and(Jp==-1) )
                    nbr_kap_b = s_ID_nbr[7];
                // Get the fluid node behind this boundary node.
                Ip = (4 + (Ip % 4)) % 4;
                Jp = (4 + (Jp % 4)) % 4;
                nbr_kap_c = Ip + 4*Jp;
                f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 5*n_maxcells];
                // ULI.
                if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                // DLI.
                if (dQ >= (ufloat_g_t)(0.5) && dQ < (ufloat_g_t)(1.0))
                    f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                if (force_type == 1)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fpx[threadIdx.x] += (f_p+f_p_p);
                        s_Fpy[threadIdx.x] += (f_p+f_p_p);
                    }
                }
                if (force_type == 2)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fpx[threadIdx.x] += (0.5+dQ)*(f_p+f_p_p)+(0.5-dQ)*(f_m+f_q_p);
                        s_Fpy[threadIdx.x] += (0.5+dQ)*(f_p+f_p_p)+(0.5-dQ)*(f_m+f_q_p);
                    }
                }
                // Check if DDF 7 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 7*n_maxcells_b] / dx_L_g;
                // Pick the right neighbor block for this cell (p).
                nbr_kap_b = i_kap_b;
                Ip = I + 1;
                Jp = J + 1;
                // Consider nbr 1.
                if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = s_ID_nbr[1];
                // Consider nbr 2.
                if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                    nbr_kap_b = s_ID_nbr[2];
                // Consider nbr 5.
                if ( (Ip==4)and(Jp==4) )
                    nbr_kap_b = s_ID_nbr[5];
                // Get the fluid node behind this boundary node.
                Ip = (4 + (Ip % 4)) % 4;
                Jp = (4 + (Jp % 4)) % 4;
                nbr_kap_c = Ip + 4*Jp;
                f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 7*n_maxcells];
                // ULI.
                if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                // DLI.
                if (dQ >= (ufloat_g_t)(0.5) && dQ < (ufloat_g_t)(1.0))
                    f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                if (force_type == 1)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fmx[threadIdx.x] += (f_q+f_q_p);
                        s_Fmy[threadIdx.x] += (f_q+f_q_p);
                    }
                }
                if (force_type == 2)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fmx[threadIdx.x] += (0.5+dQ)*(f_q+f_q_p) + (0.5-dQ)*(f_m+f_p_p);
                        s_Fmy[threadIdx.x] += (0.5+dQ)*(f_q+f_q_p) + (0.5-dQ)*(f_m+f_p_p);
                    }
                }
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
            
            // Retrieve the DDF. Use correct order of access this time.
            f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            
            //
            // Impose boundary conditions.
            // Do this only if on the boundary.
            //
            // Pick the right neighbor block for this cell (p).
            nbr_kap_b = i_kap_b;
            Ip = I + 1;
            Jp = J + -1;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                nbr_kap_b = s_ID_nbr[1];
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                nbr_kap_b = s_ID_nbr[4];
            // Consider nbr 8.
            if ( (Ip==4)and(Jp==-1) )
                nbr_kap_b = s_ID_nbr[8];
            ub = u;
            if (nbr_kap_b == -2 && I > 0)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I > 0)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+1*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(1), (ufloat_t)(-1), (ufloat_t)(0), cdotu);
            
            // Pick the right neighbor block for this cell (pb).
            nbr_kap_b = i_kap_b;
            Ip = I + -1;
            Jp = J + 1;
            // Consider nbr 2.
            if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                nbr_kap_b = s_ID_nbr[2];
            // Consider nbr 3.
            if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                nbr_kap_b = s_ID_nbr[3];
            // Consider nbr 6.
            if ( (Ip==-1)and(Jp==4) )
                nbr_kap_b = s_ID_nbr[6];
            ub = u;
            if (nbr_kap_b == -2 && I > 0)
                ub = u + (ufloat_t)(0.5)*(u - s_u[(I-1)+4*J+0*M_TBLOCK]);
            vb = v;
            if (nbr_kap_b == -2 && I > 0)
                vb = v + (ufloat_t)(0.5)*(v - s_u[(I-1)+4*J+1*M_TBLOCK]);
            if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(1.0/36.0), (ufloat_t)(-1), (ufloat_t)(1), (ufloat_t)(0), cdotu);
            
            //
            // Impose curved boundary conditions.
            // Do this only if adjacent to a solid cell (by checking the cell mask).
            // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
            //
            if (valid_mask == -2)
            {
                // Store old values of DDFs p and q.
                f_p_p = f_p;
                f_q_p = f_q;
                // Check if DDF 8 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 8*n_maxcells_b] / dx_L_g;
                // Pick the right neighbor block for this cell (pb).
                nbr_kap_b = i_kap_b;
                Ip = I + -1;
                Jp = J + 1;
                // Consider nbr 2.
                if ( (Ip>=0)and(Ip<4)and(Jp==4) )
                    nbr_kap_b = s_ID_nbr[2];
                // Consider nbr 3.
                if ( (Ip==-1)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = s_ID_nbr[3];
                // Consider nbr 6.
                if ( (Ip==-1)and(Jp==4) )
                    nbr_kap_b = s_ID_nbr[6];
                // Get the fluid node behind this boundary node.
                Ip = (4 + (Ip % 4)) % 4;
                Jp = (4 + (Jp % 4)) % 4;
                nbr_kap_c = Ip + 4*Jp;
                f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 8*n_maxcells];
                // ULI.
                if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                // DLI.
                if (dQ >= (ufloat_g_t)(0.5) && dQ < (ufloat_g_t)(1.0))
                    f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                if (force_type == 1)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fpx[threadIdx.x] += (f_p+f_p_p);
                        s_Fmy[threadIdx.x] += (f_p+f_p_p);
                    }
                }
                if (force_type == 2)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fpx[threadIdx.x] += (0.5+dQ)*(f_p+f_p_p)+(0.5-dQ)*(f_m+f_q_p);
                        s_Fmy[threadIdx.x] += (0.5+dQ)*(f_p+f_p_p)+(0.5-dQ)*(f_m+f_q_p);
                    }
                }
                // Check if DDF 6 is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 6*n_maxcells_b] / dx_L_g;
                // Pick the right neighbor block for this cell (p).
                nbr_kap_b = i_kap_b;
                Ip = I + 1;
                Jp = J + -1;
                // Consider nbr 1.
                if ( (Ip==4)and(Jp>=0)and(Jp<4) )
                    nbr_kap_b = s_ID_nbr[1];
                // Consider nbr 4.
                if ( (Ip>=0)and(Ip<4)and(Jp==-1) )
                    nbr_kap_b = s_ID_nbr[4];
                // Consider nbr 8.
                if ( (Ip==4)and(Jp==-1) )
                    nbr_kap_b = s_ID_nbr[8];
                // Get the fluid node behind this boundary node.
                Ip = (4 + (Ip % 4)) % 4;
                Jp = (4 + (Jp % 4)) % 4;
                nbr_kap_c = Ip + 4*Jp;
                f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 6*n_maxcells];
                // ULI.
                if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                    f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                // DLI.
                if (dQ >= (ufloat_g_t)(0.5) && dQ < (ufloat_g_t)(1.0))
                    f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                if (force_type == 1)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fmx[threadIdx.x] += (f_q+f_q_p);
                        s_Fpy[threadIdx.x] += (f_q+f_q_p);
                    }
                }
                if (force_type == 2)
                {
                    if (n_maxblocks_b > 0 && dQ > 0)
                    {
                        s_Fmx[threadIdx.x] += (0.5+dQ)*(f_q+f_q_p) + (0.5-dQ)*(f_m+f_p_p);
                        s_Fpy[threadIdx.x] += (0.5+dQ)*(f_q+f_q_p) + (0.5-dQ)*(f_m+f_p_p);
                    }
                }
            }
            // Write fi* to global memory.
            if (valid_mask != -1)
            {
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_q;
            }
            
            if (n_maxblocks_b > 0 && force_type > 0 && block_mask > -1)
            {
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
                    cblock_f_Ff_solid[block_mask + 0*n_maxblocks_b] = s_Fpx[0];
                    cblock_f_Ff_solid[block_mask + 1*n_maxblocks_b] = s_Fmx[0];
                    cblock_f_Ff_solid[block_mask + 2*n_maxblocks_b] = s_Fpy[0];
                    cblock_f_Ff_solid[block_mask + 3*n_maxblocks_b] = s_Fmy[0];
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ImposeBC_D2Q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_ImposeBC_D2Q9<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, mesh->n_solidb, dxf_vec[L], (ufloat_g_t)dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_X_b[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->c_cblock_f_Ff_solid[i_dev], mesh->geometry_init, S_FORCE_TYPE);
	}

	return 0;
}

