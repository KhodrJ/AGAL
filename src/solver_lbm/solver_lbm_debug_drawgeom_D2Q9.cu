/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sat May 24 06:03:37 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Debug_DrawGeometry_D2Q9(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,int n_maxcells_b,int n_maxblocks_b,ufloat_t dx_L,ufloat_t tau_L,int *__restrict__ id_set_idev_L,int *__restrict__ cells_ID_mask,ufloat_t *__restrict__ cells_f_F,ufloat_g_t *__restrict__ cells_f_X_b,ufloat_t *__restrict__ cells_f_F_aux,ufloat_t *__restrict__ cblock_f_X,int *__restrict__ cblock_ID_nbr,int *__restrict__ cblock_ID_nbr_child,int *__restrict__ cblock_ID_mask,int *__restrict__ cblock_ID_onb,int *__restrict__ cblock_ID_onb_solid,double *__restrict__ cblock_f_Ff_solid,bool geometry_init,bool compute_forces)
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
    ufloat_t f_q = (ufloat_t)(0.0);
    ufloat_t f_m = (ufloat_t)(0.0);
    ufloat_g_t dist_p = (ufloat_g_t)(0.0);
    ufloat_t rho = (ufloat_t)(0.0);
    ufloat_t u = (ufloat_t)(0.0);
    ufloat_t v = (ufloat_t)(0.0);
    ufloat_t w = (ufloat_t)(0.0);
    ufloat_t ub = (ufloat_t)(0.0);
    ufloat_t vb = (ufloat_t)(0.0);
    ufloat_t wb = (ufloat_t)(0.0);
    ufloat_t cdotu = (ufloat_t)(0.0);
    ufloat_t udotu = (ufloat_t)(0.0);
    bool near_geom = false;
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
                // p = 1
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 1*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(0.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 3*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(0.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 2
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 2*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 4*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 5
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 5*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 7*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 8
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 8*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 6*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                printf("plot(%17.15f,%17.15f,'k*');\n",x,y);
                printf("plot([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'k-');\n",
                x-dx_L/2.0,x+dx_L/2.0,x+dx_L/2.0,x-dx_L/2.0,x-dx_L/2.0,
                y-dx_L/2.0,y-dx_L/2.0,y+dx_L/2.0,y+dx_L/2.0,y-dx_L/2.0
                );
            }
        }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Debug_DrawGeometry_D2Q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Debug_DrawGeometry_D2Q9<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, mesh->n_solidb, dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_X_b[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->c_cblock_f_Ff_solid[i_dev], mesh->geometry_init, compute_forces);
	}

	return 0;
}

