/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Tue Jul 22 17:29:04 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Debug_DrawGeometry_D3Q27
(
	const int n_ids_idev_L,
	const long int n_maxcells,
	const int n_maxcblocks,
	const int n_maxcells_b,
	const int n_maxblocks_b,
	const ufloat_t dx_L,
	const int *__restrict__ id_set_idev_L,
	const int *__restrict__ cells_ID_mask,
	const ufloat_t *__restrict__ cells_f_F,
	const ufloat_g_t *__restrict__ cells_f_X_b,
	const ufloat_t *__restrict__ cells_f_F_aux,
	const ufloat_t *__restrict__ cblock_f_X,
	const int *__restrict__ cblock_ID_nbr,
	const int *__restrict__ cblock_ID_nbr_child,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_onb,
	const int *__restrict__ cblock_ID_onb_solid,
	const bool geometry_init,
	const bool compute_forces
)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = (threadIdx.x / 4) / 4;
    ufloat_t x __attribute__((unused)) = (ufloat_t)(0.0);
    ufloat_t y __attribute__((unused)) = (ufloat_t)(0.0);
    ufloat_t z __attribute__((unused)) = (ufloat_t)(0.0);
    int i_kap_b = -1;
    int valid_block = -1;
    int block_mask = -1;
    int valid_mask = -1;
    ufloat_g_t dist_p = (ufloat_g_t)(0.0);
    bool near_geom __attribute__((unused)) = false;
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
                z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
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
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(0.000000000000000),z,z+dist_p*(0.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 2*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(0.000000000000000),z,z+dist_p*(0.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 3
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 3*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(0.000000000000000));
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
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(0.000000000000000));
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
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(0.000000000000000),z,z+dist_p*(1.000000000000000));
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
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(0.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 7
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 7*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(0.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 8*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(0.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 9
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 9*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(0.000000000000000),z,z+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 10*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(0.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 11
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 11*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 12*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 13
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 13*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(0.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 14*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(0.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 15
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 15*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(0.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 16*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(0.000000000000000),z,z+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 17
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 17*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 18*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(0.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 19
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 19*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 20*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 21
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 21*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 22*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 23
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 23*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 24*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                //
                // p = 26
                //
                
                //
                // Impose curved boundary conditions.
                // Do this only if adjacent to a solid cell (by checking the cell mask).
                // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
                //
                if (valid_mask == -2)
                {
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 26*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(1.000000000000000),y,y+dist_p*(-1.000000000000000),z,z+dist_p*(-1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                    // If computing forces, add the contributions of DDFs entering the geometry.
                    dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + 25*n_maxcells_b];
                    if (dist_p > 0)
                    {
                        near_geom = true;
                        printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*(-1.000000000000000),y,y+dist_p*(1.000000000000000),z,z+dist_p*(1.000000000000000));
                    }
                    if (dist_p >= dx_L)
                    {
                        printf("ERROR\n");
                    }
                }
                
                if (valid_mask == -2)
                {
                    printf("plot3(%17.15f,%17.15f,%17.5f,'k*');\n",x,y,z);
                }
                else
                {
                    if (valid_mask == -1)
                        printf("plot3(%17.15f,%17.15f,%17.5f,'r*');\n",x,y,z);
                }
            }
        }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Debug_DrawGeometry_D3Q27(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Debug_DrawGeometry_D3Q27<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, mesh->n_solidb, dxf_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_X_b[i_dev], mesh->c_cells_f_F_aux[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->geometry_init, compute_forces);
	}

	return 0;
}

