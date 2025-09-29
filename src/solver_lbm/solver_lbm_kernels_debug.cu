/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_Debug_DrawGeometry
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
    const ufloat_g_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_ID_nbr_child,
    const int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_onb,
    const int *__restrict__ cblock_ID_onb_solid,
    const bool geometry_init,
    const bool compute_forces
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int N_Q = LP->N_Q;
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Check if block is on the boundary.
        int valid_block=cblock_ID_onb[i_kap_b];
        
        // Latter condition is added only if n>0.
        if (valid_block==1)
        {
            // Get cell indices.
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            int K = 0;
            if (N_DIM==3)
                K = (threadIdx.x / 4) / 4;
            
            // Get cell masks. If on the boundary, also get mapped blocked index to later retrieve link lengths.
            int block_mask = -1;
            int valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            if (geometry_init)
                block_mask = cblock_ID_onb_solid[i_kap_b];
            
            // Compute cell coordinates and retrieve macroscopic properties.
            ufloat_t x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*(static_cast<ufloat_t>(0.5) + I);
            ufloat_t y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*(static_cast<ufloat_t>(0.5) + J);
            ufloat_t z = static_cast<ufloat_t>(0.0);
            if (N_DIM==3)
                z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*(static_cast<ufloat_t>(0.5) + K);
            __syncthreads();
            
            // Loop over possible directions.
            for (int p = 1; p < N_Q; p++)
            {
                // Only consider half of the directions.
                if ( (N_DIM==2 && (p==1||(p+1)%3==0))   ||   (N_DIM==3 && (p==26||((p-1)%2==0&&p<25))) )
                {
                    if (valid_mask == V_CELLMASK_BOUNDARY)
                    {
                        // Along direction p.
                        ufloat_t dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + p*n_maxcells_b];
                        if (dist_p > 0)
                        {
                            if (N_DIM==2)
                                printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*V_CONN_ID[p+0*27],y,y+dist_p*V_CONN_ID[p+1*27]);
                            else // N_DIM==3
                                printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*V_CONN_ID[p+0*27],y,y+dist_p*V_CONN_ID[p+1*27],z,z+dist_p*V_CONN_ID[p+2*27]);
                        }
                        if (dist_p >= dx_L)
                            printf("ERROR\n");
                        
                        // Along direction pb.
                        dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells_b];
                        if (dist_p > 0)
                        {
                            if (N_DIM==2)
                                printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n", x,x+dist_p*V_CONN_ID[LBMpb[p]+0*27],y,y+dist_p*V_CONN_ID[LBMpb[p]+1*27]);
                            else // N_DIM==3
                                printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n", x,x+dist_p*V_CONN_ID[LBMpb[p]+0*27],y,y+dist_p*V_CONN_ID[LBMpb[p]+1*27],z,z+dist_p*V_CONN_ID[LBMpb[p]+2*27]);
                        }
                        if (dist_p >= dx_L)
                            printf("ERROR\n");
                    }
                }
            }
            
            if (N_DIM==2)
            {
                printf("plot(%17.15f,%17.15f,'k*');\n",x,y);
                printf(
                    "plot([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'k-');\n",
                    x-dx_L/2.0,x+dx_L/2.0,x+dx_L/2.0,x-dx_L/2.0,x-dx_L/2.0,
                    y-dx_L/2.0,y-dx_L/2.0,y+dx_L/2.0,y+dx_L/2.0,y-dx_L/2.0
                );
            }
            else
            {
                if (valid_mask == V_CELLMASK_BOUNDARY) printf("plot3(%17.15f,%17.15f,%17.5f,'k*');\n",x,y,z);
                else
                {
                    if (valid_mask == V_CELLMASK_SOLID)  printf("plot3(%17.15f,%17.15f,%17.5f,'r*');\n",x,y,z);
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Debug_DrawGeometry(int i_dev, int L)
{
    if (mesh->n_ids[i_dev][L] > 0)
    {
        Cu_Debug_DrawGeometry<ufloat_t,ufloat_g_t,AP,LP><<<mesh->n_ids[i_dev][L],M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, mesh->n_solidb, dxf_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_X_b[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->geometry_init, compute_forces);
    }

    return 0;
}

