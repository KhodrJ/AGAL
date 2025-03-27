/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Mar 23 06:25:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

template <typename ufloat_t, const ArgsPack *AP>
__global__
void Cu_Collide_D2Q9(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t tau_L,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,int *cblock_ID_nbr,int *cblock_ID_nbr_child,int *cblock_ID_mask,int *cblock_ID_onb)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int nbr_kap_c = -1;
    int block_on_boundary = -1;
    int nbr_1 __attribute__((unused)) = -1;
    int nbr_2 __attribute__((unused)) = -1;
    int nbr_3 __attribute__((unused)) = -1;
    int nbr_4 __attribute__((unused)) = -1;
    int nbr_5 __attribute__((unused)) = -1;
    int nbr_6 __attribute__((unused)) = -1;
    int nbr_7 __attribute__((unused)) = -1;
    int nbr_8 __attribute__((unused)) = -1;
    f_0 = (ufloat_t)0.0;
    f_1 = (ufloat_t)0.0;
    f_2 = (ufloat_t)0.0;
    f_3 = (ufloat_t)0.0;
    f_4 = (ufloat_t)0.0;
    f_5 = (ufloat_t)0.0;
    f_6 = (ufloat_t)0.0;
    f_7 = (ufloat_t)0.0;
    f_8 = (ufloat_t)0.0;
    ufloat_t cdotu = (ufloat_t)0.0;
    ufloat_t udotu = (ufloat_t)0.0;
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
            block_on_boundary=cblock_ID_mask[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(block_on_boundary>=-2) # TODO))
        {
            // Retrieve DDFs in alternating index. Compute macroscopic properties.
            f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
            f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
            f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
            f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
            f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
            rho = f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8;
            u = ((N_Pf(0)*f_0+N_Pf(1)*f_1+N_Pf(0)*f_2+N_Pf(-1)*f_3+N_Pf(0)*f_4+N_Pf(1)*f_5+N_Pf(-1)*f_6+N_Pf(-1)*f_7+N_Pf(1)*f_8)) / rho_kap;
            v = ((N_Pf(0)*f_0+N_Pf(0)*f_1+N_Pf(1)*f_2+N_Pf(0)*f_3+N_Pf(-1)*f_4+N_Pf(1)*f_5+N_Pf(1)*f_6+N_Pf(-1)*f_7+N_Pf(-1)*f_8)) / rho_kap;
            udotu = u*u + v*v;
            
            // Apply the turbulence model.
            
            // Perform collision.
            cdotu = (ufloat_t)(0.0);
            f_0 = omegap*f_0 + omega*rho*(ufloat_t)0.444444444444444*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u);
            f_1 = omegap*f_1 + omega*rho*(ufloat_t)0.111111111111111*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (v);
            f_2 = omegap*f_2 + omega*rho*(ufloat_t)0.111111111111111*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(u);
            f_3 = omegap*f_3 + omega*rho*(ufloat_t)0.111111111111111*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(v);
            f_4 = omegap*f_4 + omega*rho*(ufloat_t)0.111111111111111*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u+v);
            f_5 = omegap*f_5 + omega*rho*(ufloat_t)0.027777777777778*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (v)-(u);
            f_6 = omegap*f_6 + omega*rho*(ufloat_t)0.027777777777778*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = -(u+v);
            f_7 = omegap*f_7 + omega*rho*(ufloat_t)0.027777777777778*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            cdotu = (u)-(v);
            f_8 = omegap*f_8 + omega*rho*(ufloat_t)0.027777777777778*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
            
            // Apply no-slip or anti-bounce back boundary conditions. Free slip is implemented elsewhere.
            
            block_on_boundary = cblock_ID_onb[i_kap_b];
            if (block_on_boundary)
            {
                nbr_1 = cblock_ID_nbr[i_kap + 1*n_maxcblocks];
                nbr_2 = cblock_ID_nbr[i_kap + 2*n_maxcblocks];
                nbr_3 = cblock_ID_nbr[i_kap + 3*n_maxcblocks];
                nbr_4 = cblock_ID_nbr[i_kap + 4*n_maxcblocks];
                nbr_5 = cblock_ID_nbr[i_kap + 5*n_maxcblocks];
                nbr_6 = cblock_ID_nbr[i_kap + 6*n_maxcblocks];
                nbr_7 = cblock_ID_nbr[i_kap + 7*n_maxcblocks];
                nbr_8 = cblock_ID_nbr[i_kap + 8*n_maxcblocks];
                //
                // nbr 1
                //
                // Consider DDF 1
                // Consider DDF 5
                // Consider DDF 8
                //
                // nbr 2
                //
                // Consider DDF 2
                // Consider DDF 5
                // Consider DDF 6
                //
                // nbr 3
                //
                // Consider DDF 3
                // Consider DDF 6
                // Consider DDF 7
                //
                // nbr 4
                //
                // Consider DDF 4
                // Consider DDF 7
                // Consider DDF 8
                //
                // nbr 5
                //
                // Consider DDF 5
                //
                // nbr 6
                //
                // Consider DDF 6
                //
                // nbr 7
                //
                // Consider DDF 7
                //
                // nbr 8
                //
                // Consider DDF 8
            }
            
            // Write DDFs in proper index.
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7;
            cells_f_F[i_kap*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8;
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Collide_D2Q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collide_D2Q9<ufloat_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

