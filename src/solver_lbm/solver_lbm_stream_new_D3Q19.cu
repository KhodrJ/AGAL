/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Tue Jul  8 00:01:40 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Stream_Original_D3Q19(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,int n_maxcells_b,int *__restrict__ id_set_idev_L,int *__restrict__ cells_ID_mask,ufloat_t *__restrict__ cells_f_F,ufloat_g_t *__restrict__ cells_f_X_b,int *__restrict__ cblock_ID_nbr,int *__restrict__ cblock_ID_nbr_child,int *__restrict__ cblock_ID_mask,int *__restrict__ cblock_ID_onb_solid,bool geometry_init)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int Ip = I;
    int J = (threadIdx.x / 4) % 4;
    int Jp = J;
    int K = (threadIdx.x / 4) / 4;
    int Kp = K;
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int nbr_kap_c = -1;
    int nbr_kap_b = -1;
    int nbr_1 __attribute__((unused)) = -1;
    int nbr_2 __attribute__((unused)) = -1;
    int nbr_3 __attribute__((unused)) = -1;
    int nbr_4 __attribute__((unused)) = -1;
    int nbr_5 __attribute__((unused)) = -1;
    int nbr_6 __attribute__((unused)) = -1;
    int nbr_7 __attribute__((unused)) = -1;
    int nbr_8 __attribute__((unused)) = -1;
    int nbr_9 __attribute__((unused)) = -1;
    int nbr_10 __attribute__((unused)) = -1;
    int nbr_11 __attribute__((unused)) = -1;
    int nbr_12 __attribute__((unused)) = -1;
    int nbr_13 __attribute__((unused)) = -1;
    int nbr_14 __attribute__((unused)) = -1;
    int nbr_15 __attribute__((unused)) = -1;
    int nbr_16 __attribute__((unused)) = -1;
    int nbr_17 __attribute__((unused)) = -1;
    int nbr_18 __attribute__((unused)) = -1;
    int valid_block = 0;
    int valid_mask = 0;
    ufloat_t F_p = (ufloat_t)0.0;
    ufloat_t F_pb = (ufloat_t)0.0;
    ufloat_g_t dQ = (ufloat_g_t)(-1.0);
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
            // Load neighbor block indices.
            nbr_1 = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
            nbr_2 = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
            nbr_3 = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
            nbr_4 = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
            nbr_5 = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
            nbr_6 = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
            nbr_7 = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
            nbr_8 = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
            nbr_9 = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
            nbr_10 = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
            nbr_11 = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
            nbr_12 = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
            nbr_13 = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
            nbr_14 = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
            nbr_15 = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
            nbr_16 = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
            nbr_17 = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
            nbr_18 = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
            valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            if (geometry_init && valid_mask == -2)
                valid_block = cblock_ID_onb_solid[i_kap_b];
            
            //
            // DDF 1
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 1;
            Jp = J + 0;
            Kp = K + 0;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_1;
            
            // Correct the neighbor cell indices.
            Ip = (4 + (Ip % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 2*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 2*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = F_pb;
            }
            
            //
            // DDF 3
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 0;
            Jp = J + 1;
            Kp = K + 0;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_3;
            
            // Correct the neighbor cell indices.
            Jp = (4 + (Jp % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 4*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 4*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = F_pb;
            }
            
            //
            // DDF 5
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 5*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 0;
            Jp = J + 0;
            Kp = K + 1;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 5.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = nbr_5;
            
            // Correct the neighbor cell indices.
            Kp = (4 + (Kp % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 6*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 6*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = F_pb;
            }
            
            //
            // DDF 7
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 7*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 1;
            Jp = J + 1;
            Kp = K + 0;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_1;
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_3;
            // Consider nbr 7.
            if ( (Ip==4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_7;
            
            // Correct the neighbor cell indices.
            Ip = (4 + (Ip % 4)) % 4;
            Jp = (4 + (Jp % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 8*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 8*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = F_pb;
            }
            
            //
            // DDF 9
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 9*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 1;
            Jp = J + 0;
            Kp = K + 1;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_1;
            // Consider nbr 5.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = nbr_5;
            // Consider nbr 9.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = nbr_9;
            
            // Correct the neighbor cell indices.
            Ip = (4 + (Ip % 4)) % 4;
            Kp = (4 + (Kp % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 10*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 10*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = F_pb;
            }
            
            //
            // DDF 11
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 11*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 0;
            Jp = J + 1;
            Kp = K + 1;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_3;
            // Consider nbr 5.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==4) )
                nbr_kap_b = nbr_5;
            // Consider nbr 11.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==4) )
                nbr_kap_b = nbr_11;
            
            // Correct the neighbor cell indices.
            Jp = (4 + (Jp % 4)) % 4;
            Kp = (4 + (Kp % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 12*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 12*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = F_pb;
            }
            
            //
            // DDF 13
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 13*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 1;
            Jp = J + -1;
            Kp = K + 0;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_1;
            // Consider nbr 4.
            if ( (Ip>=0)and(Ip<4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_4;
            // Consider nbr 13.
            if ( (Ip==4)and(Jp==-1)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_13;
            
            // Correct the neighbor cell indices.
            Ip = (4 + (Ip % 4)) % 4;
            Jp = (4 + (Jp % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 14*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 14*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = F_pb;
            }
            
            //
            // DDF 15
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 15*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 1;
            Jp = J + 0;
            Kp = K + -1;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 1.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_1;
            // Consider nbr 6.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = nbr_6;
            // Consider nbr 15.
            if ( (Ip==4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = nbr_15;
            
            // Correct the neighbor cell indices.
            Ip = (4 + (Ip % 4)) % 4;
            Kp = (4 + (Kp % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 16*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 16*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = F_pb;
            }
            
            //
            // DDF 17
            //
            // Load DDFs in current cells.
            F_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
            if (valid_mask == -2)
                dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 17*n_maxcells_b];
            
            // Compute neighbor cell indices.
            Ip = I + 0;
            Jp = J + 1;
            Kp = K + -1;
            
            // Assign the correct neighbor cell-block ID.
            nbr_kap_b = i_kap_b;
            // Consider nbr 3.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp>=0)and(Kp<4) )
                nbr_kap_b = nbr_3;
            // Consider nbr 6.
            if ( (Ip>=0)and(Ip<4)and(Jp>=0)and(Jp<4)and(Kp==-1) )
                nbr_kap_b = nbr_6;
            // Consider nbr 17.
            if ( (Ip>=0)and(Ip<4)and(Jp==4)and(Kp==-1) )
                nbr_kap_b = nbr_17;
            
            // Correct the neighbor cell indices.
            Jp = (4 + (Jp % 4)) % 4;
            Kp = (4 + (Kp % 4)) % 4;
            nbr_kap_c = Ip + 4*Jp + 16*Kp;
            
            // Retrieve neighboring DDFs, if applicable.
            F_pb = -1;
            if ( nbr_kap_b>=0 )
                F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 18*n_maxcells];
            
            // Exchange, if applicable.
            if ( valid_mask != -1 && F_pb>=0 && dQ < 0)
            {
                cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 18*n_maxcells] = F_p;
                cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = F_pb;
            }
            
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Stream_Original_D3Q19(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Stream_Original_D3Q19<ufloat_t,ufloat_g_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_X_b[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->geometry_init);
	}

	return 0;
}

