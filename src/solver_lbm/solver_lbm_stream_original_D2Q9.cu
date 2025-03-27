/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Mar 23 06:36:17 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, const ArgsPack *AP>
__global__
void Cu_Stream_Original_D2Q9(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,int *cblock_ID_nbr,int *cblock_ID_nbr_child,int *cblock_ID_mask)
{
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ ufloat_t s_F_p[(4+2)*(4+2)];
    __shared__ ufloat_t s_F_pb[(4+2)*(4+2)];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int i_kap_b = -1;
    int i_kap_bc = -1;
    int i_Q = -1;
    int nbr_i_q = -1;
    int nbr_j_q = -1;
    int nbr_kap_b = -1;
    int nbr_Q_b = -1;
    int nbr_id_global_1 = -1;
    int nbr_id_global_2 = -1;
    int nbr_id_global_5 = -1;
    int nbr_id_1 = -1;
    int nbr_id_2 = -1;
    int nbr_id_5 = -1;
    int nbr_Q_1 = -1;
    int nbr_Q_2 = -1;
    int nbr_Q_5 = -1;
    int block_on_boundary = -1;
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
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(block_on_boundary>-2)))
        {
            nbr_id_global_1 = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
            nbr_id_global_2 = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
            nbr_id_global_5 = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
            for (int i_q = 0; i_q < Nqx; i_q += 1)
            {
                for (int j_q = 0; j_q < Nqx; j_q += 1)
                {
                    i_Q = i_q + Nqx*j_q;
                    nbr_id_1 = i_kap_b;
                    nbr_i_q = i_q + 1; if (nbr_i_q == Nqx) nbr_i_q = 0;
                    nbr_j_q = j_q + 0; if (nbr_j_q == Nqx) nbr_j_q = 0;
                    nbr_Q_1 = nbr_i_q + Nqx*nbr_j_q;
                    nbr_id_2 = i_kap_b;
                    nbr_i_q = i_q + 0; if (nbr_i_q == Nqx) nbr_i_q = 0;
                    nbr_j_q = j_q + 1; if (nbr_j_q == Nqx) nbr_j_q = 0;
                    nbr_Q_2 = nbr_i_q + Nqx*nbr_j_q;
                    nbr_id_5 = i_kap_b;
                    nbr_i_q = i_q + 1; if (nbr_i_q == Nqx) nbr_i_q = 0;
                    nbr_j_q = j_q + 1; if (nbr_j_q == Nqx) nbr_j_q = 0;
                    nbr_Q_5 = nbr_i_q + Nqx*nbr_j_q;
                    // Nbr 1.
                    if ((i_q==Nqx-1))
                    {
                        nbr_id_1 = nbr_id_global_1;
                    }
                    // Nbr 2.
                    if ((j_q==Nqx-1))
                    {
                        nbr_id_2 = nbr_id_global_2;
                    }
                    // Nbr 5.
                    if ((i_q==Nqx-1)and(j_q<Nqx-1))
                    {
                        nbr_id_5 = nbr_id_global_1;
                    }
                    if ((i_q<Nqx-1)and(j_q==Nqx-1))
                    {
                        nbr_id_5 = nbr_id_global_2;
                    }
                    if ((i_q==Nqx-1)and(j_q==Nqx-1))
                    {
                        nbr_id_5 = nbr_id_global_5;
                    }
                    
                    //
                    // DDFs p=1, pb=Lpb(1).
                    //
                    for (int q = 0; q < 3; q += 1)
                    {
                        if (threadIdx.x + q*16 < 36)
                        {
                            s_F_p[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                            s_F_pb[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                        }
                    }
                    __syncthreads();
                    s_F_p[(I+1)+6*(J+1)+0] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells];
                    s_F_pb[(I+1)+6*(J+1)+0] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                    //\t Nbr 1 participates in an exchange for p=1.
                    nbr_Q_b = nbr_Q_1;
                    nbr_kap_b = nbr_id_1;
                    if (nbr_kap_b >= 0)
                    {
                        if ((I==0))
                        {
                            s_F_pb[(4+1) + 6*(J+1) + 0] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 3*n_maxcells];
                        }
                    }
                    __syncthreads();
                    // Main writes.
                    if (cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] > -1)
                    {
                        if (s_F_pb[(I+1+1)+6*(J+1+0)+0] >= 0)
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 1*n_maxcells] = s_F_pb[(I+1+1)+6*(J+1+0)+0];
                        }
                        if (s_F_p[(I+1+-1)+6*(J+1+0)+0] >= 0)
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 3*n_maxcells] = s_F_p[(I+1+-1)+6*(J+1+0)+0];
                        }
                    }
                    // Neighbor writes.
                    //\t Nbr 1 participates in an exchange for p=1.
                    nbr_Q_b = nbr_Q_1;
                    nbr_kap_b = nbr_id_1;
                    if ((nbr_kap_b >=0) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and ((I==0)))
                    {
                        if (s_F_p[ (I+1+-1+(1*4)) + 6*(J+1+0+(0*4)) + 0 ] >= 0)
                        {
                            cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 3*n_maxcells] = s_F_p[ (I+1+-1+(1*4)) + 6*(J+1+0+(0*4)) + 0 ];
                        }
                    }
                    __syncthreads();
                    
                    
                    
                    //
                    // DDFs p=2, pb=Lpb(2).
                    //
                    for (int q = 0; q < 3; q += 1)
                    {
                        if (threadIdx.x + q*16 < 36)
                        {
                            s_F_p[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                            s_F_pb[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                        }
                    }
                    __syncthreads();
                    s_F_p[(I+1)+6*(J+1)+0] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells];
                    s_F_pb[(I+1)+6*(J+1)+0] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                    //\t Nbr 2 participates in an exchange for p=2.
                    nbr_Q_b = nbr_Q_2;
                    nbr_kap_b = nbr_id_2;
                    if (nbr_kap_b >= 0)
                    {
                        if ((J==0))
                        {
                            s_F_pb[(I+1) + 6*(4+1) + 0] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 4*n_maxcells];
                        }
                    }
                    __syncthreads();
                    // Main writes.
                    if (cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] > -1)
                    {
                        if (s_F_pb[(I+1+0)+6*(J+1+1)+0] >= 0)
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 2*n_maxcells] = s_F_pb[(I+1+0)+6*(J+1+1)+0];
                        }
                        if (s_F_p[(I+1+0)+6*(J+1+-1)+0] >= 0)
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 4*n_maxcells] = s_F_p[(I+1+0)+6*(J+1+-1)+0];
                        }
                    }
                    // Neighbor writes.
                    //\t Nbr 2 participates in an exchange for p=2.
                    nbr_Q_b = nbr_Q_2;
                    nbr_kap_b = nbr_id_2;
                    if ((nbr_kap_b >=0) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and ((J==0)))
                    {
                        if (s_F_p[ (I+1+0+(0*4)) + 6*(J+1+-1+(1*4)) + 0 ] >= 0)
                        {
                            cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 4*n_maxcells] = s_F_p[ (I+1+0+(0*4)) + 6*(J+1+-1+(1*4)) + 0 ];
                        }
                    }
                    __syncthreads();
                    
                    
                    
                    //
                    // DDFs p=5, pb=Lpb(5).
                    //
                    for (int q = 0; q < 3; q += 1)
                    {
                        if (threadIdx.x + q*16 < 36)
                        {
                            s_F_p[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                            s_F_pb[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                        }
                    }
                    __syncthreads();
                    s_F_p[(I+1)+6*(J+1)+0] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells];
                    s_F_pb[(I+1)+6*(J+1)+0] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                    //\t Nbr 1 participates in an exchange for p=5.
                    nbr_Q_b = nbr_Q_1;
                    nbr_kap_b = nbr_id_1;
                    if (nbr_kap_b >= 0)
                    {
                        if ((I==0))
                        {
                            s_F_pb[(4+1) + 6*(J+1) + 0] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                        }
                    }
                    //\t Nbr 2 participates in an exchange for p=5.
                    nbr_Q_b = nbr_Q_2;
                    nbr_kap_b = nbr_id_2;
                    if (nbr_kap_b >= 0)
                    {
                        if ((J==0))
                        {
                            s_F_pb[(I+1) + 6*(4+1) + 0] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                        }
                    }
                    //\t Nbr 5 participates in an exchange for p=5.
                    nbr_Q_b = nbr_Q_5;
                    nbr_kap_b = nbr_id_5;
                    if (nbr_kap_b >= 0)
                    {
                        if ((I==0)and(J==0))
                        {
                            s_F_pb[(4+1) + 6*(4+1) + 0] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells];
                        }
                    }
                    __syncthreads();
                    // Main writes.
                    if (cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] > -1)
                    {
                        if (s_F_pb[(I+1+1)+6*(J+1+1)+0] >= 0)
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 5*n_maxcells] = s_F_pb[(I+1+1)+6*(J+1+1)+0];
                        }
                        if (s_F_p[(I+1+-1)+6*(J+1+-1)+0] >= 0)
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 7*n_maxcells] = s_F_p[(I+1+-1)+6*(J+1+-1)+0];
                        }
                    }
                    // Neighbor writes.
                    //\t Nbr 1 participates in an exchange for p=5.
                    nbr_Q_b = nbr_Q_1;
                    nbr_kap_b = nbr_id_1;
                    if ((nbr_kap_b >=0) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and ((I==0)))
                    {
                        if (s_F_p[ (I+1+-1+(1*4)) + 6*(J+1+-1+(0*4)) + 0 ] >= 0)
                        {
                            cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells] = s_F_p[ (I+1+-1+(1*4)) + 6*(J+1+-1+(0*4)) + 0 ];
                        }
                    }
                    //\t Nbr 2 participates in an exchange for p=5.
                    nbr_Q_b = nbr_Q_2;
                    nbr_kap_b = nbr_id_2;
                    if ((nbr_kap_b >=0) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and ((J==0)))
                    {
                        if (s_F_p[ (I+1+-1+(0*4)) + 6*(J+1+-1+(1*4)) + 0 ] >= 0)
                        {
                            cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells] = s_F_p[ (I+1+-1+(0*4)) + 6*(J+1+-1+(1*4)) + 0 ];
                        }
                    }
                    //\t Nbr 5 participates in an exchange for p=5.
                    nbr_Q_b = nbr_Q_5;
                    nbr_kap_b = nbr_id_5;
                    if ((nbr_kap_b >=0) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and ((I==0)and(J==0)))
                    {
                        if (s_F_p[ (I+1+-1+(1*4)) + 6*(J+1+-1+(1*4)) + 0 ] >= 0)
                        {
                            cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 7*n_maxcells] = s_F_p[ (I+1+-1+(1*4)) + 6*(J+1+-1+(1*4)) + 0 ];
                        }
                    }
                    __syncthreads();
                    
                    
                    
                    //
                    // DDFs p=8, pb=Lpb(8).
                    //
                    for (int q = 0; q < 3; q += 1)
                    {
                        if (threadIdx.x + q*16 < 36)
                        {
                            s_F_p[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                            s_F_pb[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                        }
                    }
                    __syncthreads();
                    s_F_p[(I+1)+6*(J+1)+0] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                    s_F_pb[(I+1)+6*(J+1)+0] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                    //\t Nbr 1 participates in an exchange for p=8.
                    nbr_Q_b = nbr_Q_1;
                    nbr_kap_b = nbr_id_1;
                    if (nbr_kap_b >= 0)
                    {
                        if ((I==0))
                        {
                            s_F_pb[(4+1) + 6*(J+1) + 0] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 6*n_maxcells];
                        }
                    }
                    //\t Nbr 2 participates in an exchange for pb=Lpb(8).
                    nbr_Q_b = nbr_Q_2;
                    nbr_kap_b = nbr_id_2;
                    if (nbr_kap_b >= 0)
                    {
                        if ((J==0))
                        {
                            s_F_p[(I+1) + 6*(4+1) + 0] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 8*n_maxcells];
                        }
                    }
                    __syncthreads();
                    // Main writes.
                    if (cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] > -1)
                    {
                        if (s_F_pb[(I+1+1)+6*(J+1+-1)+0] >= 0)
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 8*n_maxcells] = s_F_pb[(I+1+1)+6*(J+1+-1)+0];
                        }
                        if (s_F_p[(I+1+-1)+6*(J+1+1)+0] >= 0)
                        {
                            cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + 6*n_maxcells] = s_F_p[(I+1+-1)+6*(J+1+1)+0];
                        }
                    }
                    // Neighbor writes.
                    //\t Nbr 1 participates in an exchange for p=8.
                    nbr_Q_b = nbr_Q_1;
                    nbr_kap_b = nbr_id_1;
                    if ((nbr_kap_b >=0) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and ((I==0)))
                    {
                        if (s_F_p[ (I+1+-1+(1*4)) + 6*(J+1+1+(0*4)) + 0 ] >= 0)
                        {
                            cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 6*n_maxcells] = s_F_p[ (I+1+-1+(1*4)) + 6*(J+1+1+(0*4)) + 0 ];
                        }
                    }
                    //\t Nbr 2 participates in an exchange for pb=Lpb(8).
                    nbr_Q_b = nbr_Q_2;
                    nbr_kap_b = nbr_id_2;
                    if ((nbr_kap_b >=0 ) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and ((J==0)))
                    {
                        if (s_F_pb[ (I+1+1+(0*4)) + 6*(J+1+-1+(1*4)) + 0 ] >= 0)
                        {
                            cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + 8*n_maxcells] = s_F_pb[ (I+1+1+(0*4)) + 6*(J+1+-1+(1*4)) + 0 ];
                        }
                    }
                    __syncthreads();
                    
                    
                    
                    // Finished.
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Stream_Original_D2Q9(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Stream_Original_D2Q9<ufloat_t,AP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev]);
	}

	return 0;
}

