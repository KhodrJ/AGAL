# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_compute_forces_cv
FILE_DIR ../solver_lbm/

ROUTINE_NAME ComputeForces_CV
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var

KERNEL_REQUIRE const int is_root                             | L==N_LEVEL_START
KERNEL_REQUIRE const int n_ids_idev_L                        | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE const long int n_maxcells
KERNEL_REQUIRE const int n_maxcblocks
KERNEL_REQUIRE const ufloat_t dx_L                           | dxf_vec[L]
KERNEL_REQUIRE const ufloat_t dv_L                           | dvf_vec[L]
KERNEL_REQUIRE const ufloat_t otau_0                         | dxf_vec[N_LEVEL_START]/tau_vec[N_LEVEL_START]
KERNEL_REQUIRE const int *__restrict__ id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE const int *__restrict__ cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE const ufloat_t *__restrict__ cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE const ufloat_t *__restrict__ cblock_f_X       | mesh->c_cblock_f_X[i_dev]
KERNEL_REQUIRE const int *__restrict__ cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE const int *__restrict__ cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE const int *__restrict__ cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE const int *__restrict__ cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cblock_f_Ff            | mesh->c_cblock_f_Ff[i_dev]
KERNEL_REQUIRE const bool geometry_init                      | mesh->geometry_init
KERNEL_REQUIRE const int order                               | S_FORCE_TYPE
KERNEL_REQUIRE const ufloat_t cv_xm                          | S_FORCEVOLUME_Xm
KERNEL_REQUIRE const ufloat_t cv_xM                          | S_FORCEVOLUME_XM
KERNEL_REQUIRE const ufloat_t cv_ym                          | S_FORCEVOLUME_Ym
KERNEL_REQUIRE const ufloat_t cv_yM                          | S_FORCEVOLUME_YM
KERNEL_REQUIRE const ufloat_t cv_zm                          | S_FORCEVOLUME_Zm
KERNEL_REQUIRE const ufloat_t cv_zM                          | S_FORCEVOLUME_ZM

ROUTINE_TEMPLATE_PARAMS int post_step
ROUTINE_TEMPLATE_VALS 0
ROUTINE_TEMPLATE_VALS 1
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==0
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==1



#
# Kernel definition.
#

REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ ufloat_t s_Fpx[M_TBLOCK];
REG __shared__ ufloat_t s_Fmx[M_TBLOCK];
REG __shared__ ufloat_t s_Fpy[M_TBLOCK];
REG __shared__ ufloat_t s_Fmy[M_TBLOCK];
INIF Ldim==3
    REG __shared__ ufloat_t s_Fpz[M_TBLOCK];
    REG __shared__ ufloat_t s_Fmz[M_TBLOCK];
END_INIF
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int I = threadIdx.x % 4;
REG int J = (threadIdx.x / 4) % 4;
INIF Ldim==3
    REG int K = (threadIdx.x / 4) / 4;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
#REG int valid_block = -1;
REG int valid_mask = -10;
INFOR p 1   0 Lsize 1
    REG ufloat_t f_<p> = (ufloat_t)(0.0);
END_INFOR
REG ufloat_t rho = (ufloat_t)(0.0);
REG ufloat_t rhou = (ufloat_t)(0.0);
REG ufloat_t rhov = (ufloat_t)(0.0);
INIF Ldim==3
    REG ufloat_t rhow = (ufloat_t)(0.0);
END_INIF
REG ufloat_t x = (ufloat_t)(0.0);
REG ufloat_t y = (ufloat_t)(0.0);
INIF Ldim==3
    REG ufloat_t z = (ufloat_t)(0.0);
END_INIF
REG bool participatesV = false;
REG bool participatesS = false;
REG ufloat_t omeg = otau_0;
REG ufloat_t omegp = (ufloat_t)(1.0) - omeg;
REG ufloat_t cdotu = (ufloat_t)(0.0);
REG ufloat_t udotu = (ufloat_t)(0.0);



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
#TEMPLATE ARG (valid_block==1)
#TEMPLATE ARG REG valid_block=cblock_ID_onb[i_kap_b];


// Compute cell coordinates and retrieve macroscopic properties.
REG valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
REG i_kap_bc = cblock_ID_nbr_child[i_kap_b];
REG s_Fpx[threadIdx.x] = 0;
REG s_Fmx[threadIdx.x] = 0;
REG s_Fpy[threadIdx.x] = 0;
REG s_Fmy[threadIdx.x] = 0;
INIF Ldim==3
    REG s_Fpz[threadIdx.x] = 0;
    REG s_Fmz[threadIdx.x] = 0;
END_INIF
__syncthreads();
<

// Load the cell coordinates. Check if the current cell participates.
REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
INIF Ldim==3
    REG z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
END_INIF
INIF Ldim==2
    REG participatesV = CheckPointInRegion2D(x,y,cv_xm,cv_xM,cv_ym,cv_yM) && valid_mask != -1;
INELSE
    REG participatesV = CheckPointInRegion3D(x,y,z,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM) && valid_mask != -1;
END_INIF
<

// Load the DDFs. Compute the momentum in all cells.
INFOR p 1   0 Lsize 1
    REG-v f_<p> = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
END_INFOR
#OUTIF (post_step == 0 && L == 0) // Post-collision.
    #REG-vsvz rhou = ( gNz(gSUM(i,0 Lsize 1,Lc0(<i>)*f_<i>)) );
    #REG-vsvz rhov = ( gNz(gSUM(i,0 Lsize 1,Lc1(<i>)*f_<i>)) );
    #INIF Ldim==3
        #REG-vsvz rhow = ( gNz(gSUM(i,0 Lsize 1,Lc2(<i>)*f_<i>)) );
    #END_INIF
#END_OUTIF
#OUTIF (post_step == 1 || L > 0) // Post-streaming.
REG-vsvz rho = ( gNz(gSUM(i,0 Lsize 1,f_<i>)) );
REG-vsvz rhou = ( gNz(gSUM(i,0 Lsize 1,Lc0(<i>)*f_<i>)) );
REG-vsvz rhov = ( gNz(gSUM(i,0 Lsize 1,Lc1(<i>)*f_<i>)) );
INIF Ldim==3
    REG-vsvz rhow = ( gNz(gSUM(i,0 Lsize 1,Lc2(<i>)*f_<i>)) );
END_INIF
#END_OUTIF
<

// Add force contributions in the volume.
OUTIF (participatesV && post_step==0 && i_kap_bc<0)
    OUTIF (rhou > 0)
        REG s_Fpx[threadIdx.x] += rhou;
    END_OUTIF
    OUTIF (rhou < 0)
        REG s_Fmx[threadIdx.x] += rhou;
    END_OUTIF
    OUTIF (rhov > 0)
        REG s_Fpy[threadIdx.x] += rhov;
    END_OUTIF
    OUTIF (rhov < 0)
        REG s_Fmy[threadIdx.x] += rhov;
    END_OUTIF
    INIF Ldim==3
        OUTIF (rhow > 0)
            REG s_Fpz[threadIdx.x] += rhow;
        END_OUTIF
        OUTIF (rhow < 0)
            REG s_Fmz[threadIdx.x] += rhow;
        END_OUTIF
    END_INIF
END_OUTIF
OUTIF (participatesV && post_step==1 && i_kap_bc<0)
    OUTIF (rhou > 0)
        REG s_Fmx[threadIdx.x] += rhou;
    END_OUTIF
    OUTIF (rhou < 0)
        REG s_Fpx[threadIdx.x] += rhou;
    END_OUTIF
    OUTIF (rhov > 0)
        REG s_Fmy[threadIdx.x] += rhov;
    END_OUTIF
    OUTIF (rhov < 0)
        REG s_Fpy[threadIdx.x] += rhov;
    END_OUTIF
    INIF Ldim==3
        OUTIF (rhow > 0)
            REG s_Fmz[threadIdx.x] += rhow;
        END_OUTIF
        OUTIF (rhow < 0)
            REG s_Fpz[threadIdx.x] += rhow;
        END_OUTIF
    END_INIF
END_OUTIF
<

// Perform collisions right here, but don't store.
OUTIF (post_step==0)
    REG rhou /= rho;
    REG rhov /= rho;
    INIF Ldim==3
        REG rhow /= rho;
    END_INIF
    REG-vmz udotu = gNz(rhou*rhou + rhov*rhov + gI(Ldim-2)*rhow*rhow);
    INFOR p 1   1 Lsize 1
        REG-vz cdotu = gNz(Lc0(<p>)*rhou + Lc1(<p>)*rhov + Lc2(<p>)*rhow);
        REG-vm f_<p> = f_<p>*omegp + ( (ufloat_t)(gD(Lw(<p>)))*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
    END_INFOR
END_OUTIF
<

// Now check if DDFs are leaving the CV.
OUTIF (post_step==0 && is_root)
    INFOR p 1   1 Lsize 1
        INIF Ldim==2
        REG-v participatesS = !CheckPointInRegion2D(x+Lc0(<p>)*dx_L,y+Lc1(<p>)*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
        INELSE
            REG-v participatesS = !CheckPointInRegion3D(x+Lc0(<p>)*dx_L,y+Lc1(<p>)*dx_L,z+Lc2(<p>)*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
        END_INIF
        
        OUTIF (participatesS && participatesV)
            INIF (Lc0(<p>) > 0)
                REG s_Fmx[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc0(<p>) < 0)
                REG s_Fpx[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc1(<p>) > 0)
                REG s_Fmy[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc1(<p>) < 0)
                REG s_Fpy[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc2(<p>) > 0)
                REG s_Fmz[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc2(<p>) < 0)
                REG s_Fpz[threadIdx.x] += f_<p>;
            END_INIF
        END_OUTIF
    END_INFOR
END_OUTIF
<

// Now check if DDFs are entering the CV.
OUTIF (post_step==1 && is_root)
    INFOR p 1   1 Lsize 1
        INIF Ldim==2
        REG-v participatesS = !CheckPointInRegion2D(x+Lc0(Lpb(<p>))*dx_L,y+Lc1(Lpb(<p>))*dx_L,cv_xm,cv_xM,cv_ym,cv_yM);
        INELSE
            REG-v participatesS = !CheckPointInRegion3D(x+Lc0(Lpb(<p>))*dx_L,y+Lc1(Lpb(<p>))*dx_L,z+Lc2(Lpb(<p>))*dx_L,cv_xm,cv_xM,cv_ym,cv_yM,cv_zm,cv_zM);
        END_INIF
        
        OUTIF (participatesS && participatesV)
            INIF (Lc0(Lpb(<p>)) > 0)
                REG-v s_Fmx[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc0(Lpb(<p>)) < 0)
                REG-v s_Fpx[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc1(Lpb(<p>)) > 0)
                REG-v s_Fmy[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc1(Lpb(<p>)) < 0)
                REG-v s_Fpy[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc2(Lpb(<p>)) > 0)
                REG-v s_Fmz[threadIdx.x] += f_<p>;
            END_INIF
            INIF (Lc2(Lpb(<p>)) < 0)
                REG-v s_Fpz[threadIdx.x] += f_<p>;
            END_INIF
        END_OUTIF
    END_INFOR
END_OUTIF
<


#OUTIF (participatesV)
    // Reductions for the sums of force contributions in this cell-block.
    __syncthreads();
    REG for (int s=blockDim.x/2; s>0; s>>=1)
    REG {
        OUTIF (threadIdx.x < s)
                REG s_Fpx[threadIdx.x] = s_Fpx[threadIdx.x] + s_Fpx[threadIdx.x + s];
                REG s_Fmx[threadIdx.x] = s_Fmx[threadIdx.x] + s_Fmx[threadIdx.x + s];
                REG s_Fpy[threadIdx.x] = s_Fpy[threadIdx.x] + s_Fpy[threadIdx.x + s];
                REG s_Fmy[threadIdx.x] = s_Fmy[threadIdx.x] + s_Fmy[threadIdx.x + s];
                INIF Ldim==3
                    REG s_Fpz[threadIdx.x] = s_Fpz[threadIdx.x] + s_Fpz[threadIdx.x + s];
                    REG s_Fmz[threadIdx.x] = s_Fmz[threadIdx.x] + s_Fmz[threadIdx.x + s];
                END_INIF
        END_OUTIF
        REG __syncthreads();
    REG }
    
    // Store the sums of contributions in global memory; this will be reduced further later.
    OUTIF (threadIdx.x == 0)
        OUTIF (post_step == 0)
            REG cblock_f_Ff[i_kap_b + 0*n_maxcblocks] = s_Fpx[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 1*n_maxcblocks] = s_Fmx[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 2*n_maxcblocks] = s_Fpy[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 3*n_maxcblocks] = s_Fmy[0]*dv_L;
            INIF Ldim==3
                REG cblock_f_Ff[i_kap_b + 4*n_maxcblocks] = s_Fpz[0]*dv_L;
                REG cblock_f_Ff[i_kap_b + 5*n_maxcblocks] = s_Fmz[0]*dv_L;
            END_INIF
        OUTELSE
            REG cblock_f_Ff[i_kap_b + 0*n_maxcblocks] += s_Fpx[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 1*n_maxcblocks] += s_Fmx[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 2*n_maxcblocks] += s_Fpy[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 3*n_maxcblocks] += s_Fmy[0]*dv_L;
            INIF Ldim==3
                REG cblock_f_Ff[i_kap_b + 4*n_maxcblocks] += s_Fpz[0]*dv_L;
                REG cblock_f_Ff[i_kap_b + 5*n_maxcblocks] += s_Fmz[0]*dv_L;
            END_INIF
        END_OUTIF
    END_OUTIF
#END_OUTIF


TEMPLATE NEW_BLOCK
END_TEMPLATE

