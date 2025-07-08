# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_debug_drawgeom
FILE_DIR ../solver_lbm/

ROUTINE_NAME Debug_DrawGeometry
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L

KERNEL_REQUIRE int n_ids_idev_L                        | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE int n_maxcells_b                        | mesh->n_maxcells_b
KERNEL_REQUIRE int n_maxblocks_b                       | mesh->n_solidb
KERNEL_REQUIRE ufloat_t dx_L                           | dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L                          | tau_vec[L]
KERNEL_REQUIRE int *__restrict__ id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *__restrict__ cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE ufloat_g_t *__restrict__ cells_f_X_b    | mesh->c_cells_f_X_b[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cells_f_F_aux    | mesh->c_cells_f_F_aux[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cblock_f_X       | mesh->c_cblock_f_X[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_onb_solid   | mesh->c_cblock_ID_onb_solid[i_dev]
KERNEL_REQUIRE bool geometry_init                      | mesh->geometry_init
KERNEL_REQUIRE bool compute_forces                     | compute_forces



#
# Kernel definition.
#

REG constexpr int Nqx = AP->Nqx;
REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG constexpr int N_DIM = AP->N_DIM;
REG constexpr int N_Q_max = AP->N_Q_max;

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ int s_ID_nbr[N_Q_max];
REG __shared__ ufloat_t s_u[3*M_TBLOCK];
REG __shared__ double s_Fpx[M_TBLOCK];
REG __shared__ double s_Fmx[M_TBLOCK];
REG __shared__ double s_Fpy[M_TBLOCK];
REG __shared__ double s_Fmy[M_TBLOCK];
REG __shared__ double s_Fpz[M_TBLOCK];
REG __shared__ double s_Fmz[M_TBLOCK];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int I = threadIdx.x % 4;
REG int Ip = I;
REG int J = (threadIdx.x / 4) % 4;
REG int Jp = J;
INIF Ldim==3
    REG int K = (threadIdx.x / 4) / 4;
    REG int Kp = K;
END_INIF
REG ufloat_t x __attribute__((unused)) = (ufloat_t)(0.0);
REG ufloat_t y __attribute__((unused)) = (ufloat_t)(0.0);
REG ufloat_t z __attribute__((unused)) = (ufloat_t)(0.0);
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int nbr_kap_b = -1;
REG int nbr_kap_c = -1;
REG int valid_block = -1;
REG int block_mask = -1;
REG int valid_mask = -1;
REG ufloat_t f_p = (ufloat_t)(0.0);
REG ufloat_t f_q = (ufloat_t)(0.0);
REG ufloat_t f_m = (ufloat_t)(0.0);
REG ufloat_g_t dist_p = (ufloat_g_t)(0.0);
REG ufloat_t rho = (ufloat_t)(0.0);
REG ufloat_t u = (ufloat_t)(0.0);
REG ufloat_t v = (ufloat_t)(0.0);
REG ufloat_t w = (ufloat_t)(0.0);
REG ufloat_t ub = (ufloat_t)(0.0);
REG ufloat_t vb = (ufloat_t)(0.0);
REG ufloat_t wb = (ufloat_t)(0.0);
REG ufloat_t cdotu = (ufloat_t)(0.0);
REG ufloat_t udotu = (ufloat_t)(0.0);
REG bool near_geom = false;



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (valid_block==1)
TEMPLATE ARG REG valid_block=cblock_ID_onb[i_kap_b];


// Compute cell coordinates and retrieve macroscopic properties.
REG valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
OUTIFL (geometry_init)
    REG block_mask = cblock_ID_onb_solid[i_kap_b];
INIF Ldim==2
    REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
    REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
INELSE
    REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
    REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
    REG z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
END_INIF
OUTIF (valid_block == 1 && threadIdx.x == 0)
    INFOR p 1   1 gI(3^Ldim) 1
        REG s_ID_nbr[<p>] = cblock_ID_nbr[i_kap_b + <p>*n_maxcblocks];
    END_INFOR
END_OUTIF
REG __syncthreads();
<


<
INFOR p 1   1 Lsize 1
    INIF (<p>==0 or ( (Ldim==2 and (<p>==1 or (<p>+1)%3==0)) or (Ldim==3 and ((((<p>-1)%2==0)and(<p><25))or(<p>==26))) ))
        //
        // p = <p>
        //
        <
        
        //
        // Impose curved boundary conditions.
        // Do this only if adjacent to a solid cell (by checking the cell mask).
        // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
        //
        INIF (<p> > 0)
            OUTIF (valid_mask == -2)
                #OUTIF (cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + <p>*n_maxcells_b] > 0 && cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells_b] > 0)
                    #REG printf("BC ERROR\n");
                #END_OUTIF
            
                # For p.
                REG-v dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + <p>*n_maxcells_b];
                OUTIF (dist_p > 0)
                    REG near_geom = true;
                    INIF Ldim==2
                        REG-vm printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n",   x,x+dist_p*(gD(Lc0(<p>))),y,y+dist_p*(gD(Lc1(<p>))));
                    INELSE
                        REG-vm printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n",   x,x+dist_p*(gD(Lc0(<p>))),y,y+dist_p*(gD(Lc1(<p>))),z,z+dist_p*(gD(Lc2(<p>))));
                    END_INIF
                END_OUTIF
                OUTIF (dist_p >= dx_L)
                    REG printf("ERROR\n");
                END_OUTIF
                
                # For q.
                // If computing forces, add the contributions of DDFs entering the geometry.
                REG-v dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells_b];
                OUTIF (dist_p > 0)
                    REG near_geom = true;
                    INIF Ldim==2
                        REG-vm printf("plot([%17.15f %17.15f],[%17.15f %17.15f], 'k-');\n",   x,x+dist_p*(gD(Lc0(Lpb(<p>)))),y,y+dist_p*(gD(Lc1(Lpb(<p>)))));
                    INELSE
                        REG-vm printf("plot3([%17.15f %17.15f],[%17.15f %17.15f],[%17.15f %17.15f], 'b-');\n",   x,x+dist_p*(gD(Lc0(Lpb(<p>)))),y,y+dist_p*(gD(Lc1(Lpb(<p>)))),z,z+dist_p*(gD(Lc2(Lpb(<p>)))));
                    END_INIF
                END_OUTIF
                OUTIF (dist_p >= dx_L)
                    REG printf("ERROR\n");
                END_OUTIF

            END_OUTIF
        END_INIF
        
        <
    END_INIF
END_INFOR

#OUTIF (near_geom)
    INIF Ldim==2
        REG printf("plot(%17.15f,%17.15f,'k*');\n",x,y);
        REG printf("plot([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'k-');\n",
            x-dx_L/2.0,x+dx_L/2.0,x+dx_L/2.0,x-dx_L/2.0,x-dx_L/2.0,
            y-dx_L/2.0,y-dx_L/2.0,y+dx_L/2.0,y+dx_L/2.0,y-dx_L/2.0
        );
    INELSE
        OUTIF (valid_mask == -2)
            REG printf("plot3(%17.15f,%17.15f,%17.5f,'k*');\n",x,y,z);
        OUTELSE
            OUTIFL (valid_mask == -1)
                REG printf("plot3(%17.15f,%17.15f,%17.5f,'r*');\n",x,y,z);
            END_OUTIFL
        END_OUTIF
        #REG printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'k-');\n",
            #x-dx_L/2.0,x-dx_L/2.0,x-dx_L/2.0,x-dx_L/2.0,x-dx_L/2.0,
            #y-dx_L/2.0,y+dx_L/2.0,y+dx_L/2.0,y-dx_L/2.0,y-dx_L/2.0,
            #z-dx_L/2.0,z-dx_L/2.0,z+dx_L/2.0,z+dx_L/2.0,z-dx_L/2.0
        #);
        #REG printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'k-');\n",
            #x+dx_L/2.0,x+dx_L/2.0,x+dx_L/2.0,x+dx_L/2.0,x+dx_L/2.0,
            #y-dx_L/2.0,y+dx_L/2.0,y+dx_L/2.0,y-dx_L/2.0,y-dx_L/2.0,
            #z-dx_L/2.0,z-dx_L/2.0,z+dx_L/2.0,z+dx_L/2.0,z-dx_L/2.0
        #);
        #REG printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'k-');\n",
            #x-dx_L/2.0,x+dx_L/2.0,x+dx_L/2.0,x-dx_L/2.0,x-dx_L/2.0,
            #y-dx_L/2.0,y-dx_L/2.0,y-dx_L/2.0,y-dx_L/2.0,y-dx_L/2.0,
            #z-dx_L/2.0,z-dx_L/2.0,z+dx_L/2.0,z+dx_L/2.0,z-dx_L/2.0
        #);
        #REG printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'k-');\n",
            #x-dx_L/2.0,x+dx_L/2.0,x+dx_L/2.0,x-dx_L/2.0,x-dx_L/2.0,
            #y+dx_L/2.0,y+dx_L/2.0,y+dx_L/2.0,y+dx_L/2.0,y+dx_L/2.0,
            #z-dx_L/2.0,z-dx_L/2.0,z+dx_L/2.0,z+dx_L/2.0,z-dx_L/2.0
        #);
    END_INIF
#END_OUTIF

TEMPLATE NEW_BLOCK
END_TEMPLATE

