# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
#FILE_NAME testname
#FILE_DIR ./output/

ROUTINE_NAME Average
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var
ROUTINE_REQUIRE ufloat_t tau_L
ROUTINE_REQUIRE ufloat_t tau_ratio_L

KERNEL_REQUIRE int n_ids_idev_L           | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t tau_L
KERNEL_REQUIRE ufloat_t tau_ratio         | tau_ratio_L
KERNEL_REQUIRE int *id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]

ROUTINE_TEMPLATE_PARAMS int ave_type
ROUTINE_TEMPLATE_VALS 0
ROUTINE_TEMPLATE_VALS 1
ROUTINE_TEMPLATE_VALS 2
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_INTERFACE
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_BLOCK
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_GRID





# Kernel definition.

#REG __shared__ int s_ID_cblock[M_TBLOCK];
#REG int I_kap = threadIdx.x % Nbx;
#REG int J_kap = (threadIdx.x / Nbx) % Nbx;
#INIF Ldim==3
    #int K_kap = (threadIdx.x / Nbx) / Nbx;
#END_INIF
#REG ufloat_t x_kap = N_Pf(0.0);
#REG ufloat_t y_kap = N_Pf(0.0);
#INIF Ldim==3
	#REG ufloat_t z_kap = N_Pf(0.0);
#END_INIF
#REG int i_kap_b = -1;
#REG int i_kap_bc = -1;
#REG int nbr_kap_b = -1;
#REG int block_on_boundary = -1;
#INFOR p 1   0 Lsize 1
    #REG ufloat_t f_<p> = N_Pf(0.0);
#END_INFOR
#REG ufloat_t rho_kap = N_Pf(0.0);
#REG ufloat_t u_kap = N_Pf(0.0);
#REG ufloat_t v_kap = N_Pf(0.0);
#INIF Ldim==3
    #REG ufloat_t w_kap = N_Pf(0.0);
#END_INIF
#REG ufloat_t cdotu = N_Pf(0.0);
#REG ufloat_t udotu = N_Pf(0.0);
#REG ufloat_t omeg = dx_L / tau_L;
#REG ufloat_t omegp = N_Pf(1.0) - omeg;
#<

#TEMPLATE
#TEMPLATE NAME PRIMARY_ORIGINAL
#TEMPLATE ARG (i_kap_bc<0)||(block_on_boundary==1)

#// Retrieve DDFs and compute macroscopic properties.
#REG block_on_boundary = cblock_ID_onb[i_kap_b];
#REG x_kap = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*(N_Pf(0.5) + I_kap);
#REG y_kap = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*(N_Pf(0.5) + J_kap);
#INIF Ldim==3
    #REG z_kap = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*(N_Pf(0.5) + K_kap);
#END_INIF
#INFOR p 1   0 Lsize 1
    #REG-v f_<p> = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
#END_INFOR
#REG-vs rho_kap = gSUM(i,0 Lsize 1,f_<i>);
#REG-vsvz u_kap = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc0(<i>))*f_<i>))) / rho_kap;
#REG-vsvz v_kap = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc1(<i>))*f_<i>))) / rho_kap;
#INIF Ldim==2
    #REG-vmz udotu = u_kap*u_kap + v_kap*v_kap;
#INELSE
    #REG-vsvz w_kap = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc2(<i>))*f_<i>))) / rho_kap;
    #REG udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
#END_INIF

#TEMPLATE NEW_BLOCK
#END_TEMPLATE
















#INFOR p 1   1 Lsize 1

#INIF (Ldim==2)
    #// p=<p>
    #REG-vz tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / (gNz((Lc0(<p>))*nx + (Lc1(<p>))*ny));
    #OUTIF (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        #REG-vz tmpy = gNz( vyp + tmp*Lc1(<p>) );
        #REG-vz tmp = gNz( vxp + tmp*Lc0(<p>) );
        #OUTIF (CheckPointInLine(tmp, tmpy, vx1, vy1, vx2, vy2))
            #OUTIFL (tmp < dist_<p> || dist_<p> < 0)
                #REG dist_<p> = tmp;
            #END_OUTIFL
            
            ##REG-v printf("plot([%17.15f %17.15f],[%17.15f %17.15f],'r');\n",vxp,vxp+(ufloat_g_t)(Lc0(<p>))*dx_L,vyp,vyp+(ufloat_g_t)(Lc1(<p>))*dx_L);
            ##REG-v printf("plot(%17.15f,%17.15f,'b*');\n",tmp,tmpy);
        #END_OUTIF
    #END_OUTIF
#INELSE
    #// p=<p>
    #REG-vz tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (gNz((Lc0(<p>))*nx + (Lc1(<p>))*ny + (Lc2(<p>))*nz));
    #OUTIF (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        #REG-vz tmpy = gNz( vyp + tmp*Lc1(<p>) );
        #REG-vz tmpz = gNz( vzp + tmp*Lc2(<p>) );
        #REG-vz tmp = gNz( vxp + tmp*Lc0(<p>) );
        #OUTIF (CheckPointInTriangle(tmp, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            #OUTIFL (tmp < dist_<p> || dist_<p> < 0)
                #REG dist_<p> = tmp;
            #END_OUTIFL
        #END_OUTIF
    #END_OUTIF
#END_INIF

#END_INFOR

#INIF Ldim==2
    #INFOR p 1   1 9 1
        #REG-m cells_f_X_b[i_kap_b*M_CBLOCK + threadIdx.x + gI(<p>-1)*n_maxcells_b] = dist_<p>;
    #END_INFOR
#INELSE
    #INFOR p 1   1 27 1
        #REG-m cells_f_X_b[i_kap_b*M_CBLOCK + threadIdx.x + gI(<p>-1)*n_maxcells_b] = dist_<p>;
    #END_INFOR
#END_INIF

INFOR p 1   1 Lsize 1
    //
    // p = <p>
    //
    <
    
    REG b_id_p = -8;
    REG dist_p = -1;
    OUTFOR p 1   0 n_f 1
        #// p = <p>
        REG f_p = binned_face_ids[N_f+p];
        REG vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        REG vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        REG vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        #REG vx1 = s_face[0 + p*12]; vy1 = s_face[1 + p*12]; vz1 = s_face[2 + p*12];
        #REG vx2 = s_face[3 + p*12]; vy2 = s_face[4 + p*12]; vz2 = s_face[5 + p*12];
        #REG vx3 = s_face[6 + p*12]; vy3 = s_face[7 + p*12]; vz3 = s_face[8 + p*12];
        #REG nx = s_face[9 + p*12]; ny = s_face[10 + p*12]; nz = s_face[11 + p*12];
        INIF (Ldim==2)
            REG nx = vy2-vy1; ny = vx1-vx2;
            REG-vz tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / (gNz((Lc0(<p>))*nx + (Lc1(<p>))*ny));
                OUTIF (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
                    REG-vz tmpy = gNz( vyp + tmp*Lc1(<p>) );
                    REG-vz tmpx = gNz( vxp + tmp*Lc0(<p>) );
                    OUTIF (CheckPointInLine(tmpx, tmpy, vx1, vy1, vx2, vy2))
                        OUTIFL (tmp < dist_p || dist_p < 0)
                            REG dist_p = tmp;
                        END_OUTIFL
                        
                        #REG-v printf("plot([%17.15f %17.15f],[%17.15f %17.15f],'r');\n",vxp,vxp+(ufloat_g_t)(Lc0(<p>))*dx_L,vyp,vyp+(ufloat_g_t)(Lc1(<p>))*dx_L);
                        #REG-v printf("plot(%17.15f,%17.15f,'b*');\n",tmp,tmpy);
                    END_OUTIF
                END_OUTIF
        INELSE
            REG ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
            REG ex2 = vx3-vx1; ey2 = vy3-vy1;  ez2 = vz3-vz1;
            REG Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
            REG tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp;  ny /= tmp; nz /= tmp;
            REG-vz tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (gNz((Lc0(<p>))*nx + (Lc1(<p>))*ny + (Lc2(<p>))*nz));
            OUTIF (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
                REG-vz tmpy = gNz( vyp + tmp*Lc1(<p>) );
                REG-vz tmpz = gNz( vzp + tmp*Lc2(<p>) );
                REG-vz tmpx = gNz( vxp + tmp*Lc0(<p>) );
                OUTIF (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
                    OUTIFL (tmp < dist_p || dist_p < 0)
                        REG dist_p = tmp;
                    END_OUTIFL
                END_OUTIF
            END_OUTIF
        END_INIF
    END_OUTFOR
    
    REG cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + <p>*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    REG cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + <p>*n_maxcells_b] = dist_p;
    
    OUTIFL (dist_p > 0)
        REG check_cell_mask = true;
    END_OUTIFL
    <
END_INFOR


REG-v gIMP(InterpM3D(0)
REG-v InterpM3D(0)


#INFOR p 1   1 Lsize 1
        #INIF (Ldim==2)
            #REG-vz tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny) / (gNz((Lc0(<p>))*nx + (Lc1(<p>))*ny)); if (tmp > (ufloat_g_t)0.0 && tmp < dx_L) { tmpy = gNz( vyp + tmp*Lc1(<p>) ); tmp = gNz( vxp + tmp*Lc0(<p>) ); if (CheckPointInLine(tmp, tmpy, vx1, vy1, vx2, vy2)) { add_face = true; } } // p=<p>
        #INELSE
            #REG-vz tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (gNz((Lc0(<p>))*nx + (Lc1(<p>))*ny + (Lc2(<p>))*nz)); if (tmp > (ufloat_g_t)0.0 && tmp < dx_L) { tmpy = gNz( vyp + tmp*Lc1(<p>) ); tmpz = gNz( vzp + tmp*Lc2(<p>) ); tmp = gNz( vxp + tmp*Lc0(<p>) ); if (CheckPointInTriangle(tmp, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2)) { add_face = true; } } // p = <p>
        #END_INIF
#END_INFOR










#INFOR p 1   1 Lsize 1

    #//
    #// p == <p>
    #//
    #<

    #// [-]      Compute neighbor cell indices.
    #REG-v Ip = I + Lc0(<p>);
    #REG-v Jp = J + Lc1(<p>);
    #INIF Ldim==3
            #REG-v Kp = K + Lc2(<p>);
    #END_INIF
    
    #// [-]      Assign the correct neighbor cell-block ID.
    #REG nbr_kap_b = i_kap_b;
    #INFOR q 1   1 Lsize 1
            #INIF ((Lc0(<p>)==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(<p>)==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(<p>)==Lc2(<q>) or Lc2(<q>)==0))
                    #// [-][-]   Consider nbr <q>.
                    #OUTIFL ( gNa( gCOND(Lc0(<q>): 1,(Ip==4),-1,(Ip==-1),def.,(Ip>=0)and(Ip<4)) and \
                                    #gCOND(Lc1(<q>): 1,(Jp==4),-1,(Jp==-1),def.,(Jp>=0)and(Jp<4)) and \
                                    #gCOND(Ldim: 3,gCOND(Lc2(<q>): 1,(Kp==4),-1,(Kp==-1),def.,(Kp>=0)and(Kp<4)), def.,true) )\
                    #)
                            #REG nbr_kap_b = s_ID_nbr[<q>];
                    #END_OUTIFL
            #END_INIF
    #END_INFOR
    
    #// [-]      Correct the neighbor cell indices.
    #INIF (Lc0(<p>) != 0)
        #REG Ip = (4 + (Ip % 4)) % 4;
    #END_INIF
    #INIF (Lc1(<p>) != 0)
        #REG Jp = (4 + (Jp % 4)) % 4;
    #END_INIF
    #INIF (Lc2(<p>) != 0)
            #REG Kp = (4 + (Kp % 4)) % 4;
    #END_INIF
    #INIF Ldim==2
            #REG nbr_kap_c = Ip + 4*Jp;
    #INELSE
            #REG nbr_kap_c = Ip + 4*Jp + 16*Kp;
    #END_INIF
    
    #// [-]      Check the mask of the cell, it is exists.
    #OUTIFL (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == -1)
        #REG near_a_solid_cell = true;
    #END_OUTIFL
    
#END_INFOR
