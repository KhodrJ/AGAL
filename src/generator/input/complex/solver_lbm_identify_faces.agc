# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
#FILE_NAME testname
#FILE_DIR ./output/

ROUTINE_NAME IdentifyFaces
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var

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
