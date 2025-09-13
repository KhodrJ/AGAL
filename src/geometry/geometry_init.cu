/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_Init()
{
    // From input.
    Lx                      = parser->params_dbl["L_c"];
    Ly                      = parser->params_dbl["L_fy"]*Lx;
    Lz                      = parser->params_dbl["L_fz"]*Lx;
    Nx                      = parser->params_int["Nx"];
    input_dir               = parser->params_str["I_DIR_NAME"];
    output_dir              = parser->params_str["P_DIR_NAME"];
    G_NEAR_WALL_DISTANCE    = parser->params_dbl["G_NEAR_WALL_DISTANCE"];
    G_FILENAME              = parser->params_str["G_FILENAME"];
    G_LOADTYPE              = parser->params_int["G_LOADTYPE"];
    G_PRINT                 = parser->params_int["G_PRINT"];
    G_BIN_DENSITY           = parser->params_int["G_BIN_DENSITY"];
    G_BIN_LEVELS            = parser->params_int["G_BIN_LEVELS"];
    G_BIN_SPEC              = parser->params_int["G_BIN_SPEC"];
    MAX_LEVELS_WALL         = parser->params_int["MAX_LEVELS_WALL"];
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_Init_Arrays_IndexLists_CPU()
{
    if (v_geom_f_node_X.size() > 0)
    {
        init_index_lists = 1;
        G_UpdateCounts();
        geom_f_node_X = new ufloat_g_t[3*n_nodes_a];
        geom_ID_face = new int[3*n_faces_a];
        geom_ID_face_attr = new ufloat_g_t[n_faces_a];
        for (int j = 0; j < n_nodes_a; j++)
        {
            geom_f_node_X[j + 0*n_nodes_a] = 0.0;
            geom_f_node_X[j + 1*n_nodes_a] = 0.0;
            geom_f_node_X[j + 2*n_nodes_a] = 0.0;
            if (j < n_nodes)
            {
                geom_f_node_X[j + 0*n_nodes_a] = v_geom_f_node_X[j];
                geom_f_node_X[j + 1*n_nodes_a] = v_geom_f_node_Y[j];
                geom_f_node_X[j + 2*n_nodes_a] = v_geom_f_node_Z[j];
            }
        }
        for (int j = 0; j < n_faces_a; j++)
        {
            geom_ID_face[j + 0*n_faces_a] = -1;
            geom_ID_face[j + 1*n_faces_a] = -1;
            geom_ID_face[j + 2*n_faces_a] = -1;
            if (j < n_faces)
            {
                geom_ID_face[j + 0*n_faces_a] = v_geom_ID_face_1[j];
                geom_ID_face[j + 1*n_faces_a] = v_geom_ID_face_2[j];
                geom_ID_face[j + 2*n_faces_a] = v_geom_ID_face_3[j];
            }
        }
    }
    else
        std::cout << "[-] Index lists already built, reset first to rebuild..." << std::endl;
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_Init_Arrays_CoordsList_CPU()
{
    if (v_geom_f_face_1_X.size() > 0)
    {
        // Constants.
        constexpr int N_DIM = AP->N_DIM;
        ufloat_g_t odenom = (ufloat_g_t)1.0 / (ufloat_g_t)N_DIM;
        ufloat_g_t eps = EPS<ufloat_g_t>();
        
        // Update tracker variable and face count.
        G_RefineFaces_Length();
        
        std::cout << "[-] Initializing the CPU coords list array..." << std::endl;
        geom_f_face_X = new ufloat_g_t[16*n_faces_a];
        geom_f_face_Xt = new ufloat_g_t[16*n_faces_a];
        
        for (int j = 0; j < n_faces_a; j++)
        {
            for (int p = 0; p < 16; p++) geom_f_face_X[j + p*n_faces_a] = 0.0;
            for (int p = 0; p < 16; p++) geom_f_face_Xt[p + j*16] = 0.0;
            if (j < n_faces)
            {
                // Load vertices from vectors.
                vec3<ufloat_g_t> v1
                (
                    v_geom_f_face_1_X[j],
                    v_geom_f_face_1_Y[j],
                    v_geom_f_face_1_Z[j]
                );
                vec3<ufloat_g_t> v2
                (
                    v_geom_f_face_2_X[j],
                    v_geom_f_face_2_Y[j],
                    v_geom_f_face_2_Z[j]
                );
                vec3<ufloat_g_t> v3
                (
                    v_geom_f_face_3_X[j],
                    v_geom_f_face_3_Y[j],
                    v_geom_f_face_3_Z[j]
                );
                
                // Compute face normals.
                vec3<ufloat_g_t> n = FaceNormalUnit<ufloat_g_t,N_DIM>(v1,v2,v3);
                vec3<ufloat_g_t> e1 = UnitV(v2-v1);
                vec3<ufloat_g_t> e2 = UnitV(v3-v2);
                vec3<ufloat_g_t> e3 = UnitV(v1-v3);
                
                // Compute edge normals.
                vec3<ufloat_g_t> en1 = UnitV(CrossV(e1,n));
                vec3<ufloat_g_t> en2 = UnitV(CrossV(e2,n));
                vec3<ufloat_g_t> en3 = UnitV(CrossV(e3,n));
                
                // Adjust vertices so that cell-face links with dirty shortcut to account for round-off errors.
                vec3<ufloat_g_t> vc = (v1 + v2 + v3)*odenom;
                v1 = v1 + UnitV(v1-vc)*eps;
                v2 = v2 + UnitV(v2-vc)*eps;
                v3 = v3 + UnitV(v3-vc)*eps;
                
                // Write vertices.
                geom_f_face_X[j + 0*n_faces_a] = v1.x;
                geom_f_face_X[j + 1*n_faces_a] = v1.y;
                geom_f_face_X[j + 2*n_faces_a] = v1.z;
                geom_f_face_X[j + 3*n_faces_a] = v2.x;
                geom_f_face_X[j + 4*n_faces_a] = v2.y;
                geom_f_face_X[j + 5*n_faces_a] = v2.z;
                geom_f_face_X[j + 6*n_faces_a] = v3.x;
                geom_f_face_X[j + 7*n_faces_a] = v3.y;
                geom_f_face_X[j + 8*n_faces_a] = v3.z;
                //
                geom_f_face_Xt[0 + j*16] = v1.x;
                geom_f_face_Xt[1 + j*16] = v1.y;
                geom_f_face_Xt[2 + j*16] = v1.z;
                geom_f_face_Xt[3 + j*16] = v2.x;
                geom_f_face_Xt[4 + j*16] = v2.y;
                geom_f_face_Xt[5 + j*16] = v2.z;
                geom_f_face_Xt[6 + j*16] = v3.x;
                geom_f_face_Xt[7 + j*16] = v3.y;
                geom_f_face_Xt[8 + j*16] = v3.z;
                
                // Write normals.
                geom_f_face_X[j + 9*n_faces_a] = n.x;
                geom_f_face_X[j + 10*n_faces_a] = n.y;
                geom_f_face_X[j + 11*n_faces_a] = n.z;
                //
                geom_f_face_Xt[9 + j*16] = n.x;
                geom_f_face_Xt[10 + j*16] = n.y;
                geom_f_face_Xt[11 + j*16] = n.z;
                
                // Write edges.
//                 geom_f_face_X[j + 12*n_faces_a] = ex1;
//                 geom_f_face_X[j + 13*n_faces_a] = ey1;
//                 geom_f_face_X[j + 14*n_faces_a] = ez1;
//                 geom_f_face_X[j + 15*n_faces_a] = ex2;
//                 geom_f_face_X[j + 16*n_faces_a] = ey2;
//                 geom_f_face_X[j + 17*n_faces_a] = ez2;
//                 geom_f_face_X[j + 18*n_faces_a] = ex3;
//                 geom_f_face_X[j + 19*n_faces_a] = ey3;
//                 geom_f_face_X[j + 20*n_faces_a] = ez3;
                //
//                 geom_f_face_X[12 + j*32] = ex1;
//                 geom_f_face_X[13 + j*32] = ey1;
//                 geom_f_face_X[14 + j*32] = ez1;
//                 geom_f_face_X[15 + j*32] = ex2;
//                 geom_f_face_X[16 + j*32] = ey2;
//                 geom_f_face_X[17 + j*32] = ez2;
//                 geom_f_face_X[18 + j*32] = ex3;
//                 geom_f_face_X[19 + j*32] = ey3;
//                 geom_f_face_X[20 + j*32] = ez3;
                
                // Write edge normals.
//                 geom_f_face_X[j + 21*n_faces_a] = enx1;
//                 geom_f_face_X[j + 22*n_faces_a] = eny1;
//                 geom_f_face_X[j + 23*n_faces_a] = enz1;
//                 geom_f_face_X[j + 24*n_faces_a] = enx2;
//                 geom_f_face_X[j + 25*n_faces_a] = eny2;
//                 geom_f_face_X[j + 26*n_faces_a] = enz2;
//                 geom_f_face_X[j + 27*n_faces_a] = enx3;
//                 geom_f_face_X[j + 28*n_faces_a] = eny3;
//                 geom_f_face_X[j + 29*n_faces_a] = enz3;
                //
//                 geom_f_face_X[21 + j*32] = enx1;
//                 geom_f_face_X[22 + j*32] = eny1;
//                 geom_f_face_X[23 + j*32] = enz1;
//                 geom_f_face_X[24 + j*32] = enx2;
//                 geom_f_face_X[25 + j*32] = eny2;
//                 geom_f_face_X[26 + j*32] = enz2;
//                 geom_f_face_X[27 + j*32] = enx3;
//                 geom_f_face_X[28 + j*32] = eny3;
//                 geom_f_face_X[29 + j*32] = enz3;
            }
        }
        
        // Allocate memory on the GPU to store geometry data and copy the CPU data.
        std::cout << "[-] Initializing the GPU coords list array..." << std::endl;
        gpuErrchk( cudaMalloc((void **)&c_geom_f_face_X, 16*n_faces_a*sizeof(ufloat_g_t)) );
        gpuErrchk( cudaMalloc((void **)&c_geom_f_face_Xt, 16*n_faces_a*sizeof(ufloat_g_t)) );
        gpuErrchk( cudaMemcpy(c_geom_f_face_X, geom_f_face_X, 16*n_faces_a*sizeof(ufloat_g_t), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(c_geom_f_face_Xt, geom_f_face_Xt, 16*n_faces_a*sizeof(ufloat_g_t), cudaMemcpyHostToDevice) );
        std::cout << "[-] Finished copying the coords list array to the GPU..." << std::endl;
        cudaDeviceSynchronize();
        
        // Record that coordinates lists have been initialized.
        init_coords_list = 1;
    }
    else
        std::cout << "[-] Coords list already built, reset first to rebuild..." << std::endl;
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_UpdateCounts()
{
    if (v_geom_f_node_X.size() > 0)
    {
        n_nodes = v_geom_f_node_X.size();
        n_faces = v_geom_ID_face_1.size();
        n_nodes_a = n_nodes + 128-(n_nodes%128);
        n_faces_a = n_faces + 128-(n_faces%128);
    }
    if (v_geom_f_face_1_X.size() > 0)
    {
        n_faces = v_geom_f_face_1_X.size();
        n_nodes = 3*n_faces;
        n_nodes_a = n_nodes + 128-(n_nodes%128);
        n_faces_a = n_faces + 128-(n_faces%128);
    }
    
    if (v_geom_f_node_X.size() == 0 && v_geom_f_face_1_X.size() == 0)
        std::cout << "[-] Warning: neither set of lists has been loaded. Count is zero..." << std::endl;
    
    return 0;
}
