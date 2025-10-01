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
    G_VIS_TREE              = parser->params_int["G_VIS_TREE"];
    MAX_LEVELS_WALL         = parser->params_int["MAX_LEVELS_WALL"];
    
    // Corrections.
    if (G_BIN_LEVELS > MAX_LEVELS_WALL)
        G_BIN_LEVELS = MAX_LEVELS_WALL;
    
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
        ufloat_g_t eps = static_cast<ufloat_g_t>(0.0);
        
        // Update tracker variable and face count.
        G_RefineFaces_Length();
        
        std::cout << "[-] Initializing the CPU coords list array..." << std::endl;
        geom_f_face_X = new ufloat_g_t[NVDP*n_faces_a];
        geom_f_face_Xt = new ufloat_g_t[NVDP*n_faces_a];
        
        for (int j = 0; j < n_faces_a; j++)
        {
            // Reset array data to zero.
            for (int p = 0; p < NVDP; p++) geom_f_face_X[j + p*n_faces_a] = static_cast<ufloat_g_t>(0.0);
            for (int p = 0; p < NVDP; p++) geom_f_face_Xt[p + j*NVDP] = static_cast<ufloat_g_t>(0.0);
            
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
                geom_f_face_Xt[0 + j*NVDP] = v1.x;
                geom_f_face_Xt[1 + j*NVDP] = v1.y;
                geom_f_face_Xt[2 + j*NVDP] = v1.z;
                geom_f_face_Xt[3 + j*NVDP] = v2.x;
                geom_f_face_Xt[4 + j*NVDP] = v2.y;
                geom_f_face_Xt[5 + j*NVDP] = v2.z;
                geom_f_face_Xt[6 + j*NVDP] = v3.x;
                geom_f_face_Xt[7 + j*NVDP] = v3.y;
                geom_f_face_Xt[8 + j*NVDP] = v3.z;
                
                // Write normals.
                geom_f_face_X[j + 9*n_faces_a] = n.x;
                geom_f_face_X[j + 10*n_faces_a] = n.y;
                geom_f_face_X[j + 11*n_faces_a] = n.z;
                //
                geom_f_face_Xt[9 + j*NVDP] = n.x;
                geom_f_face_Xt[10 + j*NVDP] = n.y;
                geom_f_face_Xt[11 + j*NVDP] = n.z;
            }
        }
        
        // Allocate memory on the GPU to store geometry data and copy the CPU data.
        std::cout << "[-] Initializing the GPU coords list array..." << std::endl;
        gpuErrchk( cudaMalloc((void **)&c_geom_f_face_X, NVDP*n_faces_a*sizeof(ufloat_g_t)) );
        gpuErrchk( cudaMemcpy(c_geom_f_face_X, geom_f_face_X, NVDP*n_faces_a*sizeof(ufloat_g_t), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMalloc((void **)&c_geom_f_face_Xt, NVDP*n_faces_a*sizeof(ufloat_g_t)) );
        gpuErrchk( cudaMemcpy(c_geom_f_face_Xt, geom_f_face_Xt, NVDP*n_faces_a*sizeof(ufloat_g_t), cudaMemcpyHostToDevice) );
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
