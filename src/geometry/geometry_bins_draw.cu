/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry_bins.h"

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_DrawBinsAndFaces ]=======================================================  */
/*                                                                                    */
/*  This is a debug routine that generates a MATLAB script in which the various       */
/*  bins and faces assignments can be plotted.                                        */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::Bins::G_DrawBinsAndFaces(int L)
{
    // Initial parameters. Open the output file.
    int n_faces_a = geometry->n_faces_a;
    ufloat_g_t *geom_f_face_X = geometry->geom_f_face_X;
    ufloat_g_t Lx0g __attribute__((unused)) = Lx0g_vec[L + 0*n_levels];
    ufloat_g_t Ly0g __attribute__((unused)) = Lx0g_vec[L + 1*n_levels];
    ufloat_g_t Lz0g __attribute__((unused)) = Lx0g_vec[L + 2*n_levels];
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    std::ofstream out2D = std::ofstream("debug_bins_2D.m");
    std::ofstream out3D = std::ofstream("debug_bins_3D.m");
    std::cout << "Drawing bins..." << std::endl;
    out2D << "print_all = true;" << std::endl;
    out2D << "print_bin = 1;" << std::endl;
    out3D << "print_all = true;" << std::endl;
    out3D << "print_bin = 1;" << std::endl;
    
    // Set axis.
    if (N_DIM==2)
    {
        out2D << "axis([0 1 0 1]); hold on" << std::endl;
        out3D << "axis([0 1 0 1]); hold on" << std::endl;
    }
    else
    {
        out2D << "axis([0 1 0 1 0 1]); hold on" << std::endl;
        out3D << "axis([0 1 0 1 0 1]); hold on" << std::endl;
    }
    
    int counter_2D = 1;
    int counter_3D = 1;
    int kmax = N_DIM==2?1:n_bin_density[L];
    for (int k = 0; k < kmax; k++)
    {
        for (int j = 0; j < n_bin_density[L]; j++)
        {
            for (int i = 0; i < n_bin_density[L]; i++)
            {
                // Identify the bin.
                int global_bin_id = i + n_bin_density[L]*j + n_bin_density[L]*n_bin_density[L]*k;
                
                // Get the number of faces in the bin.
                int n_f = binned_face_ids_n_3D[L][global_bin_id];
                int N_f = 0;
                //if (global_bin_id == 4683)
                if (n_f > 0)
                {
                    N_f = binned_face_ids_N_3D[L][global_bin_id];
                    out3D << "% Bin #" << counter_3D << std::endl;
                    out3D << "if (print_bin == " << counter_3D << " || print_all == true)" << std::endl;
                    counter_3D++;
                }
                //if (global_bin_id == 4683)
                // If there are faces to draw, draw the bin too. Each bin gets its own unique color.
                if (n_f > 0)
                {
                    c0 = (double)(std::rand() % 256) / 256.0;
                    c1 = (double)(std::rand() % 256) / 256.0;
                    c2 = (double)(std::rand() % 256) / 256.0;
                    if (N_DIM==2)
                        DebugDrawSquareInMATLAB(out3D, i*Lx0g, (i+1)*Lx0g, j*Ly0g, (j+1)*Ly0g, c0, c1, c2);
                    else
                        DebugDrawCubeInMATLAB(out3D, i*Lx0g, (i+1)*Lx0g, j*Ly0g, (j+1)*Ly0g, k*Lz0g, (k+1)*Lz0g, c0, c1, c2);
                    //if (counter_3D==2)
                    out3D << "hold on;\n";
                    
                    for (int p = 0; p < n_f; p++)
                    {
                        int f_p = binned_face_ids_3D[L][N_f+p];
                        ufloat_g_t vx1 = geom_f_face_X[f_p + 0*n_faces_a];
                        ufloat_g_t vy1 = geom_f_face_X[f_p + 1*n_faces_a];
                        ufloat_g_t vz1 = geom_f_face_X[f_p + 2*n_faces_a];
                        ufloat_g_t vx2 = geom_f_face_X[f_p + 3*n_faces_a];
                        ufloat_g_t vy2 = geom_f_face_X[f_p + 4*n_faces_a];
                        ufloat_g_t vz2 = geom_f_face_X[f_p + 5*n_faces_a];
                        ufloat_g_t vx3 = geom_f_face_X[f_p + 6*n_faces_a];
                        ufloat_g_t vy3 = geom_f_face_X[f_p + 7*n_faces_a];
                        ufloat_g_t vz3 = geom_f_face_X[f_p + 8*n_faces_a];
                        
                        // Draw the faces in the current bin.
                        if (N_DIM==2)
                            DebugDrawLineInMATLAB(out3D, vx1, vy1, vx2, vy2, c0, c1, c2);
                        else
                            DebugDrawTriangleInMATLAB(out3D, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, c0, c1, c2);
                    }
                    
                    out3D << "end" << std::endl;
                }
            }
            
            
            
            // Identify the bin.
            int global_bin_id_2D = j + n_bin_density[L]*k;
            
            // Get the number of faces in the bin.
            int n_f_2D = binned_face_ids_n_2D[L][global_bin_id_2D];
            int N_f_2D = 0;
            if (n_f_2D > 0)
            {
                N_f_2D = binned_face_ids_N_2D[L][global_bin_id_2D];
                out2D << "% Bin #" << counter_2D << std::endl;
                counter_2D++;
            }
            
            // If there are faces to draw, draw the bin too. Each bin gets its own unique color.
            if (n_f_2D > 0)
            {
                c0 = (double)(std::rand() % 256) / 256.0;
                c1 = (double)(std::rand() % 256) / 256.0;
                c2 = (double)(std::rand() % 256) / 256.0;
                if (N_DIM==2)
                    DebugDrawSquareInMATLAB(out2D, 0, Lx, j*Ly0g, (j+1)*Ly0g, c0, c1, c2);
                else
                    DebugDrawCubeInMATLAB(out2D, 0, Lx, j*Ly0g, (j+1)*Ly0g, k*Lz0g, (k+1)*Lz0g, c0, c1, c2);
                if (counter_2D==2)
                    out2D << "hold on;\n";
                
                for (int p = 0; p < n_f_2D; p++)
                {
                    int f_p = binned_face_ids_2D[L][N_f_2D+p];
                    ufloat_g_t vx1 = geom_f_face_X[f_p + 0*n_faces_a];
                    ufloat_g_t vy1 = geom_f_face_X[f_p + 1*n_faces_a];
                    ufloat_g_t vz1 = geom_f_face_X[f_p + 2*n_faces_a];
                    ufloat_g_t vx2 = geom_f_face_X[f_p + 3*n_faces_a];
                    ufloat_g_t vy2 = geom_f_face_X[f_p + 4*n_faces_a];
                    ufloat_g_t vz2 = geom_f_face_X[f_p + 5*n_faces_a];
                    ufloat_g_t vx3 = geom_f_face_X[f_p + 6*n_faces_a];
                    ufloat_g_t vy3 = geom_f_face_X[f_p + 7*n_faces_a];
                    ufloat_g_t vz3 = geom_f_face_X[f_p + 8*n_faces_a];
                    
                    // Draw the faces in the current bin.
                    if (N_DIM==2)
                        DebugDrawLineInMATLAB(out2D, vx1, vy1, vx2, vy2, c0, c1, c2);
                    else
                        DebugDrawTriangleInMATLAB(out2D, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, c0, c1, c2);
                }
            }
        }
    }
    
    // Close the file.
    std::cout << "Finished drawing bins..." << std::endl;
    out2D.close();
    out3D.close();
    
    return 0;
}
