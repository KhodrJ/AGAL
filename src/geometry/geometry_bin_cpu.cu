/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry_bins.h"

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_MakeBinsCPU ]============================================================  */
/*                                                                                    */
/*  Performs a uniform spatial binning of geometry faces inside of the domain in      */
/*  parallel on the CPU. Faces outside of the domain are filtered out. The result     */
/*  is the allocation of memory for and filling of three sets of arrays: 1)           */
/*  c_binned_ids_v/b, a set of contiguous binned faces such that the first batch      */
/*  correspond to the faces of bin 0, the second batch corresponds to bin 1 and so    */
/*  on, 2) c_binned_ids_n_v/b, the sizes of the n_bins_2D/b bins, and 3)               */
/*  c_binned_ids_N_v/b, the starting indices for the faces of each bin in             */
/*  c_binned_ids_v/b. The set of arrays with '_v' corresponds to a 2D binning which   */
/*  enables a raycast algorithm for solid-cell identification. The one with '_b'      */
/*  corresponds to the 3D binning, where the bins are extended in volume by an        */
/*  amount dx specified by the mesh resolution and which is used to restrict the      */
/*  search-space when cells are computing the lengths of cut-links across the         */
/*  geometry.                                                                         */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::Bins::G_MakeBinsCPU(int L)
{
    // Some constants.
    ufloat_g_t *geom_f_face_X = geometry->geom_f_face_X;
    int n_faces = geometry->n_faces;
    int n_faces_a = geometry->n_faces_a;
    ufloat_g_t Lx0g __attribute__((unused)) = Lx/static_cast<ufloat_g_t>(n_bin_density);
    ufloat_g_t Ly0g __attribute__((unused)) = Ly/static_cast<ufloat_g_t>(n_bin_density);
    ufloat_g_t Lz0g __attribute__((unused)) = Lz/static_cast<ufloat_g_t>(n_bin_density);
    ufloat_g_t eps __attribute__((unused)) = EPS<ufloat_g_t>();
    bool C2D = false; if (G_BIN_APPROACH==0) C2D = true;
    bool C3D = false; if (G_BIN_APPROACH==0) C3D = true;
    
    // Proceed only if there are actual faces loaded in the current object.
    tic_simple("");
    if (v_geom_f_face_1_X.size() > 0)
    {
        // Declare and allocate std::vector<int> bin arrays, which will be updated during traversal.
        n_bins_2D[L] = 1; for (int d = 0; d < N_DIM-1; d++) n_bins_2D[L] *= n_bin_density;
        n_bins_3D[L] = 1; for (int d = 0; d < N_DIM; d++)   n_bins_3D[L] *= n_bin_density;
        std::vector<int> *bins_a_2D = new std::vector<int>[n_bins_2D[L]];
        std::vector<int> *bins_a_3D = new std::vector<int>[n_bins_3D[L]];
        
        // Traverse faces and identify the bins they should go in.
        std::cout << "Starting CPU binning..." << std::endl;
        cudaDeviceSynchronize();
        for (int j = 0; j < n_faces_a; j++)
        {
            if (j < n_faces)
            {
                // Load face vertices from coordinate list.
                ufloat_g_t vx1 = geom_f_face_X[j + 0*n_faces_a];
                ufloat_g_t vy1 = geom_f_face_X[j + 1*n_faces_a];
                ufloat_g_t vz1 = geom_f_face_X[j + 2*n_faces_a];
                ufloat_g_t vx2 = geom_f_face_X[j + 3*n_faces_a];
                ufloat_g_t vy2 = geom_f_face_X[j + 4*n_faces_a];
                ufloat_g_t vz2 = geom_f_face_X[j + 5*n_faces_a];
                ufloat_g_t vx3 = geom_f_face_X[j + 6*n_faces_a];
                ufloat_g_t vy3 = geom_f_face_X[j + 7*n_faces_a];
                ufloat_g_t vz3 = geom_f_face_X[j + 8*n_faces_a];
                
                if (N_DIM==2)
                {
                    // Get the bounding box.
                    bool C = true;
                    ufloat_g_t vBx_m = std::min({vx1,vx2});
                    ufloat_g_t vBx_M = std::max({vx1,vx2});
                    ufloat_g_t vBy_m = std::min({vy1,vy2});
                    ufloat_g_t vBy_M = std::max({vy1,vy2});
                    if ((vBx_m<-dx&&vBx_M<-dx) || (vBx_m>Lx+dx&&vBx_M>Lx+dx))
                        C = false;
                    if ((vBy_m<-dx&&vBy_M<-dx) || (vBy_m>Ly+dx&&vBy_M>Ly+dx))
                        C = false;
                    
                    // Identify the bin indices of the lower and upper bounds.
                    int bin_id_xl = std::max((int)(vBx_m*n_bin_density)-1, 0);
                    int bin_id_yl = std::max((int)(vBy_m*n_bin_density)-1, 0);
                    int bin_id_xL = std::min((int)(vBx_M*n_bin_density)+2, n_bin_density);
                    int bin_id_yL = std::min((int)(vBy_M*n_bin_density)+2, n_bin_density);
                    
                    // Traverse bin indices and add this face to the corresponding vectors.
                    if (C)
                    {
                        for (int J = bin_id_yl; J < bin_id_yL; J++)
                        {
                            for (int I = bin_id_xl; I < bin_id_xL; I++)
                            {
                                if (G_BIN_APPROACH==1)
                                    C3D = IncludeInBin<ufloat_g_t,2>(I*Lx0g-dx,(I+1)*Lx0g+dx,J*Ly0g-dx,(J+1)*Ly0g+dx,0,0,vBx_m,vBx_M,vBy_m,vBy_M,0,0,vx1,vy1,0,vx2,vy2,0,0,0,0);
                                
                                if (C3D)
                                    bins_a_3D[I+G_BIN_DENSITY*J].push_back(j);
                            }
                            
                            if (G_BIN_APPROACH==1)
                                C2D = IncludeInBin<ufloat_g_t,2>(-eps,Lx+eps,J*Ly0g-eps,(J+1)*Ly0g+eps,0,0,vBx_m,vBx_M,vBy_m,vBy_M,0,0,vx1,vy1,0,vx2,vy2,0,0,0,0);
                            
                            if (C2D)
                                bins_a_2D[J].push_back(j);
                        }
                    }
                }
                else // N_DIM==3
                {
                    // Get bounding box (safe version)
                    bool C = true;
                    ufloat_g_t vBx_m = std::min(std::min(vx1, vx2), vx3);
                    ufloat_g_t vBx_M = std::max(std::max(vx1, vx2), vx3);
                    ufloat_g_t vBy_m = std::min(std::min(vy1, vy2), vy3);
                    ufloat_g_t vBy_M = std::max(std::max(vy1, vy2), vy3);
                    ufloat_g_t vBz_m = std::min(std::min(vz1, vz2), vz3);
                    ufloat_g_t vBz_M = std::max(std::max(vz1, vz2), vz3);
                    if ((vBx_m<-dx&&vBx_M<-dx) || (vBx_m>Lx+dx&&vBx_M>Lx+dx))
                        C = false;
                    if ((vBy_m<-dx&&vBy_M<-dx) || (vBy_m>Ly+dx&&vBy_M>Ly+dx))
                        C = false;
                    if ((vBz_m<-dx&&vBz_M<-dx) || (vBz_m>Lz+dx&&vBz_M>Lz+dx))
                        C = false;
                    
                    // Identify the bin indices of the lower and upper bounds.
                    int bin_id_xl = std::max((int)(vBx_m*G_BIN_DENSITY)-1, 0);
                    int bin_id_yl = std::max((int)(vBy_m*G_BIN_DENSITY)-1, 0);
                    int bin_id_zl = std::max((int)(vBz_m*G_BIN_DENSITY)-1, 0);
                    int bin_id_xL = std::min((int)(vBx_M*G_BIN_DENSITY)+2, G_BIN_DENSITY);
                    int bin_id_yL = std::min((int)(vBy_M*G_BIN_DENSITY)+2, G_BIN_DENSITY);
                    int bin_id_zL = std::min((int)(vBz_M*G_BIN_DENSITY)+2, G_BIN_DENSITY);
                    
                    // Traverse bin indices and add this face to the corresponding vectors.
                    if (C)
                    {
                        for (int K = bin_id_zl; K < bin_id_zL; K++)
                        {
                            for (int J = bin_id_yl; J < bin_id_yL; J++)
                            {
                                for (int I = bin_id_xl; I < bin_id_xL; I++)
                                {
                                    if (G_BIN_APPROACH==1)
                                        C3D = IncludeInBin<ufloat_g_t,3>(I*Lx0g-dx,(I+1)*Lx0g+dx,J*Ly0g-dx,(J+1)*Ly0g+dx,K*Lz0g-dx,(K+1)*Lz0g+dx,vBx_m,vBx_M,vBy_m,vBy_M,vBz_m,vBz_M,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3);
                                    
                                    if (C3D)
                                        bins_a_3D[I+G_BIN_DENSITY*J+G_BIN_DENSITY*G_BIN_DENSITY*K].push_back(j);
                                }
                                
                                if (G_BIN_APPROACH==1)
                                    C2D = IncludeInBin<ufloat_g_t,3>(-eps,Lx+eps,J*Ly0g-eps,(J+1)*Ly0g+eps,K*Lz0g-eps,(K+1)*Lz0g+eps,vBx_m,vBx_M,vBy_m,vBy_M,vBz_m,vBz_M,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3);
                                    
                                if (C2D)
                                    bins_a_2D[J+G_BIN_DENSITY*K].push_back(j);
                            }
                        }
                    }
                }
            }
            
        }
        cudaDeviceSynchronize();
        std::cout << "Elapsed time (CPU): " << toc_simple("",T_US) << std::endl;
        
        
        // Insert binned faces into GPU memory.
        std::vector<int> bins_n_2D;
        std::vector<int> bins_N_2D;
        std::vector<int> bins_f_2D;
        std::vector<int> bins_n_3D;
        std::vector<int> bins_N_3D;
        std::vector<int> bins_f_3D;
        int Npv = 0;
        int Npb = 0;
        const int PADDING = 4;
        for (int p = 0; p < n_bins_2D[L]; p++)
        {
            int npv = bins_a_2D[p].size();
            bins_n_2D.push_back(npv);
            bins_N_2D.push_back(Npv);
            if (npv > 0)
            {
                for (int k = 0; k < npv; k++)
                    bins_f_2D.push_back(bins_a_2D[p][k]);
                
                int rem = PADDING-npv%PADDING;
                for (int k = 0; k < rem; k++)
                    bins_f_2D.push_back(-1);
                
                Npv += npv + rem;
            }
        }
        for (int p = 0; p < n_bins_3D[L]; p++)
        {
            int npb = bins_a_3D[p].size();
            bins_n_3D.push_back(npb);
            bins_N_3D.push_back(Npb);
            if (npb > 0)
            {
                for (int k = 0; k < npb; k++)
                    bins_f_3D.push_back(bins_a_3D[p][k]);
                
                int rem = PADDING-npb%PADDING;
                for (int k = 0; k < rem; k++)
                    bins_f_3D.push_back(-1);
                
                Npb += npb + rem;
            }
        }
        
        // Now copy final vector data to the GPU.
        //
        // 2D
        binned_face_ids_n_2D[L] = new int[n_bins_2D[L]];
        binned_face_ids_N_2D[L] = new int[n_bins_2D[L]];
        binned_face_ids_2D[L] = new int[bins_f_2D.size()];
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_2D[L], n_bins_2D[L]*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_2D[L], n_bins_2D[L]*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_2D[L], bins_f_2D.size()*sizeof(int)) );
        for (int p = 0; p < n_bins_2D[L]; p++)
        {
            binned_face_ids_n_2D[L][p] = bins_n_2D[p];
            binned_face_ids_N_2D[L][p] = bins_N_2D[p];
        }
        for (int p = 0; p < bins_f_2D.size(); p++)
        {
            binned_face_ids_2D[L][p] = bins_f_2D[p];
        }
        gpuErrchk( cudaMemcpy(c_binned_face_ids_n_2D[L], binned_face_ids_n_2D[L], n_bins_2D[L]*sizeof(int), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(c_binned_face_ids_N_2D[L], binned_face_ids_N_2D[L], n_bins_2D[L]*sizeof(int), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(c_binned_face_ids_2D[L], binned_face_ids_2D[L], bins_f_2D.size()*sizeof(int), cudaMemcpyHostToDevice) );
        //
        // 3D
        binned_face_ids_n_3D[L] = new int[n_bins_3D[L]];
        binned_face_ids_N_3D[L] = new int[n_bins_3D[L]];
        binned_face_ids_3D[L] = new int[bins_f_3D.size()];
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_3D[L], n_bins_3D[L]*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_3D[L], n_bins_3D[L]*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_3D[L], bins_f_3D.size()*sizeof(int)) );
        for (int p = 0; p < n_bins_3D[L]; p++)
        {
            binned_face_ids_n_3D[L][p] = bins_n_3D[p];
            binned_face_ids_N_3D[L][p] = bins_N_3D[p];
        }
        for (int p = 0; p < bins_f_3D.size(); p++)
        {
            binned_face_ids_3D[L][p] = bins_f_3D[p];
        }
        gpuErrchk( cudaMemcpy(c_binned_face_ids_n_3D[L], binned_face_ids_n_3D[L], n_bins_3D[L]*sizeof(int), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(c_binned_face_ids_N_3D[L], binned_face_ids_N_3D[L], n_bins_3D[L]*sizeof(int), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(c_binned_face_ids_3D[L], binned_face_ids_3D[L], bins_f_3D.size()*sizeof(int), cudaMemcpyHostToDevice) );
        
        
        // DEBUG (2D)
//         std::cout << "Finished CPU binning, starting debugging..." << std::endl;
//         std::cout << "APPROACH: ALT 2D" << std::endl;
//         for (int p = 0; p < n_bins_2D; p++)
//         {
//             int Nbpv = binned_face_ids_N_2D[p];
//             int npbv = binned_face_ids_n_2D[p];
//             int npb = bins_a_2D[p].size();
//             if (npb > 0)
//             {
//                 std::cout << "Bin #" << p << ": ";
//                 bool same = true;
//                 
//                 if (npb != npbv)
//                     same = false;
//                 else
//                 {
//                     for (int K = 0; K < npb; K++)
//                     {
//                         if (bins_a_2D[p][K] != binned_face_ids_2D[Nbpv + K])
//                             same = false;
//                     }
//                 }
//                 if (same)
//                     std::cout << "SAME" << std::endl;
//                 else
//                     std::cout << "NOT THE SAME (" << npb-npbv << ")" << std::endl;
//             }
//         }
        // DEBUG (3D)
//         std::cout << "APPROACH: ALT 3D" << std::endl;
//         for (int p = 0; p < n_bins_3D; p++)
//         {
//             int Nbpv = binned_face_ids_N_3D[p];
//             int npbv = binned_face_ids_n_3D[p];
//             int npb = bins_a_3D[p].size();
//             if (npb > 0)
//             {
//                 std::cout << "Bin #" << p << ": ";
//                 bool same = true;
//                 
//                 if (npb != npbv)
//                     same = false;
//                 else
//                 {
//                     for (int K = 0; K < npb; K++)
//                     {
//                         if (bins_a_3D[p][K] != binned_face_ids_3D[Nbpv + K])
//                             same = false;
//                     }
//                 }
//                 if (same)
//                     std::cout << "SAME" << std::endl;
//                 else
//                     std::cout << "NOT THE SAME (" << npb-npbv << ")" << std::endl;
//             }
//         }
        
        
        // Free memory used for CPU-side bin arrays.
        delete[] bins_a_2D;
        delete[] bins_a_3D;
    }
    else
    {
        std::cout << "ERROR: Could not make bins...there are no faces..." << std::endl;
    }
    
    return 0;
}
