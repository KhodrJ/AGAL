/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

/*
         .d88888b.           888                      888             
        d88P" "Y88b          888                      888             
        888     888          888                      888             
        888     888 888  888 888888 88888b.  888  888 888888          
        888     888 888  888 888    888 "88b 888  888 888             
        888     888 888  888 888    888  888 888  888 888             
        Y88b. .d88P Y88b 888 Y88b.  888 d88P Y88b 888 Y88b.           
88888888 "Y88888P"   "Y88888  "Y888 88888P"   "Y88888  "Y888 88888888 
                                    888                               
                                    888                               
                                    888                               
*/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Print_VTHB_Patch(int i_dev, int iter)
{
    const int M_PATCH = Nxi[0];
    int M_PATCH_VOL = 1;
    for (int d = 0; d < N_DIM; d++)
        M_PATCH_VOL *= M_PATCH;
    
    
    // New variables.
    vtkNew<vtkOverlappingAMR> data;
    int patches_per_level[N_PRINT_LEVELS_PATCH];
    double global_origin[3] = {0,0,0};
    
    // Get the number of levels in which there are a non-zero number of blocks.
    // Also, compute the number of patches on each level.
    int n_levels_nonzero_patches = 0;
    patches_per_level[0] = n_ids[i_dev][0];
    std::unordered_map<int, std::vector<int>> patch2patch[N_PRINT_LEVELS_PATCH];
    for (int L = 0; L < N_PRINT_LEVELS_PATCH; L++)
    {
        // If there are blocks on this level, add to level counter.
        if (n_ids[i_dev][L] > 0)
            n_levels_nonzero_patches++;
        
        // Precompute adjusted resolutions.
        int n_cells_L = 1;
        int Nxi_L[3] = {1,1,1};   // Number of cells in a dense representation of level L.
        int Nxp_L[3] = {1,1,1};   // Number of patches in a dense representation of level L.
        int mult_Nx = static_cast<int>( pow(2.0, static_cast<double>(L)) );
        for (int d = 0; d < N_DIM; d++)
        {
            Nxi_L[d] = Nxi[d]*mult_Nx;
            Nxp_L[d] = Nxi_L[d]/M_PATCH;
            n_cells_L *= Nxi_L[d];
        }
        
        // Loop over blocks on this level and add to patches.
        for (int kap = 0; kap < n_ids[i_dev][L]; kap++)
        {
            // ID of the kap'th cblock.
            int i_kap_b = id_set[i_dev][L*n_maxcblocks + kap];
            
            // Load block coordinates.
            ufloat_t x = cblock_f_X[i_dev][i_kap_b + 0*n_maxcblocks] + static_cast<ufloat_t>(0.5)*dxf_vec[L];
            ufloat_t y = cblock_f_X[i_dev][i_kap_b + 1*n_maxcblocks] + static_cast<ufloat_t>(0.5)*dxf_vec[L];
            ufloat_t z = static_cast<ufloat_t>(0.0);
            if (N_DIM==3)
                z = cblock_f_X[i_dev][i_kap_b + 2*n_maxcblocks] + static_cast<ufloat_t>(0.5)*dxf_vec[L];
            
            // Compute the global patch indices.
            int i_patch_x = static_cast<int>(x * Nxp_L[0]);
            int i_patch_y = static_cast<int>(y * Nxp_L[1]);
            int i_patch_z = 0;
            if (N_DIM==3)
                i_patch_z = static_cast<int>(z * Nxp_L[2]);
            int i_patch_g = i_patch_x + Nxp_L[0]*i_patch_y + Nxp_L[0]*Nxp_L[1]*i_patch_z;
            
            // Add this block to the appropriate patch by mapping the patch global index.
            patch2patch[L][i_patch_g].push_back(kap);
        }
        
        // Adjust the number of patches on this level.
        patches_per_level[L] = patch2patch[L].size();
    }
    
    
    // Initialize AMR object.
    data->Initialize(n_levels_nonzero_patches, patches_per_level);
    data->SetOrigin(global_origin);
    data->SetGridDescription(N_DIM==2?VTK_XY_PLANE:VTK_XYZ_GRID);
    
    
    // Parameters.
    int n_dim_lattice[3] = {M_PATCH+1, M_PATCH+1, (N_DIM==2) ? 1:(M_PATCH+1)};
    double origin_kap[3] = {0,0,0};
    double u_kap[M_TBLOCK*(7+1)];
    for (int i = 0; i < M_TBLOCK*(7+1); i++)
        u_kap[i] = 0.0;
    
    
    // For each level, insert all existing blocks.
    for (int L = 0; L < std::min(n_levels_nonzero_patches, N_PRINT_LEVELS_PATCH); L++)
    {
        // Precompute adjusted resolutions (again; yes, I'm lazy).
        std::cout << "Filling in level " << L << "..." << std::endl;
        int n_cells_L = 1;
        int Nxi_L[3] = {1,1,1};
        int Nxp_L[3] = {1,1,1};
        int mult_Nx = static_cast<int>( pow(2.0, static_cast<double>(L)) );
        for (int d = 0; d < N_DIM; d++)
        {
            Nxi_L[d] = Nxi[d]*mult_Nx;
            Nxp_L[d] = Nxi_L[d]/M_PATCH;
            n_cells_L *= Nxi_L[d];
        }
        
        // Construct spacing array for level L.
        double dxf_L = (double)dxf_vec[L];
        double h_L_kap[3] = {dxf_L, dxf_L, N_DIM==2?(double)dx:dxf_L};
        
        // Only insert spacing array if there are actual blocks to insert on this level.
        if (n_ids[i_dev][L] > 0)
            data->SetSpacing(L, h_L_kap);
        
        // Loop over patches and add the block data in a nested loop.
        int counter = 0;
        for (const auto &pkap: patch2patch[L])
        {
            // pkap.first is the global index.
            // pkap.second is the vector of block Ids in that patch.
            
            // Get global coordinates based on global index.
            int i_patch_x = pkap.first % Nxp_L[0];
            int i_patch_y = (pkap.first / Nxp_L[1]) % Nxp_L[0];
            int i_patch_z = 0;
            if (N_DIM==3)
                i_patch_z = (pkap.first / Nxp_L[1]) / Nxp_L[0];
            
            // Adjust origin of current patch (based on lower-left corner per the global index).
            origin_kap[0] = static_cast<double>(i_patch_x) * static_cast<double>(M_PATCH)*dxf_vec[L];
            origin_kap[1] = static_cast<double>(i_patch_y) * static_cast<double>(M_PATCH)*dxf_vec[L];
            origin_kap[2] = static_cast<double>(i_patch_z) * static_cast<double>(M_PATCH)*dxf_vec[L];
            
            // Initialize uniform grid defining the patch.
            vtkNew<vtkUniformGrid> grid_kap;
            grid_kap->Initialize();
            grid_kap->SetOrigin(origin_kap);
            grid_kap->SetSpacing(h_L_kap);
            grid_kap->SetDimensions(n_dim_lattice);
            
            // Define VTK arrays.
                // Debug.
            vtkNew<vtkDoubleArray> data_kap_ref;
            data_kap_ref->SetName("Ref ID");
            data_kap_ref->SetNumberOfComponents(1);
            data_kap_ref->SetNumberOfTuples(M_PATCH_VOL);
                // Debug.
            vtkNew<vtkDoubleArray> data_kap_onb;
            data_kap_onb->SetName("On Boundary?");
            data_kap_onb->SetNumberOfComponents(1);
            data_kap_onb->SetNumberOfTuples(M_PATCH_VOL);
                // Cell mask.
            vtkNew<vtkDoubleArray> data_kap_mask_cell;
            data_kap_mask_cell->SetName("Cell Mask");
            data_kap_mask_cell->SetNumberOfComponents(1);
            data_kap_mask_cell->SetNumberOfTuples(M_PATCH_VOL);
                // Block mask.
            vtkNew<vtkDoubleArray> data_kap_mask_cblock;
            data_kap_mask_cblock->SetName("Block Mask");
            data_kap_mask_cblock->SetNumberOfComponents(2);
            data_kap_mask_cblock->SetNumberOfTuples(M_PATCH_VOL);
                // Density.
            vtkNew<vtkDoubleArray> data_kap_sc;
            data_kap_sc->SetName("Density");
            data_kap_sc->SetNumberOfComponents(1);
            data_kap_sc->SetNumberOfTuples(M_PATCH_VOL);
                // Velocity.
            vtkNew<vtkDoubleArray> data_kap_v;
            data_kap_v->SetName("Velocity");
            data_kap_v->SetNumberOfComponents(3);
            data_kap_v->SetNumberOfTuples(M_PATCH_VOL);
                // Vorticity.
            vtkNew<vtkDoubleArray> data_kap_w;
            data_kap_w->SetName("Vorticity");
            data_kap_w->SetNumberOfComponents(3);
            data_kap_w->SetNumberOfTuples(M_PATCH_VOL);
                // Q-Criterion.
            vtkNew<vtkDoubleArray> data_kap_Q;
            data_kap_Q->SetName("Q-Criterion");
            data_kap_Q->SetNumberOfComponents(1);
            data_kap_Q->SetNumberOfTuples(M_PATCH_VOL);
            
            // By default, blank all cells.
            for (int k = 0; k < M_PATCH_VOL; k++)
               grid_kap->BlankCell(k);
            
            // Now fill the patch with block data.
            for (const int &kap: pkap.second)
            {
                // kap is the block Id in the Id set on this level.
                
                // ID of the kap'th cblock.
                int i_kap_b = id_set[i_dev][L*n_maxcblocks + kap];
                
                // Load block coordinates (again).
                ufloat_t x = cblock_f_X[i_dev][i_kap_b + 0*n_maxcblocks] + static_cast<ufloat_t>(0.5)*dxf_vec[L];
                ufloat_t y = cblock_f_X[i_dev][i_kap_b + 1*n_maxcblocks] + static_cast<ufloat_t>(0.5)*dxf_vec[L];
                ufloat_t z = static_cast<ufloat_t>(0.0);
                if (N_DIM==3)
                    z = cblock_f_X[i_dev][i_kap_b + 2*n_maxcblocks] + static_cast<ufloat_t>(0.5)*dxf_vec[L];
                
                // Reset intermediate computation array.
                for (int i = 0; i < M_TBLOCK*(7+1); i++)
                    u_kap[i] = 0.0;
                M_ComputeOutputProperties(i_dev, 0, i_kap_b, dxf_vec[L], u_kap);
                
                // Insert quantities in arrays.
                for (int t = 0; t < M_TBLOCK; t++)
                {
                    // Get cell's local indices.
                    int I = t%4;
                    int J = (t/4)%4;
                    int K = 0;
                    if (N_DIM==3)
                        K = (t/4)/4;
                    
                    // Get the celll's coordinates.
                    double x_t = static_cast<double>(x) + static_cast<double>(I)*dxf_vec[L];
                    double y_t = static_cast<double>(y) + static_cast<double>(J)*dxf_vec[L];
                    double z_t = static_cast<double>(0.0);
                    if (N_DIM==3)
                        z_t = static_cast<double>(z) + static_cast<double>(K)*dxf_vec[L];
                    
                    // Compute the indices of its cells within the patch.
                    int ix = static_cast<int>(x_t * Nxi_L[0]) % M_PATCH;
                    int iy = static_cast<int>(y_t * Nxi_L[1]) % M_PATCH;
                    int iz = static_cast<int>(z_t * Nxi_L[2]) % M_PATCH;
                    int i_global = ix + M_PATCH*iy + M_PATCH*M_PATCH*iz;
                    grid_kap->UnBlankCell(i_global);
                    
                    // Debug.
                    data_kap_ref->SetTuple1(i_global, (double)i_kap_b); //(double)cblock_ID_ref[i_dev][i_kap_b]);
                    
                    // On boundary?
                    data_kap_onb->SetTuple1(i_global, (double)cblock_ID_onb[i_dev][i_kap_b]);
                    
                    // Cell mask.
                    data_kap_mask_cell->SetTuple1(i_global, (double)cells_ID_mask[i_dev][i_kap_b*M_CBLOCK + t]);
                    
                    // Block mask.
                    data_kap_mask_cblock->SetTuple2(i_global, (double)cblock_ID_mask[i_dev][i_kap_b], (double)cblock_ID_mask[i_dev][i_kap_b + 1*n_maxcblocks]);
                    
                    // Density.
                    data_kap_sc->SetTuple1(i_global,
                        u_kap[t + 0*M_TBLOCK]
                    );
                    
                    // Velocity.
                    if (L > 0 || (L == 0 && N_PROBE_AVE == 0))
                    {
                        data_kap_v->SetTuple3(i_global,
                            u_kap[t + 1*M_TBLOCK],
                            u_kap[t + 2*M_TBLOCK],
                            u_kap[t + 3*M_TBLOCK]
                        );
                    }
                    else
                    {
                        data_kap_v->SetTuple3(i_global,
                            (double)cells_f_U_mean[i_dev][i_kap_b*M_CBLOCK + t + 0*n_ids[i_dev][0]*M_CBLOCK],
                            (double)cells_f_U_mean[i_dev][i_kap_b*M_CBLOCK + t + 1*n_ids[i_dev][0]*M_CBLOCK],
                            (double)cells_f_U_mean[i_dev][i_kap_b*M_CBLOCK + t + 2*n_ids[i_dev][0]*M_CBLOCK]
                        );
                    }
                    
                    // Vorticity.
                    data_kap_w->SetTuple3(i_global, 
                        u_kap[t + 4*M_TBLOCK],
                        u_kap[t + 5*M_TBLOCK],
                        u_kap[t + 6*M_TBLOCK]
                    );
                    
                    // Q-Criterion.
                    data_kap_Q->SetTuple1(i_global,
                        u_kap[t + 7*M_TBLOCK]
                    );
                }
            }
            
            // Add the data arrays to the uniform grid.
            grid_kap->GetCellData()->AddArray(data_kap_ref);
            grid_kap->GetCellData()->AddArray(data_kap_onb);
            grid_kap->GetCellData()->AddArray(data_kap_mask_cell);
            grid_kap->GetCellData()->AddArray(data_kap_mask_cblock);
            grid_kap->GetCellData()->AddArray(data_kap_sc);
            grid_kap->GetCellData()->AddArray(data_kap_v);
            grid_kap->GetCellData()->AddArray(data_kap_w);
            grid_kap->GetCellData()->AddArray(data_kap_Q);
            
            // Create the vtkAMRBox and insert it into AMR object.
            vtkAMRBox box_kap(origin_kap, n_dim_lattice, h_L_kap, data->GetOrigin(), data->GetGridDescription());
            data->SetAMRBox(L, counter, box_kap);
            data->SetDataSet(L, counter, grid_kap);
            counter++;
        }
    }
    
    
    // Write the AMR object.
    std::cout << "Finished building VTK dataset, writing..." << std::endl;
    std::string fileName = output_dir + std::string("out_") + std::to_string(iter+1) + ".vthb";
    vtkNew<vtkXMLUniformGridAMRWriter> writer;
    writer->SetInputData(data);
    writer->SetFileName(fileName.c_str());
    writer->Write();
    std::cout << "Finished writing VTK dataset..." << std::endl;
    
    return 0;
}
