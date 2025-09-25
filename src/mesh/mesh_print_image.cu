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
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Print_ImageData(int i_dev, int iter)
{
    // Allocate memory for a cell-block's macroscopic properties.
    double u_kap[M_TBLOCK*(7+1)];
    for (int i = 0; i < M_TBLOCK*(7+1); i++)
        u_kap[i] = 0.0;
    
    // Loop over levels and make a uniform grid for each of them.
    for (int L = 0; L < N_PRINT_LEVELS_IMAGE; L++)
    {
        if (n_ids[i_dev][L] > 0)
        {
            // Declare uniform grid.
            vtkNew<vtkUniformGrid> grid_L;
            
            // Adjust resolution.
            int n_cells_L = 1;
            int Nxi_L[3] = {1,1,1};
            int mult_Nx = static_cast<int>( pow(2.0, static_cast<double>(L)) );
            for (int d = 0; d < N_DIM; d++)
            {
                Nxi_L[d] = Nxi[d]*mult_Nx;
                n_cells_L *= Nxi_L[d];
            }
            
            // Define attributes.
            int n_dim_lattice[3] = {Nxi_L[0]+1, Nxi_L[1]+1, (N_DIM==2) ? 1:(Nxi_L[2]+1)};
            double origin_kap[3] = {0,0,0};
            double dxf_L = static_cast<double>(dxf_vec[L]);
            double dxfo2_L = 0.5*static_cast<double>(dxf_vec[L]);
            double h_L_kap[3] = {dxf_L, dxf_L, (N_DIM==2) ? static_cast<double>(dx):dxf_L};
            
            // Initialize the uniform grid.
            grid_L->Initialize();
            grid_L->SetOrigin(origin_kap);
            grid_L->SetSpacing(h_L_kap);
            grid_L->SetDimensions(n_dim_lattice);
            
            // Data arrays.
            //
            //     Cell mask.
            vtkNew<vtkDoubleArray> data_kap_mask_cell;
            data_kap_mask_cell->SetName("Cell Mask");
            data_kap_mask_cell->SetNumberOfComponents(1);
            data_kap_mask_cell->SetNumberOfTuples(n_cells_L);
            //
            //     Block mask.
            vtkNew<vtkDoubleArray> data_kap_mask_cblock;
            data_kap_mask_cblock->SetName("Block Mask");
            data_kap_mask_cblock->SetNumberOfComponents(1);
            data_kap_mask_cblock->SetNumberOfTuples(n_cells_L);
            //
            //     AMR Id.
            vtkNew<vtkDoubleArray> data_kap_mask_amr_id;
            data_kap_mask_amr_id->SetName("AMR Id");
            data_kap_mask_amr_id->SetNumberOfComponents(1);
            data_kap_mask_amr_id->SetNumberOfTuples(n_cells_L);
            //
            // Density.
            vtkNew<vtkDoubleArray> data_kap_sc;
            data_kap_sc->SetName("Density");
            data_kap_sc->SetNumberOfComponents(1);
            data_kap_sc->SetNumberOfTuples(n_cells_L);
            //
            // Velocity.
            vtkNew<vtkDoubleArray> data_kap_v;
            data_kap_v->SetName("Velocity");
            data_kap_v->SetNumberOfComponents(3);
            data_kap_v->SetNumberOfTuples(n_cells_L);
            //
            // Vorticity.
            vtkNew<vtkDoubleArray> data_kap_w;
            data_kap_w->SetName("Vorticity");
            data_kap_w->SetNumberOfComponents(3);
            data_kap_w->SetNumberOfTuples(n_cells_L);
            //
            //     Reset.
            for (int k = 0; k < n_cells_L; k++)
            {
                data_kap_mask_cell->SetTuple1(k, 0.0);
                data_kap_mask_cblock->SetTuple1(k, 0.0);
                data_kap_mask_amr_id->SetTuple1(k, 0.0);
            }
            
            // Loop over blocks on this grid level and fill the data arrays.
            for (int kap = 0; kap < n_ids[i_dev][L]; kap++)
            {
                int i_kap_b = id_set[i_dev][L*n_maxcblocks + kap];
                
                // Reset intermediate computation array.
                for (int i = 0; i < M_TBLOCK*(7+1); i++)
                    u_kap[i] = 0.0;
                M_ComputeOutputProperties(i_dev, 0, i_kap_b, dxf_vec[L], u_kap);
                
                for (int t = 0; t < M_CBLOCK; t++)
                {
                    // Get cell's local indices.
                    int I = t%4;
                    int J = (t/4)%4;
                    int K = 0;
                    if (N_DIM==3)
                        K = (t/4)/4;
                    
                    // Get cell's global coordinates.
                    ufloat_t x = cblock_f_X[i_dev][i_kap_b + 0*n_maxcblocks] + dxf_vec[L]*I + dxfo2_L;
                    ufloat_t y = cblock_f_X[i_dev][i_kap_b + 1*n_maxcblocks] + dxf_vec[L]*J + dxfo2_L;
                    ufloat_t z = static_cast<ufloat_t>(0.0);
                    if (N_DIM==3)
                        z = cblock_f_X[i_dev][i_kap_b + 2*n_maxcblocks] + dxf_vec[L]*K + dxfo2_L;
                    
                    // Get cell's global index.
                    int ix = static_cast<int>(x * static_cast<ufloat_t>(Nxi_L[0]));
                    int iy = static_cast<int>(y * static_cast<ufloat_t>(Nxi_L[1]));
                    int iz = 0;
                    if (N_DIM==3)
                        iz = static_cast<int>(z * static_cast<ufloat_t>(Nxi_L[2]));
                    int i_global = ix + Nxi_L[0]*iy + Nxi_L[0]*Nxi_L[1]*iz;
                    
                    // Cell mask.
                    data_kap_mask_cell->SetTuple1(i_global, static_cast<double>(cells_ID_mask[i_dev][i_kap_b*M_CBLOCK + t]));
                    
                    // Block mask.
                    data_kap_mask_cblock->SetTuple1(i_global, static_cast<double>(cblock_ID_mask[i_dev][i_kap_b]));
                    
                    // AMR Id.
                    data_kap_mask_amr_id->SetTuple1(i_global, static_cast<double>(i_kap_b));
                    
                    // Density.
                    data_kap_sc->SetTuple1(i_global,
                        u_kap[t + 0*M_TBLOCK]
                    );
                    
                    // Velocity.
                    data_kap_v->SetTuple3(i_global,
                        u_kap[t + 1*M_TBLOCK],
                        u_kap[t + 2*M_TBLOCK],
                        u_kap[t + 3*M_TBLOCK]
                    );
                    
                    // Vorticity.
                    data_kap_w->SetTuple3(i_global, 
                        u_kap[t + 4*M_TBLOCK],
                        u_kap[t + 5*M_TBLOCK],
                        u_kap[t + 6*M_TBLOCK]
                    );
                }
            }
            
            // Attach data arrays to uniform grid.
            grid_L->GetCellData()->AddArray(data_kap_mask_cell);
            grid_L->GetCellData()->AddArray(data_kap_mask_cblock);
            grid_L->GetCellData()->AddArray(data_kap_mask_amr_id);
            grid_L->GetCellData()->AddArray(data_kap_sc);
            grid_L->GetCellData()->AddArray(data_kap_v);
            grid_L->GetCellData()->AddArray(data_kap_w);
            
            // Print the uniform grid.
            std::cout << "Finished building VTK dataset, writing (L = " << L << ")..." << std::endl;
            std::string fileName = output_dir + std::string("out_") + std::to_string(iter+1) + "_" + std::to_string(L) + ".vti";
            vtkNew<vtkXMLImageDataWriter> writer;
            writer->SetInputData(grid_L);
            writer->SetFileName(fileName.c_str());
            writer->SetDataModeToBinary();
            writer->Write();
            std::cout << "Finished writing VTK dataset (" << L << ")..." << std::endl;
        }
    }
    
    return 0;
}


