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

int Mesh::M_Print_VTHB(int i_dev, int iter)
{
	// New variables.
	vtkNew<vtkOverlappingAMR> data;
	int blocks_per_level[N_PRINT_LEVELS];
	for (int L = 0; L < N_PRINT_LEVELS; L++)
		blocks_per_level[L] = n_ids[i_dev][L];
	double global_origin[3] = {0,0,0};
	
	// Get the number of levels in which there are a non-zero number of blocks.
	int n_levels_nonzero_blocks = 1;
#if (MAX_LEVELS>1)
	for (int L = 1; L < N_PRINT_LEVELS; L++)
	{
		if (n_ids[i_dev][L] > 0)
			n_levels_nonzero_blocks++;
	}
#endif
	
	// Initialize AMR object.
	data->Initialize(n_levels_nonzero_blocks, blocks_per_level);
	data->SetOrigin(global_origin);
	data->SetGridDescription(N_DIM==2?VTK_XY_PLANE:VTK_XYZ_GRID);
	
	// Parameters.
	//int n_dim_box[3] = {Nbx, Nbx, N_DIM==2?1:Nbx};
	int n_dim_lattice[3] = {Nbx+1, Nbx+1, N_DIM==2?1:Nbx+1};
	double origin_kap[3] = {0,0,0};
	double u_kap[M_CBLOCK*(6+1)]; for (int i = 0; i < M_CBLOCK*(6+1); i++) u_kap[i] = 0.0; 
	double yplus_kap[M_CBLOCK]; for (int i = 0; i < M_CBLOCK; i++) yplus_kap[i] = 0.0;
	
	// For each level, insert all existing blocks.
	for (int L = 0; L < std::min(n_levels_nonzero_blocks, N_PRINT_LEVELS); L++)
	{
		// Construct spacing array for level L.
		double dxf_L = (double)dxf_vec[L];
		double h_L_kap[3] = {dxf_L, dxf_L, N_DIM==2?(double)dx:dxf_L};
		
		// Only insert spacing array if there are actual blocks to insert on this level.
		if (n_ids[i_dev][L] > 0)
			data->SetSpacing(L, h_L_kap);
		
		// For all blocks on level L, build vtkAMRBox and insert in AMR object.
		int kap_counter = 0;
		for (int kap = 0; kap < n_ids[i_dev][L]; kap++)
		{
			// ID of the kap'th cblock.
			int i_kap = id_set[i_dev][L*n_maxcblocks + kap];
			
			//if (cblock_ID_ref[i_dev][i_kap] == V_REF_ID_UNREFINED)
			{
				// Set origin of block (based on lower-left corner stored in cblock_f_X).
				for (int d = 0; d < N_DIM; d++)
					origin_kap[d] = cblock_f_X[i_dev][i_kap + d*n_maxcblocks];
				
				// Initialize uniform grid defining the block.
				vtkNew<vtkUniformGrid> grid_kap;
				grid_kap->Initialize();
				grid_kap->SetOrigin(origin_kap);
				grid_kap->SetSpacing(h_L_kap);
				grid_kap->SetDimensions(n_dim_lattice);
				
				// Fill data in the uniform grid defining the block.
					// Debug.
				vtkNew<vtkDoubleArray> data_kap_dbg;
				data_kap_dbg->SetName("debug - ref ID");
				data_kap_dbg->SetNumberOfComponents(1);
				data_kap_dbg->SetNumberOfTuples(M_CBLOCK);
				int min_nbr_id = 0;
				#pragma unroll
				for (int p = 0; p < N_Q_max; p++)
				{
					int nbr_id_p = cblock_ID_nbr[i_dev][i_kap + p*n_maxcblocks];
					//if (nbr_id_p == N_SKIPID)
					//	nbr_id_p = -1;
					if (nbr_id_p < 0 && min_nbr_id == 0)
						min_nbr_id = nbr_id_p;
					if (nbr_id_p > min_nbr_id && nbr_id_p < 0)
						min_nbr_id = nbr_id_p;
				}
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					//data_kap_dbg->SetTuple1(kap_i, (double)cblock_ID_ref[i_dev][i_kap]);
					data_kap_dbg->SetTuple1(kap_i, (double)cells_ID_mask[i_dev][i_kap*M_CBLOCK + kap_i]);
					//data_kap_dbg->SetTuple1(kap_i, (double)cblock_ID_mask[i_dev][i_kap]);
					//data_kap_dbg->SetTuple1(kap_i, (double)min_nbr_id);
					//data_kap_dbg->SetTuple1(kap_i, (double)cblock_f_X[i_dev][i_kap + 0*n_maxcblocks] + (kap_i + 0.5)*dx);
				}
				grid_kap->GetCellData()->AddArray(data_kap_dbg);
					// Compute macroscopic properties.
				for (int i = 0; i < M_CBLOCK*(6+1); i++) u_kap[i] = 0.0;
				for (int i = 0; i < M_CBLOCK; i++) yplus_kap[i] = 0.0;
				M_ComputeOutputProperties(i_dev, i_kap, dxf_vec[L], u_kap, yplus_kap);
					// Density.
				vtkNew<vtkDoubleArray> data_kap_sc;
				data_kap_sc->SetName("Density");
				data_kap_sc->SetNumberOfComponents(1);
				data_kap_sc->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					data_kap_sc->SetTuple1(kap_i,
						u_kap[kap_i + 0*M_CBLOCK]
					);
				}
				grid_kap->GetCellData()->AddArray(data_kap_sc);
					// Velocity.
				vtkNew<vtkDoubleArray> data_kap_v;
				data_kap_v->SetName("Velocity");
				data_kap_v->SetNumberOfComponents(3);
				data_kap_v->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					if (L > 0 || (L == 0 && N_PROBE_AVE == 0))
					{
						data_kap_v->SetTuple3(kap_i,
							u_kap[kap_i + 1*M_CBLOCK],
							u_kap[kap_i + 2*M_CBLOCK],
							u_kap[kap_i + 3*M_CBLOCK]
						);
					}
					else
					{
						data_kap_v->SetTuple3(kap_i,
							(double)cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_ids[i_dev][0]*M_CBLOCK],
							(double)cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_ids[i_dev][0]*M_CBLOCK],
							(double)cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_ids[i_dev][0]*M_CBLOCK]
						);
					}
				}
				grid_kap->GetCellData()->AddArray(data_kap_v);
					// Vorticity.
				vtkNew<vtkDoubleArray> data_kap_w;
				data_kap_w->SetName("Vorticity");
				data_kap_w->SetNumberOfComponents(3);
				data_kap_w->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					data_kap_w->SetTuple3(kap_i, 
						u_kap[kap_i + 4*M_CBLOCK],
						u_kap[kap_i + 5*M_CBLOCK],
						u_kap[kap_i + 6*M_CBLOCK]
					);
				}
				grid_kap->GetCellData()->AddArray(data_kap_w);
#if (S_LES==1)
					// Y+
				vtkNew<vtkDoubleArray> data_kap_yp;
				data_kap_yp->SetName("Y+");
				data_kap_yp->SetNumberOfComponents(1);
				data_kap_yp->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					data_kap_yp->SetTuple1(kap_i, 
						yplus_kap[kap_i]
					);
				}
				grid_kap->GetCellData()->AddArray(data_kap_yp);
#endif
				
				// Create the vtkAMRBox and insert it into AMR object.
				vtkAMRBox box_kap(origin_kap, n_dim_lattice, h_L_kap, data->GetOrigin(), data->GetGridDescription());
				data->SetAMRBox(L, kap_counter, box_kap);
				data->SetDataSet(L, kap_counter, grid_kap);
				kap_counter++;
			}
		}
	}
	
	// Write the AMR object.
	std::cout << "Finished building VTK dataset, writing..." << std::endl;
	std::string fileName = P_DIR_NAME + std::string("out_") + std::to_string(iter+1) + ".vthb";
	vtkNew<vtkXMLUniformGridAMRWriter> writer;
	writer->SetInputData(data);
	writer->SetFileName(fileName.c_str());
	writer->Write();
	std::cout << "Finished writing VTK dataset..." << std::endl;
	
	return 0;
}
