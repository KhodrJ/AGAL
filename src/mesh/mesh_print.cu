/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

int Mesh::M_Print_FillBlock(int i_dev, int *Is, int i_kap, int L, double dx_f, int *mult_f, int *Nxi_f, double *out_u, double *out_yplus, double *tmp_data)
{
	// (I,J,K) are the indices for the top parent. All child indices moving forward are built from these.
	int child_0 = cblock_ID_nbr_child[i_dev][i_kap];
	
	if (child_0 >= 0 && L+1 < N_PRINT_LEVELS) // Has children, keep traversing.
	{
#if (N_DIM==3)
		for (int xk = 0; xk < 2; xk++)
#else
		int xk = 0;
#endif
		{
			for (int xj = 0; xj < 2; xj++)
			{
				for (int xi = 0; xi < 2; xi++)
				{
					Is[L+1 + 0*N_PRINT_LEVELS] = xi;
					Is[L+1 + 1*N_PRINT_LEVELS] = xj;
					Is[L+1 + 2*N_PRINT_LEVELS] = xk;
					int xc = xi + 2*xj + 4*xk;
					M_Print_FillBlock(i_dev, Is, child_0+xc, L+1, dx_f, mult_f, Nxi_f, out_u, out_yplus, tmp_data
					);
				}
			}
		}
	}
	else // No children, print here.
	{
		// Get the macroscopic properties for this block.
		M_ComputeProperties(i_dev, i_kap, dxf_vec[L], out_u, out_yplus);
		
		// Modify the cell values in the region defined by the leaf block.
#if (N_DIM==3)
		for (int k = 0; k < Nbx; k++)
#else
		int k = 0;
#endif
		{
			for (int j = 0; j < Nbx; j++)
			{
				for (int i = 0; i < Nbx; i++)
				{
					int Ip = 0;
					int Jp = 0;
					int Kp = 0;
					for (int l = 0; l < L+1; l++)
					{
						Ip += mult_f[l]*Nbx*Is[l + 0*N_PRINT_LEVELS];
						Jp += mult_f[l]*Nbx*Is[l + 1*N_PRINT_LEVELS];
						Kp += mult_f[l]*Nbx*Is[l + 2*N_PRINT_LEVELS];
					}
					
#if (N_DIM==3)
					for (int kk = 0; kk < mult_f[L]; kk++)
#else
					int kk = 0;
#endif
					{
						for (int jj = 0; jj < mult_f[L]; jj++)
						{
							for (int ii = 0; ii < mult_f[L]; ii++)
							{
								int kap_i = i + Nbx*j + Nbx*Nbx*k;
								int Ipp = Ip + i*mult_f[L] + ii;
								int Jpp = Jp + j*mult_f[L] + jj;
								int Kpp = Kp + k*mult_f[L] + kk;
								long int Id = Ipp + Nxi_f[0]*Jpp + Nxi_f[0]*Nxi_f[1]*Kpp;
								
								tmp_data[Id + 0*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = out_u[kap_i + 0*M_CBLOCK];
								tmp_data[Id + 1*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = out_u[kap_i + 1*M_CBLOCK];
								tmp_data[Id + 2*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = out_u[kap_i + 2*M_CBLOCK];
								tmp_data[Id + 3*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = out_u[kap_i + 3*M_CBLOCK];
								tmp_data[Id + 4*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = out_u[kap_i + 4*M_CBLOCK];
								tmp_data[Id + 5*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = out_u[kap_i + 5*M_CBLOCK];
								tmp_data[Id + 6*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = out_u[kap_i + 6*M_CBLOCK];
								tmp_data[Id + 7*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = sqrt(out_u[kap_i + 1*M_CBLOCK]*out_u[kap_i + 1*M_CBLOCK] + out_u[kap_i + 2*M_CBLOCK]*out_u[kap_i + 2*M_CBLOCK] + out_u[kap_i + 3*M_CBLOCK]*out_u[kap_i + 3*M_CBLOCK]);
								tmp_data[Id + 8*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = sqrt(out_u[kap_i + 4*M_CBLOCK]*out_u[kap_i + 4*M_CBLOCK] + out_u[kap_i + 5*M_CBLOCK]*out_u[kap_i + 5*M_CBLOCK] + out_u[kap_i + 6*M_CBLOCK]*out_u[kap_i + 6*M_CBLOCK]);
								tmp_data[Id + 9*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = L;
								tmp_data[Id + 10*Nxi_f[0]*Nxi_f[1]*Nxi_f[2]] = i_kap;
							}
						}
					}
				}
			}
		}
	}
		
	return 0;
}

int Mesh::M_Print(int i_dev, int iter)
{
	// Parameters.
		// Resolution multiplier.
	int mult = 1;
	double dx_f = dxf_vec[0];
	for (int L = 1; L < N_PRINT_LEVELS; L++) 
	{
		mult *= 2;
		dx_f *= 0.5;
	}
	int *mult_f = new int[N_PRINT_LEVELS];
	for (int L = 0; L < N_PRINT_LEVELS; L++) mult_f[L] = pow(2.0, (double)(N_PRINT_LEVELS-1-L));
		// Resolution array.
	int Nxi_f[3];
	for (int d = 0; d < 3; d++) Nxi_f[d] = Nxi[d];
	for (int d = 0; d < N_DIM; d++) Nxi_f[d] *= mult;
	int vol = Nxi_f[0]*Nxi_f[1]*Nxi_f[2];
		// Tracker for child indices.
	int *Is = new int[N_PRINT_LEVELS*3];
	for (int Ld = 0; Ld < N_PRINT_LEVELS*3; Ld++) Is[Ld] = 0;
	
	// Cell data arrays.
	int n_data = 3+3+1+1+1+1+1;
	double *tmp_data = new double[n_data*vol];
	double *tmp_data_b = new double[n_data*vol];
	for (long int p = 0; p < n_data*vol; p++) tmp_data[p] = 0.0;
		// Density.
	vtkNew<vtkDoubleArray> cell_data_density;
	cell_data_density->SetName("Density");
	cell_data_density->SetNumberOfComponents(1);
	cell_data_density->SetNumberOfTuples(vol);
		// Velocity.
	vtkNew<vtkDoubleArray> cell_data_velocity;
	cell_data_velocity->SetName("Velocity");
	cell_data_velocity->SetNumberOfComponents(3);
	cell_data_velocity->SetNumberOfTuples(vol);
		// Vorticity.
	vtkNew<vtkDoubleArray> cell_data_vorticity;
	cell_data_vorticity->SetName("Vorticity");
	cell_data_vorticity->SetNumberOfComponents(3);
	cell_data_vorticity->SetNumberOfTuples(vol);
		// Velocity Magnitude.
	vtkNew<vtkDoubleArray> cell_data_velmag;
	cell_data_velmag->SetName("Velocity Magnitude");
	cell_data_velmag->SetNumberOfComponents(1);
	cell_data_velmag->SetNumberOfTuples(vol);
		// Vorticity Magnitude.
	vtkNew<vtkDoubleArray> cell_data_vortmag;
	cell_data_vortmag->SetName("Vorticity Magnitude");
	cell_data_vortmag->SetNumberOfComponents(1);
	cell_data_vortmag->SetNumberOfTuples(vol);
		// AMR Level.
	vtkNew<vtkDoubleArray> cell_data_level;
	cell_data_level->SetName("AMR Level");
	cell_data_level->SetNumberOfComponents(1);
	cell_data_level->SetNumberOfTuples(vol);
		// Block Id.
	vtkNew<vtkDoubleArray> cell_data_blockid;
	cell_data_blockid->SetName("Block Id");
	cell_data_blockid->SetNumberOfComponents(1);
	cell_data_blockid->SetNumberOfTuples(vol);
		// Temporary arrays.
	double out_u[M_CBLOCK*(6+1)]; for (int i = 0; i < M_CBLOCK*(6+1); i++) out_u[i] = 0.0; 
	double out_yplus[M_CBLOCK]; for (int i = 0; i < M_CBLOCK; i++) out_yplus[i] = 0.0;
	
	// Traverse the grid and fill data arrays.
	for (int kap = 0; kap < n_ids[i_dev][0]; kap++)
	{
		Is[0*N_PRINT_LEVELS] = coarse_I[i_dev][kap];
		Is[1*N_PRINT_LEVELS] = coarse_J[i_dev][kap];
		Is[2*N_PRINT_LEVELS] = coarse_K[i_dev][kap];
		
		M_Print_FillBlock(i_dev, Is, kap, 0, dx_f, mult_f, Nxi_f, out_u, out_yplus, tmp_data);
	}
	
	// Smoothing.
	int n_smooths = 8;
	for (int kap = 0; kap < vol; kap++)
	{
		for (int p = 0; p < 7; p++)
			tmp_data_b[kap + p*vol] = tmp_data[kap + p*vol];
	}
	for (int i = 0; i < n_smooths; i++)
	{
		std::cout << "Smoothing iteration " << i << "..." << std::endl;
#if (N_DIM==2)
		for (int J = 1; J < Nxi_f[1]-1; J++)
		{
			for (int I = 1; I < Nxi_f[0]-1; I++)
			{
				int kap = (I) + Nxi_f[0]*(J);
				for (int p = 0; p < 9; p++)
				{
					tmp_data_b[kap + p*vol] = (
						tmp_data[(I+1) + Nxi_f[0]*(J) + p*vol] +
						tmp_data[(I-1) + Nxi_f[0]*(J) + p*vol] +
						tmp_data[(I) + Nxi_f[0]*(J+1) + p*vol] +
						tmp_data[(I) + Nxi_f[0]*(J-1) + p*vol]
					)/4.0;
				}
			}
		}
#else
		for (int K = 1; K < Nxi_f[2]-1; K++)
		{
			for (int J = 1; J < Nxi_f[1]-1; J++)
			{
				for (int I = 1; I < Nxi_f[0]-1; I++)
				{
					int kap = (I) + Nxi_f[0]*(J) + Nxi_f[0]*Nxi_f[1]*(K);
					for (int p = 0; p < 9; p++)
					{
						tmp_data_b[kap + p*vol] = (
							tmp_data[(I+1) + Nxi_f[0]*(J) + Nxi_f[0]*Nxi_f[1]*(K) + p*vol] +
							tmp_data[(I-1) + Nxi_f[0]*(J) + Nxi_f[0]*Nxi_f[1]*(K) + p*vol] +
							tmp_data[(I) + Nxi_f[0]*(J+1) + Nxi_f[0]*Nxi_f[1]*(K) + p*vol] +
							tmp_data[(I) + Nxi_f[0]*(J-1) + Nxi_f[0]*Nxi_f[1]*(K) + p*vol] +
							tmp_data[(I) + Nxi_f[0]*(J) + Nxi_f[0]*Nxi_f[1]*(K+1) + p*vol] +
							tmp_data[(I) + Nxi_f[0]*(J) + Nxi_f[0]*Nxi_f[1]*(K-1) + p*vol]
						)/6.0;
					}
				}
			}
		}
#endif
		for (int kap = 0; kap < vol; kap++)
		{
			for (int p = 0; p < 7; p++)
				tmp_data[kap + p*vol] = tmp_data_b[kap + p*vol];
		}
	}
	
	// Insert data in VTK arrays.
	for (long int kap = 0; kap < vol; kap++)
	{
		cell_data_density->SetTuple1(kap,
			tmp_data[kap+ 0*vol]
		);
		cell_data_velocity->SetTuple3(kap,
			tmp_data[kap+ 1*vol],
			tmp_data[kap+ 2*vol],
			tmp_data[kap+ 3*vol]
		);
		cell_data_vorticity->SetTuple3(kap, 
			tmp_data[kap+ 4*vol],
			tmp_data[kap+ 5*vol],
			tmp_data[kap+ 6*vol]
		);
		cell_data_velmag->SetTuple1(kap, 
			tmp_data[kap+ 7*vol]
		);
		cell_data_vortmag->SetTuple1(kap, 
			tmp_data[kap+ 8*vol]
		);
		cell_data_level->SetTuple1(kap,
			tmp_data[kap+ 9*vol]
		);
		cell_data_blockid->SetTuple1(kap,
			tmp_data[kap+ 10*vol]
		);
	}
	
	// Image data.
		// Parameters and initialization.
	double origin[3] = {0.0,0.0,0.0};
	double spacing[3] = {dx_f, dx_f, dx_f};
	vtkNew<vtkUniformGrid> grid;
	vtkNew<vtkCellDataToPointData> cell_to_points;
	vtkNew<vtkContourFilter> contour;
		// Set up image data grid.
	grid->Initialize();
	grid->SetOrigin(origin);
	grid->SetSpacing(spacing);
	grid->SetDimensions(Nxi_f[0]+1, Nxi_f[1]+1, N_DIM==2?1:Nxi_f[2]+1);
	grid->GetCellData()->AddArray(cell_data_density);
	grid->GetCellData()->AddArray(cell_data_velocity);
	grid->GetCellData()->AddArray(cell_data_vorticity);
	grid->GetCellData()->AddArray(cell_data_velmag);
	grid->GetCellData()->AddArray(cell_data_vortmag);
	grid->GetCellData()->AddArray(cell_data_level);
	grid->GetCellData()->AddArray(cell_data_blockid);
	
	// Image data processing.
		// Convert cell data to point data.
	cell_to_points->SetInputData(grid);
	cell_to_points->Update();
		// Contour for vorticity magnitude.
	cell_to_points->GetImageDataOutput()->GetPointData()->SetActiveScalars("Vorticity Magnitude");
	contour->SetInputConnection(0, cell_to_points->GetOutputPort(0));
	contour->SetValue(0, 0.1);
	
	// Offscreen rendering.
		// Setup offscreen rendering.
	vtkNew<vtkNamedColors> colors;
	vtkNew<vtkGraphicsFactory> graphics_factory;
	graphics_factory->SetOffScreenOnlyMode(1);
	graphics_factory->SetUseMesaClasses(1);
		// Create mapper.
	vtkNew<vtkPolyDataMapper> mapper;
	mapper->SetInputConnection(contour->GetOutputPort(0));
		// Create actor.
	vtkNew<vtkActor> actor;
	actor->SetMapper(mapper);
	actor->GetProperty()->SetColor(colors->GetColor3d("White").GetData());
		// Create renderer.
	vtkNew<vtkRenderer> renderer;
	vtkNew<vtkRenderWindow> renderWindow;
	renderWindow->SetOffScreenRendering(1);
	renderWindow->AddRenderer(renderer);
		// Create camera.
	double cam_pos[3] = {1.8, -2.5, 1.25};
	//double cam_view_up[3] = {-0.066475, 0.21161, 0.975091};
	double cam_view_up[3] = {0.0, 0.0, 1.0};
	double cam_focal_point[3] = {0.5, 0.5, 0.5};
	vtkNew<vtkCamera> camera;
	renderer->SetActiveCamera(camera);
	camera->SetPosition(cam_pos);
	camera->SetViewUp(cam_view_up);
	camera->SetFocalPoint(cam_focal_point);
		// Add actor to scene and render.
	renderer->AddActor(actor);
	renderer->SetBackground(colors->GetColor3d("SlateGray").GetData());
	renderWindow->Render();
		// Print to PNG.
	vtkNew<vtkWindowToImageFilter> windowToImageFilter;
	windowToImageFilter->SetInput(renderWindow);
	windowToImageFilter->Update();
	vtkNew<vtkPNGWriter> photographer;
	std::string photo_name = P_DIR_NAME + std::string("shot_") + std::to_string(iter+1) + ".png";
	photographer->SetFileName(photo_name.c_str());
	photographer->SetInputConnection(windowToImageFilter->GetOutputPort());
	photographer->Write();
	
	// Write grid.
	std::cout << "Finished building VTK dataset, writing..." << std::endl;
	std::string file_name = P_DIR_NAME + std::string("out_") + std::to_string(iter+1) + ".vti";
	vtkNew<vtkXMLImageDataWriter> writer;
	writer->SetInputData(cell_to_points->GetImageDataOutput());
	writer->SetFileName(file_name.c_str());
	writer->Write();
	std::cout << "Finished writing VTK dataset..." << std::endl;
	
	// Free allocations.
	delete[] mult_f;
	delete[] Is;
	delete[] tmp_data;
	delete[] tmp_data_b;
	
	return 0;
}
