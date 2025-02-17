#include "geometry.h"

int Geometry::G_Dest()
{
	std::cout << " o====================================================================================" << std::endl;
	std::cout << " | Deleting: Geometry Object                                                          " << std::endl;
	std::cout << " o====================================================================================" << std::endl;
	
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
		G_Dest_Arrays_IndexLists(i_dev);
	
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
		G_Dest_Arrays_CoordsList(i_dev);
	
	return 0;
}

int Geometry::G_Dest_Arrays_IndexLists(int i_dev)
{
	if (init_index_lists)
	{
		// Free all the memory allocated on the CPU.
		delete[] geom_f_node_X[i_dev];
		delete[] geom_ID_face[i_dev];
		delete[] geom_ID_face_attr[i_dev];
		
		// Free all the memory allocated on the GPU.
		gpuErrchk( cudaFree(c_geom_f_node_X[i_dev]) );
		gpuErrchk( cudaFree(c_geom_ID_face[i_dev]) );
		gpuErrchk( cudaFree(c_geom_ID_face_attr[i_dev]) );
		
		init_index_lists = 0;
	}
	
	return 0;
}

int Geometry::G_Dest_Arrays_CoordsList(int i_dev)
{
	if (init_coords_list)
	{
		// Free all the memory allocated on the CPU.
		delete[] geom_f_face_X[i_dev];
		
		// Free all the memory allocated on the GPU.
		gpuErrchk( cudaFree(c_geom_f_face_X[i_dev]) );
		
		init_coords_list = 0;
	}
	
	return 0;
}
