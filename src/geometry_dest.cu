#include "geometry.h"

int Geometry::G_Dest()
{
	std::cout << " o====================================================================================" << std::endl;
	std::cout << " | Deleting: Geometry Object                                                          " << std::endl;
	std::cout << " o====================================================================================" << std::endl;
	
	if (init_index_lists)
	{
		// Free all the memory allocated on the CPU.
		for (int i_dev = 0; i_dev < N_DEV; i_dev++)
		{
			delete[] geom_f_node_X[i_dev];
			delete[] geom_ID_face[i_dev];
			delete[] geom_ID_face_attr[i_dev];
		}
		
		// Free all the memory allocated on the GPU.
		for (int i_dev = 0; i_dev < N_DEV; i_dev++)
		{
			gpuErrchk( cudaFree(c_geom_f_node_X[i_dev]) );
			gpuErrchk( cudaFree(c_geom_ID_face[i_dev]) );
			gpuErrchk( cudaFree(c_geom_ID_face_attr[i_dev]) );
		}
		
		init_index_lists = 0;
	}
	
	return 0;
}

int Geometry::G_Dest_Arrays_IndexLists(int i_dev)
{
	if (init_index_lists)
	{
		delete[] geom_f_node_X[i_dev];
		delete[] geom_ID_face[i_dev];
		delete[] geom_ID_face_attr[i_dev];
		
		gpuErrchk( cudaFree(c_geom_f_node_X[i_dev]) );
		gpuErrchk( cudaFree(c_geom_ID_face[i_dev]) );
		gpuErrchk( cudaFree(c_geom_ID_face_attr[i_dev]) );
		
		init_index_lists = 0;
	}
	else
		std::cout << "[-] Warning: no geometry to reset..." << std::endl;
	
	return 0;
}
