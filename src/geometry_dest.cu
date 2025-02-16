#include "geometry.h"

int Geometry::G_Dest()
{
	std::cout << " o====================================================================================" << std::endl;
	std::cout << " | Deleting: Geometry Object                                                          " << std::endl;
	std::cout << " o====================================================================================" << std::endl;
	
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
	
	return 0;
}
