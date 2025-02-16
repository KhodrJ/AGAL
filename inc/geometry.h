#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "cppspec.h"

class Geometry
{
	private:
	
	// o====================================================================================
	// | Routines.
	// o====================================================================================
	
	int G_Init(int *L, std::string output_dir_);
	int G_Dest();
	
	
	
	
	
	public:
	
	// o====================================================================================
	// | Geometry parameters.
	// o====================================================================================
	
	float           Lx                      = 1.0F;         ///< Length of domain in x-axis (in meters).
	float           Ly                      = 1.0F;         ///< Length of domain in y-axis as a fraction of @ref Lx.
	float           Lz                      = 1.0F;         ///< Length of domain in z-axis as a fraction of @ref Lx.
	std::string     output_dir;                             ///< Output directory.
	
	// o====================================================================================
	// | CPU paramters.
	// o====================================================================================
	
	//! Number of nodes. [DEPRECATED]
	int		n_nodes[N_DEV];
	
	//! Number of faces. [DEPRECATED]
	int		n_faces[N_DEV];
	
	//! Array of geometry node locations.
	double		*geom_f_node_X[N_DEV];
	
	//! Array of face indices.
	int		*geom_ID_face[N_DEV];
	
	//! Array of geometry face attributes attributes.
	double		*geom_ID_face_attr[N_DEV];
	
	// o====================================================================================
	// | GPU paramters.
	// o====================================================================================
	
	////! GPU counterpart of @ref cblock_ID_face.
	//int		*c_cblock_ID_face[N_DEV];
	
	//! GPU counterpart of @ref geom_f_node_X.
	double		*c_geom_f_node_X[N_DEV];
	
	//! GPU counterpart of @ref geom_ID_face.
	int		*c_geom_ID_face[N_DEV];
	
	//! GPU counterpart of @ref geom_ID_face_attr.
	double		*c_geom_ID_face_attr[N_DEV];
	
	// o====================================================================================
	// | Routines.
	// o====================================================================================
	
	int G_ImportBoundariesFromTextFile(int i_dev);
	int G_ImportBoundariesFromSTLFile(int i_dev);
	int PrintSTL(int i_dev);
	int PrintOBJ(int i_dev);
	
	Geometry(int *L, std::string output_dir_)
	{
		G_Init(L, output_dir_);
	}
	
	~Geometry()
	{
		G_Dest();
	}
};

#endif
