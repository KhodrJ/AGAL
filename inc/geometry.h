#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "cppspec.h"

class Mesh;

class Geometry
{
	private:
	
	// o====================================================================================
	// | Internal parameters.
	// o====================================================================================
	
	int             init_index_lists        = 0;            ///< Indicates if index lists were initialized.
	int             init_coords_list        = 0;            ///< Indicates if face coordinates list was initialized.
	
	// o====================================================================================
	// | Routines.
	// o====================================================================================
	
	int G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
	int G_Dest();
	
	
	
	
	
	public:
	
	Mesh *mesh;
	
	// o====================================================================================
	// | Geometry parameters.
	// o====================================================================================
	
	float           Lx                      = 1.0F;         ///< Length of domain in x-axis (in meters).
	float           Ly                      = 1.0F;         ///< Length of domain in y-axis as a fraction of @ref Lx.
	float           Lz                      = 1.0F;         ///< Length of domain in z-axis as a fraction of @ref Lx.
	std::string     input_dir;                              ///< Input directory.
	std::string     output_dir;                             ///< Output directory.
	
	// o====================================================================================
	// | CPU parameters.
	// o====================================================================================
	
	//! Number of nodes.
	int             n_nodes[N_DEV];
	
	//! Number of vertices rounded to 32 for alignment.
	int             n_nodes_a[N_DEV];
	
	//! Number of faces.
	int             n_faces[N_DEV];
	
	//! Number of faces rounded to 32 for alignment.
	int             n_faces_a[N_DEV];
	
	//! Array of geometry node locations.
	double          *geom_f_node_X[N_DEV];
	
	//! Array of face indices.
	int             *geom_ID_face[N_DEV];
	
	//! Array of face vertices.
	double          *geom_f_face_X[N_DEV];
	
	//! Array of geometry face attributes attributes.
	double          *geom_ID_face_attr[N_DEV];
	
	std::vector<double>     v_geom_f_node_X;
	std::vector<double>     v_geom_f_node_Y;
	std::vector<double>     v_geom_f_node_Z;
	std::vector<int>        v_geom_ID_face_1;
	std::vector<int>        v_geom_ID_face_2;
	std::vector<int>        v_geom_ID_face_3;
	std::vector<double>     v_geom_f_face_1_X;
	std::vector<double>     v_geom_f_face_1_Y;
	std::vector<double>     v_geom_f_face_1_Z;
	std::vector<double>     v_geom_f_face_2_X;
	std::vector<double>     v_geom_f_face_2_Y;
	std::vector<double>     v_geom_f_face_2_Z;
	std::vector<double>     v_geom_f_face_3_X;
	std::vector<double>     v_geom_f_face_3_Y;
	std::vector<double>     v_geom_f_face_3_Z;
	
	// o====================================================================================
	// | GPU parameters.
	// o====================================================================================
	
	//! GPU counterpart of @ref geom_f_node_X.
	double          *c_geom_f_node_X[N_DEV];
	
	//! GPU counterpart of @ref geom_ID_face.
	int             *c_geom_ID_face[N_DEV];
	
	//! GPU counterpart of @ref geom_f_face_X.
	double          *c_geom_f_face_X[N_DEV];
	
	//! GPU counterpart of @ref geom_ID_face_attr.
	double          *c_geom_ID_face_attr[N_DEV];
	
	// o====================================================================================
	// | Routines.
	// o====================================================================================
	
	int G_ClearVectors();
	int G_ImportBoundariesFromTextFile(int i_dev);
	int G_ImportBoundariesFromSTLFile(int i_dev);
	int G_ImportSTL_ASCII(std::string filename);
	int G_PrintSTL(int i_dev);
	int G_PrintOBJ(int i_dev);
	
	int G_UpdateCounts(int i_dev);
	int G_Init_Arrays_IndexLists_CPU(int i_dev);
	int G_Init_Arrays_CoordsList_CPU(int i_dev);
	int G_Dest_Arrays_IndexLists(int i_dev);
	int G_Dest_Arrays_CoordsList(int i_dev);
	int G_Convert_IndexListsToCoordList(int i_dev);
	int G_Convert_CoordListToIndexLists(int i_dev);
	
	int G_AddBoundingBox(double ax, double bx, double ay, double by, double az, double bz);
	int G_AddRectangle(double ax, double bx, double ay, double by);
	int G_AddCircle(int N, double cx, double cy, double R);
	int G_AddPrism(double ax, double bx, double ay, double by, double az, double bz);
	int G_AddSphere(int N1, int N2, double cx, double cy, double cz, double R);
	int G_AddNACA002D(int N, double t, double ax, double bx, double ay, double by, int te);
	int G_AddNACA003D(int N, double t, double ax, double bx, double ay, double by, double az, double bz, int te);
	
	
	
	
	
	Geometry(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str)
	{
		G_Init(params_int, params_dbl, params_str);
	}
	
	~Geometry()
	{
		G_Dest();
	}
};

#endif
