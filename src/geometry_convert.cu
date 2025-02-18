#include "geometry.h"

int Geometry::G_Convert_IndexListsToCoordList(int i_dev)
{
	// Reset the coords list if it's non-empty.
	if (v_geom_f_face_1_X.size() > 0)
	{
		v_geom_f_face_1_X.clear();
		v_geom_f_face_1_Y.clear();
		v_geom_f_face_1_Z.clear();
		v_geom_f_face_2_X.clear();
		v_geom_f_face_2_Y.clear();
		v_geom_f_face_2_Z.clear();
		v_geom_f_face_3_X.clear();
		v_geom_f_face_3_Y.clear();
		v_geom_f_face_3_Z.clear();
	}
	
	// Now fill it with the index list data.
	for (int j = 0; j < v_geom_ID_face_1.size(); j++)
	{
		int f1 = v_geom_ID_face_1[j];
		int f2 = v_geom_ID_face_2[j];
		int f3 = v_geom_ID_face_3[j];
		ufloat_g_t v1x = v_geom_f_node_X[f1];
		ufloat_g_t v1y = v_geom_f_node_Y[f1];
		ufloat_g_t v1z = v_geom_f_node_Z[f1];
		ufloat_g_t v2x = v_geom_f_node_X[f2];
		ufloat_g_t v2y = v_geom_f_node_Y[f2];
		ufloat_g_t v2z = v_geom_f_node_Z[f2];
		ufloat_g_t v3x = v_geom_f_node_X[f3];
		ufloat_g_t v3y = v_geom_f_node_Y[f3];
		ufloat_g_t v3z = v_geom_f_node_Z[f3];
		
		v_geom_f_face_1_X.push_back(v1x);
		v_geom_f_face_1_Y.push_back(v1y);
		v_geom_f_face_1_Z.push_back(v1z);
		v_geom_f_face_2_X.push_back(v2x);
		v_geom_f_face_2_Y.push_back(v2y);
		v_geom_f_face_2_Z.push_back(v2z);
		v_geom_f_face_3_X.push_back(v3x);
		v_geom_f_face_3_Y.push_back(v3y);
		v_geom_f_face_3_Z.push_back(v3z);
	}
	
	return 0;
}

// TODO
int Geometry::G_Convert_CoordListToIndexLists(int i_dev)
{
	for (int j = 0; j < n_faces[i_dev]; j++)
	{
		
	}
	
	return 0;
}
