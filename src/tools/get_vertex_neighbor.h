#ifndef ZOMBIE_GET_VERTEX_NEIGHBOR_H
#define ZOMBIE_GET_VERTEX_NEIGHBOR_H

#include <Eigen/Dense>
#include <surface_mesh/Surface_mesh.h>

namespace Zombie
{
	// Get a vector contains 1-ring neighbor faces of each vertices
	//
	// Inputs:
	//	mesh  surface mesh object
	//
	// Outputs:
	//
	//   neighF #V 1-ring neighbor faces of each vertices
	void get_neighbor_faces(const surface_mesh::Surface_mesh& mesh,
		std::vector<std::vector<int>>& neighF);
	
	// Get a vector contains 1-ring neighbor vertices of each vertices
	//
	// Inputs:
	//	mesh  surface mesh object
	//
	// Outputs:
	//
	//   neighF #V 1-ring neighbor vertices of each vertices
	void get_neighbor_vertices(const surface_mesh::Surface_mesh& mesh,
		std::vector<std::vector<int>>& neighV);	
	
}
#include "get_vertex_neighbor.cpp"

#endif 