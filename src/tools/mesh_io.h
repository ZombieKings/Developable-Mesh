#ifndef ZOMBIE_MESH_IO_H
#define ZOMBIE_MESH_IO_H

#include <Eigen/Dense>
#include <surface_mesh/Surface_mesh.h>

namespace Zombie
{
	// Compute normals per face for a triangle mesh 
	//
	// Inputs:
	//	mesh  surface mesh object
	//
	// Outputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	template<typename DerivedV, typename DerivedF>
	void mesh2matrix(const surface_mesh::Surface_mesh& mesh,
		Eigen::PlainObjectBase<DerivedV>& V,
		Eigen::PlainObjectBase<DerivedF>& F);

	// Compute normals per face for a triangle mesh 
	//
	// Inputs:
	//	mesh  surface mesh object
	//
	// Outputs:
	//   V  3 by #V list of vertices position
	//
	//   E  3 by #E list of mesh edges
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	template<typename DerivedV, typename DerivedE, typename DerivedF>
	void mesh2matrix(const surface_mesh::Surface_mesh& mesh,
		Eigen::PlainObjectBase<DerivedV>& V,
		Eigen::PlainObjectBase<DerivedE>& E,
		Eigen::PlainObjectBase<DerivedF>& F);


	//void mesh_neighbor(const surface_mesh::Surface_mesh& mesh,
	//	Eigen::PlainObjectBase<DerivedV>& V,
	//	Eigen::PlainObjectBase<DerivedE>& E,
	//	Eigen::PlainObjectBase<DerivedF>& F);

}
#include "mesh_io.cpp"

#endif 