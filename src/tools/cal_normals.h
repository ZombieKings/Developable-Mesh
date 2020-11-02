#ifndef ZOMBIE_CAL_NORMALS_H
#define ZOMBIE_CAL_NORMALS_H

#include <Eigen/Dense>

namespace Zombie
{
	// Compute normals per vertex for a triangle mesh 
	//
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	// Outputs:
	//   N  3 by #V list of internal angles for vertices
	template<typename DerivedV, typename DerivedF, typename DerivedN>
	void cal_normal_per_vertex(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedN>& N);

	// Compute normals per vertex for a triangle mesh 
	//
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	//   FN  3 by #F list of faces' normal (must be triangles)
	//
	// Outputs:
	//   N  3 by #V list of internal angles for vertices
	template<typename DerivedV, typename DerivedF, typename DerivedFN, typename DerivedN>
	void cal_normal_per_vertex(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		const Eigen::MatrixBase<DerivedFN>& FN,
		Eigen::PlainObjectBase<DerivedN>& N);

	// Compute normals per face for a triangle mesh 
	//
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	// Outputs:
	//   N  3 by #F list of internal angles for faces
	template<typename DerivedV, typename DerivedF, typename DerivedN>
	void cal_normal_per_face(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedN>& N);
}
#include "cal_normals.cpp"

#endif 