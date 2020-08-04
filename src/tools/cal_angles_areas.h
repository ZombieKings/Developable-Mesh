#ifndef ZOMBIE_CAL_ANGLES_AREAS_H
#define ZOMBIE_CAL_ANGLES_AREAS_H

#include <Eigen/Core>

namespace Zombie
{
	// Compute angles for a triangle mesh 
	//
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	// Outputs:
	//   vecAngles  #V list of internal angles for vertices
	template<typename DerivedV, typename DerivedF,typename DerivedA>
	void cal_angles(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedA>& vecAngles);
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	// Outputs:
	//   vecAngles	#V list of internal angles for vertices
	//
	//   matAngles  3 by #F list of internal angles for triangles,
	//			 rows correspond to face vertices fv[0, 1, 2].
	template<typename DerivedV, typename DerivedF, typename DerivedvA, typename DerivedmA>
	void cal_angles(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedvA>& vecAngles,
		Eigen::PlainObjectBase<DerivedmA>& matAngles);

	// Compute angles and areas for a triangle mesh 
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	// Outputs:
	//   vecAngles	#V list of internal angles for vertices.
	//
	//   vecAreas	#V list of mix areas for vertices.
	//
	//   matAngles  3 by #F list of internal angles for triangles,
	//			 rows correspond to face vertices fv[0, 1, 2].
	//Note:
	//	Mix areas(i) = 1/3 * sum of 1-ring triangles' area over vertex(i).
	template<typename DerivedV, typename DerivedF, typename DerivedvA, typename DerivedvAr, typename DerivedmA>
	void cal_angles_and_areas(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedvA>& vecAngles,
		Eigen::PlainObjectBase<DerivedvAr>& vecAreas,
		Eigen::PlainObjectBase<DerivedmA>& matAngles);

	// Compute angles and areas for a triangle mesh 
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	// Outputs:
	//   vecAngles	#V list of internal angles for vertices.
	//
	//   vecAreas	#V list of mix areas for vertices.
	//
	//   matAngles  3 by #F list of internal angles for triangles,
	//			 rows correspond to face vertices fv[0, 1, 2].
	//Note:
	//	Mix areas(i) = 1/3 * sum of 1-ring triangles' area over vertex(i).
	template<typename DerivedV, typename DerivedF, typename DerivedvA, typename DerivedvAr>
	void cal_angles_and_areas(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedvA>& vecAngles,
		Eigen::PlainObjectBase<DerivedvAr>& vecAreas);

	// Compute angles for a triangle mesh with edge lengths
	//
	// Inputs:
	//   L  3 by #F list of edge lengths for mesh faces (must be triangles)
	//
	// Outputs:
	//   matAngles  3 by #F list of internal angles for triangles,
	//			 rows correspond to face vertices fv[0, 1, 2].
	template<typename DerivedL, typename DerivedmA>
	void cal_angles_and_areas_with_edges(const Eigen::MatrixBase<DerivedL>& L,
		Eigen::PlainObjectBase<DerivedmA>& matAngles);

	void cal_angles_and_areas_with_edges(int Vnum,
		const Eigen::Matrix3Xi& F,
		const Eigen::Matrix3Xd& matLength,
		Eigen::VectorXd& vecAreas,
		Eigen::VectorXd& vecAngles,
		Eigen::Matrix3Xd& matAngles);
}

#include "cal_angles_areas.cpp"

#endif 