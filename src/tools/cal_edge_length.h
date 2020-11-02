#ifndef ZOMBIE_CAL_EDGE_LENGTH_H
#define ZOMBIE_CAL_EDGE_LENGTH_H

#include <Eigen/Dense>

namespace Zombie
{
	// Compute edge lengths for a triangle mesh 
	//
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//		or 2 by #E list of edges
	//
	// Outputs:
	//   L  1|3 by #F list of edge lengths 
	//     for edges, rows of lengths
	//     for triangles, rows correspond to edges [1,2],[2,0],[0,1]
	template <typename DerivedV, typename DerivedF, typename DerivedL>
	void cal_edge_length_per_face(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedL>& L);

	template <typename DerivedV, typename DerivedF, typename DerivedL>
	void cal_squard_edge_lengths(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedL>& L);
}

#include "cal_edge_length.cpp"

#endif 