#ifndef ZOMBIE_CAL_LAPLACIAN_H
#define ZOMBIE_CAL_LAPLACIAN_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Zombie
{
	// Compute cotangent-weighted Laplace operator matrix for a triangle mesh 
	//
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	//	 RowsPerV 1|3, denote every coefficient corespondant every vertex or every dimension of vertex
	// Outputs:
	//   L  #V * RowsPerV by #V * RowsPerV sparse Laplace matrix
	template <typename DerivedV, typename DerivedF, typename T>
	void cal_cot_laplace(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		const int RowsPerV,
		Eigen::SparseMatrix<T>& L);
	// Inputs:
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	//   matAngles  3 by #F list of internal angles for triangles,
	//			 rows correspond to face vertices fv[0, 1, 2].
	//
	//   vecAreas	#V list of mix areas for vertices.
	//
	//	 RowsPerV 1|3, denote every coefficient corespondant every vertex or every dimension of vertex
	// Outputs:
	//   L  #V * RowsPerV by #V * RowsPerV sparse Laplace matrix
	template <typename DerivedF, typename DerivedA, typename DerivedAr, typename T>
	void cal_cot_laplace(int Vnum,
		const Eigen::MatrixBase<DerivedF>& F,
		const Eigen::MatrixBase<DerivedA>& matAngles,
		const Eigen::MatrixBase<DerivedAr>& vecAreas,
		const int RowsPerV,
		Eigen::SparseMatrix<T>& L);

	// Compute uniform-weighted Laplace operator matrix for a triangle mesh 
	//
	// Inputs:
	//   Vnum  number of vertices
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	//	 RowsPerV 1|3, denote every coefficient corespondant every vertex or every dimension of vertex
	// Outputs:
	//   L  #V * RowsPerV by #V * RowsPerV sparse Laplace matrix
	template <typename DerivedF, typename T>
	void cal_uni_laplace(int Vnum,
		const Eigen::MatrixBase<DerivedF>& F,
		const int RowsPerV,
		Eigen::SparseMatrix<T>& L);
}

#include "cal_laplacian.cpp"

#endif 