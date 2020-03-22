#ifndef ZOMBIE_CAL_LAPLACIAN_H
#define ZOMBIE_CAL_LAPLACIAN_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Zombie
{
	template <typename DerivedF, typename DerivedAngle, typename AreasVector, typename IndexVector, typename T>
	void cal_cot_laplace(const Eigen::MatrixBase<DerivedF>& F,
		const Eigen::MatrixBase<DerivedAngle>& mAngles,
		const AreasVector& areas,
		const IndexVector& interVidx,
		Eigen::SparseMatrix<T>& L);


	template <typename DerivedF, typename IndexVector, typename T>
	void cal_uni_laplace(const Eigen::MatrixBase<DerivedF>& F,
		int Vnum, const IndexVector& interVidx,
		Eigen::SparseMatrix<T>& L);
}

#include "cal_laplacian.cpp"

#endif 