#ifndef ZOMBIE_CAL_JACOBIAN_OF_ANGLE_H
#define ZOMBIE_CAL_JACOBIAN_OF_ANGLE_H

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace Zombie
{
	template<typename DerivedV, typename DerivedF, typename DerivedVType, typename DerivedA, typename T>
	void cal_jacobian_of_angle(
		const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		const Eigen::VectorXi& VType,
		const Eigen::MatrixBase<DerivedA>& mAngles,
		Eigen::SparseMatrix<T>& JoA);
}

#include "cal_Jacobian_of_angle.cpp"

#endif 