#ifndef ZOMBIE_CAL_EDGE_LENGTH_H
#define ZOMBIE_CAL_EDGE_LENGTH_H

#include <Eigen/Dense>

namespace Zombie
{
	template <typename DerivedV, typename DerivedE, typename DerivedL>
	void cal_edge_length(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedE>& E,
		Eigen::PlainObjectBase<DerivedL>& L);
}

#include "cal_edge_length.cpp"

#endif 