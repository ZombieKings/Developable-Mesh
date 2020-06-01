#include "cal_edge_length.h"

template <typename DerivedV, typename DerivedE, typename DerivedL>
void Zombie::cal_edge_length(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedE>& E,
	Eigen::PlainObjectBase<DerivedL>& L)
{
	const int Enum = E.cols();
	L.setConstant(Enum, 0);
	for (int i = 0, i < Enum; ++i)
		L(i) = V.col(E(1, i)) - V.col(E(0, i)).norm();
}