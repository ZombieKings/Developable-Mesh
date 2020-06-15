#include "cal_edge_length.h"

template <typename DerivedV, typename DerivedF, typename DerivedL>
void Zombie::cal_edge_length_per_face(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	Eigen::PlainObjectBase<DerivedL>& L)
{
	typedef typename DerivedF::Index Index;
	const int DIM = F.rows();
	switch (DIM)
	{
	case 2: {
		const int Enum = F.cols();
		L.resize(Enum, 1);
		for (int i = 0; i < Enum; ++i)
			L(i) = (V.col(F(1, i)) - V.col(F(0, i))).norm();
		break;
	}
	case 3: {
		assert(V.rows() == 3 && "This function is only for triangle mesh!");
		const int Fnum = F.cols();
		L.resize(3, Fnum);
		for (Index f = 0; f < Fnum; ++f)
		{
			const auto& fv = F.col(f);
			L(0, f) = (V.col(fv[1]) - V.col(fv[2])).norm();
			L(1, f) = (V.col(fv[2]) - V.col(fv[0])).norm();
			L(2, f) = (V.col(fv[0]) - V.col(fv[1])).norm();
		}
		break;
	}
	}
}