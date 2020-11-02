#include "cal_normals.h"

template<typename DerivedV, typename DerivedF, typename DerivedN>
void Zombie::cal_normal_per_vertex(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	Eigen::PlainObjectBase<DerivedN>& N)
{

	Eigen::Matrix<typename DerivedV::Scalar, 3, Eigen::Dynamic> PFN;
	Zombie::cal_normal_per_face(V, F, PFN);
	Zombie::cal_normal_per_vertex(V, F, PFN, N);
}

template<typename DerivedV, typename DerivedF, typename DerivedFN, typename DerivedN>
void Zombie::cal_normal_per_vertex(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::MatrixBase<DerivedFN>& FN,
	Eigen::PlainObjectBase<DerivedN>& N)
{
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");

	N.setConstant(DIM, V.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
		for(int vi = 0; vi < 3; ++vi)
			N.col(F(vi, f)) += FN.col(f);

	N.colwise().normalize();
}

template<typename DerivedV, typename DerivedF, typename DerivedN>
void Zombie::cal_normal_per_face(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	Eigen::PlainObjectBase<DerivedN>& N)
{
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");

	N.setConstant(DIM, F.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const auto& p0 = V.col(F(0, f));
		const auto& p1 = V.col(F(1, f));
		const auto& p2 = V.col(F(2, f));
		N.col(f) = (p1 - p0).cross(p2 - p0).normalized();
	}
}
