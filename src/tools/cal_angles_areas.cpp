#include "cal_angles_areas.h"
#include "cal_edge_length.h"

#include <math.h>  

template<typename DerivedV, typename DerivedF, typename DerivedA>
void Zombie::cal_angles(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	Eigen::PlainObjectBase<DerivedA>& vecAngles)
{
	using namespace Eigen;
	using namespace std;
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");
	vecAngles.resize(V.cols());
	vecAngles.setZero();
	for (int f = 0; f < F.cols(); ++f)
	{
		const auto& fv = F.col(f);
		for (int vi = 0; vi < DIM; ++vi)
		{
			const auto& p0 = V.col(fv[vi]);
			const auto& p1 = V.col(fv[(vi + 1) % DIM]);
			const auto& p2 = V.col(fv[(vi + 2) % DIM]);
			const auto angle = acos(max(-1.0, min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			vecAngles(fv[vi]) += angle;
		}
	}
}

template<typename DerivedV, typename DerivedF, typename DerivedvA, typename DerivedmA>
void Zombie::cal_angles(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	Eigen::PlainObjectBase<DerivedvA>& vecAngles,
	Eigen::PlainObjectBase<DerivedmA>& matAngles)
{
	typedef typename DerivedV::Scalar Scalar;
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");
	matAngles.setConstant(DIM, F.cols(), 0);
	vecAngles.setConstant(V.cols(), 0);

	//Calculate edge lengths of mesh
	Eigen::Matrix<Scalar, DerivedF::RowsAtCompileTime, DerivedF::ColsAtCompileTime>L;
	cal_edge_length_per_face(V, F, L);

	//Calculate angles with edge lengths
	cal_angles_and_areas_with_edges(L, matAngles);

	//Collect angles per vertices
	for (int f = 0; f < F.cols(); ++f)
		for (int vi = 0; vi < DIM; ++vi)
			vecAngles(F(vi, f)) += matAngles(vi, f);
}

template<typename DerivedV, typename DerivedF, typename DerivedvA, typename DerivedvAr, typename DerivedmA>
void Zombie::cal_angles_and_areas(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	Eigen::PlainObjectBase<DerivedvA>& vecAngles,
	Eigen::PlainObjectBase<DerivedvAr>& vecAreas,
	Eigen::PlainObjectBase<DerivedmA>& matAngles)
{
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");
	matAngles.setConstant(DIM, F.cols(), 0);
	vecAngles.setConstant(V.cols(), 0);
	vecAreas.setConstant(V.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const auto& fv = F.col(f);
		//Mix area
		const auto area = (V.col(fv[1]) - V.col(fv[0])).cross(V.col(fv[2]) - V.col(fv[0])).norm() / 2.0;
		for (int vi = 0; vi < DIM; ++vi)
		{
			const auto& p0 = V.col(fv[vi]);
			const auto& p1 = V.col(fv[(vi + 1) % DIM]);
			const auto& p2 = V.col(fv[(vi + 2) % DIM]);
			const auto angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			matAngles(vi, f) = angle;
			//Collect information per vertices
			vecAreas(fv[vi]) += area;
			vecAngles(fv(vi)) += angle;
		}
	}
}

template<typename DerivedL, typename DerivedmA>
void Zombie::cal_angles_and_areas_with_edges(const Eigen::MatrixBase<DerivedL>& L,
	Eigen::PlainObjectBase<DerivedmA>& matAngles)
{
	typedef typename DerivedL::Index Index;
	const int DIM = L.rows();
	assert(DIM == 3 && "L should contain 3 edge length of a triangle");

	matAngles.resize(DIM, L.cols());
	for (Index i = 0; i < L.cols(); ++i)
	{
		auto& fl = L.col(i);
		for (Index j = 0; j < 3; ++j)
		{
			const auto& a = fl[j];
			const auto& b = fl[(j + 1) % DIM];
			const auto& c = fl[(j + 2) % DIM];
			auto cosA = (b * b + c * c - a * a) / (2. * b * c);
			matAngles(j, i) = std::acos(std::max(-1., std::min(1., cosA)));
		}
	}
}

void Zombie::cal_angles_and_areas_with_edges(int Vnum,
	const Eigen::Matrix3Xi& F,
	const Eigen::Matrix3Xd& matLength,
	Eigen::VectorXd& vecAreas,
	Eigen::VectorXd& vecAngles,
	Eigen::Matrix3Xd& matAngles)
{
	const int DIM = F.rows();
	matAngles.setConstant(DIM, F.cols(), 0);
	vecAngles.setConstant(Vnum, 0);
	vecAreas.setConstant(Vnum, 0);

	matAngles.resize(DIM, matLength.cols());
	for (int i = 0; i < matLength.cols(); ++i)
	{
		const auto& fl = matLength.col(i);
		const auto& fv = F.col(i);
		for (int j = 0; j < 3; ++j)
		{
			const auto& a = fl[j];
			const auto& b = fl[(j + 1) % DIM];
			const auto& c = fl[(j + 2) % DIM];
			const auto cosA = (b * b + c * c - a * a) / (2. * b * c);
			const auto angle = acos(std::max(-1., std::min(1., cosA)));
			matAngles(j, i) = angle;
			vecAngles(fv[j]) += angle;
			vecAreas(fv[j]) += sin(angle) * b * c / 2.0;
		}
	}
}
