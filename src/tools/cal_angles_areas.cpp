#include "cal_angles_areas.h"
#include "cal_edge_length.h"

#include <math.h>  

template<typename DerivedV, typename DerivedF, typename DerivedA>
void Zombie::cal_angles(
	const Eigen::MatrixBase<DerivedV>& V,
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
void Zombie::cal_angles(
	const Eigen::MatrixBase<DerivedV>& V,
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
void Zombie::cal_angles_and_areas(
	const Eigen::MatrixBase<DerivedV>& V,
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
		const auto area = (V.col(fv[1]) - V.col(fv[0])).cross(V.col(fv[2]) - V.col(fv[0])).norm() / 6.0;
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
void Zombie::cal_angles_and_areas_with_edges(
	const Eigen::MatrixBase<DerivedL>& L,
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

template<typename Derivedl, typename DerivedF, typename DerivedvAr>
void Zombie::cal_mixed_areas(
	int Vnum,
	const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::MatrixBase<Derivedl>& l,
	Eigen::PlainObjectBase<DerivedvAr>& vecAreas)
{
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");
	assert(l.cols() == DIM && "Invalid angels input!");
	vecAreas.setConstant(Vnum, 0);

	typedef typename Derivedl::Scalar Scalar;
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> A;
	cal_face_areas(l.transpose(), A);

	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> ll = l;
	ll.array().pow(2);

	// http://www.alecjacobson.com/weblog/?p=874
	// compute cosines for every angels in every faces.
	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> cosines(F.cols(), 3);
	cosines.col(0) =
		(ll.col(2).array() + ll.col(1).array() - ll.col(0).array()) / (l.col(1).array() * l.col(2).array() * 2.0);
	cosines.col(1) =
		(ll.col(0).array() + ll.col(2).array() - ll.col(1).array()) / (l.col(2).array() * l.col(0).array() * 2.0);
	cosines.col(2) =
		(ll.col(1).array() + ll.col(0).array() - ll.col(2).array()) / (l.col(0).array() * l.col(1).array() * 2.0);
	cosines.array() = (abs(cosines.array()) >= 1e-10).select(cosines.array(), 0);

	// compute barycentric for every faces.
	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> barycentric = cosines.array() * l.array();
	// nomalize barycentric
	barycentric = (barycentric.array().colwise() / barycentric.rowwise().sum().array()).eval();
	// compute partial areas
	barycentric.array().colwise() *= A.array();

	// compute partial areas for every vertices in face
	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> quads(barycentric.rows(), barycentric.cols());
	for (int i = 0; i < 3; ++i)
		quads.col(i) = (barycentric.col((i + 1) % 3) + barycentric.col((i + 2) % 3)) * 0.5;

	for (int i = 0; i < cosines.rows(); ++i)
	{
		// check obtuse angle
		for (int j = 0; j < 3; ++j)
		{
			if (cosines(i, j) < 0)
			{
				quads(i, j) = 0.5 * A(i);
				quads(i, (j + 1) % 3) = 0.25 * A(i);
				quads(i, (j + 2) % 3) = 0.25 * A(i);
			}
		}
		// add final partial area to every vertices.
		for (int j = 0; j < 3; ++j)
			vecAreas(F(j, i)) += quads(i, j);
	}
}

template <typename DerivedV, typename DerivedF, typename DerivedAr>
void Zombie::cal_face_areas(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	Eigen::PlainObjectBase<DerivedAr>& Ar)
{
	const int dim = F.rows();
	assert(dim == 3 && "Only support triangles");
	const size_t m = F.cols();

	// Projected area helper
	const auto& proj_doublearea =
		[&V, &F](const int x, const int y, const int f)
		->typename DerivedV::Scalar
	{
		auto rx = V(x, F(0, f)) - V(x, F(2, f));
		auto sx = V(x, F(1, f)) - V(x, F(2, f));
		auto ry = V(y, F(0, f)) - V(y, F(2, f));
		auto sy = V(y, F(1, f)) - V(y, F(2, f));
		return rx * sy - ry * sx;
	};

	Ar = DerivedAr::Zero(m, 1);
	for (size_t f = 0; f < m; f++)
	{
		for (int d = 0; d < 3; d++)
		{
			const auto dblAd = proj_doublearea(d, (d + 1) % 3, f);
			Ar(f) += dblAd * dblAd;
		}
	}
	Ar = Ar.array().sqrt().eval() * .5;
}

template <typename Derivedl, typename DerivedAr>
void Zombie::cal_face_areas(
	const Eigen::MatrixBase<Derivedl>& l,
	Eigen::PlainObjectBase<DerivedAr>& Ar)
{
	typedef typename Derivedl::Scalar Scalar;
	const int dim = l.rows();
	assert(dim == 3 && "Only support triangles");
	const size_t m = l.cols();

	Ar.resize(m, 1);
	for (int i = 0; i < m; ++i)
	{
		const Scalar arg =
			(l(0, i) + (l(1, i) + l(2, i))) *
			(l(2, i) - (l(0, i) - l(1, i))) *
			(l(2, i) + (l(0, i) - l(1, i))) *
			(l(0, i) + (l(1, i) - l(2, i)));
		Ar(i) = 0.25 * sqrt(arg);
		assert(Ar(i) == Ar(i) && "DOUBLEAREA() PRODUCED NaN");
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
			vecAreas(fv[j]) += sin(angle) * b * c / 6.0;
		}
	}
}
