#include "cal_laplacian.h"
#include "cal_angles_areas.h"

template <typename DerivedV, typename DerivedF, typename T>
void Zombie::cal_cot_laplace(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	const int RowsPerV,
	Eigen::SparseMatrix<T>& L)
{
	typedef typename DerivedF::Index Index;
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");
	assert((RowsPerV == 1 || RowsPerV == DIM) && "One row per vertices or one row per dimension!");
	const int Vnum = V.cols();

	//Calculate angles and areas.
	Eigen::Matrix<T, Eigen::Dynamic, 1>vA;
	Eigen::Matrix<T, Eigen::Dynamic, 1>vAr;
	Eigen::Matrix<T, DerivedF::RowsAtCompileTime, DerivedF::ColsAtCompileTime>mA;
	cal_angles_and_areas(V, F, vA, vAr, mA);

	//Construct cotangent-weighted Laplace operator matrix.
	cal_cot_laplace(Vnum, F, mA, vAr, RowsPerV, L);
}

template <typename DerivedF, typename DerivedA, typename DerivedAr, typename T>
void Zombie::cal_cot_laplace(int Vnum,
	const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::MatrixBase<DerivedA>& matAngles,
	const Eigen::MatrixBase<DerivedAr>& vecAreas,
	const int RowsPerV,
	Eigen::SparseMatrix<T>& L)
{
	typedef typename DerivedF::Index Index;
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");
	assert((RowsPerV == 1 || RowsPerV == DIM) && "One row per vertices or one row per dimension!");

	//Construct cotangent-weighted Laplace operator matrix.
	L.resize(Vnum * RowsPerV, Vnum * RowsPerV);
	L.reserve(10 * Vnum * RowsPerV);
	std::vector<Eigen::Triplet<T>> triple;
	triple.reserve(10 * Vnum);
	for (Index i = 0; i < F.cols(); ++i)
	{
		const auto& fv = F.col(i);
		const auto& ca = matAngles.col(i);
		for (size_t vi = 0; vi < DIM; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % DIM];
			const int fv2 = fv[(vi + 2) % DIM];
			const T coeff1 = -1. / std::tan(ca[(vi + 2) % DIM]) / 2. / vecAreas(fv[vi]);
			const T coeff2 = -1. / std::tan(ca[(vi + 1) % DIM]) / 2. / vecAreas(fv[vi]);
			const T coeff0 = -coeff1 - coeff2;
			for (int j = 0; j < RowsPerV; ++j)
			{
				triple.push_back(Eigen::Triplet<T>(fv0 * RowsPerV + j, fv0 * RowsPerV + j, coeff0));
				triple.push_back(Eigen::Triplet<T>(fv0 * RowsPerV + j, fv1 * RowsPerV + j, coeff1));
				triple.push_back(Eigen::Triplet<T>(fv0 * RowsPerV + j, fv2 * RowsPerV + j, coeff2));
			}
		}
	}
	L.setFromTriplets(triple.begin(), triple.end());
}


template <typename DerivedF, typename T>
void Zombie::cal_uni_laplace(int Vnum,
	const Eigen::MatrixBase<DerivedF>& F,
	const int RowsPerV,
	Eigen::SparseMatrix<T>& L)
{
	typedef typename DerivedF::Index Index;
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");
	assert((RowsPerV == 1 || RowsPerV == DIM) && "One row per vertices or one row per dimension!");

	//Construct uniform-weighted Laplace operator matrix.
	L.resize(Vnum * RowsPerV, Vnum * RowsPerV);
	std::vector<Eigen::Triplet<T>> triple;
	triple.reserve(10 * Vnum);
	for (Index i = 0; i < F.cols(); ++i)
	{
		const auto& fv = F.col(i);
		for (size_t vi = 0; vi < DIM; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % DIM];
			const int fv2 = fv[(vi + 2) % DIM];
			for (int j = 0; j < RowsPerV; ++j)
			{
				triple.push_back(Eigen::Triplet<T>(fv0 * RowsPerV + j, fv0 * RowsPerV + j, 1.0));
				triple.push_back(Eigen::Triplet<T>(fv0 * RowsPerV + j, fv1 * RowsPerV + j, -0.5));
				triple.push_back(Eigen::Triplet<T>(fv0 * RowsPerV + j, fv2 * RowsPerV + j, -0.5));
			}
		}
	}
	L.setFromTriplets(triple.begin(), triple.end());
}

