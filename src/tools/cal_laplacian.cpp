#include "cal_laplacian.h"

template <typename DerivedF, typename DerivedAngle, typename AreasVector, typename IndexVector, typename T>
void Zombie::cal_cot_laplace(const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::MatrixBase<DerivedAngle>& mAngles,
	const AreasVector& areas,
	const IndexVector& interVidx,
	Eigen::SparseMatrix<T>& L)
{
	//计算固定边界的cot权拉普拉斯系数矩阵
	std::vector<Eigen::Triplet<T>> triple;
	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::VectorXi& fv = F.col(i);
		const Eigen::VectorXd& ca = mAngles.col(i);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (interVidx(fv0) != -1)
			{
				const T temp0 = (1.0 / std::tan(ca[(vi + 1) % 3]) + 1.0 / std::tan(ca[(vi + 2) % 3])) / (2.0 * areas(fv0));
				const T temp1 = -1.0 / std::tan(ca[(vi + 2) % 3]) / (2.0 * areas(fv0));
				const T temp2 = -1.0 / std::tan(ca[(vi + 1) % 3]) / (2.0 * areas(fv0));
				for (int j = 0; j < 3; ++j)
				{
					triple.push_back(Eigen::Triplet<T>(fv0 * 3 + j, fv0 * 3 + j, temp0));
					triple.push_back(Eigen::Triplet<T>(fv0 * 3 + j, fv1 * 3 + j, temp1));
					triple.push_back(Eigen::Triplet<T>(fv0 * 3 + j, fv2 * 3 + j, temp2));
				}
			}
		}
	}
	L.resize(areas.size() * 3, areas.size() * 3);
	L.setFromTriplets(triple.begin(), triple.end());
}

template <typename DerivedF, typename IndexVector, typename T>
void Zombie::cal_uni_laplace(const Eigen::MatrixBase<DerivedF>& F,
	int Vnum,
	const IndexVector& interVidx,
	Eigen::SparseMatrix<T>& L)
{
	//计算固定边界的cot权拉普拉斯系数矩阵
	std::vector<Eigen::Triplet<T>> triple;
	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::VectorXi& fv = F.col(i);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (interVidx(fv0) != -1)
			{
				for (int j = 0; j < 3; ++j)
				{
					triple.push_back(Eigen::Triplet<T>(fv0 * 3 + j, fv0 * 3 + j, 1.0));
					triple.push_back(Eigen::Triplet<T>(fv0 * 3 + j, fv1 * 3 + j, -0.5));
					triple.push_back(Eigen::Triplet<T>(fv0 * 3 + j, fv2 * 3 + j, -0.5));
				}
			}
		}
	}
	L.resize(Vnum * 3, Vnum * 3);
	L.setFromTriplets(triple.begin(), triple.end());
}

