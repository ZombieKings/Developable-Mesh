#include <math.h>  

//计算所有内部顶点内角和的雅可比矩阵
template<typename DerivedV, typename DerivedF, typename DerivedVType, typename DerivedA, typename T>
void Zombie::cal_jacobian_of_angle(
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::VectorXi& VType,
	const Eigen::MatrixBase<DerivedA>& mAngles,
	Eigen::SparseMatrix<T>& JoA)
{
	typedef Eigen::Triplet<T> Tri;
	typedef Eigen::Matrix<3, 1, T> PosVector;

	std::vector<Tri> triple;
	triple.reserve(F.cols() * 12);
	//高斯曲率的梯度
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//记录当前面信息
		const Eigen::Vector3i& fv = F.col(fit);
		const PosVector& ca = mAngles.col(fit);

		//计算各角及各边长
		PosVector length;
		for (int i = 0; i < 3; ++i)
			length(i) = (V.col(fv[(i + 1) % 3]) - V.col(fv[i])).norm();

		//对一个面片内每个顶点计算相关系数
		for (int i = 0; i < 3; ++i)
		{
			const PosVector& p0 = V.col(fv[i]);
			const PosVector& p1 = V.col(fv[(i + 1) % 3]);
			const PosVector& p2 = V.col(fv[(i + 2) % 3]);

			//判断顶点fv是否为内部顶点，边界顶点不参与计算
			//theta(vp)对vp求偏微分的系数
			PosVector v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
			PosVector v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
			//theta(vq)对vp求偏微分的系数
			PosVector v01 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
			PosVector v02 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
			//系数项
			if (VType(fv[(i + 1) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
					if (v11[j])
						triple.push_back(Tri(fv[(i + 1) % 3] * 3 + j, fv[(i + 1) % 3] * 3 + j, v11[j]));
			//系数项
			if (VType(fv[(i + 2) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
					if (v22[j])
						triple.push_back(Tri(fv[(i + 2) % 3] * 3 + j, fv[(i + 2) % 3] * 3 + j, v22[j]));

			if (VType(fv[i]) != -1)
			{
				for (int j = 0; j < 3; ++j)
				{
					if (v01[j])
						triple.push_back(Tri(fv[(i + 1) % 3] * 3 + j, fv[i] * 3 + j, v01[j]));
					if (v02[j])
						triple.push_back(Tri(fv[(i + 2) % 3] * 3 + j, fv[i] * 3 + j, v02[j]));
				}
			}
		}
	}
	JoA.resize(V.cols() * 3, V.cols() * 3);
	JoA.setFromTriplets(triple.begin(), triple.end());
}
