#include "func_opt.h"

using namespace std;

namespace func_opt
{
	my_function::my_function(const surface_mesh::Surface_mesh& mesh, double eps, double w1, double w2, double w3)
		:eps_(eps),w1_(w1),w2_(w2),w3_(w3)
	{
		interVidx_.resize(mesh.n_vertices() + 1);
		memset(interVidx_.data(), -1, sizeof(int) * interVidx_.size());
		int count = 0;
		for (const auto& vit : mesh.vertices())
		{
			if (!mesh.is_boundary(vit))
			{
				interV_.push_back(vit.idx());
				interVidx_(vit.idx()) = count++;
			}
			else
			{
				boundV_.push_back(vit.idx());
			}
		}
		interVidx_(mesh.n_vertices()) = count;

		Vnum_ = mesh.n_vertices();
		mesh2matrix(mesh, V_, F_);	

		makeF2V(F_,Vnum_, F2V_);
		F2Vt_ = F2V_.transpose();

		cal_grad(V_, F_, Grad_);
		preGradX_.resize(F_.cols() * 3, 1);
		preGradX_ = F2V_ * Grad_ * (V_.row(0).transpose());

		preGradY_.resize(F_.cols() * 3, 1);
		preGradY_ = F2V_ * Grad_ * (V_.row(1).transpose());

		preGradZ_.resize(F_.cols() * 3, 1);
		preGradZ_ = F2V_ * Grad_ * (V_.row(2).transpose());
	}

	size_t my_function::dim(void) const
	{
		return V_.cols();
	}

	int my_function::val(const double* x, double& v)
	{
		v = 0;
		MatrixType curV = Eigen::Map<MatrixTypeConst>(x, 3, Vnum_);
		cal_angles_and_areas(curV, F_, boundV_, mAngles_, vAngles_, areas_);

		//高斯曲率1范数
		for (size_t i = 0; i < interV_.size(); ++i)
		{
			const double k = vAngles_(interV_[i]) - 2.0 * M_PI;
			v += w1_ * sqrt(k * k + eps_);
		}

		//梯度坐标差
		cal_grad(curV, F_, Grad_);
		Eigen::VectorXd gradX(F2V_ * Grad_ * (curV.row(0).transpose()));
		Eigen::VectorXd gradY(F2V_ * Grad_ * (curV.row(1).transpose()));
		Eigen::VectorXd gradZ(F2V_ * Grad_ * (curV.row(2).transpose()));
		//gradX.resize(F_.cols * 3, 1);
		//gradX = Grad_ * (curV.row(0).transpose());
		//gradY.resize(F_.cols * 3, 1);
		//gradY = Grad_ * (curV.row(1).transpose());
		//gradZ.resize(F_.cols * 3, 1);
		//gradZ = Grad_ * (curV.row(2).transpose());

		v += w2_ * (gradX - preGradX_).squaredNorm();
		v += w2_ * (gradY - preGradY_).squaredNorm();
		v += w2_ * (gradZ - preGradZ_).squaredNorm();

		//边界项
		for (int i = 0; i < boundV_.size(); ++i)
		{
			const PosVector temp = w3_ * (curV.col(boundV_[i]) - V_.col(boundV_[i]));
			v += temp.squaredNorm();
		}

		preGradX_ = gradX;
		preGradY_ = gradY;
		preGradZ_ = gradZ;
		return 0;
	}

	int my_function::gra(const double* x, double* g)
	{
		MatrixType curV = Eigen::Map<MatrixTypeConst>(x, 3, Vnum_);
		cal_angles_and_areas(curV, F_, boundV_, mAngles_, vAngles_, areas_);

		Eigen::Map<VectorType> Gradient(g, Vnum_ * 3, 1);

		//高斯曲率1范数的梯度
		cal_gaussian_gradient(curV, F_, interVidx_, mAngles_, vAngles_, Gau_);
		for (int i = 0; i < Gau_.outerSize(); ++i)
		{
			for (Eigen::SparseMatrix<DataType>::InnerIterator it(Gau_, i); it; ++it)
			{
				Gradient(i) += w1_ * it.value();
			}
		}

		//梯度域坐标的梯度
		cal_grad(curV, F_, Grad_);
		Gradt_ = Grad_.transpose();
		VectorType gradX(F2V_ * Grad_ * (curV.row(0).transpose()));
		VectorType gradY(F2V_ * Grad_ * (curV.row(1).transpose()));
		VectorType gradZ(F2V_ * Grad_ * (curV.row(2).transpose()));
		VectorType tempX = Gradt_ * F2Vt_ * (gradX - preGradX_);
		VectorType tempY = Gradt_ * F2Vt_ * (gradY - preGradY_);
		VectorType tempZ = Gradt_ * F2Vt_ * (gradZ - preGradZ_);

		for (int i = 0; i < curV.cols(); ++i)
		{
			Gradient(i * 3) += 2.0 * w2_ * tempX(i);
			Gradient(i * 3 + 1) += 2.0 * w2_ * tempY(i);
			Gradient(i * 3 + 2) += 2.0 * w2_ * tempZ(i);
		}

		//边界项
		for (int i = 0; i < boundV_.size(); ++i)
		{
			const PosVector temp = 2 * w3_ * (curV.col(boundV_[i]) - V_.col(boundV_[i]));
			for (int k = 0; k < 3; ++k)
				Gradient(boundV_[i] * 3 + k) += temp(k);
		}

		preGradX_ = gradX;
		preGradY_ = gradY;
		preGradZ_ = gradZ;
		return 0;
	}

	int my_function::val_gra(const double* x, double& v, double* g)
	{
		MatrixType curV = Eigen::Map<MatrixTypeConst>(x, 3, Vnum_);
		cal_angles_and_areas(curV, F_, boundV_, mAngles_, vAngles_, areas_);

		Eigen::Map<Eigen::VectorXd> Gradient(g, Vnum_ * 3, 1);
		//高斯曲率1范数
		for (size_t i = 0; i < interV_.size(); ++i)
		{
			const double k = vAngles_(interV_[i]) - 2.0 * M_PI;
			v += w1_ * sqrt(k * k + eps_);
		}

		cal_gaussian_gradient(curV, F_, interVidx_, mAngles_, vAngles_, Gau_);
		for (int i = 0; i < Gau_.outerSize(); ++i)
		{
			for (Eigen::SparseMatrix<DataType>::InnerIterator it(Gau_, i); it; ++it)
			{
				Gradient(i) += w1_ * it.value();
			}
		}

		//梯度坐标差
		cal_grad(curV, F_, Grad_);
		Gradt_ = Grad_.transpose();

		Eigen::VectorXd gradX(F2V_ * Grad_ * (curV.row(0).transpose()));
		Eigen::VectorXd gradY(F2V_ * Grad_ * (curV.row(1).transpose()));
		Eigen::VectorXd gradZ(F2V_ * Grad_ * (curV.row(2).transpose()));

		v += w2_ * (gradX - preGradX_).squaredNorm();
		v += w2_ * (gradY - preGradY_).squaredNorm();
		v += w2_ * (gradZ - preGradZ_).squaredNorm();

		Eigen::VectorXd tempX = Gradt_ * F2Vt_ * (gradX - preGradX_);
		Eigen::VectorXd tempY = Gradt_ * F2Vt_ * (gradY - preGradY_);
		Eigen::VectorXd tempZ = Gradt_ * F2Vt_ * (gradZ - preGradZ_);

		//Eigen::SparseMatrix<DataType> Grad,Gradt;
		//cal_grad(curV, F_, Grad);
		//Gradt = Grad.transpose();

		//Eigen::VectorXd gradX(Grad * (curV.row(0).transpose()));
		//Eigen::VectorXd gradY(Grad * (curV.row(1).transpose()));
		//Eigen::VectorXd gradZ(Grad * (curV.row(2).transpose()));

		//v += w2_ * (gradX - preGradX_).squaredNorm();
		//v += w2_ * (gradY - preGradY_).squaredNorm();
		//v += w2_ * (gradZ - preGradZ_).squaredNorm();

		//Eigen::VectorXd tempX = 2 * w2_ * Gradt * (gradX - preGradX_);
		//Eigen::VectorXd tempY = 2 * w2_ * Gradt * (gradY - preGradY_);
		//Eigen::VectorXd tempZ = 2 * w2_ * Gradt * (gradZ - preGradZ_);

		for (int i = 0; i < curV.cols(); ++i)
		{
			Gradient(i * 3) += 2 * w2_ * tempX(i);
			Gradient(i * 3 + 1) += 2 * w2_ * tempY(i);
			Gradient(i * 3 + 2) += 2 * w2_ * tempZ(i);
		}

		//边界项
		for (int i = 0; i < boundV_.size(); ++i)
		{
			const PosVector temp = w3_* (curV.col(boundV_[i]) - V_.col(boundV_[i]));
			v += temp.squaredNorm();
			for (int k = 0; k < 3; ++k)
				Gradient(boundV_[i] * 3 + k) += 2 * temp(k);
		}

		preGradX_ = gradX;
		preGradY_ = gradY;
		preGradZ_ = gradZ;
		std::cout << Gradient;
		return 0;
	}

	int my_function::hes(const double* x, Eigen::SparseMatrix<double>& h)
	{
		std::vector<Tri> tripletH;
		Eigen::SparseMatrix<DataType> Gaut = Gau_.transpose();
		Eigen::SparseMatrix<DataType> GautGau = Gaut * Gau_;
		h += GautGau;

		Eigen::SparseMatrix<DataType> GtG = Gradt_ * Grad_;
		h += 2 * w2_ * GtG;

		Eigen::SparseMatrix<DataType> I;
		I.resize(V_.cols(), V_.cols());
		I.setIdentity();
		h += 2 * w3_ * I;

		h.resize(V_.cols(), V_.cols());

		for (size_t i = 0; i < interV_.size(); ++i)
		{
			const double k = vAngles_(interV_[i]) - 2.0 * M_PI;
			const double k2e = k * k + eps_;
			const double temp = w1_ * eps_ / sqrt(k2e * k2e * k2e);
			Gau_.row(interV_[i]) *= temp;
		}

		//if (h.nonZeros() == 0)
		//{
		//	std::vector<Tri> triplets;
		//	size_t num_triplets(L_.nonZeros()), num_non_const(0);
		//	for (int k = 0; k < L_.outerSize(); ++k)
		//	{
		//		for (Eigen::SparseMatrix<double>::InnerIterator it(L_, k); it; ++it)
		//		{
		//			const double val = 2 * w2_ * it.value();
		//			for (size_t j = 0; j < 3; ++j)
		//			{
		//				triplets.push_back(Eigen::Triplet<double>(3 * k + j, it.row() * 3 + j, val));
		//			}
		//		}
		//		for (size_t j = 0; j < 3; ++j)
		//		{
		//			triplets.push_back(Eigen::Triplet<double>(3 * k + j, 3 * k + j, 2 * w3_));
		//		}
		//	}
		//	h.resize(Vnum_ * 3, Vnum_ * 3);
		//	h.setFromTriplets(triplets.begin(), triplets.end());
		//}
		//else
		//{
		//	for (int i = 0; i < Gau_.rows(); ++i)
		//	{
		//		const double k = vAngles_(interV_[i]) - 2.0 * M_PI;
		//		const double k2e = k * k + eps_;
		//		const double temp = w1_ * eps_ / sqrt(k2e * k2e * k2e);
		//		const Eigen::MatrixXd H = temp * Gau_.row(i) * Gau_.row(i).transpose();
		//		h += H;
		//	}
		//}
		return 0;
	}

	void my_function::mesh2matrix(const surface_mesh::Surface_mesh & mesh, MatrixType & V, Eigen::Matrix3Xi & F)
	{
		F.resize(3, mesh.n_faces());
		V.resize(3, mesh.n_vertices());

		Eigen::VectorXi flag;
		flag.resize(mesh.n_vertices());
		flag.setZero();
		for (auto fit : mesh.faces())
		{
			int i = 0;
			for (auto fvit : mesh.vertices(fit))
			{
				//save faces informations
				F(i++, fit.idx()) = fvit.idx();
				//save vertices informations
				if (!flag(fvit.idx()))
				{
					const Eigen::Vector3f& temp = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data());
					V.col(fvit.idx()) = temp.cast<double>();
					flag(fvit.idx()) = 1;
				}
			}
		}
	}


	void my_function::makeF2V(const Eigen::Matrix3Xi& F, int Vnum, Eigen::SparseMatrix<DataType>& F2V)
	{
		std::vector<Tri> tripleF;
		for (int i = 0; i < F.cols(); ++i)
		{
			const Eigen::Vector3i& fv = F.col(i);

			for (int j = 0; j < 3; ++j)
			{
				for (int k = 0; k < 3; ++k)
				{
					tripleF.push_back(Tri(fv[j] * 3 + k, i * 3 + k, 1));
				}
			}
		}
		F2V.resize(Vnum * 3, F.cols() * 3);
		F2V.setFromTriplets(tripleF.begin(), tripleF.end());
	}

	void my_function::cal_angles_and_areas(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const std::vector<int>& boundIdx, MatrixType& matAngles, VectorType& vecAngles, VectorType& areas)
	{
		matAngles.resize(3, F.cols());
		matAngles.setZero();
		vecAngles.resize(V.cols());
		vecAngles.setZero();
		areas.resize(V.cols());
		areas.setZero();
		for (int f = 0; f < F.cols(); ++f)
		{
			const Eigen::Vector3i& fv = F.col(f);

			//Mix area
			double area = (V.col(fv[1]) - V.col(fv[0])).cross(V.col(fv[2]) - V.col(fv[0])).norm() / 6.0;

			for (size_t vi = 0; vi < 3; ++vi)
			{
				areas(fv[vi]) += area;
				const PosVector& p0 = V.col(fv[vi]);
				const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
				const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
				const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
				matAngles(vi, f) = angle;
				vecAngles(F(vi, f)) += angle;
			}
		}
	}

	void my_function::cal_grad(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& G)
	{
		std::vector<Tri> tripleG;
		tripleG.reserve(F.cols() * 3 * 4);
		for (int i = 0; i < F.cols(); ++i)
		{
			const Eigen::Vector3i& fv = F.col(i);

			//三角形各边向量
			const PosVector v21 = V.col(fv[2]) - V.col(fv[1]);
			const PosVector v02 = V.col(fv[0]) - V.col(fv[2]);
			const PosVector v10 = V.col(fv[1]) - V.col(fv[0]);
			const PosVector n = v21.cross(v02);
			const double dblA = n.norm();
			PosVector u = n / dblA;

			PosVector B10 = u.cross(v10);
			B10.normalize();
			B10 *= v10.norm() / dblA;
			//PosVector B10 = n.normalized().cross(v10).normalized() * v10.norm() / dblA;
			for (int j = 0; j < 3; ++j)
			{
				tripleG.push_back(Tri(i * 3 + j, fv[1], B10(j)));
				tripleG.push_back(Tri(i * 3 + j, fv[0], -B10(j)));
			}

			PosVector B02 = u.cross(v02);
			B02.normalize();
			B02 *= v02.norm() / dblA;
			//PosVector B02 = n.normalized().cross(v02).normalized() * v02.norm() / dblA;
			for (int j = 0; j < 3; ++j)
			{
				tripleG.push_back(Tri(i * 3 + j, fv[2], B02(j)));
				tripleG.push_back(Tri(i * 3 + j, fv[0], -B02(j)));
			}
		}
		G.resize(F.cols() * 3, V.cols());
		G.setFromTriplets(tripleG.begin(), tripleG.end());
	}

	//void my_function::cal_grad_pos(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& gradXpos, MatrixType& gradYpos, MatrixType& gradZpos)
	//{
	//	cal_grad(V, F, Grad_);
	//	Gradt_ = Grad_.transpose();

	//	Eigen::VectorXd gradX(F2V_ * Grad_ * (V.row(0).transpose()));
	//	Eigen::VectorXd gradY(F2V_ * Grad_ * (V.row(1).transpose()));
	//	Eigen::VectorXd gradZ(F2V_ * Grad_ * (V.row(2).transpose()));

	//	v += w2_ * (gradX - preGradX_).squaredNorm();
	//	v += w2_ * (gradY - preGradY_).squaredNorm();
	//	v += w2_ * (gradZ - preGradZ_).squaredNorm();

	//	Eigen::VectorXd tempX = 2 * w2_ * Gradt_ * (gradX - preGradX_);
	//	Eigen::VectorXd tempY = 2 * w2_ * Gradt_ * (gradY - preGradY_);
	//	Eigen::VectorXd tempZ = 2 * w2_ * Gradt_ * (gradZ - preGradZ_);
	//}

	void my_function::cal_cot_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& A, const Eigen::VectorXi& interVidx, Eigen::SparseMatrix<DataType>& L)
	{
		//计算固定边界的cot权拉普拉斯系数矩阵
		std::vector<Tri> triple;
		triple.reserve(F.cols() * 3 * 3);

		VectorType areas;
		areas.resize(interVidx(V.cols()));
		areas.setZero();
		for (int j = 0; j < F.cols(); ++j)
		{
			const Eigen::Vector3i& fv = F.col(j);
			const PosVector& ca = A.col(j);

			//Mix area
			const PosVector& p0 = V.col(fv[0]);
			const PosVector& p1 = V.col(fv[1]);
			const PosVector& p2 = V.col(fv[2]);
			const DataType area = ((p1 - p0).cross(p2 - p0)).norm() / 2.0;

			for (size_t vi = 0; vi < 3; ++vi)
			{
				const int fv0 = fv[vi];
				const int fv1 = fv[(vi + 1) % 3];
				const int fv2 = fv[(vi + 2) % 3];
				if (interVidx(fv0) != -1)
					areas(interVidx(fv0)) += area / 3.0;
				triple.push_back(Tri(interVidx(fv0), fv0, 1.0 / std::tan(ca[(vi + 1) % 3]) + 1.0 / std::tan(ca[(vi + 2) % 3])));
				triple.push_back(Tri(interVidx(fv0), fv1, -1.0 / std::tan(ca[(vi + 2) % 3])));
				triple.push_back(Tri(interVidx(fv0), fv2, -1.0 / std::tan(ca[(vi + 1) % 3])));
			}
		}

		L.resize(interVidx(V.cols()), V.cols());
		L.setFromTriplets(triple.begin(), triple.end());

		for (int k = 0; k < L.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<DataType>::InnerIterator it(L, k); it; ++it)
			{
				if (interVidx(it.index()) != -1)
					it.valueRef() /= (2.0 * areas(it.index()));
			}
		}
	}

	void my_function::cal_uni_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& L)
	{
		std::vector<Tri> tripleL;
		tripleL.reserve(F.cols() * 9);
		for (int j = 0; j < F.cols(); ++j)
		{
			const Eigen::Vector3i& fv = F.col(j);
			for (size_t vi = 0; vi < 3; ++vi)
			{
				const PosVector& p0 = V.col(fv[vi]);
				const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
				const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
				tripleL.push_back(Tri(fv[vi], fv[vi], 1));
				tripleL.push_back(Tri(fv[vi], fv[(vi + 1) % 3], -0.5f));
				tripleL.push_back(Tri(fv[vi], fv[(vi + 2) % 3], -0.5f));
			}
		}
		L.resize(V.cols(), V.cols());
		L.setFromTriplets(tripleL.begin(), tripleL.end());
	}

	//void my_function::cal_gaussian_gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const VectorType& interVidx, const MatrixType& mAngles, const VectorType& vAngles, Eigen::VectorXd& grad)
	//{
	//	//高斯曲率1范数的梯度
	//	for (int fit = 0; fit < F_.cols(); ++fit)
	//	{
	//		//记录当前面信息
	//		const Eigen::Vector3i& fv = F_.col(fit);
	//		const Eigen::Vector3d& ca = mAngles.col(fit);
	//		Eigen::Matrix3d p;
	//		for (int i = 0; i < 3; ++i)
	//			p.col(i) = V.col(fv[i]);
	//
	//		//计算各角及各边长
	//		Eigen::Vector3d length;
	//		for (int i = 0; i < 3; ++i)
	//		{
	//			length(i) = (p.col((i + 1) % 3) - p.col(i)).norm();
	//		}
	//
	//		//对每个顶点计算相关系数
	//		for (int i = 0; i < 3; ++i)
	//		{
	//			//Mix area
	//			const Eigen::Vector3d& p0 = p.col(i);
	//			const Eigen::Vector3d& p1 = p.col((i + 1) % 3);
	//			const Eigen::Vector3d& p2 = p.col((i + 2) % 3);
	//
	//			//判断顶点fv是否为内部顶点，边界顶点不参与计算
	//			if (interVidx_(fv[(i + 1) % 3]) != -1)
	//			{
	//				//对vp求偏微分的系数
	//				Eigen::Vector3d v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
	//				//对vq求偏微分的系数
	//				Eigen::Vector3d v10 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
	//				//系数项
	//				double K = vAngles(fv[(i + 1) % 3]) - 2.0 * M_PI;
	//				double coeffK = K / sqrtf(K * K + eps_);
	//				for (int j = 0; j < 3; ++j)
	//				{
	//					if (v11[j]) grad(fv[(i + 1) % 3] * 3 + j) += w1_ * coeffK * v11[j];
	//					if (v10[j]) grad(fv[i] * 3 + j) += w1_ * coeffK * v10[j];
	//				}
	//			}
	//
	//			if (interVidx_(fv[(i + 2) % 3]) != -1)
	//			{
	//				//对vp求偏微分的系数
	//				Eigen::Vector3d v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
	//				//对vq求偏微分的系数
	//				Eigen::Vector3d v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
	//				//系数项
	//				double K = vAngles(fv[(i + 2) % 3]) - 2.0 * M_PI;
	//				double coeffK = K / sqrtf(K * K + eps_);
	//				for (int j = 0; j < 3; ++j)
	//				{
	//					if (v22[j]) grad(fv[(i + 2) % 3] * 3 + j) += w1_ * coeffK * v22[j];
	//					if (v20[j]) grad(fv[i] * 3 + j) += w1_ * coeffK * v20[j];
	//				}
	//			}
	//		}
	//	}
	//}
	//
	void my_function::cal_gaussian_gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, const VectorType& vAngles, Eigen::SparseMatrix<DataType>& mGradient)
	{
		std::vector<Tri> triple;
		//高斯曲率1范数的梯度
		for (int fit = 0; fit < F_.cols(); ++fit)
		{
			//记录当前面信息
			const Eigen::Vector3i& fv = F_.col(fit);
			const Eigen::Vector3d& ca = mAngles.col(fit);
			Eigen::Matrix3d p;
			for (int i = 0; i < 3; ++i)
				p.col(i) = V.col(fv[i]);

			//计算各角及各边长
			Eigen::Vector3d length;
			for (int i = 0; i < 3; ++i)
			{
				length(i) = (p.col((i + 1) % 3) - p.col(i)).norm();
			}

			//对每个顶点计算相关系数
			for (int i = 0; i < 3; ++i)
			{
				//Mix area
				const Eigen::Vector3d& p0 = p.col(i);
				const Eigen::Vector3d& p1 = p.col((i + 1) % 3);
				const Eigen::Vector3d& p2 = p.col((i + 2) % 3);

				//判断顶点fv是否为内部顶点，边界顶点不参与计算
				if (interVidx(fv[(i + 1) % 3]) != -1)
				{
					//对vp求偏微分的系数
					Eigen::Vector3d v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
					//对vq求偏微分的系数
					Eigen::Vector3d v10 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
					//系数项
					const double K = vAngles(fv[(i + 1) % 3]) - 2.0 * M_PI;
					const double coeffK = K / sqrtf(K * K + eps_);
					for (int j = 0; j < 3; ++j)
					{
						if (v11[j])
							triple.push_back(Tri((fv[(i + 1) % 3]) * 3 + j, fv[(i + 1) % 3] * 3 + j, coeffK * v11[j]));
						if (v10[j])
							triple.push_back(Tri((fv[(i + 1) % 3]) * 3 + j, fv[i] * 3 + j, coeffK * v10[j]));
					}
				}

				if (interVidx(fv[(i + 2) % 3]) != -1)
				{
					//对vp求偏微分的系数
					Eigen::Vector3d v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
					//对vq求偏微分的系数
					Eigen::Vector3d v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
					//系数项
					const double K = vAngles(fv[(i + 2) % 3]) - 2.0 * M_PI;
					const double coeffK = K / sqrtf(K * K + eps_);
					for (int j = 0; j < 3; ++j)
					{
						if (v22[j])
							triple.push_back(Tri((fv[(i + 2) % 3]) * 3 + j, fv[(i + 2) % 3] * 3 + j, coeffK * v22[j]));
						if (v20[j])
							triple.push_back(Tri((fv[(i + 2) % 3]) * 3 + j, fv[i] * 3 + j, coeffK * v20[j]));
					}
				}
			}
		}
		mGradient.resize((Vnum_ * 3), Vnum_ * 3);
		mGradient.setFromTriplets(triple.begin(), triple.end());
	}

	double gra_err(function& f, double* x)
	{
		return 0;
	}

	// assume grad is accurate
	double hes_err(function& f, double* x)
	{
		return 0;
	}
}
