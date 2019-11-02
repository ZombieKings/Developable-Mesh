#include "func_opt.h"

using namespace std;

namespace func_opt
{
	my_function::my_function(const surface_mesh::Surface_mesh& mesh, double eps, double w1, double w2, double w3)
		:topology_(mesh), eps_(eps), w1_(w1), w2_(w2), w3_(w3)
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

		//cal_topo(F_, Vnum_, F2V_, V2V_);
		//F2Vt_ = F2V_.transpose();

		//Eigen::SparseMatrix<DataType> FGrad;
		//cal_face_grad(V_, F_, FGrad);
		//Grad_ = F2V_ * FGrad;

		//preGradX_.resize(F_.cols() * 3, 1);
		//preGradX_ = Grad_ * (V_.row(0).transpose());

		//preGradY_.resize(F_.cols() * 3, 1);
		//preGradY_ = Grad_ * (V_.row(1).transpose());

		//preGradZ_.resize(F_.cols() * 3, 1);
		//preGradZ_ = Grad_ * (V_.row(2).transpose());
	}

	size_t my_function::dim(void) const
	{
		return Vnum_ * 3;
	}

	int my_function::val(const double* x, double& v)
	{
		v = 0;
		MatrixType curV = Eigen::Map<MatrixTypeConst>(x, 3, Vnum_);
		cal_angles_and_areas(curV, F_, boundV_, mAngles_, vAngles_, areas_);

		//��˹����1����
		for (size_t i = 0; i < interV_.size(); ++i)
		{
			const double K(vAngles_(interV_[i]) - 2.0 * M_PI);
			const double temp = normtype_ ? sqrt(K * K + eps_) : K * K / (K * K + eps_);
			v += w1_ * temp;
		}

		////�ݶ������
		//Eigen::SparseMatrix<DataType> FGrad;
		//cal_face_grad(curV, F_, FGrad);
		//Grad_ = F2V_ * FGrad;
		//Eigen::VectorXd gradX(Grad_ * (curV.row(0).transpose()));
		//Eigen::VectorXd gradY(Grad_ * (curV.row(1).transpose()));
		//Eigen::VectorXd gradZ(Grad_ * (curV.row(2).transpose()));

		//v += w2_ * (gradX - preGradX_).squaredNorm();
		//v += w2_ * (gradY - preGradY_).squaredNorm();
		//v += w2_ * (gradZ - preGradZ_).squaredNorm();

		//�߽���
		for (int i = 0; i < boundV_.size(); ++i)
		{
			const PosVector temp = w3_ * (curV.col(boundV_[i]) - V_.col(boundV_[i]));
			v += temp.squaredNorm();
		}

		//preGradX_ = gradX;
		//preGradY_ = gradY;
		//preGradZ_ = gradZ;
		return 0;
	}

	int my_function::gra(const double* x, double* g)
	{
		MatrixType curV = Eigen::Map<MatrixTypeConst>(x, 3, Vnum_);
		cal_angles_and_areas(curV, F_, boundV_, mAngles_, vAngles_, areas_);

		Eigen::Map<VectorType> Gradient(g, Vnum_ * 3, 1);
		Gradient.setZero();

		//��˹����1�������ݶ�
		cal_gaussian_gradient(curV, F_, interVidx_, mAngles_, vAngles_, Gau_);
		for (int i = 0; i < Gau_.outerSize(); ++i)
		{
			const double K(vAngles_(i) - 2.0 * M_PI);
			const double K2e(K * K + eps_);
			const double temp = normtype_ ? K / sqrt(K2e) : 2 * K * eps_ / (K2e * K2e);
			for (Eigen::SparseMatrix<DataType>::InnerIterator it(Gau_, i); it; ++it)
			{
				Gradient(it.row()) += w1_ * temp * it.value();
			}
		}

		////�ݶ���������ݶ�
		//Eigen::SparseMatrix<DataType> FGrad;
		//cal_face_grad(curV, F_, FGrad);
		//Grad_ = F2V_ * FGrad;
		//Gradt_ = Grad_.transpose();
		//GtG_ = Gradt_ * Grad_;
		//VectorType gradX(Grad_ * (curV.row(0).transpose()));
		//VectorType gradY(Grad_ * (curV.row(1).transpose()));
		//VectorType gradZ(Grad_ * (curV.row(2).transpose()));
		//VectorType tempX = Gradt_ * (gradX - preGradX_);
		//VectorType tempY = Gradt_ * (gradY - preGradY_);
		//VectorType tempZ = Gradt_ * (gradZ - preGradZ_);

		//for (int i = 0; i < curV.cols(); ++i)
		//{
		//	Gradient(i * 3) += 2.0 * w2_ * tempX(i);
		//	Gradient(i * 3 + 1) += 2.0 * w2_ * tempY(i);
		//	Gradient(i * 3 + 2) += 2.0 * w2_ * tempZ(i);
		//}

		//�߽���
		for (int i = 0; i < boundV_.size(); ++i)
		{
			const PosVector temp = 2 * w3_ * (curV.col(boundV_[i]) - V_.col(boundV_[i]));
			for (int k = 0; k < 3; ++k)
				Gradient(boundV_[i] * 3 + k) += temp(k);
		}

		//preGradX_ = gradX;
		//preGradY_ = gradY;
		//preGradZ_ = gradZ;
		return 0;
	}

	int my_function::val_gra(const double* x, double& v, double* g)
	{
		MatrixType curV = Eigen::Map<MatrixTypeConst>(x, 3, Vnum_);
		cal_angles_and_areas(curV, F_, boundV_, mAngles_, vAngles_, areas_);

		//��˹����1����
		for (size_t i = 0; i < interV_.size(); ++i)
		{
			const double K(vAngles_(interV_[i]) - 2.0 * M_PI);
			const double temp = normtype_ ? sqrt(K * K + eps_) : K * K / (K * K + eps_);
			v += w1_ * temp;
		}

		////�ݶ������
		//Eigen::SparseMatrix<DataType> FGrad;
		//cal_face_grad(curV, F_, FGrad);
		//Grad_ = F2V_ * FGrad;
		//std::cout << Grad_ << std::endl;
		//Gradt_ = Grad_.transpose();
		//GtG_ = Gradt_ * Grad_;
		//std::cout << GtG_ << std::endl;

		//Eigen::VectorXd gradX(Grad_ * (curV.row(0).transpose()));
		//Eigen::VectorXd gradY(Grad_ * (curV.row(1).transpose()));
		//Eigen::VectorXd gradZ(Grad_ * (curV.row(2).transpose()));
		//v += w2_ * (gradX - preGradX_).squaredNorm();
		//v += w2_ * (gradY - preGradY_).squaredNorm();
		//v += w2_ * (gradZ - preGradZ_).squaredNorm();

		Eigen::Map<Eigen::VectorXd> Gradient(g, Vnum_ * 3, 1);
		Gradient.setZero();

		//��˹����1�������ݶ�
		cal_gaussian_gradient(curV, F_, interVidx_, mAngles_, vAngles_, Gau_);
		//std::cout << Gau_ << std::endl;

		for (int i = 0; i < Gau_.outerSize(); ++i)
		{
			const double K(vAngles_(i / 3) - 2.0 * M_PI);
			const double K2e(K * K + eps_);
			const double temp = normtype_ ? K / sqrt(K2e) : 2 * K * eps_ / (K2e * K2e);
			for (Eigen::SparseMatrix<DataType>::InnerIterator it(Gau_, i); it; ++it)
			{
				Gradient(it.row()) += w1_ * temp * it.value();
			}
		}

		////�ݶ��������ݶ�
		//Eigen::VectorXd tempX = Gradt_ * (gradX - preGradX_);
		//Eigen::VectorXd tempY = Gradt_ * (gradY - preGradY_);
		//Eigen::VectorXd tempZ = Gradt_ * (gradZ - preGradZ_);
		//for (int i = 0; i < curV.cols(); ++i)
		//{
		//	Gradient(i * 3) += 2 * w2_ * tempX(i);
		//	Gradient(i * 3 + 1) += 2 * w2_ * tempY(i);
		//	Gradient(i * 3 + 2) += 2 * w2_ * tempZ(i);
		//}

		//�߽���
		for (int i = 0; i < boundV_.size(); ++i)
		{
			const PosVector temp = (curV.col(boundV_[i]) - V_.col(boundV_[i]));
			v += w3_ * temp.squaredNorm();
			for (int k = 0; k < 3; ++k)
				Gradient(boundV_[i] * 3 + k) += 2 * w3_ * temp(k);
		}

		//preGradX_ = gradX;
		//preGradY_ = gradY;
		//preGradZ_ = gradZ;
		//std::cout << Gradient << std::endl;
		return 0;
	}

	int my_function::hes(const double* x, Eigen::SparseMatrix<double>& h)
	{
		//{
		//	std::vector<Tri> tripletH;
		//	for (size_t i = 0; i < interV_.size(); i++)
		//	{
		//		const double K = vAngles_(interV_[i]) - 2.0 * M_PI;
		//		const double K2e = (K * K + eps_);
		//		const double temp = normtype_ ? eps_ / sqrt(K2e * K2e * K2e) : 2 * eps_ * (eps_ - 3 * K * K) / (K2e * K2e * K2e);

		//		const VectorType& gc(Gau_.col(interV_[i]));
		//		Eigen::MatrixXd H = w1_ * temp * gc * gc.transpose();

		//		for (int j = 0; j < H.cols(); ++j)
		//		{
		//			for (int k = 0; k < H.rows(); ++k)
		//			{
		//				if (H(k, j) != 0)
		//				{
		//					tripletH.push_back(Tri(k, j, H(k, j)));
		//				}
		//			}
		//		}
		//	}

		//	for (int i = 0; i < GtG_.outerSize(); ++i)
		//	{
		//		for (Eigen::SparseMatrix<DataType>::InnerIterator it(GtG_, i); it; ++it)
		//		{
		//			tripletH.push_back(Tri(it.row(), it.col(), 2 * w2_ * it.value()));
		//		}
		//	}

		//	for (size_t i = 0; i < boundV_.size(); ++i)
		//	{
		//		tripletH.push_back(Tri(i * 3, i * 3, 2 * w3_));
		//		tripletH.push_back(Tri(i * 3 + 1, i * 3 + 1, 2 * w3_));
		//		tripletH.push_back(Tri(i * 3 + 2, i * 3 + 2, 2 * w3_));
		//	}

		//	h.resize(dim(), dim());
		//	h.setFromTriplets(tripletH.begin(), tripletH.end());
		//	//std::cout << h << std::endl;
		//}

		if (h.nonZeros() == 0)
		{
			std::vector<Eigen::Triplet<double>> triplets;
			size_t num_triplets(GtG_.nonZeros() + boundV_.size() * 3), num_non_const(9 * degrees_.squaredNorm());
			num_triplets += num_non_const;

			triplets.reserve(num_triplets);
			indices_.resize(topology_.n_vertices());
			surface_mesh::Surface_mesh::Vertex_around_vertex_circulator vv_it, vv_end;
			for (auto vi : topology_.vertices())
			{
				const int vidx = vi.idx();
				if (interVidx_[vidx] == -1)
				{
					const size_t i3 = vidx * 3;
					for (size_t j = 0; j < 3; ++j)
					{
						triplets.push_back(Eigen::Triplet<double>(i3 + j, i3 + j, 2 * w3_));
					}
				}
				else
				{
					std::vector<size_t>& index = indices_[vidx];
					index.reserve(topology_.valence(vi) + 1);
					index.push_back(vi.idx());
					vv_it = vv_end = topology_.vertices(vi);
					do
					{
						index.push_back((*vv_it).idx());
					} while (++vv_it != vv_end);

					for (auto i : index)
					{
						size_t i3 = i * 3;
						for (auto j : index)
						{
							size_t j3 = i * 3;
							for (size_t ki = 0; ki < 3; ++ki)
							{
								for (size_t kj = 0; kj < 3; ++kj)
								{
									triplets.push_back(Eigen::Triplet<double>(i3 + ki, j3 + kj, 0));
									triplets.push_back(Eigen::Triplet<double>(i3 + ki, j3 + kj, 0));
								}
							}
						}
					}
				}
			}

			h.resize(dim(), dim());
			h.setFromTriplets(triplets.begin(), triplets.end());

			ch_ptr_cache_.reserve(boundV_.size() * 3);
			int ch_count(0);
			for (auto vi : topology_.vertices())
			{
				if (interVidx_[vi.idx()] == -1)
				{
					int i3 = vi.idx() * 3;
					for (size_t j = 0; j < 3; ++j)
					{
						ch_ptr_cache_.push_back(&h.coeffRef(i3 + j, i3 + j));
					}
				}
			}

			nch_ptr_cache_.reserve(degrees_.sum() * 3 * degrees_.sum() * 3);
			for (auto vidx : interV_)
			{
				std::vector<size_t>& index = indices_[vidx];
				for (auto i : index)
				{
					size_t i3 = i * 3;
					for (auto j : index)
					{
						size_t j3 = i * 3;
						for (size_t ki = 0; ki < 3; ++ki)
						{
							for (size_t kj = 0; kj < 3; ++kj)
							{
								nch_ptr_cache_.push_back(&h.coeffRef(i3 + ki, j3 + kj));
							}
						}
					}
				}

			}
		}
		else
		{
			for (size_t i = 0; i < ch_ptr_cache_.size(); ++i)
			{
				*ch_ptr_cache_[i] = 2 * w3_;
			}

			int nch_count = 0;
			for (size_t i = 0; i < interV_.size(); i++)
			{
				const double K = vAngles_(interV_[i]) - 2.0 * M_PI;
				const double K2e = (K * K + eps_);
				const double temp = normtype_ ? eps_ / sqrt(K2e * K2e * K2e) : 2 * eps_ * (eps_ - 3 * K * K) / (K2e * K2e * K2e);

				const VectorType& gc(Gau_.col(interV_[i]));
				Eigen::MatrixXd H = w1_ * temp * gc * gc.transpose();
				std::vector<size_t>& index = indices_[interV_[i]];
				for (auto i : index)
				{
					size_t i3 = i * 3;
					for (auto j : index)
					{
						size_t j3 = i * 3;
						for (size_t ki = 0; ki < 3; ++ki)
						{
							for (size_t kj = 0; kj < 3; ++kj)
							{
								*nch_ptr_cache_[nch_count++] += H(i3 + ki, j3 + kj);
							}
						}
					}
				}
			}

			//for (int i = 0; i < GtG_.outerSize(); ++i)
			//{
			//	for (Eigen::SparseMatrix<DataType>::InnerIterator it(GtG_, i); it; ++it)
			//	{
			//		h.coeffRef(it.row(), it.col()) += 2 * w2_ * it.value();
			//	}
			//}
		}

		//if (h.nonZeros() == 0)
		//{
		//	std::vector<Eigen::Triplet<double>> triplets;
		//	size_t num_triplets(GtG_.nonZeros() + boundV_.size() * 3), num_non_const(9 * degrees_.squaredNorm());
		//	num_triplets += num_non_const;
		//
		//	triplets.reserve(num_triplets);		
		//	indices_.resize(topology_.n_vertices());
		//	surface_mesh::Surface_mesh::Vertex_around_vertex_circulator vv_it, vv_end;
		//	for (auto vi : topology_.vertices())
		//	{
		//		const int vidx = vi.idx();
		//		if (interVidx_[vidx] == -1)
		//		{
		//			const size_t i3 = vidx * 3;
		//			for (size_t j = 0; j < 3; ++j)
		//			{
		//				triplets.push_back(Eigen::Triplet<double>(i3 + j, i3 + j, 2 * w3_));
		//			}
		//		}
		//		else
		//		{
		//			std::vector<size_t>& index = indices_[vidx];
		//			index.reserve(topology_.valence(vi) + 1);
		//			index.push_back(vi.idx());
		//			vv_it = vv_end = topology_.vertices(vi);
		//			do
		//			{
		//				index.push_back((*vv_it).idx());
		//			} while (++vv_it != vv_end);
		//
		//			for (auto i : index)
		//			{
		//				size_t i3 = i * 3;
		//				for (auto j : index)
		//				{
		//					size_t j3 = j * 3;
		//					for (size_t ki = 0; ki < 3; ++ki)
		//					{
		//						for (size_t kj = 0; kj < 3; ++kj)
		//						{
		//							triplets.push_back(Eigen::Triplet<double>(i3 + ki, j3 + kj, 0));
		//						}
		//					}
		//				}
		//			}
		//		}
		//	}
		//
		//	h.resize(dim(), dim());
		//	h.setFromTriplets(triplets.begin(), triplets.end());
		//
		//	ch_ptr_cache_.reserve(boundV_.size() * 3);		
		//	nch_ptr_cache_.reserve(num_non_const);
		//	int ch_count(0);
		//	for (auto vi : topology_.vertices())
		//	{
		//		if (interVidx_[vi.idx()] != -1)
		//		{
		//			std::vector<size_t>& index = indices_[vi.idx()];
		//			for (auto i : index)
		//			{
		//				size_t i3 = i * 3;
		//				for (auto j : index)
		//				{
		//					size_t j3 = j * 3;
		//					for (size_t ki = 0; ki < 3; ++ki)
		//					{
		//						for (size_t kj = 0; kj < 3; ++kj)
		//						{
		//							nch_ptr_cache_.push_back(&h.coeffRef(i3 + ki, j3 + kj));
		//						}
		//					}
		//				}
		//			}
		//		}
		//		else
		//		{
		//			int i3 = vi.idx() * 3;
		//			for (size_t j = 0; j < 3; ++j)
		//			{
		//				ch_ptr_cache_.push_back(&h.coeffRef(i3 + j, i3 + j));
		//			}
		//		}
		//	}
		//}
		//else
		//{
		//	for (size_t i = 0; i < ch_ptr_cache_.size(); ++i)
		//	{
		//		*ch_ptr_cache_[i] = 2 * w3_;
		//	}
		//	size_t ptr(0);
		//	for (auto vi : topology_.vertices())
		//	{
		//		const size_t vidx = vi.idx();
		//		if (interVidx_[vi.idx()] != -1)
		//		{
		//			const double K = vAngles_(vidx) - 2.0 * M_PI;
		//			const double K2e = (K * K + eps_);
		//			const double temp = normtype_ ? eps_ / sqrt(K2e * K2e * K2e) : 2 * eps_ * (eps_ - 3 * K * K) / (K2e * K2e * K2e);
		//
		//			const VectorType& gc(Gau_.col(vidx));
		//			Eigen::MatrixXd H = w1_ * temp * gc * gc.transpose();
		//
		//			std::vector<size_t>& index = indices_[vi.idx()];
		//			int n = index.size();
		//			for (size_t i = 0; i < n; ++i)
		//			{
		//				const size_t i3 = index[i] * 3;
		//				for (size_t j = 0; j < n; ++j)
		//				{
		//					const size_t j3 = index[j] * 3;
		//					for (size_t ki = 0; ki < 3; ++ki)
		//					{
		//						for (size_t kj = 0; kj < 3; ++kj)
		//						{
		//							*nch_ptr_cache_[ptr] += H(i3 + ki, j3 + kj);
		//							*nch_ptr_cache_[ptr++] += 2 * w2_ * GtG_.coeff(index[i], index[j]);
		//						}
		//					}
		//				}
		//			}
		//		}
		//	}
		//}
		return 0;
	}

	void my_function::mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F)
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

	void my_function::cal_topo(const Eigen::Matrix3Xi& F, int Vnum, Eigen::SparseMatrix<DataType>& F2V, Eigen::SparseMatrix<bool>& V2V)
	{
		std::vector<Tri> tripleF;
		std::vector<Eigen::Triplet<bool>> tripleV;
		for (int i = 0; i < F.cols(); ++i)
		{
			const Eigen::Vector3i& fv = F.col(i);

			for (int j = 0; j < 3; ++j)
			{
				for (int k = 0; k < 3; ++k)
				{
					if (interVidx_(fv[j]) != -1)
					{
						tripleF.push_back(Tri(fv[j] * 3 + k, i * 3 + k, 1));
					}
				}
				tripleV.push_back(Eigen::Triplet<bool>(fv[j], fv[j], true));
				tripleV.push_back(Eigen::Triplet<bool>(fv[j], fv[(j + 1) % 3], true));
				tripleV.push_back(Eigen::Triplet<bool>(fv[j], fv[(j + 2) % 3], true));
			}
		}
		F2V.resize(Vnum * 3, F.cols() * 3);
		F2V.setFromTriplets(tripleF.begin(), tripleF.end());
		V2V.resize(Vnum, Vnum);
		V2V.setFromTriplets(tripleV.begin(), tripleV.end());

		degrees_.resize(Vnum);
		degrees_.setZero();
		for (int k = 0, i = 0; k < V2V.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<bool>::InnerIterator it(V2V, k); it; ++it)
			{
				degrees_(it.row()) += 1;
			}
		}
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

	void my_function::cal_face_grad(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& G)
	{
		std::vector<Tri> tripleG;
		tripleG.reserve(F.cols() * 3 * 4);
		for (int i = 0; i < F.cols(); ++i)
		{
			const Eigen::Vector3i& fv = F.col(i);

			//�����θ�������
			const PosVector v21 = V.col(fv[2]) - V.col(fv[1]);
			const PosVector v02 = V.col(fv[0]) - V.col(fv[2]);
			const PosVector v10 = V.col(fv[1]) - V.col(fv[0]);
			const PosVector n = v21.cross(v02);
			const double dblA = n.norm();
			PosVector u = n / dblA;

			PosVector B10 = u.cross(v10);
			B10.normalize();
			B10 *= v10.norm() / dblA;
			for (int j = 0; j < 3; ++j)
			{
				tripleG.push_back(Tri(i * 3 + j, fv[1], B10(j)));
				tripleG.push_back(Tri(i * 3 + j, fv[0], -B10(j)));
			}

			PosVector B02 = u.cross(v02);
			B02.normalize();
			B02 *= v02.norm() / dblA;
			for (int j = 0; j < 3; ++j)
			{
				tripleG.push_back(Tri(i * 3 + j, fv[2], B02(j)));
				tripleG.push_back(Tri(i * 3 + j, fv[0], -B02(j)));
			}
		}
		G.resize(F.cols() * 3, V.cols());
		G.setFromTriplets(tripleG.begin(), tripleG.end());
	}

	void my_function::cal_cot_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& A, const Eigen::VectorXi& interVidx, Eigen::SparseMatrix<DataType>& L)
	{
		//����̶��߽��cotȨ������˹ϵ������
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

	void my_function::cal_gaussian_gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, const VectorType& vAngles, Eigen::SparseMatrix<DataType>& mGradient)
	{
		std::vector<Tri> triple;
		//��˹����1�������ݶ�
		for (int fit = 0; fit < F_.cols(); ++fit)
		{
			//��¼��ǰ����Ϣ
			const Eigen::Vector3i& fv = F_.col(fit);
			const PosVector& ca = mAngles.col(fit);

			//������Ǽ����߳�
			PosVector length;
			for (int i = 0; i < 3; ++i)
			{
				length(i) = (V.col(fv[(i + 1) % 3]) - V.col(fv[i])).norm();
			}

			//��ÿ������������ϵ��
			for (int i = 0; i < 3; ++i)
			{
				//Mix area
				const PosVector& p0 = V.col(fv[i]);
				const PosVector& p1 = V.col(fv[(i + 1) % 3]);
				const PosVector& p2 = V.col(fv[(i + 2) % 3]);

				//�ж϶���fv�Ƿ�Ϊ�ڲ����㣬�߽綥�㲻�������
				if (interVidx(fv[(i + 1) % 3]) != -1)
				{
					//��vp��ƫ΢�ֵ�ϵ��
					PosVector v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
					//��vq��ƫ΢�ֵ�ϵ��
					PosVector v10 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
					//ϵ����
					for (int j = 0; j < 3; ++j)
					{
						triple.push_back(Tri(fv[(i + 1) % 3] * 3 + j, (fv[(i + 1) % 3]), v11[j]));
						triple.push_back(Tri(fv[i] * 3 + j, (fv[(i + 1) % 3]), v10[j]));
					}
				}

				if (interVidx(fv[(i + 2) % 3]) != -1)
				{
					//��vp��ƫ΢�ֵ�ϵ��
					PosVector v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
					//��vq��ƫ΢�ֵ�ϵ��
					PosVector v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
					//ϵ����
					for (int j = 0; j < 3; ++j)
					{
						triple.push_back(Tri(fv[(i + 2) % 3] * 3 + j, (fv[(i + 2) % 3]), v22[j]));
						triple.push_back(Tri(fv[i] * 3 + j, (fv[(i + 2) % 3]), v20[j]));
					}
				}
			}
		}
		mGradient.resize(Vnum_ * 3, Vnum_ * 3);
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