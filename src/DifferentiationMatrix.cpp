// [12/17/2019]
// 所有内部顶点内角和的雅可比矩阵

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <surface_mesh/Surface_mesh.h>

typedef double DataType;
typedef Eigen::Triplet<DataType> Tri;
typedef Eigen::Vector3d PosVector;
typedef Eigen::VectorXd VectorType;
typedef Eigen::Matrix3Xd MatrixType;
typedef Eigen::SparseMatrix<DataType> SparseMatrixType;
typedef const Eigen::Matrix3Xd MatrixTypeConst;

std::vector<int> interV_;
std::vector<int> boundV_;
Eigen::VectorXi interVidx_;
Eigen::Matrix3Xi matF_;
MatrixType matV_;

VectorType vecAngles_;
MatrixType matAngles_;
VectorType areas_;
using namespace surface_mesh;

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F, std::vector<std::vector<int>>& Adj)
{
	F.resize(3, mesh.n_faces());
	V.resize(3, mesh.n_vertices());

	Eigen::VectorXi flag;
	flag.setConstant(mesh.n_vertices(), 0);
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
				V.col(fvit.idx()) = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data()).cast<DataType>();
				flag(fvit.idx()) = 1;
			}
		}
	}

	for (auto vit : mesh.vertices())
	{
		std::vector<int> adj_idx;
		for (auto vvit : mesh.vertices(vit))
		{
			adj_idx.push_back(vvit.idx());
		}
		Adj.push_back(adj_idx);
	}
}

void cal_angles(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, VectorType& vecAngles)
{
	vecAngles.setConstant(V.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			if (interVidx(fv[vi]) != -1)
			{
				const PosVector& p0 = V.col(fv[vi]);
				const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
				const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
				const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
				vecAngles(fv[vi]) += angle;
			}
		}
	}
}

void cal_angles_and_areas(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixType& matAngles, VectorType& vecAngles, VectorType& areas)
{
	matAngles.setConstant(3, F.cols(), 0);
	vecAngles.setConstant(V.cols(), 0);
	areas.setConstant(V.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		//Mix area
		double area = (V.col(fv[1]) - V.col(fv[0])).cross(V.col(fv[2]) - V.col(fv[0])).norm() / 6.0f;

		for (size_t vi = 0; vi < 3; ++vi)
		{
			const PosVector& p0 = V.col(fv[vi]);
			const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
			const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			matAngles(vi, f) = angle;
			if (interVidx(fv(vi)) != -1)
			{
				areas(fv[vi]) += area;
				vecAngles(fv(vi)) += angle;
			}
		}
	}
}

//计算所有内部顶点内角和的雅可比矩阵
void cal_anglesum_Jacobian(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, SparseMatrixType& mGradient)
{
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
		{
			length(i) = (V.col(fv[(i + 1) % 3]) - V.col(fv[i])).norm();
		}

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
			if (interVidx(fv[(i + 1) % 3]) != -1)
			{
				for (int j = 0; j < 3; ++j)
				{
					if (v11[j])
						triple.push_back(Tri(fv[(i + 1) % 3] * 3 + j, fv[(i + 1) % 3] * 3 + j, v11[j]));
				}
			}
			//系数项
			if (interVidx(fv[(i + 2) % 3]) != -1)
			{
				for (int j = 0; j < 3; ++j)
				{
					if (v22[j])
						triple.push_back(Tri(fv[(i + 2) % 3] * 3 + j, fv[(i + 2) % 3] * 3 + j, v22[j]));
				}
			}

			if (interVidx(fv[i]) != -1)
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
	mGradient.resize(V.cols() * 3, V.cols() * 3);
	mGradient.setFromTriplets(triple.begin(), triple.end());
}

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Load failed!" << std::endl;
	}

	//收集内部顶点下标
	interVidx_.setConstant(mesh.n_vertices() + 1, -1);
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

	//网格初始信息收集
	std::vector<std::vector<int>> adj;
	mesh2matrix(mesh, matV_, matF_, adj);
	cal_angles_and_areas(matV_, matF_, interVidx_, matAngles_, vecAngles_, areas_);
	SparseMatrixType G;
	cal_anglesum_Jacobian(matV_, matF_, interVidx_, matAngles_, G);
	VectorType vecG;
	vecG.setConstant(matV_.cols() * 3, 0);
	for (int i = 0; i < G.rows(); ++i)
	{
		vecG(i) = G.row(i).sum();
	}
	const double step = 1e-4;
	const double sum = vecAngles_.sum();
	VectorType numDG;
	numDG.setConstant(matV_.cols() * 3, 0);
	for (int i = 0; i < matV_.cols(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			VectorType temp;
			matV_(j, i) += step;
			cal_angles(matV_, matF_, interVidx_, temp);
			numDG(i * 3 + j) = (temp.sum() - sum) / step;
			matV_(j, i) -= step;
		}
	}

	//for(int i =0;i<vecG.size();++i)
	//	std::cout << vecG(i) << " " << numDG(i) << std::endl;

	for (int i = 0; i < vecG.size(); ++i)
		std::cout << abs(vecG(i) - numDG(i)) << std::endl;
	return 1;
}

