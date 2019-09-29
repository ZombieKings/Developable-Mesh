// [9/27/2019]
// 内角和的偏微分方程系数矩阵

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <surface_mesh/Surface_mesh.h>

using namespace surface_mesh;

int mesh2mat(const Surface_mesh& mesh, Eigen::MatrixX3f& L);
int cal_diff_mat(const Surface_mesh& mesh, Eigen::SparseMatrix<float>& D, Eigen::VectorXf& b);

int main()
{
	Surface_mesh mesh;
	mesh.read("1.off");

	Surface_mesh target(mesh);
	target.position(Surface_mesh::Vertex(target.n_vertices() / 2)).z += 20;

	Eigen::SparseMatrix<float> D;
	Eigen::VectorXf b;
	cal_diff_mat(mesh, D, b);

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
	Eigen::SparseMatrix<float> tempAT = D.transpose().eval();
	Eigen::SparseMatrix<float> A = (D * tempAT).eval();
	solver.compute(A);

	Eigen::VectorXf tempx;
	tempx = solver.solve(b);
	Eigen::VectorXf result;
	result = (tempAT * tempx).eval();

	std::cout << result << std::endl;
	return 1;
}

int mesh2mat(const Surface_mesh& mesh, Eigen::MatrixX3f& interV)
{
	interV.resize(mesh.n_vertices(), 3);
	for (size_t j = 0; j < mesh.n_vertices(); ++j)
	{
		interV.row(j) = Eigen::Map<const Eigen::RowVector3f>(mesh.position(Surface_mesh::Vertex(j)).data());
	}

	return 1;
}

int cal_diff_mat(const Surface_mesh& mesh, Eigen::SparseMatrix<float>& D, Eigen::VectorXf& b)
{
	std::vector<int> boundidx;
	std::vector<int> interidx;
	Eigen::VectorXi interidx_re;
	interidx_re.resize(mesh.n_vertices());
	interidx_re.setConstant(-1);
	int count = 0;
	for (auto vit : mesh.vertices())
	{
		if (mesh.is_boundary(vit))
		{
			boundidx.push_back(vit.idx());
		}
		else
		{
			interidx.push_back(vit.idx());
			interidx_re(vit.idx()) = count++;
		}
	}

	std::vector<Eigen::Triplet<float>> triple;

	Eigen::VectorXf angles;
	angles.resize(mesh.n_vertices());
	angles.setZero();
	for (auto fit : mesh.faces())
	{
		int i = 0;
		std::vector <Point> p;
		std::vector <int> index;
		for (auto fvit : mesh.vertices(fit))
		{
			index.push_back(fvit.idx());
			p.push_back(mesh.position(fvit));
		}
		Eigen::Vector3f length;
		Eigen::Vector3f angle;
		for (int i = 0; i < 3; ++i)
		{
			length(i) = norm(p[(i + 1) % 3] - p[i]);
			angle(i) = std::acos(std::max(-1.0f, std::min(1.0f, dot((p[(i + 1) % 3] - p[i]).normalize(), (p[(i + 2) % 3] - p[i]).normalize()))));
		}

		for (int i = 0; i < 3; ++i)
		{
			angles(index[i]) += angle(i);
			if (interidx_re(index[(i + 1) % 3]) >= 0)
			{
				Vector<Scalar, 3> v11 = (p[i] - p[(i + 1) % 3]) / (tan(angle(i)) * length(i) * length(i));
				Vector<Scalar, 3> v10 = (p[i] - p[(i + 2) % 3]) / (sin(angle(i)) * length(i) * length((i + 2) % 3)) - v11;
				for (int j = 0; j < 3; ++j)
				{
					if (v11[j])
						triple.push_back(Eigen::Triplet<float>(interidx_re(index[(i + 1) % 3]), index[(i + 1) % 3] * 3 + j, v11[j]));
					if (v10[j])
						triple.push_back(Eigen::Triplet<float>(interidx_re(index[(i + 1) % 3]), index[i] * 3 + j, v10[j]));
				}
			}
			if (interidx_re(index[(i + 2) % 3]) >= 0)
			{
				Vector<Scalar, 3> v22 = (p[i] - p[(i + 2) % 3]) / (tan(angle(i)) * length((i + 2) % 3) * length((i + 2) % 3));
				Vector<Scalar, 3> v20 = (p[i] - p[(i + 1) % 3]) / (sin(angle(i)) * length(i) * length((i + 2) % 3)) - v22;
				for (int j = 0; j < 3; ++j)
				{
					if (v22[j])
						triple.push_back(Eigen::Triplet<float>(interidx_re(index[(i + 2) % 3]), index[(i + 2) % 3] * 3 + j, v22[j]));
					if (v20[j])
						triple.push_back(Eigen::Triplet<float>(interidx_re(index[(i + 2) % 3]), index[i] * 3 + j, v20[j]));
				}
			}
		}
	}

	D.resize(interidx.size(), mesh.n_vertices() * 3);
	b.resize(interidx.size());

	D.setFromTriplets(triple.begin(), triple.end());
	for (size_t i = 0; i < interidx.size(); ++i)
	{
		b(i) = angles(interidx[i]);
	}
	return 1;
}