#include <iostream>
#include <map>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include <surface_mesh/Surface_mesh.h>

#include "myVisualizer.h"

using namespace surface_mesh;

int mesh2mat(const Surface_mesh& mesh, Eigen::MatrixX3f &L);
int cal_cot_laplace(const Surface_mesh& mesh, Eigen::SparseMatrix<float> &L);
int UpdateMesh(const Surface_mesh& source, const Surface_mesh& target, Eigen::MatrixX3f& result);


int main()
{
	Surface_mesh mesh;
	mesh.read("test10.off");

	Surface_mesh target(mesh);
	target.position(Surface_mesh::Vertex(target.n_vertices() / 2)).z += 20;

	Eigen::MatrixX3f resultM;
	UpdateMesh(mesh, target, resultM);

	//update mesh 
	auto points = mesh.get_vertex_property<Point>("v:point");
	for (auto vit : mesh.vertices())
	{
		points[vit] = Point(resultM(vit.idx(), 0), resultM(vit.idx(), 1), resultM(vit.idx(), 2));
	}

	myVisualizer MV;
	MV.LoadMesh(mesh);
	MV.Run();
	return 1;
}

int mesh2mat(const Surface_mesh& mesh, Eigen::MatrixX3f &interV)
{
	interV.resize(mesh.n_vertices(), 3);
	for (size_t j = 0; j < mesh.n_vertices(); ++j)
	{
		interV.row(j) = Eigen::Map<const Eigen::RowVector3f>(mesh.position(Surface_mesh::Vertex(j)).data());
	}

	return 1;
}

int UpdateMesh(const Surface_mesh& source, const Surface_mesh& target, Eigen::MatrixX3f& result)
{
	Eigen::SparseMatrix<float, Eigen::RowMajor> A;
	Eigen::MatrixX3f b;

	Eigen::SparseMatrix<float> L;
	cal_cot_laplace(source, L);

	Eigen::MatrixX3f matS;
	mesh2mat(source, matS);
	Eigen::MatrixX3f matT;
	mesh2mat(target, matT);

	for (size_t i = 0; i < source.n_vertices(); ++i)
	{
		if (source.is_boundary(Surface_mesh::Vertex(i)))
		{
			L.row(i) *= 0;
			L.coeffRef(i, i) = 1;
		}
		else
		{
			matS.row(i).setZero();
		}
	}

	A.resize(matT.rows() * 2, matT.rows());
	A.topRows(matT.rows()) = L;
	for (size_t i = 0; i < matT.rows(); ++i)
	{
		A.coeffRef(i + matT.rows(), i) = 1;
	}
	A.makeCompressed();

	b.resize(matT.rows() * 2, 3);
	b.setZero();
	b.topRows(matT.rows()) = matS;
	b.bottomRows(matT.rows()) = matT;

	Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> solver;
	solver.compute(A);
	result = solver.solve(b);

	return 1;
}

int cal_cot_laplace(const Surface_mesh& mesh, Eigen::SparseMatrix<float> &L)
{
	std::vector<Eigen::Triplet<float>> triple;

	Eigen::VectorXf areas;
	areas.resize(mesh.n_vertices());
	areas.setZero();
	double sum_area = 0;
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

		//Mix area
		float area = norm(cross((p[1] - p[0]), (p[2] - p[0]))) / 6.0f;

		//Cot
		Eigen::Vector3f angle;
		for (int i = 0; i < 3; ++i)
		{
			angle(i) = std::acos(std::max(-1.0f, std::min(1.0f, dot((p[(i + 1) % 3] - p[i]).normalize(), (p[(i + 2) % 3] - p[i]).normalize()))));
		}

		for (int i = 0; i < 3; ++i)
		{
			areas(index[i]) += area;
			triple.push_back(Eigen::Triplet<float>(index[i], index[i], 1.0f / tan(angle[(i + 1) % 3]) + 1.0f / tan(angle[(i + 2) % 3])));
			triple.push_back(Eigen::Triplet<float>(index[i], index[(i + 2) % 3], -1.0f / tan(angle[(i + 1) % 3])));
			triple.push_back(Eigen::Triplet<float>(index[i], index[(i + 1) % 3], -1.0f / tan(angle[(i + 2) % 3])));
		}
	}

	int nInter = 0;
	Eigen::VectorXf mark;
	mark.resize(mesh.n_vertices());
	mark.setZero();
	for (size_t i = 0; i < mesh.n_vertices(); ++i)
	{
		if (!mesh.is_boundary(Surface_mesh::Vertex(i)))
		{
			mark(i) = 1;
			++nInter;
		}
	}
	sum_area = areas.dot(mark) / float(nInter);

	L.resize(mesh.n_vertices(), mesh.n_vertices());
	L.setFromTriplets(triple.begin(), triple.end());

	for (int r = 0; r < L.cols(); ++r)
	{
		L.row(r) *= sum_area / (2.0f * areas(r));
	}

	return 1;
}
