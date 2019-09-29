#include <iostream>
#include <map>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include <surface_mesh/Surface_mesh.h>

using namespace surface_mesh;

Eigen::VectorXf update_d_;
std::vector<int> interV;
std::vector<int> boundV;
Eigen::VectorXi interVidx;
int mesh2mat(const Surface_mesh& mesh, Eigen::MatrixX3f& L);
void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& vertices_mat, Eigen::Matrix3Xi& faces_mat);

int cal_cot_laplace(const Surface_mesh& mesh, Eigen::SparseMatrix<float>& L);
int UpdateMesh(const Surface_mesh& source, const Surface_mesh& target, Eigen::MatrixX3f& result);
void cal_angles(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xf& A);
void cal_laplace(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& A, Eigen::SparseMatrix<float>& L, Eigen::MatrixX3f& b);

int main()
{
	Surface_mesh mesh;
	mesh.read("2.off");

	//收集内部顶点下标
	interV.clear();
	interVidx.resize(mesh.n_vertices());
	interVidx.setOnes();
	interVidx *= -1;
	int count = 0;
	for (const auto& vit : mesh.vertices())
	{
		if (!mesh.is_boundary(vit))
		{
			interV.push_back(vit.idx());
			interVidx(vit.idx()) = count++;
		}
		else
		{
			boundV.push_back(vit.idx());
		}
	}

	Surface_mesh target(mesh);
	target.position(Surface_mesh::Vertex(target.n_vertices() / 2)).z += 20;

	update_d_.resize(mesh.n_vertices() * 3);
	update_d_.setZero();
	update_d_(mesh.n_vertices() / 2 * 3 + 2) = 20;

	Eigen::MatrixX3f resultM;
	UpdateMesh(mesh, target, resultM);

	//update mesh
	auto points = mesh.get_vertex_property<Point>("v:point");
	for (auto vit : mesh.vertices())
	{
		points[vit] = Point(resultM(vit.idx(), 0), resultM(vit.idx(), 1), resultM(vit.idx(), 2));
	}

	return 1;
}

int mesh2mat(const Surface_mesh& mesh, Eigen::MatrixX3f& V)
{
	V.resize(mesh.n_vertices(), 3);
	for (size_t j = 0; j < mesh.n_vertices(); ++j)
	{
		V.row(j) = Eigen::Map<const Eigen::RowVector3f>(mesh.position(Surface_mesh::Vertex(j)).data());
	}

	return 1;
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& vertices_mat, Eigen::Matrix3Xi& faces_mat)
{
	faces_mat.resize(3, mesh.n_faces());
	vertices_mat.resize(3, mesh.n_vertices());

	Eigen::VectorXi flag;
	flag.resize(mesh.n_vertices());
	flag.setZero();
	for (auto fit : mesh.faces())
	{
		int i = 0;
		for (auto fvit : mesh.vertices(fit))
		{
			//save faces informations
			faces_mat(i++, fit.idx()) = fvit.idx();
			//save vertices informations
			if (!flag(fvit.idx()))
			{
				vertices_mat.col(fvit.idx()) = Eigen::Map<const Eigen::Vector3f>(mesh.position(Surface_mesh::Vertex(fvit.idx())).data());
				flag(fvit.idx()) = 1;
			}
		}
	}
}

int UpdateMesh(const Surface_mesh& source, const Surface_mesh& target, Eigen::MatrixX3f& result)
{
	//Eigen::SparseMatrix<float, Eigen::RowMajor> A;
	//Eigen::MatrixX3f b;

	//Eigen::SparseMatrix<float> L;
	//cal_cot_laplace(source, L);
	Eigen::Matrix3Xf V;
	Eigen::Matrix3Xi F;
	Eigen::Matrix3Xf Angle;
	mesh2matrix(source, V, F);
	cal_angles(V, F, Angle);
	//Eigen::MatrixX3f matT;
	//mesh2mat(target, matT);

	Eigen::MatrixX3f b;
	Eigen::SparseMatrix<float> L;
	cal_laplace(V, F, Angle, L, b);

	//for (size_t i = 0; i < source.n_vertices(); ++i)
	//{
	//	if (source.is_boundary(Surface_mesh::Vertex(i)))
	//	{
	//		L.row(i) *= 0;
	//		L.coeffRef(i, i) = 1;
	//	}
	//	else
	//	{
	//		matS.row(i).setZero();
	//	}
	//}

	//A.resize(matT.rows() * 2, matT.rows());
	//A.topRows(matT.rows()) = L;
	//for (size_t i = 0; i < matT.rows(); ++i)
	//{
	//	A.coeffRef(i + matT.rows(), i) = 1;
	//}
	//A.makeCompressed();

	//b.resize(matT.rows() * 2, 3);
	//b.setZero();
	//b.topRows(matT.rows()) = matS;
	//b.bottomRows(matT.rows()) = matT;

	//Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> solver;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
	solver.compute((L.transpose() * L).eval());
	result = solver.solve(L.transpose() * b);

	return 1;
}

void cal_angles(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xf& A)
{
	A.resize(3, F.cols());
	for (int f = 0; f < F.cols(); ++f) {
		const Eigen::Vector3i& fv = F.col(f);
		for (size_t vi = 0; vi < 3; ++vi) {
			const Eigen::VectorXf& p0 = V.col(fv[vi]);
			const Eigen::VectorXf& p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::VectorXf& p2 = V.col(fv[(vi + 2) % 3]);
			const float angle = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			A(vi, f) = angle;
		}
	}
}

int cal_cot_laplace(const Surface_mesh& mesh, Eigen::SparseMatrix<float>& L)
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

void cal_laplace(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& A, Eigen::SparseMatrix<float>& L, Eigen::MatrixX3f& b)
{
	//计算固定边界的拉普拉斯系数矩阵
	std::vector<Eigen::Triplet<float>> triple;

	Eigen::VectorXf areas;
	areas.resize(V.cols());
	areas.setZero();
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		const Eigen::Vector3f& ca = A.col(j);

		//Mix area
		const Eigen::Vector3f& p0 = V.col(fv[0]);
		const Eigen::Vector3f& p1 = V.col(fv[1]);
		const Eigen::Vector3f& p2 = V.col(fv[2]);
		float area = ((p1 - p0).cross(p2 - p0)).norm() / 6.0f;

		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];

			if (interVidx(fv0) != -1)
			{
				areas(fv0) += area;
				triple.push_back(Eigen::Triplet<float>(fv0, fv0, 1.0f / std::tan(ca[(vi + 1) % 3]) + 1.0f / std::tan(ca[(vi + 2) % 3])));
				triple.push_back(Eigen::Triplet<float>(fv0, fv1, -1.0f / std::tan(ca[(vi + 2) % 3])));
				triple.push_back(Eigen::Triplet<float>(fv0, fv2, -1.0f / std::tan(ca[(vi + 1) % 3])));
			}
		}
	}
	for (size_t i = 0; i < boundV.size(); ++i)
	{
		triple.push_back(Eigen::Triplet<float>(boundV[i], boundV[i], 1000));
	}

	//下半部分单位矩阵
	for (int i = 0; i < V.cols(); ++i)
	{
		triple.push_back(Eigen::Triplet<float>(i + V.cols(), i, 1));
	}

	L.resize(V.cols() * 2, V.cols());
	L.setFromTriplets(triple.begin(), triple.end());

	float sum_area = areas.sum() / float(interV.size());

	for (int r = 0; r < interV.size(); ++r)
	{
		L.row(interV[r]) *= sum_area / (2.0f * areas(interV[r]));
	}

	b.resize(V.cols() * 2, 3);
	b.setZero();

	//固定边界
	for (size_t ib = 0; ib < boundV.size(); ++ib)
	{
		b.row(boundV[ib]) = V.col(boundV[ib]).transpose() * 1000;
	}
	//变形目标
	for (int r = 0; r < V.cols(); ++r)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			b(r + V.cols(), i) = V(i, r) + update_d_(r * 3 + i);
		}
	}
}