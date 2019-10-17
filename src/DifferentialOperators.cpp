#include "DifferentialOperators.h"

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Laod failed!" << std::endl;
	}

	std::vector<int> interV;
	std::vector<int> boundV;
	Eigen::VectorXi interVidx;
	interVidx.resize(mesh.n_vertices() + 1);
	memset(interVidx.data(), -1, sizeof(int) * interVidx.size());
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
	interVidx(mesh.n_vertices()) = count;
	MatrixType matV;
	Eigen::Matrix3Xi matF;
	mesh2matrix(mesh, matV, matF);

	MatrixType gradX;
	MatrixType gradY;
	MatrixType gradZ;
	cal_grad_pos(matV, matF, gradX, gradY, gradZ);

	std::cout << gradX << std::endl;
	return 1;
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F)
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
				V.col(fvit.idx()) = Eigen::Map<const VectorType>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data());
				flag(fvit.idx()) = 1;
			}
		}
	}
}

void cal_angles(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& A)
{
	A.resize(3, F.cols());
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const VectorType& p0 = V.col(fv[vi]);
			const VectorType& p1 = V.col(fv[(vi + 1) % 3]);
			const VectorType& p2 = V.col(fv[(vi + 2) % 3]);
			const DataType angle = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			A(vi, f) = angle;
		}
	}
}

void cal_cot_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& A, const Eigen::VectorXi& interVidx, Eigen::SparseMatrix<DataType>& L)
{
	//计算固定边界的cot权拉普拉斯系数矩阵
	std::vector<Tri> triple;

	Eigen::VectorXf areas;
	areas.resize(V.cols());
	areas.setZero();
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		const VectorType& ca = A.col(j);

		//Mix area
		const VectorType& p0 = V.col(fv[0]);
		const VectorType& p1 = V.col(fv[1]);
		const VectorType& p2 = V.col(fv[2]);
		const DataType area = ((p1 - p0).cross(p2 - p0)).norm() / 2.0f;

		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (interVidx(fv0) != -1)
				areas(fv0) += area / 3.0f;
			triple.push_back(Tri(fv0, fv0, 1.0f / std::tan(ca[(vi + 1) % 3]) + 1.0f / std::tan(ca[(vi + 2) % 3])));
			triple.push_back(Tri(fv0, fv1, -1.0f / std::tan(ca[(vi + 2) % 3])));
			triple.push_back(Tri(fv0, fv2, -1.0f / std::tan(ca[(vi + 1) % 3])));
		}
	}

	L.resize(V.cols(), V.cols());
	L.setFromTriplets(triple.begin(), triple.end());

	for (int k = 0; k < L.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<DataType>::InnerIterator it(L, k); it; ++it)
		{
			if (interVidx(it.index()) != -1)
				it.valueRef() /= (2.0f * areas(it.index()));
		}
	}
}

void cal_uni_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& L)
{
	std::vector<Tri> tripleL;
	tripleL.reserve(F.cols() * 9);
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const VectorType& p0 = V.col(fv[vi]);
			const VectorType& p1 = V.col(fv[(vi + 1) % 3]);
			const VectorType& p2 = V.col(fv[(vi + 2) % 3]);
			tripleL.push_back(Tri(fv[vi], fv[vi], 1));
			tripleL.push_back(Tri(fv[vi], fv[(vi + 1) % 3], -0.5f));
			tripleL.push_back(Tri(fv[vi], fv[(vi + 2) % 3], -0.5f));
		}
	}
	L.resize(V.cols(), V.cols());
	L.setFromTriplets(tripleL.begin(), tripleL.end());
}

void cal_grad_pos(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& gradX, MatrixType& gradY, MatrixType& gradZ)
{
	gradX.resize(3, V.cols());
	gradY.resize(3, V.cols());
	gradZ.resize(3, V.cols());
	gradX.setZero();
	gradY.setZero();
	gradZ.setZero();

	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::Vector3i& fv = F.col(i);
		Eigen::Matrix<DataType, 3, 3> P;
		for (int j = 0; j < 3; ++j)
		{
			P.col(j) = V.col(fv[j]);
		}
		const VectorType nor = (P.col(1) - P.col(0)).cross(P.col(2) - P.col(0));
		const DataType area = nor.norm() / 2.0f;
		for (int j = 0; j < 3; ++j)
		{
			const VectorType gradTemp((nor.cross(P.col((j + 2) % 3) - P.col((j + 1) % 3))) / (area * 2.0));
			gradX.col(fv[j]) += P(0, j) * gradTemp;
			gradY.col(fv[j]) += P(1, j) * gradTemp;
			gradZ.col(fv[j]) += P(2, j) * gradTemp;
		}
	}
}
