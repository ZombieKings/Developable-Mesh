#include "func_opt.h"
#include "newton_solver.h"

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xd& vertices_mat, Eigen::Matrix3Xi& faces_mat);

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Laod failed!" << std::endl;
	}
	std::cout << mesh.n_vertices() << std::endl;
	std::cout << mesh.n_faces() << std::endl;
	Eigen::Matrix3Xd matV;
	Eigen::Matrix3Xi matF;
	mesh2matrix(mesh, matV, matF);

	func_opt::my_function f(mesh, 0.001, 1.0, 1.0, 1.0);
	opt_solver::newton_solver solver;
	solver.set_f(f);
	//solver.solve_sqp(matV.data());
	solver.solve(matV.data());
	return 1;
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xd& vertices_mat, Eigen::Matrix3Xi& faces_mat)
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
				const Eigen::Vector3f& temp = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data());
				vertices_mat.col(fvit.idx()) = temp.cast<double>();
				flag(fvit.idx()) = 1;
			}
		}
	}
}
