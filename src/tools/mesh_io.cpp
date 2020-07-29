#include "mesh_io.h"

template<typename DerivedV, typename DerivedF>
void Zombie::mesh2matrix(const surface_mesh::Surface_mesh& mesh,
	Eigen::PlainObjectBase<DerivedV>& V,
	Eigen::PlainObjectBase<DerivedF>& F)
{
	assert(mesh.n_faces() != 0 && "No faces in input mesh!");
	assert(mesh.n_vertices() != 0 && "No vertices in input mesh!");

	F.resize(Eigen::NoChange, mesh.n_faces());
	V.resize(Eigen::NoChange, mesh.n_vertices());

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
				V.col(fvit.idx()) = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data()).cast<typename DerivedV::Scalar>();
				flag(fvit.idx()) = 1;
			}
		}
	}
}
