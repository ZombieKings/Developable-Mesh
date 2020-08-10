#include "get_vertex_neighbor.h"

void Zombie::get_neighbor_faces(const surface_mesh::Surface_mesh& mesh,	std::vector<std::vector<int>>& neighF)
{
	neighF.reserve(mesh.n_vertices());
	for (auto vit : mesh.vertices())
	{
		std::vector<int> vnf;
		for (auto fit : mesh.faces(vit))
		{
			vnf.push_back(fit.idx());
		}
		neighF.push_back(vnf);
	}
}

void Zombie::get_neighbor_vertices(const surface_mesh::Surface_mesh& mesh, std::vector<std::vector<int>>& neighV)
{
	neighV.reserve(mesh.n_vertices());
	for (auto vit : mesh.vertices())
	{
		std::vector<int> vnv;
		for (auto vvit : mesh.vertices(vit))
		{
			vnv.push_back(vvit.idx());
		}
		neighV.push_back(vnv);
	}
}
