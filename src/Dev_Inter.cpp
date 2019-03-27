#include "Dev_Inter.h"

int Dev_Inter::Deformation()
{

}

const surface_mesh::Surface_mesh& Dev_Inter::Get_Result() const
{
	return cur_mesh_;
}

int Dev_Inter::BuildMetrix()
{

}

int Dev_Inter::SolveProblem()
{

}

float Dev_Inter::Cal_Curvature(const Surface_mesh::Vertex& v)
{

}

float Dev_Inter::Cal_EInter(const Surface_mesh::Vertex& v)
{

}

float Dev_Inter::Cal_Elength(const Surface_mesh::Edge& v)
{

}

int Dev_Inter::Cal_Weights(Surface_mesh X0, double errD, double errL, double errI, const Surface_mesh& X0r, double errDr, double errLr, double errIr)
{

}

float Dev_Inter::Cal_Theta(const Surface_mesh::Vertex& v)
{
	float theta = 0.0;
	for (auto ve = cur_mesh_.vertices(v).begin(); ve != cur_mesh_.vertices(v).end();)
	{
		auto vf = ve;
		++ve;
		auto l = (cur_mesh_.position(*ve) - cur_mesh_.position(v)) * (cur_mesh_.position(*vf) - cur_mesh_.position(v));
		float s = 0.0;
		for (int i = 0; i < 3; ++i)
			s += l[i];
		dot()
		theta += acos(s / (cur_mesh_.edge_length(cur_mesh_.edge(vf.halfedge())) * cur_mesh_.edge_length(cur_mesh_.edge(ve.halfedge()))));
	}
}

