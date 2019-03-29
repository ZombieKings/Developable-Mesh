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
	assert(!cur_mesh_.is_boundary(v));

	float sum_theta = 0.0f;
	float area = 0.0f;
	float K = 0.0f;
	//calculate summarize of internal angles and Voronoi area
	for (auto it_v1 = cur_mesh_.vertices(v).begin(); it_v1 != cur_mesh_.vertices(v).end();)
	{
		auto it_v2 = it_v1;
		++it_v1;
		//2 adjacent vectors 
		auto e1 = (cur_mesh_.position(*it_v1) - cur_mesh_.position(v));
		auto e2 = (cur_mesh_.position(*it_v2) - cur_mesh_.position(v));
		//calculate theta 
		float theta = acos(dot(e1, e2) / (norm(e1) * norm(e2)));
		//accumulate
		sum_theta += theta;
		area += norm(e1) * norm(e2) * sin(theta);
	}
	area /= 3;

	//calculate Gaussion curvature
	return K = (2 * M_PI - sum_theta) / area;
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
