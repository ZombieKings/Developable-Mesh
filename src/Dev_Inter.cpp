#include "Dev_Inter.h"

Dev_Inter::Dev_Inter(Surface_mesh input_mesh, std::vector<Point> input_anchor, std::vector<unsigned int> input_anchor_idx)
{
	ori_mesh_ = input_mesh;
	anchor_position_ = input_anchor;
	anchor_idx_ = input_anchor_idx;

	//Collect the index of internal vertices
	for (auto vit : ori_mesh_.vertices())
		if (!ori_mesh_.is_boundary(vit))
			inter_p_.push_back(vit.idx());

	//Adapt the coefficient matrix based on the input datas 
	coeff_A_.reserve(inter_p_.size() * 3 * 18 + input_mesh.n_edges() * 3 * 2 + input_anchor.size() * 3 * 3);
	right_b_.resize(inter_p_.size() + input_mesh.n_edges() + input_anchor.size() * 3);
}

int Dev_Inter::Deformation()
{
	size_t it_count = 0;
	return 1;
}

const surface_mesh::Surface_mesh& Dev_Inter::Get_Result() const
{
	return cur_mesh_;
}

int Dev_Inter::BuildMetrix()
{
	size_t cur_row = 0; //row of current processing
	//For every internal vertices calculate K
	for (; cur_row < inter_p_.size(); ++cur_row)
	{
		Surface_mesh::Vertex v(inter_p_[cur_row]);
		Cal_CurvatureCoeff(v, cur_row);
	}

	//For every edges calculate length errors
	for (auto eit : cur_mesh_.edges())
	{
		Cal_LengthCoeff(eit, cur_row++);
	}

	//For every anuchor vertices calculate interpolation errors
	for (size_t j = 0; j < anchor_idx_.size(); ++j)
	{
		Cal_InterCoeff(j, cur_row);
		cur_row += 3;
	}
	return 1;
}

int Dev_Inter::SolveProblem()
{
	return 1;
}

Eigen::SparseMatrix<float> Dev_Inter::Col_Length(const Surface_mesh& input_mesh) const
{
	Eigen::SparseMatrix<float> mat_length(input_mesh.n_vertices(), input_mesh.n_vertices());
	std::vector<Eigen::Triplet<float>> tri_temp;
	for (auto eit : input_mesh.edges())
	{
		unsigned int p = input_mesh.vertex(eit, 1).idx();
		unsigned int q = input_mesh.vertex(eit, 0).idx();
		p > q ? tri_temp.push_back(Eigen::Triplet<float>(p, q, input_mesh.edge_length(eit))) : tri_temp.push_back(Eigen::Triplet<float>(q, p, input_mesh.edge_length(eit)));
	}
	mat_length.setFromTriplets(tri_temp.begin(), tri_temp.end());
	mat_length.makeCompressed();
	return mat_length;
}

int Dev_Inter::Cal_CurvatureCoeff(const Surface_mesh::Vertex& v, size_t num)
{
	//计算高斯曲率相关的系数
	Vec3 coeff_vp(0, 0, 0); //当前顶点系数
	Vec3 coeff_vq(0, 0, 0);	//相邻顶点系数
	float cot_l = 0.0f, cot_r = 0.0f; //左右两侧角的cotangent值
	float sum_theta = 0.0f; //内角和
	float area = 0.0f; //voronoi 面积

	//初始化
	auto p = v;
	auto q = cur_mesh_.vertices(p).begin();
	auto pre_q = q;
	--pre_q;
	auto nex_q = q;
	++nex_q;
	while (q != cur_mesh_.vertices(p).end())
	{
		//要使用到的向量 
		auto vpq = (cur_mesh_.position(*q) - cur_mesh_.position(p));
		auto vpqn = (cur_mesh_.position(*nex_q) - cur_mesh_.position(p));
		auto vqqn = (cur_mesh_.position(*nex_q) - cur_mesh_.position(*q));
		auto vqqp = (cur_mesh_.position(*pre_q) - cur_mesh_.position(*q));

		//计算要用到的数值
		//计算内角
		float theta = acos(dot(vpq, vpqn) / (norm(vpq) * norm(vpqn)));

		//叉乘的范数
		float cro_pqqn, cro_pqqp;
		cro_pqqn = norm(cross(vpq, vqqn));
		cro_pqqp = norm(cross(vpq, vqqp));

		//cotangent
		cot_l = dot(vpq, vqqp) / cro_pqqp;
		cot_r = dot(vpq, vqqn) / cro_pqqn;

		//邻接顶点系数
		coeff_vq = (cot_l + cot_r) / sqrnorm(vpq)*vpq - vqqn / cro_pqqn - vqqn / cro_pqqp;
		coeff_vq = coeff_vq * cur_mesh_.position(*q);
		vec2mat(tri_Coeff_, w1_*coeff_vq, num, (*q).idx());

		//累加当前顶点
		sum_theta += theta;
		area += cro_pqqn / 2;
		coeff_vp += ((cot_l + cot_r) / norm(vpq))*(-vpq);

		//计算下一个相邻顶点
		pre_q = q;
		q = nex_q;
		++nex_q;
	}
	temp_Area_.push_back(area / 3);

	//当前顶点系数
	coeff_vp = coeff_vp * cur_mesh_.position(p);
	vec2mat(tri_Coeff_, w1_ * coeff_vp, num, p.idx());

	//计算等式右侧的系数
	right_b_(num) = w1_ * (2 * M_PI - sum_theta) / (area / 3);
	return 1;
}

float Dev_Inter::Cal_LengthCoeff(const Surface_mesh::Edge& e, size_t num)
{
	auto vertex_head = cur_mesh_.vertex(e, 1);
	auto vertex_tail = cur_mesh_.vertex(e, 0);
	auto position_h = cur_mesh_.position(vertex_head);
	auto position_t = cur_mesh_.position(vertex_tail);

	auto vec = position_t - position_h;

	vec2mat(tri_Coeff_, w2_ * position_t / norm(vec), num, vertex_tail.idx());
	vec2mat(tri_Coeff_, - w2_ * position_h / norm(vec), num, vertex_head.idx());
	right_b_(num) = 0;
	return 1;
}

float Dev_Inter::Cal_InterCoeff(size_t idx, size_t num)
{
	Surface_mesh::Vertex temp_v(anchor_idx_[idx]);
	surface_mesh::Vec3f temp_lambda = cur_mesh_.position(temp_v) - anchor_position_[idx];
		for (size_t i = 0; i < 3; ++i)
		{
			//left hand 
			tri_Coeff_.push_back(Eigen::Triplet <float>(num + i, temp_v.idx() * 3 + i, cur_mesh_.position(temp_v)[i]));
			//right hand
			right_b_(num + i) = -temp_lambda[i];
		}
	return 1;
}

int Dev_Inter::Adjust_Weights()
{
	return 1;
}

