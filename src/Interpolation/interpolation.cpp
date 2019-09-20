#include "interpolation.h"

Dev_Inter::Dev_Inter()
{
	//Collect the index of internal vertices
	for (const auto& vit : ori_mesh_.vertices())
		if (!ori_mesh_.is_boundary(vit))
			inter_p_.push_back(vit.idx());
	cur_mesh_ = ori_mesh_;

	Cal_Error();
	w1_ = w2_ = (ED_ + EI_ + EL_) * 50;
}

int Dev_Inter::Deformation()
{
	size_t it_count = 0;
	while (EI_ >= epI_ || EL_ >= epL_ || ED_ >= epD_ || it_count <= 50)
	{
		//Determine weights w1, w2
		if (!Adjust_Weights())
		{
			std::cout << "Faild to adjust weights" << std::endl;
			return 0;
		}

		//Construct linear system
		if (!BuildMetrix())
		{
			std::cout << "Faild to build coefficient matrix" << std::endl;
			return 0;
		}

		//Using Cholesky factorization to solve the linear system
		if (!SolveProblem())
		{
			std::cout << "Faild to solve system" << std::endl;
			return 0;
		}

		//Update the mesh with result scale S
		if (!Update_Mesh())
		{
			std::cout << "Faild to update mesh" << std::endl;
			return 0;
		}

		//Calculate three errors
		preEI_ = EI_;
		preED_ = ED_;
		preEL_ = EL_;
		Cal_Error();

		++it_count;
	}
	return 1;
}

int Dev_Inter::BuildMetrix()
{
	coeff_A_.setZero();
	right_b_.setZero();
	tri_Coeff_.clear();

	//row of current processing
	size_t cur_row = 0;
	//For every internal vertices calculate K
	for (; cur_row < inter_p_.size(); ++cur_row)
	{
		const Surface_mesh::Vertex v(inter_p_[cur_row]);
		Cal_CurvatureCoeff(v, cur_row);
	}

	//For every edges calculate length errors
	for (const auto& eit : cur_mesh_.edges())
	{
		Cal_LengthCoeff(eit, cur_row++);
	}

	//For every anuchor vertices calculate interpolation errors
	for (size_t j = 0; j < anchor_idx_.size(); ++j)
	{
		Cal_InterCoeff(j, cur_row);
		cur_row += 3;
	}

	//Build matrix
	coeff_A_.setFromTriplets(tri_Coeff_.begin(), tri_Coeff_.end());
	return 1;
}

int Dev_Inter::SolveProblem()
{
	//Solve the sparse linear system
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	Eigen::VectorXf tempB = coeff_A_.transpose() * right_b_;
	Eigen::SparseMatrix<double> tempA = (coeff_A_.transpose() * coeff_A_).eval();

	solver.compute(tempA);
	Eigen::SparseMatrix<double> tempU = solver.matrixU();
	if (solver.info() != Eigen::Success)
	{
		std::cout << solver.info() << std::endl;
		return 0;
	}

	scale_s_.setZero();
	scale_s_ = solver.solve(tempB);
	if (solver.info() != Eigen::Success)
	{
		std::cout << solver.info() << std::endl;
		return 0;
	}
	for (size_t i = 0; i < scale_s_.size(); ++i)
	{
		if (!isfinite(scale_s_(i)))
		{
			std::cout << "Wrong result!" << std::endl;
			return 0;
		}
	}
	return 1;
}

surface_mesh::Surface_mesh Dev_Inter::CreatMesh(size_t mesh_size)
{
	Surface_mesh mesh;
	for (int i = 0; i < mesh_size; ++i)
	{
		for (int j = 0; j < mesh_size; ++j)
		{
			mesh.add_vertex(Point(2 * i, 2 * j, 0));
		}
		if (i != (mesh_size - 1))
		{
			for (int j = 0; j < mesh_size - 1; ++j)
			{
				mesh.add_vertex(Point(2 * i + 1, 2 * j + 1, 0));
			}
		}
	}

	for (int i = 0; i < mesh_size - 1; ++i)
	{
		for (int j = 0; j < mesh_size - 1; ++j)
		{
			mesh.add_triangle(Surface_mesh::Vertex(j + i * (2 * mesh_size - 1)), Surface_mesh::Vertex(j + i * (2 * mesh_size - 1) + 1), Surface_mesh::Vertex(j + i * (2 * mesh_size - 1) + mesh_size));
			mesh.add_triangle(Surface_mesh::Vertex(j + i * (2 * mesh_size - 1) + 1), Surface_mesh::Vertex(j + (i + 1) * (2 * mesh_size - 1) + 1), Surface_mesh::Vertex(j + i * (2 * mesh_size - 1) + mesh_size));
			mesh.add_triangle(Surface_mesh::Vertex(j + i * (2 * mesh_size - 1) + mesh_size), Surface_mesh::Vertex(j + (i + 1) * (2 * mesh_size - 1) + 1), Surface_mesh::Vertex(j + (i + 1) * (2 * mesh_size - 1)));
			mesh.add_triangle(Surface_mesh::Vertex(j + i * (2 * mesh_size - 1)), Surface_mesh::Vertex(j + i * (2 * mesh_size - 1) + mesh_size), Surface_mesh::Vertex(j + (i + 1) * (2 * mesh_size - 1)));
		}
	}

	return mesh;
}

int Dev_Inter::SetConditions(const float& D, const float& I, const float& L, const float& dD, const float& dI, const float& duL, const float& ddL)
{
	//Regulate coefficients
	//terminal conditions
	epD_ = D;
	epI_ = I;
	epL_ = L;

	//weights related conditions
	deD_ = dD;
	deI_ = dI;
	ddeL_ = ddL;
	udeL_ = duL;
	return 1;
}

int Dev_Inter::Cal_CurvatureCoeff(const Surface_mesh::Vertex& v, size_t num)
{
	//差分距离
	float scale = 0.1f;

	//计算原始曲率
	right_b_(num) = -w1_ * Cal_Guassion_Curvature(v);

	//初始化
	for (size_t i = 0; i < 3; ++i)
	{
		//计算前
		cur_mesh_.position(v)[i] *= (1 - scale);
		double tempB = Cal_Guassion_Curvature(v);
		//计算后
		cur_mesh_.position(v)[i] *= ((1 + scale) / (1 - scale));
		double tempF = Cal_Guassion_Curvature(v);
		Eigen::Triplet <float> tempTri(num, v.idx() * 3 + i, w1_ * (tempF - tempB) / (2 * scale));
		tri_Coeff_.push_back(tempTri);
		//std::cout << tempTri.value() << std::endl;
		//复原
		cur_mesh_.position(v)[i] /= (1 + scale);
	}

	for (auto nit : cur_mesh_.vertices(v))
	{
		for (size_t i = 0; i < 3; ++i)
		{
			//计算前
			cur_mesh_.position(nit)[i] *= (1 - scale);
			double tempB = Cal_Guassion_Curvature(v);
			//计算后
			cur_mesh_.position(nit)[i] *= ((1 + scale) / (1 - scale));
			double tempF = Cal_Guassion_Curvature(v);
			Eigen::Triplet <float> tempTri(num, nit.idx() * 3 + i, w1_ * (tempF - tempB) / (2 * scale));
			//std::cout << tempTri.value() << std::endl;
			tri_Coeff_.push_back(tempTri);
			cur_mesh_.position(nit)[i] /= (1 + scale);
		}
	}

	return 1;
}

float Dev_Inter::Cal_LengthCoeff(const Surface_mesh::Edge& e, size_t num)
{
	auto vertex_head = cur_mesh_.vertex(e, 0);
	auto vertex_tail = cur_mesh_.vertex(e, 1);
	auto position_h = cur_mesh_.position(vertex_head);
	auto position_t = cur_mesh_.position(vertex_tail);

	auto vec = position_t - position_h;

	vec2mat(tri_Coeff_, w2_ * position_t / norm(vec), num, vertex_tail.idx());
	vec2mat(tri_Coeff_, -w2_ * position_h / norm(vec), num, vertex_head.idx());

	right_b_(num) = -w2_ * (cur_mesh_.edge_length(e) - ori_mesh_.edge_length(e));
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

void Dev_Inter::Cal_Error()
{
	//---------calculate developable error------------
	ED_ = 0;
	float area = 0.0f;
	for (size_t i = 0; i < inter_p_.size(); ++i)
	{
		float sum_theta = 0.0f; //summary internal angles
		const Surface_mesh::Vertex p(inter_p_[i]);
		auto q = cur_mesh_.vertices(p).begin();
		auto nex_q = q;
		++nex_q;
		while (q != cur_mesh_.vertices(p).end())
		{
			//necessary vectors
			auto vpq = (cur_mesh_.position(*q) - cur_mesh_.position(p));
			auto vpqn = (cur_mesh_.position(*nex_q) - cur_mesh_.position(p));

			//get the internal angle of ∠pvqn
			float theta = acos(dot(vpq, vpqn) / (norm(vpq) * norm(vpqn)));
			sum_theta += theta;
			area += norm(cross(vpq, vpqn)) / 2;
			q = nex_q;
			++nex_q;
		}
		area /= 3;
		ED_ += ((2 * M_PI - sum_theta) / area) * ((2 * M_PI - sum_theta) / area);
	}

	//For every edges calculate length errors
	EL_ = 0;
	for (const auto& eit : cur_mesh_.edges())
	{
		float lo = ori_mesh_.edge_length(eit);
		float lc = cur_mesh_.edge_length(eit);
		EL_ += sqrt(lc - lo);
	}

	//For every anuchor vertices calculate interpolation errors
	EI_ = 0;
	for (size_t i = 0; i < anchor_idx_.size(); ++i)
	{
		Surface_mesh::Vertex v(anchor_idx_[i]);
		EI_ += sqrnorm(cur_mesh_.position(v) - anchor_position_[i]);
	}
}

int Dev_Inter::Adjust_Weights()
{
	if ((EI_ - preEI_) < deI_ && EI_ > epI_)
	{
		w1_ /= 2;
		w2_ /= 2;
	}
	else if ((abs(EL_ - preEL_) / preEL_) > udeL_)
	{
		w2_ /= 2;
		cur_mesh_ = pre_mesh_;
	}
	else if ((abs(EL_ - preEL_) / preEL_) < udeL_ && EL_ > epL_)
	{
		w2_ *= 2;
	}
	else if ((ED_ - preED_) < deD_ && ED_ > epD_)
	{
		w1_ *= 2;
	}
	return 1;
}

int Dev_Inter::Update_Mesh()
{
	pre_mesh_ = cur_mesh_;
	for (auto vit : cur_mesh_.vertices())
	{
		for (size_t i = 0; i < 3; ++i)
		{
			cur_mesh_.position(vit)[i] *= (1 + scale_s_(vit.idx() + i));
		}
	}
	return 1;
}

float Dev_Inter::Cal_Guassion_Curvature(const Surface_mesh::Vertex& v)
{
	float sum_theta = 0.0f; //内角和
	float area = 0.0f; //voronoi 面积

	auto q = cur_mesh_.vertices(v).begin();
	auto nex_q = q;
	++nex_q;
	while (q != cur_mesh_.vertices(v).end())
	{
		//要使用到的向量
		auto vpq = (cur_mesh_.position(*q) - cur_mesh_.position(v));
		auto vpqn = (cur_mesh_.position(*nex_q) - cur_mesh_.position(v));

		//------计算要用到的数值-------
		//计算内角
		double temp = dot(vpq, vpqn) / (norm(vpq) * norm(vpqn));
		if (temp > 1)	temp = 1;
		if (temp < -1)	temp = -1;
		double theta = acos(temp);

		//叉乘的范数
		float cro_pqqn, cro_pqqp;
		cro_pqqn = norm(cross(vpq, vpqn));

		//累加当前顶点
		sum_theta += theta;
		area += (cro_pqqn / 2);

		//计算下一个相邻顶点
		++q;
		++nex_q;
	}
	return (2.0 * M_PI - sum_theta) / (area / 3.0);
}