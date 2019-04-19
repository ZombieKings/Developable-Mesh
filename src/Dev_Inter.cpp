#include "Dev_Inter.h"

Dev_Inter::Dev_Inter(Surface_mesh input_mesh, std::vector<Point> input_anchor, std::vector<unsigned int> input_anchor_idx) :ori_mesh_(input_mesh), anchor_position_(input_anchor), anchor_idx_(input_anchor_idx)
{
	//Collect the index of internal vertices
	for (const auto &vit : ori_mesh_.vertices())
		if (!ori_mesh_.is_boundary(vit))
			inter_p_.push_back(vit.idx());
	cur_mesh_ = ori_mesh_;
	temp_Area_.resize(inter_p_.size());

	//Adapt the coefficient matrix based on the input datas
	coeff_A_.resize(inter_p_.size() + input_mesh.n_edges() + input_anchor.size() * 3, input_mesh.n_vertices() * 3);
	coeff_A_.reserve(inter_p_.size() * 3 * 18 + input_mesh.n_edges() * 3 * 2 + input_anchor.size() * 3 * 3);
	right_b_.resize(inter_p_.size() + input_mesh.n_edges() + input_anchor.size() * 3);
	scale_s_.resize(inter_p_.size() + input_mesh.n_edges() + input_anchor.size() * 3);
}

int Dev_Inter::Deformation()
{
	size_t it_count = 0;
	Cal_Error();
	while (EI_ >= epI_ || EL_ >= epL_ || ED_ >= epD_ || it_count >= 50)
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

const surface_mesh::Surface_mesh& Dev_Inter::Get_Result() const
{
	return cur_mesh_;
}

int Dev_Inter::BuildMetrix()
{
	coeff_A_.setZero();
	right_b_.setZero();
	tri_Coeff_.clear();
	temp_Area_.clear();

	//row of current processing
	size_t cur_row = 0;
	//For every internal vertices calculate K
	for (; cur_row < inter_p_.size(); ++cur_row)
	{
		const Surface_mesh::Vertex v(inter_p_[cur_row]);
		Cal_CurvatureCoeff(v, cur_row);
	}

	//For every edges calculate length errors
	for (const auto &eit : cur_mesh_.edges())
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

	//additionally divide Voronoi area into coefficients
	for (size_t i = 0; i < inter_p_.size(); ++i)
		for (Eigen::SparseMatrix<float>::InnerIterator Iit(coeff_A_, i); Iit; ++Iit)
			Iit.valueRef() /= temp_Area_[i];
	coeff_A_.makeCompressed();
	return 1;
}

int Dev_Inter::SolveProblem()
{
	//Solve the sparse linear system
	Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> solver;
	Eigen::SparseMatrix<float> tempAT = coeff_A_.transpose();
	Eigen::VectorXf tempB = tempAT * right_b_;
	Eigen::SparseMatrix<float> tempA = tempAT * coeff_A_;
	//int rank = Matrix_Rank(coeff_A_);
	//if (rank != coeff_A_.cols())
	//{
	//	std::cout << rank << std::endl;
	//	return 0;
	//}
	
	//Eigen::MatrixXf R ;
	//if (!my_LLT(tempA, R))
	//{
	//	return 0;
	//}
	Matrix_Visualization(coeff_A_, inter_p_.size(), inter_p_.size() + cur_mesh_.n_edges());
	//Matrix_Visualization(Tempgs, 0, 0);

	solver.compute(tempA);
	if (solver.info() != Eigen::Success)
	{
		std::cout << solver.info() << std::endl;
		return 0;
	}

	scale_s_.setZero();
	scale_s_ = solver.solve(tempB);
	std::cout << scale_s_ << std::endl;
	if (solver.info() != Eigen::Success)
	{
		std::cout << solver.info() << std::endl;
		return 0;
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

Eigen::SparseMatrix<float> Dev_Inter::Col_Length(const Surface_mesh& input_mesh) const
{
	Eigen::SparseMatrix<float> mat_length(input_mesh.n_vertices(), input_mesh.n_vertices());
	std::vector<Eigen::Triplet<float>> tri_temp;
	for (const auto& eit : input_mesh.edges())
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
		auto vqp = (cur_mesh_.position(p) - cur_mesh_.position(*q));
		auto vpqn = (cur_mesh_.position(*nex_q) - cur_mesh_.position(p));
		auto vqqn = (cur_mesh_.position(*nex_q) - cur_mesh_.position(*q));
		auto vqqp = (cur_mesh_.position(*pre_q) - cur_mesh_.position(*q));

		//------计算要用到的数值-------

		//计算内角
		float theta = acos(dot(vpq, vpqn) / (norm(vpq) * norm(vpqn)));

		//叉乘的范数
		float cro_pqqn, cro_pqqp;
		cro_pqqn = norm(cross(vpq, vqqn));
		cro_pqqp = norm(cross(vpq, vqqp));

		//cotangent
		cot_l = dot(vqp, vqqp) / cro_pqqp;
		cot_r = dot(vqp, vqqn) / cro_pqqn;

		//邻接顶点系数
		coeff_vq = (cot_l + cot_r) / sqrnorm(vpq) * vqp - vqqn / cro_pqqn - vqqp / cro_pqqp;
		coeff_vq = coeff_vq * cur_mesh_.position(*q);
		vec2mat(tri_Coeff_, w1_*coeff_vq, num, (*q).idx());

		//累加当前顶点
		sum_theta += theta;
		area += (cro_pqqn / 2);
		coeff_vp += (((cot_l + cot_r) / sqrnorm(vpq)) * (vpq));

		//计算下一个相邻顶点
		++pre_q;
		++q;
		++nex_q;
	}
	temp_Area_.push_back(w1_ * area / 3);

	//当前顶点系数
	coeff_vp = coeff_vp * cur_mesh_.position(p);
	vec2mat(tri_Coeff_, w1_ * coeff_vp, num, p.idx());

	//计算等式右侧的系数
	right_b_(num) = - w1_ * (2 * M_PI - sum_theta) / (area / 3);
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
	vec2mat(tri_Coeff_, - w2_ * position_h / norm(vec), num, vertex_head.idx());

	right_b_(num) = - w2_ * (cur_mesh_.edge_length(e) - ori_mesh_.edge_length(e));
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
		ED_ += ((2 * M_PI - sum_theta) / area)*((2 * M_PI - sum_theta) / area);
	}

	//For every edges calculate length errors
	EL_ = 0;
	for (const auto &eit : cur_mesh_.edges())
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

void Dev_Inter::Matrix_Visualization(const Eigen::MatrixXf& iMatrix, double first, double second)
{
	cv::Mat Image = cv::Mat::zeros(1000, 1000, CV_8UC3);
	double W = 1000.0f / iMatrix.cols();
	double H = 1000.0f / iMatrix.rows();
	for (double i = 0; i < iMatrix.cols(); ++i)
		for (double j = 0; j < iMatrix.rows(); ++j)
			if (iMatrix(j, i) > 0)
			{
				cv::Point2d p1(i * W, j * H);
				cv::Point2d p2((i + 1) * W, (j + 1) * H);
				cv::rectangle(Image, p1, p2, cv::Scalar{ 0 , 0 , 255 }, cv::FILLED);
				//cv::rectangle(Image, p1, p2, cv::Scalar{ 255 , 255 , 255 });
			}
			else if(iMatrix(j, i) < 0)
			{
				cv::Point2d p1(i * W, j * H);
				cv::Point2d p2((i + 1) * W, (j + 1) * H);
				cv::rectangle(Image, p1, p2, cv::Scalar{ 255 , 23 , 0 }, cv::FILLED);
			}

	cv::Point2d p3(0, first * H);
	cv::Point2d p4(1000, first * H);
	cv::line(Image, p3, p4, cv::Scalar{ 255,0,0 });
	cv::Point2d p5(0, second * H);
	cv::Point2d p6(1000, second * H);
	cv::line(Image, p5, p6, cv::Scalar{ 255,0,0 });

	while (cvWaitKey(10) != 's')
	{
		cv::imshow("Matrix", Image);
	}
	return;
}

int Dev_Inter::Matrix_Rank(const Eigen::MatrixXf& iMatrix)
{
	Eigen::MatrixXf tempMat(iMatrix);
	int r = 0;
	int c = 0;
	while (r < tempMat.rows() && c < tempMat.cols())
	{
		if (tempMat(r, c))
		{
			for (int j = r + 1; j < tempMat.rows(); ++j)
			{
				if (!tempMat(j, c))
					continue;
				for (int k = c + 1; k < tempMat.cols(); ++k)
				{
					tempMat(j, k) -= tempMat(r, k) * (tempMat(j, c) / tempMat(r, c));
				}
				tempMat(j, c) = 0;
			}
			++r;
		}
		++c;
	}
	return r;
}

bool Dev_Inter::my_LLT(const Eigen::SparseMatrix<float>& in_Mat, Eigen::MatrixXf& Result)
{
	Eigen::MatrixXf tempMat(in_Mat);
	Result = Eigen::MatrixXf::Identity(in_Mat.rows(), in_Mat.cols());
	for (int i = 0; i < tempMat.cols(); ++i)
	{
		if (tempMat(i, i) == 0)
		{
			std::cout << "Zero Element in " << i << "!!!" << std::endl;
			return false;
		}
		for (int j = i + 1; j < tempMat.rows(); ++j)
		{
			if (!tempMat(j, i))
				continue;
			Result(j, i) = tempMat(j, i) / tempMat(i, i);
			tempMat(j, i) = 0;
			for (int k = (i + 1); k < tempMat.cols(); ++k)
			{
				tempMat(j, k) -= tempMat(i, k) * Result(j, i);
			}
		}
		for (int j = i; j < tempMat.rows(); ++j)
		{
			Result(j, i) *= sqrt(tempMat(i, i));
		}
	}
	return true;
}
