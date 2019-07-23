#include "least_norm.h"

int Dev_LN::Load_Mesh(const std::string &filename)
{
	Surface_mesh mesh;
	if (!mesh.read(filename))
	{
		std::cout << "Laod failed!" << std::endl;
	}
	//-----------保存构造的网格-----------
	mesh2matrix(mesh, ori_mesh_mat_, face_mat_);

	ori_mesh_.clear();
	ori_mesh_ = mesh;

	//收集内部顶点下标
	inter_p_.clear();
	inter_p_r_.resize(mesh.n_vertices());
	inter_p_r_.setOnes();
	inter_p_r_ *= mesh.n_vertices();

	int count = 0;
	for (const auto &vit : ori_mesh_.vertices())
		if (!ori_mesh_.is_boundary(vit))
		{
			inter_p_.push_back(vit.idx());
			inter_p_r_(vit.idx()) = count++;
		}
		else
		{
			boundary_p_.push_back(vit.idx());
		}

	//根据内部顶点数设置参数
	vnum_ = inter_p_.size();
	epsilon_ = std::max(inter_p_.size() * pow(10, -8), pow(10, -5));

	std::cout << "初始平均误差： " << Cal_Error(ori_mesh_mat_, 0) << std::endl;
	std::cout << "初始最大误差： " << Cal_Error(ori_mesh_mat_, 1) << std::endl;

	cur_mesh_ = ori_mesh_;
	cur_mesh_mat_ = ori_mesh_mat_;
	return 1;
}

int Dev_LN::SetCondition(double delta, size_t times)
{
	epsilon_ = delta;
	it_count_ = times;
	return 1;
}

double Dev_LN::Cal_Error(const Eigen::Matrix3Xf &V, int flag)
{
	Eigen::Matrix3Xf A_mat;
	cal_angles(V, face_mat_, A_mat);
	Eigen::VectorXf temp_angle;
	temp_angle.resize(V.cols());
	temp_angle.setZero();
	for (size_t j = 0; j < face_mat_.cols(); ++j)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			temp_angle(face_mat_(i, j)) += A_mat(i, j);
		}
	}
	if (flag)
	{
		double max = 0;
		for (int i = 0; i < inter_p_.size(); ++i)
		{
			max = abs(2.0 * M_PI - temp_angle(inter_p_[i])) > max ? abs(2.0 * M_PI - temp_angle(inter_p_[i])) : max;
		}
		return max;
	}
	else
	{
		double averange = 0;
		for (int i = 0; i < inter_p_.size(); ++i)
		{
			averange += temp_angle(inter_p_[i]);
		}
		averange = 2.0 * M_PI - averange / inter_p_.size();
		return averange;
	}
}

void Dev_LN::cal_angles(const Eigen::Matrix3Xf &V, const Eigen::Matrix3Xi &F, Eigen::Matrix3Xf &angles)
{

	angles.resize(3, F.cols());
	for (size_t j = 0; j < F.cols(); ++j) {
		const Eigen::Vector3i &fv = F.col(j);
		for (size_t vi = 0; vi < 3; ++vi) {
			const Eigen::VectorXf &p0 = V.col(fv[vi]);
			const Eigen::VectorXf &p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::VectorXf &p2 = V.col(fv[(vi + 2) % 3]);
			const float angle = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			angles(vi, j) = angle;
		}
	}
}

const surface_mesh::Surface_mesh& Dev_LN::Get_Result() const
{
	return cur_mesh_;
}

int Dev_LN::Deformation()
{
	int count = 0;
	do
	{
		if (!Build_Equation())
		{
			std::cout << "BE" << std::endl;
			return 0;
		}

		if (!Solve_Problem())
		{
			std::cout << "SP" << std::endl;
			return 0;
		}

		if (!Update_Mesh())
		{
			std::cout << "UM" << std::endl;
			return 0;
		}
		count++;
	} while (result_x_.squaredNorm() > epsilon_ && count <= it_count_);
	std::cout << "迭代次数： " << count << std::endl;
	std::cout << "平均误差： " << Cal_Error(cur_mesh_mat_, 0) << std::endl;
	std::cout << "最大误差： " << Cal_Error(cur_mesh_mat_, 1) << std::endl;
	return 1;
}

int Dev_LN::Build_Equation()
{
	//清零等式变量
	coeff_A_.resize(vnum_, vnum_ * 3);
	coeff_A_.setZero();
	right_b_.resize(vnum_);
	right_b_.setZero();
	result_x_.resize(vnum_ * 3);
	result_x_.setZero();
	tri_Coeff_.clear();

	for (int i = 0; i < inter_p_.size(); ++i)
	{
		const Surface_mesh::Vertex v(inter_p_[i]);
		Vec3 coeff_vp(0, 0, 0); //当前顶点系数
		Vec3 coeff_vq(0, 0, 0);	//相邻顶点系数
		double cot_l = 0.0f, cot_r = 0.0f; //左右两侧角的cotangent值
		double sum_theta = 0.0f; //内角和

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
			double theta = acos(std::max(-1.0, std::min(1.0, double(dot(vpq, vpqn) / (norm(vpq) * norm(vpqn))))));

			//叉乘的范数
			double cro_pqqn, cro_pqqp;
			cro_pqqn = norm(cross(vpq, vqqn));
			cro_pqqp = norm(cross(vpq, vqqp));

			//cotangent
			cot_l = dot(vqp, vqqp) / cro_pqqp;
			cot_r = dot(vqp, vqqn) / cro_pqqn;

			//邻接顶点系数
			if (inter_p_r_((*q).idx()) < vnum_)
			{
				coeff_vq = (cot_l + cot_r) / sqrnorm(vpq) * vqp - vqqn / cro_pqqn - vqqp / cro_pqqp;
				vec2mat(tri_Coeff_, coeff_vq, i, inter_p_r_((*q).idx()));
			}

			//累加当前顶点
			sum_theta += theta;
			coeff_vp += (((cot_l + cot_r) / sqrnorm(vpq)) * (vpq));

			//计算下一个相邻顶点
			++pre_q;
			++q;
			++nex_q;
		}
		//当前顶点系数
		vec2mat(tri_Coeff_, coeff_vp, i, i);

		//--------等式右侧-------------
		right_b_(i) = 2.0 * M_PI - sum_theta;
	}

	//生成稀疏矩阵
	coeff_A_.setFromTriplets(tri_Coeff_.begin(), tri_Coeff_.end());
	return 1;
}

int Dev_LN::Solve_Problem()
{
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
	Eigen::SparseMatrix<float> tempAT = Eigen::SparseMatrix<float>(coeff_A_.transpose());
	Eigen::SparseMatrix<float> tempA = (coeff_A_ * tempAT).eval();

	//solver.analyzePattern(tempA);
	//solver.compute(tempA);
	solver.compute(tempA);
	if (solver.info() != Eigen::Success)
	{
		std::cout << solver.info() << std::endl;
		return 0;
	}

	result_x_.setZero();
	Eigen::VectorXf tempx(result_x_);
	//std::cout << "right b: " << std::endl;
	//std::cout << right_b_ << std::endl;

	tempx = solver.solve(right_b_);
	result_x_ = solver.matrixU() * tempx;

	//std::cout << "result_x_:" << std::endl;
	//std::cout << result_x_ << std::endl;

	if (!result_x_.allFinite())
	{
		std::cout << "Wrong result!" << std::endl;
		return 0;
	}

	return 1;
}

int Dev_LN::Update_Mesh()
{
	//build equation systems
	Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> solver;

	//构造矩阵 
	Eigen::SparseMatrix<float,Eigen::RowMajor> coeff_ls;
	Eigen::SparseMatrix<float> cotL;
	cal_cot_laplace(cur_mesh_, cotL);

	Eigen::MatrixX3f b_ls;
	b_ls.resize(cur_mesh_mat_.cols()*2, 3);
	b_ls.setZero();
	b_ls.topRows(cur_mesh_mat_.cols()) = cur_mesh_mat_.transpose();

	Eigen::MatrixX3f tempv(cur_mesh_mat_.transpose());
	for (int i = 0; i < vnum_; ++i)
	{
		tempv(inter_p_[i], 0) += result_x_(3 * i);
		tempv(inter_p_[i], 1) += result_x_(3 * i + 1);
		tempv(inter_p_[i], 2) += result_x_(3 * i + 2);
	}
	b_ls.bottomRows(cur_mesh_mat_.cols()) = tempv;
	//pin boundary points
	for (int i = 0; i < boundary_p_.size(); ++i)
	{
		cotL.row(boundary_p_[i]) *= 0;
		cotL.coeffRef(boundary_p_[i], boundary_p_[i]) = 1;
		b_ls.row(boundary_p_[i]) = cur_mesh_mat_.col(boundary_p_[i]).transpose();
	}

	coeff_ls.resize(vnum_ * 2, vnum_);
	coeff_ls.topRows(vnum_) = cotL;
	for (size_t i = 0; i < vnum_; ++i)
	{
		coeff_ls.coeffRef(i + vnum_, i) = 1;
	}
	coeff_ls.makeCompressed();

	solver.compute(coeff_ls);

	int counter = 0;
	double pre_err = pow(10, 6), err;
	err = Cal_Error(cur_mesh_mat_, 1);
	while (err > 0.001)
	{
		if (counter > 20 && pre_err - err < 0.01)
			break;

		b_ls = tempv;

		if (solver.info() != Eigen::Success)
		{
			std::cout << solver.info() << std::endl;
		}
		tempv = solver.solve(b_ls);

		cur_mesh_mat_ = tempv.transpose();
		pre_err = err;
		err = Cal_Error(cur_mesh_mat_, 1);
		++counter;
	}

	//update mesh 
	auto points = cur_mesh_.get_vertex_property<Point>("v:point");
	for (auto vit : cur_mesh_.vertices())
	{
		points[vit] = Point(cur_mesh_mat_(0, vit.idx()), cur_mesh_mat_(1, vit.idx()), cur_mesh_mat_(2, vit.idx()));
	}

	return 1;
}

void Dev_LN::mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& vertices_mat, Eigen::Matrix3Xi& faces_mat)
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
				vertices_mat(0, fvit.idx()) = mesh.position(fvit).x;
				vertices_mat(1, fvit.idx()) = mesh.position(fvit).y;
				vertices_mat(2, fvit.idx()) = mesh.position(fvit).z;
				flag(fvit.idx()) = 1;
			}
		}
	}
}

int  Dev_LN::cal_cot_laplace(const Surface_mesh& mesh, Eigen::SparseMatrix<float> &L)
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
