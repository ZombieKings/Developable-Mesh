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

double Dev_LN::Cal_Error(const Eigen::Matrix3Xd &V, int flag)
{
	Eigen::Matrix3Xd A_mat;
	cal_angles(V, face_mat_, A_mat);
	Eigen::VectorXd temp_angle;
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
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
	Eigen::SparseMatrix<double> tempAT = Eigen::SparseMatrix<double>(coeff_A_.transpose());
	Eigen::SparseMatrix<double> tempA = (coeff_A_ * tempAT).eval();

	solver.analyzePattern(tempA);
	solver.compute(tempA);
	if (solver.info() != Eigen::Success)
	{
		std::cout << solver.info() << std::endl;
		return 0;
	}

	result_x_.setZero();
	Eigen::VectorXd tempx(result_x_);
	//std::cout << "right b: " << std::endl;
	//std::cout << right_b_ << std::endl;

	tempx = solver.solve(right_b_);
	result_x_ = tempAT * tempx;

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
	//构造矩阵 
	Eigen::SparseMatrix<double> coeff_ls;
	cal_cot_laplace(cur_mesh_mat_, face_mat_, coeff_ls);
	coeff_ls.conservativeResize(vnum_ * 2, vnum_);
	coeff_ls.reserve(Eigen::VectorXi::Constant(vnum_ * 2, 50));
	for (int i = 0; i < vnum_; ++i)
	{
		coeff_ls.insert(i + vnum_, i) = 1;
	}
	coeff_ls.makeCompressed();

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
	Eigen::MatrixX3d tempb;
	Eigen::SparseMatrix<double> tempA = (coeff_ls.transpose() * coeff_ls).eval();
	solver.compute(tempA);

	Eigen::MatrixX3d b_ls;
	b_ls.resize(2 * vnum_, 3);
	b_ls.setZero();
	Eigen::MatrixX3d tempv;
	tempv.resize(vnum_, Eigen::NoChange);
	Eigen::Matrix3Xd tempmat(cur_mesh_mat_);
	for (int i = 0; i < vnum_; ++i)
	{
		tempv(i, 0) = cur_mesh_mat_(0, inter_p_[i]) + result_x_(3 * i);
		tempv(i, 1) = cur_mesh_mat_(1, inter_p_[i]) + result_x_(3 * i + 1);
		tempv(i, 2) = cur_mesh_mat_(2, inter_p_[i]) + result_x_(3 * i + 2);
		tempmat.col(inter_p_[i]) = tempv.row(i).transpose();
	}
	int counter = 0;
	double pre_err = pow(10, 6), err;
	err = Cal_Error(tempmat, 1);
	while (err > 0.001)
	{
		if (counter > 20 && pre_err - err < 0.01)
			break;

		b_ls.bottomRows(vnum_) = tempv;

		if (solver.info() != Eigen::Success)
		{
			std::cout << solver.info() << std::endl;
		}
		tempb = (coeff_ls.transpose() * b_ls).eval();
		tempv = solver.solve(tempb);

		for (int i = 0; i < vnum_; ++i)
		{
			tempmat.col(inter_p_[i]) = tempv.row(i).transpose();
		}
		pre_err = err;
		err = Cal_Error(tempmat, 1);
		++counter;
	}

	//update mesh matrix
	for (int i = 0; i < tempv.rows(); ++i)
	{
		cur_mesh_mat_.col(inter_p_[i]) = tempv.row(i);
	}
	//update mesh 
	auto points = cur_mesh_.get_vertex_property<Point>("v:point");
	for (int i = 0; i < tempv.rows(); ++i)
	{
		points[Surface_mesh::Vertex(inter_p_[i])] = Point(tempv(i, 0), tempv(i, 1), tempv(i, 2));
	}

	////update mesh matrix
	//Eigen::MatrixX3d tempv;
	//tempv.resize(vnum_, Eigen::NoChange);
	//for (int i = 0; i < vnum_; ++i)
	//{
	//	tempv(i, 0) = cur_mesh_mat_(0, inter_p_[i]) + result_x_(3 * i);
	//	tempv(i, 1) = cur_mesh_mat_(1, inter_p_[i]) + result_x_(3 * i + 1);
	//	tempv(i, 2) = cur_mesh_mat_(2, inter_p_[i]) + result_x_(3 * i + 2);
	//}
	//for (int i = 0; i < tempv.rows(); ++i)
	//{
	//	cur_mesh_mat_.col(inter_p_[i]) = tempv.row(i);
	//}
	////update mesh 
	//auto points = cur_mesh_.get_vertex_property<Point>("v:point");
	//for (int i = 0; i < tempv.rows(); ++i)
	//{
	//	points[Surface_mesh::Vertex(inter_p_[i])] = Point(tempv(i,0), tempv(i, 1), tempv(i, 2));
	//}
	return 1;
}



void Dev_LN::mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xd& vertices_mat, Eigen::Matrix3Xi& faces_mat)
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

void Dev_LN::cal_angles(const Eigen::Matrix3Xd &V, const Eigen::Matrix3Xi &F, Eigen::Matrix3Xd &angles)
{
	angles.resize(3, F.cols());
	for (size_t j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i &fv = F.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const Eigen::VectorXd &p0 = V.col(fv[vi]);
			const Eigen::VectorXd &p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::VectorXd &p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			angles(vi, j) = angle;
		}
	}
}

int Dev_LN::cal_cot_laplace(const Eigen::MatrixXd &V, const Eigen::Matrix3Xi &F, Eigen::SparseMatrix<double> &L)
{
	Eigen::Matrix3Xd angles;
	cal_angles(V, F, angles);
	std::vector<Eigen::Triplet<double>> triple;

	Eigen::VectorXd area;
	area.resize(vnum_);
	area.setZero();
	double sum_area = 0;
	for (size_t j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i &fv = F.col(j);
		const Eigen::Vector3d &ca = angles.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const size_t j1 = (vi + 1) % 3;
			const size_t j2 = (vi + 2) % 3;
			const int fv0 = fv[vi];
			const int fv1 = fv[j1];
			const int fv2 = fv[j2];
			if (inter_p_r_(fv0) < vnum_)
			{
				triple.push_back(Eigen::Triplet<double>(inter_p_r_(fv0), inter_p_r_(fv0), 1 / tan(ca[j1]) + 1 / tan(ca[j2])));
				if (inter_p_r_(fv1) < vnum_)
				{
					triple.push_back(Eigen::Triplet<double>(inter_p_r_(fv0), inter_p_r_(fv1), -1 / tan(ca[j2])));
				}
				if (inter_p_r_(fv2) < vnum_)
				{
					triple.push_back(Eigen::Triplet<double>(inter_p_r_(fv0), inter_p_r_(fv2), -1 / tan(ca[j1])));
				}
			}
		}
		//area coefficient
		const Eigen::VectorXd &p0 = V.col(fv[0]);
		const Eigen::VectorXd &p1 = V.col(fv[1]);
		const Eigen::VectorXd &p2 = V.col(fv[2]);
		double tempA = ((p1 - p0).norm() * (p2 - p0).norm() *std::sin(ca(0))) / 2;
		for (int i = 0; i < 3; ++i)
			if (inter_p_r_(fv[i]) < vnum_)
				area(inter_p_r_(fv[i])) += tempA;

		sum_area += tempA;
	}
	L.resize(vnum_, vnum_);
	L.setFromTriplets(triple.begin(), triple.end());
	L /= (sum_area / vnum_);
	for (int i = 0; i < L.rows(); ++i)
	{
		L.row(i) *= area(i);
	}
	return 1;
}
