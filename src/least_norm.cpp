#include "Dev_Creator.h"

int Dev_Creator::Deformation()
{
	cur_mesh_ = ori_mesh_;
	cur_mesh_mat_ = ori_mesh_mat_;

	//收集内部顶点下标
	inter_p_.clear();
	bound_p_.clear();
	for (const auto &vit : ori_mesh_.vertices())
		if (!ori_mesh_.is_boundary(vit))
			inter_p_.push_back(vit.idx());

	epsilon_ = std::max(inter_p_.size()*pow(10, -8), pow(10, -5));

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

double Dev_Creator::Cal_Error(const Eigen::Matrix3Xd &V,int flag)
{
	Eigen::Matrix3Xd A_mat;
	cal_angles(V, face_mat_, A_mat);
	Eigen::VectorXd temp_angle;
	temp_angle.resize(vnum_);
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
		double max = pow(10, 6);
		for (int i = 0; i < inter_p_.size(); ++i)
		{
			max = abs(temp_angle(inter_p_[i])) < max ? abs(temp_angle(inter_p_[i])) : max;
		}
		max = 2.0 * M_PI - max;
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

const surface_mesh::Surface_mesh& Dev_Creator::Get_Result() const
{
	return cur_mesh_;
}

int Dev_Creator::Build_Equation()
{
	//清零等式变量
	coeff_A_.resize(inter_p_.size(), vnum_ * 3);
	coeff_A_.setZero();
	right_b_.resize(inter_p_.size());
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
			coeff_vq = (cot_l + cot_r) / sqrnorm(vpq) * vqp - vqqn / cro_pqqn - vqqp / cro_pqqp;
			vec2mat(tri_Coeff_, coeff_vq, i, (*q).idx());

			//累加当前顶点
			sum_theta += theta;
			coeff_vp += (((cot_l + cot_r) / sqrnorm(vpq)) * (vpq));

			//计算下一个相邻顶点
			++pre_q;
			++q;
			++nex_q;
		}
		//当前顶点系数
		vec2mat(tri_Coeff_, coeff_vp, i, inter_p_[i]);

		//--------等式右侧-------------
		right_b_(i) = 2.0 * M_PI - sum_theta;

	}

	//生成稀疏矩阵
	coeff_A_.setFromTriplets(tri_Coeff_.begin(), tri_Coeff_.end());
	return 1;
}

int Dev_Creator::Solve_Problem()
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
	//if (solver.info() != Eigen::Success)
	//{
	//	std::cout << solver.info() << std::endl;
	//	return 0;
	//}
	//std::cout << result_x_ << std::endl;

	if (!result_x_.allFinite())
	{
		std::cout << "Wrong result!" << std::endl;
		return 0;
	}

	return 1;

}

Eigen::MatrixX3d Dev_Creator::Fit_Line(const std::vector<int>& l_idx, int dense)
{
	//导入线段，并弦长参数化
	Eigen::MatrixX3d mp;
	Eigen::VectorXd vt;
	mp.resize(l_idx.size(), Eigen::NoChange);
	vt.resize(l_idx.size());
	mp.setZero();
	vt.setZero();
	for (int i = 0; i < l_idx.size(); ++i)
	{
		mp.row(i) = input_vertice_.col(l_idx[i]);
		if (i >= 1)
		{
			double templ = (input_vertice_.col(l_idx[i]) - input_vertice_.col(l_idx[i - 1])).norm();
			vt(i) = vt(i - 1) + templ;
		}
	}
	vt /= vt(l_idx.size() - 1);

	//拟合曲线
	alglib::real_1d_array tempt;
	alglib::real_1d_array tempx;
	alglib::real_1d_array tempy;
	alglib::real_1d_array tempz;
	tempt.setcontent(vt.size(), vt.data());
	tempx.setcontent(mp.rows(), mp.col(0).data());
	tempy.setcontent(mp.rows(), mp.col(1).data());
	tempz.setcontent(mp.rows(), mp.col(2).data());

	double diff = 0.1;
	alglib::spline1dinterpolant sx, sy, sz;
	alglib::spline1dfitreport x_rep, y_rep, z_rep;
	int x_info, y_info, z_info;
	alglib::spline1dfitpenalized(tempt, tempx, vt.size(), diff, x_info, sx, x_rep);
	alglib::spline1dfitpenalized(tempt, tempy, vt.size(), diff, y_info, sy, y_rep);
	alglib::spline1dfitpenalized(tempt, tempz, vt.size(), diff, z_info, sz, z_rep);

	//提取拟合结果
	Eigen::MatrixX3d result;
	result.resize(dense + 1, Eigen::NoChange);
	result.setZero();
	for (double i = 0; i <= dense; ++i)
	{
		result(i, 0) = spline1dcalc(sx, 1.0 / dense * i);
		result(i, 1) = spline1dcalc(sy, 1.0 / dense * i);
		result(i, 2) = spline1dcalc(sz, 1.0 / dense * i);
	}

	return result;
}

int Dev_Creator::Read_File(const std::string &filename)
{
	std::ifstream is(filename);
	if (!is)
		return 0;

	std::vector<std::vector<int>> edges;
	std::vector<double> tempv;
	std::string line, p;
	double  node[3];
	while (!is.eof())
	{
		std::getline(is, line);
		if (line.empty() || 13 == line[0])
			continue;
		std::istringstream instream(line);

		std::string word;
		instream >> word;

		if ("v" == word || "V" == word)
		{
			instream >> node[0] >> node[1] >> node[2];
			for (size_t j = 0; j < 3; ++j) {
				tempv.push_back(node[j]);
			}
		}
		else if ("e" == word || "E" == word)
		{
			std::vector<int>  es;
			while (!instream.eof())
			{
				instream >> p;
				es.push_back(strtoul(p.c_str(), NULL, 10));
			}
			edges.push_back(es);
		}
	}
	is.close();

	input_vertice_ = (Eigen::Map<Eigen::Matrix3Xd>(tempv.data(), 3, tempv.size() / 3)) / 10;
	input_edges_ = edges;

	return 1;
}

int Dev_Creator::CreatMesh(size_t dense)
{
	//计算网格顶点数
	vnum_ = (dense + 3)*(dense + 1);
	dense_ = dense;
	
	U.setZero();
	D.setZero();
	L.setZero();
	R.setZero();

	//-------------拟合轮廓------------
	U = Fit_Line(input_edges_[0], dense);
	D = Fit_Line(input_edges_[1], dense);
	L = Fit_Line(input_edges_[2], dense);
	R = Fit_Line(input_edges_[3], dense);

	//------------构造曲面-------------
	ori_mesh_mat_.resize(Eigen::NoChange, vnum_);
	ori_mesh_mat_.setZero();
	face_mat_.resize(Eigen::NoChange, 2 * (dense)*(dense + 2));
	face_mat_.setZero();

	//顶点信息
	//Left edge
	int cols = 0;
	for (int i = 0; i <= dense; ++i)
	{
		ori_mesh_mat_.col(cols++) = L.row(dense - i);
	}

	//interal points
	for (int i = 0; i <= dense; ++i)
	{
		for (double j = 0; j <= dense; ++j)
		{
			double temp2[3];
			for (int k = 0; k < 3; ++k)
			{
				temp2[k] = U(i, k)*(j / dense) + D(i, k)*(1 - j / dense);
			}
			ori_mesh_mat_.col(cols++) = Eigen::Vector3d(temp2[0], temp2[1], temp2[2]);
		}
	}
	//Rigeht edge
	for (int i = 0; i <= dense; ++i)
	{
		ori_mesh_mat_.col(cols++) = R.row(dense - i);
	}

	//面信息
	int count = dense;
	int col_idx = 0;
	for (size_t i = 0; i < vnum_ - (dense + 1); ++i)
	{
		if (i == count)
			//跳过末端点
			count += dense + 1;
		else
		{
			//左下三角面
			face_mat_(0, col_idx) = i;
			face_mat_(1, col_idx) = (i + dense + 1);
			face_mat_(2, col_idx++) = i + 1;
			//右上三角面
			face_mat_(0, col_idx) = i + 1;
			face_mat_(1, col_idx) = (i + dense + 1);
			face_mat_(2, col_idx++) = (i + dense + 2);
		}
	}

	//-------------求解Lx=0-------------
	Eigen::SparseMatrix<double> mL;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
	Eigen::MatrixX3d tempx;
	Eigen::MatrixX3d tempb;
	tempx.resize(vnum_, Eigen::NoChange);
	tempb.resize(vnum_, Eigen::NoChange);
	tempx.setZero();
	tempb.setZero();

	cal_topo_laplace(ori_mesh_mat_, face_mat_, mL);
	mL = (mL.transpose() * mL).eval();

	//为固定轮廓添加大权值
	const double weight = pow(10, 6);
	//left
	for (int i = 0; i < dense; ++i)
	{
		mL.coeffRef(i, i) += weight;
		tempb.row(i) += weight * ori_mesh_mat_.col(i).transpose();
		bound_p_.push_back(i);
	}
	//buttom
	for (int i = dense + 1; i < (dense + 1)*(dense + 3); i += dense + 1)
	{
		mL.coeffRef(i, i) += weight;
		tempb.row(i) += weight * ori_mesh_mat_.col(i).transpose();
		bound_p_.push_back(i);
	}
	//up
	for (int i = dense; i < (dense + 1)*(dense + 3); i += dense + 1)
	{
		mL.coeffRef(i, i) += weight;
		tempb.row(i) += weight * ori_mesh_mat_.col(i).transpose();
		bound_p_.push_back(i);
	}
	//right
	for (int i = (dense + 1)*(dense + 3) - 1; i > (dense + 1)*(dense + 2); --i)
	{
		mL.coeffRef(i, i) += weight;
		tempb.row(i) += weight * ori_mesh_mat_.col(i).transpose();
		bound_p_.push_back(i);
	}

	//求解
	solver.compute(mL);
	if (solver.info() != Eigen::Success)
	{
		std::cout << solver.info() << std::endl;
	}

	tempx = solver.solve(tempb);

	//-----------保存构造的网格-----------
	ori_mesh_mat_ = tempx.transpose();

	ori_mesh_.clear();
	for (int i = 0; i < ori_mesh_mat_.cols(); ++i)
	{
		ori_mesh_.add_vertex(Point(ori_mesh_mat_(0, i), ori_mesh_mat_(1, i), ori_mesh_mat_(2, i)));
	}
	for (int i = 0; i < face_mat_.cols(); ++i)
	{
		ori_mesh_.add_triangle(Surface_mesh::Vertex(face_mat_(0, i)), Surface_mesh::Vertex(face_mat_(1, i)), Surface_mesh::Vertex(face_mat_(2, i)));
	}


	//-----------Visualizer------------
	//将网格导入点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	for (auto vit : ori_mesh_.vertices())
	{
		cloud->push_back(pcl::PointXYZ(ori_mesh_.position(vit).x, ori_mesh_.position(vit).y, ori_mesh_.position(vit).z));
	}

	pcl::PolygonMesh::Ptr polygon_ptr(new pcl::PolygonMesh);
	toPCLPointCloud2(*cloud, polygon_ptr->cloud);

	std::vector<pcl::Vertices> polygon;
	for (auto fit : ori_mesh_.faces())
	{
		pcl::Vertices vt;
		for (auto fvit : ori_mesh_.vertices(fit))
			vt.vertices.push_back(fvit.idx());
		polygon_ptr->polygons.push_back(vt);
	}

	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addPolygonMesh(*polygon_ptr);

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}

	return 1;
}

int Dev_Creator::SetCondition(double delta, size_t times)
{
	epsilon_ = delta;
	it_count_ = times;
	return 1;
}

int Dev_Creator::Update_Mesh()
{
	//构造矩阵 
	Eigen::SparseMatrix<double> coeff_ls;
	cal_cot_laplace(cur_mesh_mat_, face_mat_, coeff_ls);
	coeff_ls.resize(vnum_ * 2, vnum_);
	coeff_ls.reserve(Eigen::VectorXi::Constant(vnum_ * 2, 1));
	for (int i = 0; i < vnum_; ++i)
	{
		coeff_ls.insert(i + vnum_, i) = 1;
	}
	coeff_ls.makeCompressed();
	Eigen::MatrixX3d b_ls;
	b_ls.resize(2 * vnum_, 3);
	b_ls.setZero();
	//const double weight = 0;
	const double weight = pow(10, 3);
	for (int i = 0; i < dense_; ++i)
	{
		coeff_ls.coeffRef(i, i) += weight;
		b_ls.row(i) += weight * cur_mesh_mat_.col(i).transpose();
	}
	//buttom
	for (int i = dense_ + 1; i < (dense_ + 1)*(dense_ + 3); i += dense_ + 1)
	{
		coeff_ls.coeffRef(i, i) += weight;
		b_ls.row(i) += weight * cur_mesh_mat_.col(i).transpose();
	}
	//up
	for (int i = dense_; i < (dense_ + 1)*(dense_ + 3); i += dense_ + 1)
	{
		coeff_ls.coeffRef(i, i) += weight;
		b_ls.row(i) += weight * cur_mesh_mat_.col(i).transpose();
	}
	//right
	for (int i = (dense_ + 1)*(dense_ + 3) - 1; i > (dense_ + 1)*(dense_ + 2); --i)
	{
		coeff_ls.coeffRef(i, i) += weight;
		b_ls.row(i) += weight * cur_mesh_mat_.col(i).transpose();
	}


	//cout << coeff_ls << endl;

	Eigen::MatrixX3d tempv;
	tempv.resize(vnum_, Eigen::NoChange);
	for (int i = 0; i < cur_mesh_mat_.cols(); ++i)
	{
		tempv(i, 0) = cur_mesh_mat_(0, i) + result_x_(3 * i);
		tempv(i, 1) = cur_mesh_mat_(1, i) + result_x_(3 * i + 1);
		tempv(i, 2) = cur_mesh_mat_(2, i) + result_x_(3 * i + 2);
	}
	int counter = 0;
	double pre_err = pow(10, 6), err;
	err = abs(Cal_Error(tempv.transpose(), 1));
	while (err > 0.001 && counter <= 20 && pre_err - err > 0.01)
	{
		b_ls.bottomRows(vnum_) = tempv;

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		Eigen::MatrixX3d tempb;
		//Eigen::SparseMatrix<double> temp_coeffT = ;
		Eigen::SparseMatrix<double> tempA = (coeff_ls.transpose()*coeff_ls).eval();
		solver.compute(tempA);

		if (solver.info() != Eigen::Success)
		{
			std::cout << solver.info() << std::endl;
		}
		tempb = (coeff_ls.transpose() * b_ls).eval();
		tempv = solver.solve(tempb);

		pre_err = err;
		err = Cal_Error(tempv.transpose(), 1);
		++counter;
	}
	cur_mesh_mat_ = tempv.transpose();
	auto points = cur_mesh_.get_vertex_property<Point>("v:point");
	for (auto vit : cur_mesh_.vertices())
	{
		points[vit] = Point(cur_mesh_mat_(0, vit.idx()), cur_mesh_mat_(1, vit.idx()), cur_mesh_mat_(2, vit.idx()));
	}
	return 1;
}

int Dev_Creator::cal_topo_laplace(const Eigen::MatrixXd &V, const Eigen::Matrix3Xi &F, Eigen::SparseMatrix<double> &L)
{
	const size_t num_faces = F.cols();
	std::vector<Eigen::Triplet<double>> triple;
	triple.reserve(num_faces * 9);
	for (size_t j = 0; j < num_faces; ++j) {
		const Eigen::Vector3i &fv = F.col(j);
		for (size_t vi = 0; vi < 3; ++vi) {
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			triple.push_back(Eigen::Triplet<double>(fv0, fv0, 1));
			triple.push_back(Eigen::Triplet<double>(fv0, fv1, -0.5));
			triple.push_back(Eigen::Triplet<double>(fv0, fv2, -0.5));
		}
	}
	L.resize(V.cols(), V.cols());
	L.setFromTriplets(triple.begin(), triple.end());
	return 1;
}

int Dev_Creator::cal_cot_laplace(const Eigen::MatrixXd &V, const Eigen::Matrix3Xi &F, Eigen::SparseMatrix<double> &L)
{
	Eigen::Matrix3Xd angles;
	cal_angles(V, F, angles);
	std::vector<Eigen::Triplet<double>> triple;
	triple.reserve(F.cols() * 9);
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
			triple.push_back(Eigen::Triplet<double>(fv0, fv0, 1 / tan(ca[j1]) + 1 / tan(ca[j2])));
			triple.push_back(Eigen::Triplet<double>(fv0, fv1, -1 / tan(ca[j2])));
			triple.push_back(Eigen::Triplet<double>(fv0, fv2, -1 / tan(ca[j1])));
		}
	}
	L.resize(V.cols(), V.cols());
	L.setFromTriplets(triple.begin(), triple.end());
	return 1;
}

void Dev_Creator::cal_angles(const Eigen::Matrix3Xd &V, const Eigen::Matrix3Xi &F, Eigen::Matrix3Xd &angles)
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

