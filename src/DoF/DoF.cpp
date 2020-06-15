#include "Dof.h"

Eigen::VectorXi VType_;
Eigen::Matrix3Xi matF_;
MatrixType matV_;
std::vector<std::vector<Eigen::Vector2i>> adjF_;
VectorType areas_;
VectorType vecA_;
MatrixType matA_;
MatrixType matN_;

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Load failed!" << std::endl;
	}
	//收集内部顶点下标
	VType_.setConstant(mesh.n_vertices(), 0);
	for (const auto& vit : mesh.vertices())
	{
		VType_(vit.idx()) = mesh.is_boundary(vit) ? -1 : 1;
	}

	//网格初始信息收集
	mesh2matrix(mesh, matV_, matF_);
	Zombie::cal_angles_and_areas(matV_, matF_, vecA_, areas_, matA_);
	Zombie::cal_normal_per_face(matV_, matF_, matN_);
	adj_face(matV_.cols(), matF_, adjF_);
	MatrixType oriV(matV_);
	VectorType oriA(vecA_);
	std::cout << "初始最大误差： " << cal_error(vecA_, VType_, 1) << std::endl;
	std::cout << "初始平均误差： " << cal_error(vecA_, VType_, 0) << std::endl;

	//--------------测试---------------
	alglib::real_1d_array xx;
	xx.attach_to_ptr(matV_.cols() * 3, matV_.data());
	double f(0.0);
	alglib::real_1d_array g;
	grad_function(xx, f, g, nullptr);

	alglib::real_1d_array x;
	//x.setcontent(matV_.cols() * 3, matV_.data());
	x.attach_to_ptr(matV_.cols() * 3, matV_.data());
	//bool flag = grad_function_test(x, grad_function, ori_function);

	double epsg = 0.0;
	double epsf = 0.0;
	double epsx = 0.0;
	double stpmax = 0.0;
	alglib::ae_int_t maxits = 0;
	alglib::minlbfgsstate state;
	alglib::minlbfgsreport rep;

	// create and tune optimizer
	alglib::minlbfgscreate(5.0, x, state);
	alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
	alglib::minlbfgssetstpmax(state, stpmax);
	VectorType vs;
	vs.resize(matV_.cols() * 3);
	memset(vs.data(), 100, sizeof(double) * vs.size());
	alglib::real_1d_array scalar;
	scalar.setcontent(vs.size(), vs.data());
	alglib::minlbfgssetscale(state, scalar);

	////OptGuard is essential at the early prototyping stages.
	//minlbfgsoptguardsmoothness(state);
	//minlbfgsoptguardgradient(state, 1);

	// first run
	alglib::minlbfgsoptimize(state, grad_function);
	alglib::real_1d_array rex;
	alglib::minlbfgsresults(state, rex, rep);

	std::cout << "迭代次数 : " << rep.iterationscount << std::endl;
	std::cout << "梯度计算次数 : " << rep.nfev << std::endl;
	std::cout << "终止情况 : " << rep.terminationtype << std::endl;

	//alglib::optguardreport ogrep;
	//minlbfgsoptguardresults(state, ogrep);
	//printf("%s\n", ogrep.badgradsuspected ? "true" : "false"); // EXPECTED: false
	//printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
	//printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false

	MatrixType curV = Eigen::Map<MatrixType>(rex.getcontent(), 3, matV_.cols());
	Zombie::cal_angles(curV, matF_, vecA_);
	std::cout << "最终最大误差： " << cal_error(vecA_, VType_, 1) << std::endl;
	std::cout << "最终平均误差： " << cal_error(vecA_, VType_, 0) << std::endl;

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	Zombie::visualize_mesh(renderer1, curV, matF_, vecA_, VType_);
	renderer1->SetViewport(0.0, 0.0, 0.5, 1.0);

	//添加文本
	auto textActor1 = vtkSmartPointer<vtkTextActor>::New();
	textActor1->SetInput("Result Mesh");
	textActor1->GetTextProperty()->SetFontSize(33);
	textActor1->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	renderer1->AddActor2D(textActor1);

	//视角设置
	renderer1->ResetCamera();
	renderWindow->AddRenderer(renderer1);

	auto renderer2 = vtkSmartPointer<vtkRenderer>::New();
	Zombie::visualize_mesh(renderer2, oriV, matF_, oriA, VType_);
	renderer2->SetViewport(0.5, 0.0, 1.0, 1.0);

	//添加文本
	auto textActor2 = vtkSmartPointer<vtkTextActor>::New();
	textActor2->SetInput("Original Mesh");
	textActor2->GetTextProperty()->SetFontSize(33);
	textActor2->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	renderer2->AddActor2D(textActor2);

	//视角设置
	renderer2->ResetCamera();
	renderWindow->AddRenderer(renderer2);

	auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(renderWindow);
	auto style = vtkInteractorStyleTrackballCamera::New();
	interactor->SetInteractorStyle(style);
	interactor->Initialize();
	//interactor->CreateRepeatingTimer(1000);

	//auto timeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	//timeCallback->SetCallback(CallbackFunction);
	//timeCallback->SetClientData(renderer1->GetActors()->GetLastActor()->GetMapper()->GetInput());

	//interactor->AddObserver(vtkCommand::TimerEvent, timeCallback);

	//开始
	renderWindow->Render();
	interactor->Start();

	return EXIT_SUCCESS;
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F)
{
	F.resize(3, mesh.n_faces());
	V.resize(3, mesh.n_vertices());

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
				V.col(fvit.idx()) = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data()).cast<DataType>();
				flag(fvit.idx()) = 1;
			}
		}
	}
}

void adj_face(int Vnum, const Eigen::Matrix3Xi& F, std::vector<std::vector<Eigen::Vector2i>>& adjF)
{
	adjF.resize(Vnum);
	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::Vector3i& fv = F.col(i);
		for (int j = 0; j < 3; ++j)
		{
			adjF[fv[j]].push_back(Eigen::Vector2i(i, j));
		}
	}
}

double cal_error(const VectorType& vecAngles, const Eigen::VectorXi& VType_, int flag)
{
	//计算最大误差或平均误差
	if (flag)
	{
		size_t idx = 0;
		double max = 0;
		for (int i = 0; i < VType_.size(); ++i)
		{
			if (VType_(i) != -1)
			{
				const double e = abs(2.0 * M_PI - vecAngles(i)) / areas_(i);
				//max = e > max ? e : max;
				if (e > max)
				{
					max = e;
					idx = i;
				}
			}
		}
		//std::cout << idx << std::endl;
		return max;
	}
	else
	{
		double averange = 0;
		int cnt = 0;
		for (size_t i = 0; i < VType_.size(); ++i)
		{
			if (VType_(i) != -1)
			{
				averange += abs(2.0 * M_PI - vecAngles(i)) / areas_(i);
				++cnt;
			}
		}
		averange /= double(cnt);
		return averange;
	}
}

void grad_function(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)
{
	alglib::real_1d_array temp(x);
	Eigen::Map<MatrixType> curV(temp.getcontent(), 3, matV_.cols());
	func = 0.0;
	VectorType Gradient;
	Gradient.setConstant(curV.cols() * 3, 0);

	VectorType vecAngles;
	VectorType vecAreas;
	MatrixType matAngles;
	MatrixType matNormals;
	Zombie::cal_angles_and_areas(curV, matF_, vecAngles, vecAreas, matAngles);
	Zombie::cal_normal_per_face(curV, matF_, matNormals);

	MatrixType matX(3, curV.cols());
	for (size_t i = 0; i < adjF_.size(); ++i)
	{
		std::vector<Eigen::Vector2i>& adjf = adjF_[i];

		Eigen::Matrix3d NNT(Eigen::Matrix3d::Zero());
		for (auto it : adjf)
		{
			const PosVector& n = matNormals.col(it(0));
			NNT += matAngles(it(1), it(0)) * (n * n.transpose());
		}

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(NNT);
		const VectorType& v(solver.eigenvalues());
		int idx(0);
		v.minCoeff(&idx);
		func += solver.eigenvalues()(idx);
		matX.col(i) = solver.eigenvectors().col(idx);
	}

	//梯度部分
	for (int i = 0; i < matF_.cols(); ++i)
	{
		const auto& fv = matF_.col(i);
		const auto& p0 = curV.col(fv[0]);
		const auto& p1 = curV.col(fv[1]);
		const auto& p2 = curV.col(fv[2]);
		const auto& n = matNormals.col(i);

		Eigen::Matrix3d e;
		e.col(0) = p1 - p0;
		e.col(1) = p2 - p1;
		e.col(2) = p0 - p2;
		const double area(e.col(0).cross(e.col(1)).norm());
		eigen_assert(area);

		for (int j = 0; j < 3; ++j)
		{
			const PosVector& xx(matX.col(fv[j]));
			double xTn = xx.dot(n);

			const PosVector c1 = xTn * xTn * n.cross((e.col(j)).normalized());
			srhs(Gradient, -c1, fv[j]);
			srhs(Gradient, c1, fv[(j + 1) % 3]);
			const PosVector c2 = xTn * xTn * n.cross((e.col((j + 2) % 3)).normalized());
			srhs(Gradient, -c2, fv[j]);
			srhs(Gradient, c2, fv[(j + 2) % 3]);
			Eigen::Matrix3d gradN((e.col((j + 1) % 3).cross(n) * n.transpose()) / area);
			const PosVector c3 = 2.0 * matAngles(j, i) * xTn * gradN.transpose() * xx;
			srhs(Gradient, c3, fv[j]);
		}
	}
	//std::cout << "Energy: " << func << std::endl;
	//std::cout << "---------------------------" << std::endl;
	////std::cout << "Gradient: " << Gradient.norm() << std::endl;
	//std::cout << "Gradient: " << Gradient << std::endl;
	//
	for (int i = 0; i < grad.length(); ++i)
	{
		grad[i] = Gradient(i);
	}
}