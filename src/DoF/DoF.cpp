#include "Dof.h"

#define MAXIT 100

int counter_ = 0;

std::vector<int> interV_;
std::vector<int> boundV_;
Eigen::VectorXi interVidx_;
Eigen::Matrix3Xi matF_;
MatrixType matV_;
VectorType orivecV_;
std::vector<std::vector<Eigen::Vector2i>> adjF_;
VectorType areas_;
VectorType vecA_;
MatrixType matA_;
MatrixType matN_;

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (counter_ < MAXIT)
	{
		//Solve(solver_, At_, matV_, matE_, matF_, matAngles_, vecAngles_, interVidx_, boundV_);

		std::cout << "----------------------" << std::endl;
		std::cout << "第" << counter_++ << "次迭代，最大误差为： " << cal_error(vecA_, interV_, 1) << "，平均误差为： " << cal_error(vecA_, interV_, 0) << std::endl;

		//--------------可视化更新---------------------
		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV_.cols());
		for (auto i = 0; i < vecA_.size(); ++i)
		{
			if (interVidx_(i) != -1)
				scalar->InsertTuple1(i, abs(2.0f * M_PI - vecA_(i)));
			else
				scalar->InsertTuple1(i, 0);
		}

		auto polydata = static_cast<vtkPolyData*>(clientData);
		auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);

		auto points = vtkSmartPointer<vtkPoints>::New();
		for (int i = 0; i < matV_.cols(); ++i)
		{
			points->InsertNextPoint(matV_.col(i).data());
		}
		polydata->SetPoints(points);
		polydata->GetPointData()->SetScalars(scalar);
		polydata->Modified();;

		iren->Render();
	}
}

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Load failed!" << std::endl;
	}
	//收集内部顶点下标
	interVidx_.setConstant(mesh.n_vertices() + 1, -1);
	int count = 0;
	for (const auto& vit : mesh.vertices())
	{
		if (!mesh.is_boundary(vit))
		{
			interV_.push_back(vit.idx());
			interVidx_(vit.idx()) = count++;
		}
		else
		{
			boundV_.push_back(vit.idx());
		}
	}
	interVidx_(mesh.n_vertices()) = count;

	//网格初始信息收集
	mesh2matrix(mesh, matV_, matF_);
	cal_angles_and_areas_and_normal(matV_, matF_, matA_, vecA_, areas_, matN_);
	adj_face(matV_.cols(), matF_, adjF_);
	MatrixType oriV(matV_);
	VectorType oriA(vecA_);
	std::cout << "初始最大误差： " << cal_error(vecA_, interV_, 1) << std::endl;
	std::cout << "初始平均误差： " << cal_error(vecA_, interV_, 0) << std::endl;

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
	cal_angles(curV, matF_, vecA_);
	std::cout << "最终最大误差： " << cal_error(vecA_, interV_, 1) << std::endl;
	std::cout << "最终平均误差： " << cal_error(vecA_, interV_, 0) << std::endl;

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer1, curV, matF_, vecA_, interVidx_);
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
	visualize_mesh(renderer2, oriV, matF_, oriA, interVidx_);
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

void cal_angles(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, VectorType& vecAngles)
{
	vecAngles.resize(V.cols());
	vecAngles.setZero();
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const PosVector& p0 = V.col(fv[vi]);
			const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
			const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			vecAngles(F(vi, f)) += angle;
		}
	}
}

void cal_angles_and_areas_and_normal(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& matAngles, VectorType& vecAngles, VectorType& areas, MatrixType& matNormal)
{
	areas.setConstant(V.cols(), 0);
	vecAngles.setConstant(V.cols(),0);
	matAngles.setConstant(3, F.cols(), 0);
	matNormal.setConstant(3, F.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);

		const PosVector e01(V.col(fv[1]) - V.col(fv[0]));
		const PosVector e02(V.col(fv[2]) - V.col(fv[0]));
		const PosVector crosstemp = e01.cross(e02);
		//Mix area
		double area = crosstemp.norm() / 6.0f;
		matNormal.col(f) = crosstemp.normalized();

		for (size_t vi = 0; vi < 3; ++vi)
		{
			areas(fv[vi]) += area;
			matNormal.col(fv[vi]) += crosstemp;
			const PosVector& p0 = V.col(fv[vi]);
			const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
			const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			matAngles(vi, f) = angle;
			vecAngles(F(vi, f)) += angle;
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

double cal_error(const VectorType& vecAngles, const std::vector<int>& interIdx, int flag)
{
	//计算最大误差或平均误差
	if (flag)
	{
		size_t idx = 0;
		double max = 0;
		for (size_t i = 0; i < interIdx.size(); ++i)
		{
			const double e = abs(2.0 * M_PI - vecAngles(interIdx[i])) / areas_(interIdx[i]);
			//max = e > max ? e : max;
			if (e > max)
			{
				max = e;
				idx = i;
			}
		}
		//std::cout << idx << std::endl;
		return max;
	}
	else
	{
		double averange = 0;
		for (size_t i = 0; i < interIdx.size(); ++i)
		{
			averange += abs(2.0 * M_PI - vecAngles(interIdx[i])) / areas_(interIdx[i]);
		}
		averange /= double(interIdx.size());
		return averange;
	}
}

void cal_cot_laplace(const Eigen::Matrix3Xi& F, MatrixTypeConst& mAngles, const VectorType& areas, const Eigen::VectorXi& interVidx, SparseMatrixType& L)
{
	//计算固定边界的cot权拉普拉斯系数矩阵
	std::vector<Tri> triple;
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		const PosVector& ca = mAngles.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (interVidx(fv0) != -1)
			{
				const DataType temp0 = (1.0 / std::tan(ca[(vi + 1) % 3]) + 1.0 / std::tan(ca[(vi + 2) % 3])) / (2.0 * areas(fv0));
				const DataType temp1 = -1.0 / std::tan(ca[(vi + 2) % 3]) / (2.0 * areas(fv0));
				const DataType temp2 = -1.0 / std::tan(ca[(vi + 1) % 3]) / (2.0 * areas(fv0));
				for (int j = 0; j < 3; ++j)
				{
					triple.push_back(Tri(fv0 * 3 + j, fv0 * 3 + j, temp0));
					triple.push_back(Tri(fv0 * 3 + j, fv1 * 3 + j, temp1));
					triple.push_back(Tri(fv0 * 3 + j, fv2 * 3 + j, temp2));
				}
			}
		}
	}
	L.resize(areas.size() * 3, areas.size() * 3);
	L.setFromTriplets(triple.begin(), triple.end());
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
	cal_angles_and_areas_and_normal(curV, matF_, matAngles, vecAngles, vecAreas, matNormals);

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
		int idx = (v(0) <= v(1)) ? 0 : 1;
		idx = (v(2) <= v(idx)) ? 2 : idx;
		func += solver.eigenvalues()(idx);
		matX.col(i) = solver.eigenvectors().col(idx);
	}

	//梯度部分
	for (int i = 0; i < matF_.cols(); ++i)
	{
		const Eigen::Vector3i& fv = matF_.col(i);
		const PosVector& p0 = curV.col(fv[0]);
		const PosVector& p1 = curV.col(fv[1]);
		const PosVector& p2 = curV.col(fv[2]);
		const PosVector& n = matNormals.col(i);

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
	std::cout << "Energy: " << func << std::endl;
	std::cout << "---------------------------" << std::endl;
	//std::cout << "Gradient: " << Gradient.norm() << std::endl;
	std::cout << "Gradient: " << Gradient << std::endl;
	
	for (int i = 0; i < grad.length(); ++i)
	{
		grad[i] = Gradient(i);
	}
}

void matrix2vtk(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, vtkPolyData* P)
{
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < V.cols(); ++i)
		points->InsertNextPoint(V.col(i).data());

	auto faces = vtkSmartPointer <vtkCellArray>::New();
	for (int i = 0; i < F.cols(); ++i)
	{
		auto triangle = vtkSmartPointer<vtkTriangle>::New();
		for (int j = 0; j < 3; ++j)
			triangle->GetPointIds()->SetId(j, F(j, i));
		faces->InsertNextCell(triangle);
	}
	P->SetPoints(points);
	P->SetPolys(faces);
}

void MakeLUT(vtkDoubleArray* Scalar, vtkLookupTable* LUT)
{
	auto ctf = vtkSmartPointer<vtkColorTransferFunction>::New();
	ctf->SetColorSpaceToHSV();
	ctf->AddRGBPoint(0.0, 0.1, 0.3, 1);
	ctf->AddRGBPoint(0.25, 0.55, 0.65, 1);
	ctf->AddRGBPoint(0.5, 1, 1, 1);
	ctf->AddRGBPoint(0.75, 1, 0.65, 0.55);
	ctf->AddRGBPoint(1.0, 1, 0.3, 0.1);

	LUT->SetNumberOfColors(256);
	for (auto i = 0; i < LUT->GetNumberOfColors(); ++i)
	{
		Eigen::Vector4d color;
		ctf->GetColor(double(i) / LUT->GetNumberOfColors(), color.data());
		color(3) = 1.0;
		LUT->SetTableValue(i, color.data());
	}
	LUT->Build();
}

void visualize_mesh(vtkRenderer* Renderer, MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const VectorType& angles, const Eigen::VectorXi& interVidx)
{
	//生成网格
	auto P = vtkSmartPointer<vtkPolyData>::New();
	matrix2vtk(V, F, P);

	auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
	scalar->SetNumberOfComponents(1);
	scalar->SetNumberOfTuples(V.cols());
	for (auto i = 0; i < angles.size(); ++i)
	{
		if (interVidx(i) != -1)
			scalar->InsertTuple1(i, abs(2.0f * M_PI - angles(i)));
		else
			scalar->InsertTuple1(i, 0);
	}
	P->GetPointData()->SetScalars(scalar);

	auto lut = vtkSmartPointer<vtkLookupTable>::New();
	MakeLUT(scalar, lut);

	auto scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
	scalarBar->SetLookupTable(lut);
	scalarBar->SetTitle("Curvature Error");
	scalarBar->SetNumberOfLabels(4);

	//网格及法向渲染器
	auto polyMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	polyMapper->SetInputData(P);
	polyMapper->SetLookupTable(lut);
	polyMapper->SetScalarRange(scalar->GetValueRange()[0], scalar->GetValueRange()[1]);
	//polyMapper->SetScalarRange(scalar->GetValueRange()[0], 2.0);

	auto polyActor = vtkSmartPointer<vtkActor>::New();
	polyActor->SetMapper(polyMapper);
	polyActor->GetProperty()->SetDiffuseColor(1, 1, 1);
	Renderer->AddActor(polyActor);
	Renderer->AddActor2D(scalarBar);
}

