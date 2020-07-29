#include "least_norm.h"

bool terminal_ = false;
unsigned int counter_ = 0;

double epsilon_ = 0;
double dqn_ = 1.;
double theta_ = 0;
double pretheta_ = 0;

int innerNum_ = 0;
Eigen::VectorXi VType_;
Eigen::Matrix3Xi matF_;
MatrixType matV_;

MatrixType oriV_;

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (dqn_ >= epsilon_ && counter_ < 50)
	{
		//----compute update vector-----
		VectorType update_d;
		compute_update_vector(matV_, matF_, VType_, innerNum_, update_d);
		if (!update_d.allFinite())
			std::cout << "Wrong result!" << std::endl;
		//---------update mesh----------
		if (theta_ >= 0.001 || counter_ <= 20 || (pretheta_ - theta_) >= 0.01)
			update_points(matV_, matF_, VType_, innerNum_, update_d);
		else
			matV_.array() += update_d.array();

		VectorType vA;
		Zombie::cal_angles(matV_, matF_, vA);
		dqn_ = update_d.squaredNorm();
		pretheta_ = theta_;
		theta_ = cal_error(vA, VType_, 1);
		std::cout << "第" << counter_++ << "次迭代，最大误差为： " << theta_ << "，平均误差为： " << cal_error(vA, VType_, 0) << std::endl;

		//--------------可视化更新---------------------
		auto polydata = static_cast<vtkPolyData*>(clientData);
		auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);

		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV_.cols());
		for (auto i = 0; i < vA.size(); ++i)
		{
			if (VType_(i) != -1)
				scalar->InsertTuple1(i, abs(2.0f * M_PI - vA(i)));
			else
				scalar->InsertTuple1(i, 0);
		}

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
	//else if (!flag)
	//{
	//	double e1 = cal_error(matV_, matF_, angle_mat, 1);
	//	double e2 = cal_error(matV_, matF_, angle_mat, 0);
	//	std::cout << "共" << counter << "次迭代，优化结果最大误差为： " << e1 << "，平均误差为： " << e2 << std::endl;
	//	flag = true;
	//}
}

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
		if (mesh.is_boundary(vit))
			VType_(vit.idx()) = -1;
		else
			VType_(vit.idx()) = innerNum_++;
	}

	//-----------保存构造的网格-----------
	Zombie::mesh2matrix(mesh, matV_, matF_);
	oriV_ = matV_;

	//adjust epsilon with number of vertices.
	epsilon_ = std::max(matV_.cols() * pow(10, -8), pow(10, -5));

	MatrixType matA;
	VectorType oriA;
	Zombie::cal_angles(matV_, matF_, oriA, matA);
	theta_ = cal_error(oriA, VType_, 1);
	std::cout << "初始最大误差： " << theta_ << std::endl;
	std::cout << "初始平均误差： " << cal_error(oriA, VType_, 1) << std::endl;

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	Zombie::visualize_mesh(renderer1, matV_, matF_, oriA, VType_);	
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
	Zombie::visualize_mesh(renderer2, oriV_, matF_, oriA, VType_);
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
	interactor->CreateRepeatingTimer(1000);

	auto timeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	timeCallback->SetCallback(CallbackFunction);
	timeCallback->SetClientData(renderer1->GetActors()->GetLastActor()->GetMapper()->GetInput());

	interactor->AddObserver(vtkCommand::TimerEvent, timeCallback);

	//开始
	renderWindow->Render();
	interactor->Start();

	return EXIT_SUCCESS;
}

double cal_error(const VectorType& vecAngles, const Eigen::VectorXi& VType, int flag)
{
	//计算最大误差或平均误差
	if (flag)
	{
		size_t idx = 0;
		double maxE = 0;
		for (int i = 0; i < VType_.size(); ++i)
			if (VType_(i) != -1)
				maxE = std::max(maxE, abs(2.0 * M_PI - vecAngles(i)));
		return maxE;
	}
	else
	{
		double averange = 0;
		int cnt = 0;
		for (size_t i = 0; i < VType_.size(); ++i)
			if (VType_(i) != -1)
			{
				averange += abs(2.0 * M_PI - vecAngles(i));
				++cnt;
			}
		averange /= double(cnt);
		return averange;
	}
}

void compute_update_vector(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& Vtype, int innerNum, VectorType& update_d) 
{
	const int Vnum = V.cols();
	const int Fnum = F.cols();
	MatrixType matA;
	VectorType vecA;
	Zombie::cal_angles(V, F, vecA, matA);
	VectorType b;

	//b = 2 * pi - theta(i)
	b.setConstant(innerNum, 0);
	for (int i = 0; i < Vtype.size(); ++i)
		if (Vtype(i) != -1)
			b(Vtype[i]) = 2. * M_PI - vecA(i);

	//Jacobian matrix of internal angle sum.
	std::vector<Tri> triA;
	triA.reserve(Fnum * 12);
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//记录当前面信息
		const auto& fv = F.col(fit);
		const auto& ca = matA.col(fit);

		//计算各角及各边长
		PosVector length;
		for (int i = 0; i < 3; ++i)
			length(i) = (V.col(fv[(i + 1) % 3]) - V.col(fv[i])).norm();

		//对一个面片内每个顶点计算相关系数
		for (int i = 0; i < 3; ++i)
		{
			const auto& p0 = V.col(fv[i]);
			const auto& p1 = V.col(fv[(i + 1) % 3]);
			const auto& p2 = V.col(fv[(i + 2) % 3]);

			//三角形内另外两个顶点v1、v2的内角对它们自身的偏微分与当前角i相关的系数
			PosVector v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
			PosVector v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));

			//顶点v0的内角i对三角形内另外两个顶点v1、v2的偏微分系数
			PosVector v01 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
			PosVector v02 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;

			//判断顶点i是否为内部顶点，边界顶点不参与计算
			if (Vtype(fv[i]) != -1)
				for (int j = 0; j < 3; ++j)
				{
					triA.push_back(Tri(Vtype(fv[i]), fv[(i + 1) % 3] * 3 + j, v01[j]));
					triA.push_back(Tri(Vtype(fv[i]), fv[(i + 2) % 3] * 3 + j, v02[j]));
				}

			//判断顶点i+1是否为内部顶点，边界顶点不参与计算
			if (Vtype(fv[(i + 1) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
					triA.push_back(Tri(Vtype(fv[(i + 1) % 3]), fv[(i + 1) % 3] * 3 + j, v11[j]));

			//判断顶点i+2是否为内部顶点，边界顶点不参与计算
			if (Vtype(fv[(i + 2) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
					triA.push_back(Tri(Vtype(fv[(i + 2) % 3]), fv[(i + 2) % 3] * 3 + j, v22[j]));
		}
	}
	SparseMatrixType A(innerNum, Vnum * 3);
	A.setFromTriplets(triA.begin(), triA.end());

	//solve least norm problam
	Eigen::SimplicialLDLT<SparseMatrixType> solver;
	solver.compute(A * A.transpose());
	if (solver.info() != Eigen::Success)
		std::cout << "solve fail" << std::endl;
	update_d = A.transpose() * solver.solve(b);
}

void update_points(MatrixType& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& Vtype, int innerNum, const VectorType& update_d)
{
	const int Vnum = V.cols();
	const int Fnum = F.cols();
	VectorType tempV = Eigen::Map<VectorType>(V.data(), Vnum * 3, 1) + update_d;

	MatrixType matA;
	VectorType vecA;
	VectorType vecAreas;
	Zombie::cal_angles_and_areas(V, F, vecA, vecAreas, matA);
	
	//Nomalized area at vp. 
	double sumAreas = 0;
	for (int i = 0; i < Vnum; ++i)
		if (Vtype(i) != -1)
			sumAreas += vecAreas(i) / 3;
	sumAreas /= innerNum;
	vecAreas.array() /= 3.0 * sumAreas;


	//Construct cotangent-weighted Laplace operator matrix.
	SparseMatrixType A(6 * Vnum, 3 * Vnum);
	VectorType b;
	b.setConstant(6 * Vnum, 0);
	std::vector<Tri> triA;
	triA.reserve(14 * Vnum);
	// ||Lvp||
	for (int i = 0; i < F.cols(); ++i)
	{
		const auto& fv = F.col(i);
		const auto& ca = matA.col(i);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			if(Vtype(fv[vi]) != -1)
			{
				const double coeff1 = -1. / std::tan(ca[(vi + 1) % 3]) / (2. * vecAreas(fv[vi]));
				const double coeff2 = -1. / std::tan(ca[(vi + 2) % 3]) / (2. * vecAreas(fv[vi]));
				const double coeff0 = - coeff1 - coeff2;
				for (int j = 0; j < 3; ++j)
				{
					triA.push_back(Tri(fv[vi] * 3 + j, fv[vi] * 3 + j, coeff0));
					triA.push_back(Tri(fv[vi] * 3 + j, fv[(vi + 1) % 3] * 3 + j, coeff1));
					triA.push_back(Tri(fv[vi] * 3 + j, fv[(vi + 2) % 3] * 3 + j, coeff2));
				}
			}
		}
	}
	// ||v - v0||
	for (int i = 0; i < Vnum; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			triA.push_back(Tri(Vnum * 3 + i * 3 + j, i * 3 + j, 1.));
			b(Vnum * 3 + i * 3 + j) = tempV(i * 3 + j);
			if (Vtype(i) == -1)
			{
				triA.push_back(Tri(i * 3 + j, i * 3 + j, 1. * 100.0));
				b(i * 3 + j) = V(j, i) * 100.0;
			}
		}
	}
	A.setFromTriplets(triA.begin(), triA.end());

	//solve least norm problam
	Eigen::SimplicialLDLT<SparseMatrixType> solver;
	solver.compute(A.transpose() *A);
	if (solver.info() != Eigen::Success)
		std::cout << "solve fail" << std::endl;
	tempV = solver.solve(A.transpose() * b);
	V = Eigen::Map<MatrixType>(tempV.data(), 3, Vnum);
}

