#include "PlanB.h"
#include "mosek_solver.h"

#define MAXIT 100

int counter_ = 0;
double wp_ = 100.0;

std::vector<int> interV_;
std::vector<int> boundV_;
Eigen::VectorXi interVidx_;
Eigen::Matrix3Xi matF_;
Eigen::Matrix2Xi matE_;
MatrixType matV_;
VectorType orivecV_;

VectorType vecAngles_;
MatrixType matAngles_;
VectorType areas_;

SolverType solver_;
SparseMatrixType At_;

double errsum_ = 0.0;

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (counter_ < MAXIT)
	{
		Solve(solver_, At_, matV_, matE_, matF_, matAngles_, vecAngles_, areas_, interVidx_, boundV_);

		std::cout << "----------------------" << std::endl;
		std::cout << "第" << counter_++ << "次迭代，最大误差为： " << cal_error(vecAngles_, interV_, 1) << "，平均误差为： " << cal_error(vecAngles_, interV_, 0) << std::endl;
		cal_angles_and_areas(matV_, matF_, interVidx_, matAngles_, vecAngles_, areas_);

		//--------------可视化更新---------------------
		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV_.cols());
		for (auto i = 0; i < vecAngles_.size(); ++i)
		{
			if (interVidx_(i) != -1)
				scalar->InsertTuple1(i, abs(2.0f * M_PI - vecAngles_(i)));
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

	boundV_.push_back(8);
	boundV_.push_back(89);
	boundV_.push_back(170);
	boundV_.push_back(249);
	boundV_.push_back(328);
	boundV_.push_back(405);
	interVidx_(8) = -1;
	interVidx_(89) = -1;
	interVidx_(170) = -1;
	interVidx_(249) = -1;
	interVidx_(328) = -1;
	interVidx_(405) = -1;

	//网格初始信息收集
	mesh2matrix(mesh, matV_, matF_, matE_);
	cal_angles_and_areas(matV_, matF_, interVidx_, matAngles_, vecAngles_, areas_);
	MatrixType oriV(matV_);
	VectorType oriA(vecAngles_);
	//orivecV_ = Eigen::Map<VectorType>(oriV.data(), oriV.cols() * 3, 1);
	//std::cout << "初始最大误差： " << cal_error(vecAngles_, interV_, 1) << std::endl;
	//std::cout << "初始平均误差： " << cal_error(vecAngles_, interV_, 0) << std::endl;
	precompute_A(matV_.cols(), matE_, interVidx_, boundV_, solver_, At_);

	Solve(solver_, At_, matV_, matE_, matF_, matAngles_, vecAngles_, areas_, interVidx_, boundV_);
	cal_angles_and_areas(matV_, matF_, interVidx_, matAngles_, vecAngles_, areas_);

	//--------------测试---------------

	//std::vector<Tri> triC;
	//for (int i = 0; i < matE_.cols(); ++i)
	//{
	//	if (interVidx_(matE_(0, i)) != -1 || interVidx_(matE_(1, i)) != -1)
	//	{
	//		for (int j = 0; j < 3; ++j)
	//		{
	//			triC.push_back(Tri(i * 3 + j, matE_(0, i) * 3 + j, 1));
	//			triC.push_back(Tri(i * 3 + j, matE_(1, i) * 3 + j, -1));
	//		}
	//	}
	//}
	//for (size_t i = 0; i < boundV_.size(); ++i)
	//{
	//	for (int j = 0; j < 3; ++j)
	//	{
	//		triC.push_back(Tri(matE_.cols() * 3 + i * 3 + j, boundV_[i] * 3 + j, wp_));
	//	}
	//}
	//SparseMatrixType L;
	//L.resize(matE_.cols() * 3 + boundV_.size() * 3, matV_.cols() * 3);
	//L.setFromTriplets(triC.begin(), triC.end());
	//At_ = L.transpose();
	//SparseMatrixType LtL_;
	//LtL_ = At_ * L;
	//Lsolver_.compute(LtL_);
	//if (Lsolver_.info() != Eigen::Success)
	//{
	//	std::cout << "Solve Failed !" << std::endl;
	//}
	//compute_scale(matV_, matE_, matF_, matAngles_, vecAngles_, interVidx_, boundV_, l_);

	//Solve(solver_, At_, matV_, matE_, matF_, matAngles_, vecAngles_, interVidx_, boundV_);
	//cal_angles_and_areas(matV_, matF_, interVidx_, matAngles_, vecAngles_, areas_);

	//Eigen::Matrix3Xi F2E(3, matF_.cols());
	//for (auto fit : mesh.faces())
	//{
	//	int i = 0;
	//	for (auto feit : mesh.halfedges(fit))
	//	{
	//		F2E(i++, fit.idx()) = mesh.edge(feit).idx();
	//	}
	//}
	////计算系数
	//int Vnum = matV_.cols();
	//SparseMatrixType A(Vnum, Vnum);
	//VectorType b(vecAngles_.array() - 2.0 * M_PI);
	//std::vector<Tri> triple;
	//for (int j = 0; j < matF_.cols(); ++j)
	//{
	//	const Eigen::Vector3i& fv = matF_.col(j);
	//	const PosVector& ca = matAngles_.col(j);
	//	for (size_t vi = 0; vi < 3; ++vi)
	//	{
	//		const int fv0 = fv[vi];
	//		const int fv1 = fv[(vi + 1) % 3];
	//		const int fv2 = fv[(vi + 2) % 3];
	//		triple.push_back(Tri(fv0, fv0, (1.0 / std::tan(ca[(vi + 1) % 3]) + 1.0 / std::tan(ca[(vi + 2) % 3])) / 2.0));
	//		triple.push_back(Tri(fv0, fv1, -1.0 / std::tan(ca[(vi + 2) % 3]) / 2.0));
	//		triple.push_back(Tri(fv0, fv2, -1.0 / std::tan(ca[(vi + 1) % 3]) / 2.0));
	//	}
	//}
	//A.setFromTriplets(triple.begin(), triple.end());
	//Eigen::SimplicialLLT<SparseMatrixType> solver;
	//solver.compute(A * A.transpose());
	//if (solver.info() != Eigen::Success)
	//{
	//	std::cout << "Scales Solve Failed !" << std::endl;
	//}
	//VectorType phi = A.transpose() * solver.solve(b);

	//VectorType s;
	//s.setConstant(matE_.cols(), 0);
	//for (int i = 0; i < matE_.cols(); ++i)
	//{
	//	const Eigen::Vector2i ev = matE_.col(i);
	//	const double p0 = phi(ev(0));
	//	const double p1 = phi(ev(1));
	//	if (p0 == p1)
	//	{
	//		s(i) = exp(p0);
	//	}
	//	else
	//	{
	//		s(i) = (exp(p0) - exp(p1)) / (p0 - p1);
	//	}
	//}	
	////根据系数计算目标边长
	//VectorType length(matE_.cols());
	//for (int i = 0; i < matE_.cols(); ++i)
	//{
	//	length(i) = (matV_.col(matE_(0, i)) - matV_.col(matE_(1, i))).norm() * s(i);
	//}

	//MatrixType EA(3, matF_.cols());
	//VectorType EAngles;
	//EAngles.setConstant(matV_.cols(), 0);
	//for (int i = 0; i < F2E.cols(); ++i)
	//{
	//	const Eigen::Vector3i& fe = F2E.col(i);

	//	for (size_t ei = 0; ei < 3; ++ei)
	//	{
	//		const double& l0 = length(fe[ei]);
	//		const double& l1 = length(fe[(ei + 1) % 3]);
	//		const double& l2 = length(fe[(ei + 2) % 3]);
	//		const double temp = (l0 * l0 + l1 * l1 - l2 * l2) / (2.0 * l0 * l1);
	//		const double angle = std::acos(std::max(-1.0, std::min(1.0, temp)));
	//		EA(ei, i) = angle;
	//		EAngles(matF_(ei, i)) += angle;
	//	}
	//}

	////double relafjdlfk = EAngles.sum() - matV_.cols() * 2.0 * M_PI;
	////std::cout << relafjdlfk;
	//std::cout << EAngles.sum() / (2.0 * M_PI);

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer1, matV_, matF_, vecAngles_, interVidx_);
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

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F, Eigen::Matrix2Xi& E)
{
	F.resize(3, mesh.n_faces());
	E.resize(2, mesh.n_edges());
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

	for (auto eit : mesh.edges())
	{
		E(0, eit.idx()) = mesh.vertex(eit, 0).idx();
		E(1, eit.idx()) = mesh.vertex(eit, 1).idx();
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

void cal_angles_and_areas(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixType& matAngles, VectorType& vecAngles, VectorType& areas)
{
	matAngles.setConstant(3, F.cols(), 0);
	vecAngles.setConstant(V.cols(), 0);
	areas.setConstant(V.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		//Mix area
		double area = (V.col(fv[1]) - V.col(fv[0])).cross(V.col(fv[2]) - V.col(fv[0])).norm() / 6.0f;

		for (size_t vi = 0; vi < 3; ++vi)
		{
			const PosVector& p0 = V.col(fv[vi]);
			const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
			const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			matAngles(vi, f) = angle;
			if (interVidx(fv(vi)) != -1)
			{
				areas(fv[vi]) += area;
				vecAngles(fv(vi)) += angle;
			}
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

void cal_gaussian_gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, SparseMatrixType& mGradient)
{
	std::vector<Tri> triple;
	triple.reserve(F.cols() * 12);
	//高斯曲率的梯度
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//记录当前面信息
		const Eigen::Vector3i& fv = F.col(fit);
		const PosVector& ca = mAngles.col(fit);

		//计算各角及各边长
		PosVector length;
		for (int i = 0; i < 3; ++i)
		{
			length(i) = (V.col(fv[(i + 1) % 3]) - V.col(fv[i])).norm();
		}

		//对一个面片内每个顶点计算相关系数
		for (int i = 0; i < 3; ++i)
		{
			const PosVector& p0 = V.col(fv[i]);
			const PosVector& p1 = V.col(fv[(i + 1) % 3]);
			const PosVector& p2 = V.col(fv[(i + 2) % 3]);

			//判断顶点fv是否为内部顶点，边界顶点不参与计算
			//theta(vp)对vp求偏微分的系数
			PosVector v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
			PosVector v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
			//theta(vq)对vp求偏微分的系数
			PosVector v01 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
			PosVector v02 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
			//系数项
			if (interVidx(fv[(i + 1) % 3]) != -1)
			{
				for (int j = 0; j < 3; ++j)
				{
					if (v11[j])
						triple.push_back(Tri(fv[(i + 1) % 3], fv[(i + 1) % 3] * 3 + j, v11[j]));
				}
			}
			if (interVidx(fv[(i + 2) % 3]) != -1)
			{
				for (int j = 0; j < 3; ++j)
				{
					if (v22[j])
						triple.push_back(Tri(fv[(i + 2) % 3], fv[(i + 2) % 3] * 3 + j, v22[j]));
				}
			}
			if (interVidx(fv[i]) != -1)
			{
				for (int j = 0; j < 3; ++j)
				{
					if (v01[j])
						triple.push_back(Tri(fv[i], fv[(i + 1) % 3] * 3 + j, v01[j]));
					if (v02[j])
						triple.push_back(Tri(fv[i], fv[(i + 2) % 3] * 3 + j, v02[j]));
				}
			}
		}
	}
	mGradient.resize(V.cols(), V.cols() * 3);
	mGradient.setFromTriplets(triple.begin(), triple.end());
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

void cal_uni_laplace(const Eigen::Matrix3Xi& F, int Vnum, const Eigen::VectorXi& interVidx, SparseMatrixType& L)
{
	//计算固定边界的cot权拉普拉斯系数矩阵
	std::vector<Tri> triple;
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (interVidx(fv0) != -1)
			{
				for (int j = 0; j < 3; ++j)
				{
					triple.push_back(Tri(fv0 * 3 + j, fv0 * 3 + j, 1.0));
					triple.push_back(Tri(fv0 * 3 + j, fv1 * 3 + j, -0.5));
					triple.push_back(Tri(fv0 * 3 + j, fv2 * 3 + j, -0.5));
				}
			}
		}
	}
	L.resize(Vnum * 3, Vnum * 3);
	L.setFromTriplets(triple.begin(), triple.end());
}

void precompute_A(int Vnum, const Eigen::Matrix2Xi& E, const Eigen::VectorXi& interVidx, const std::vector<int>& boundV, SolverType& solver, SparseMatrixType& At)
{
	std::vector<Tri> triA;
	for (int i = 0; i < E.cols(); ++i)
	{
		if (interVidx_(E(0, i)) != -1 || interVidx_(E(1, i)) != -1)
		{
			for (int j = 0; j < 3; ++j)
			{
				triA.push_back(Tri(i * 3 + j, E(0, i) * 3 + j, 1));
				triA.push_back(Tri(i * 3 + j, E(1, i) * 3 + j, -1));
			}
		}
	}
	for (size_t i = 0; i < boundV.size(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			triA.push_back(Tri(E.cols() * 3 + i * 3 + j, boundV[i] * 3 + j, wp_));
		}
	}
	SparseMatrixType A;
	A.resize(E.cols() * 3 + boundV.size() * 3, Vnum * 3);
	A.setFromTriplets(triA.begin(), triA.end());
	At = A.transpose();

	solver.compute(At * A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Solve Failed !" << std::endl;
	}
}

void compute_scale(int Vnum, const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F, MatrixTypeConst& mAngles,
	const VectorType& vecAngles, const VectorType& areas, const Eigen::VectorXi& interVidx, const std::vector<int>& boundV, VectorType& s)
{
	SparseMatrixType A(Vnum, Vnum);
	VectorType b(vecAngles.array() - 2.0 * M_PI);
	errsum_ = b.sum();
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
				b(fv0) /= areas(fv0);
				triple.push_back(Tri(fv0, fv0, (1.0 / std::tan(ca[(vi + 1) % 3]) + 1.0 / std::tan(ca[(vi + 2) % 3])) / (2.0 * areas(fv0))));
				triple.push_back(Tri(fv0, fv1, -1.0 / std::tan(ca[(vi + 2) % 3]) / (2.0 * areas(fv0))));
				triple.push_back(Tri(fv0, fv2, -1.0 / std::tan(ca[(vi + 1) % 3]) / (2.0 * areas(fv0))));
			}
		}
	}
	for (auto i : boundV)
	{
		triple.push_back(Tri(i, i, 1.0));
		b(i) = 0;
	}
	A.setFromTriplets(triple.begin(), triple.end());
	Eigen::SimplicialLLT<SparseMatrixType> solver;
	solver.compute(A * A.transpose());
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Scales Solve Failed !" << std::endl;
	}

	VectorType phi = A.transpose() * solver.solve(b);

	s.setConstant(E.cols(), 0);
	for (int i = 0; i < E.cols(); ++i)
	{
		const Eigen::Vector2i ev = E.col(i);
		const double p0 = phi(ev(0));
		const double p1 = phi(ev(1));
		if (p0 == p1)
		{
			s(i) = exp(p0);
		}
		else
		{
			s(i) = (exp(p0) - exp(p1)) / (p0 - p1);
		}
	}
}

void update_vertices(SolverType& solver, const SparseMatrixType& At, MatrixType& V, const Eigen::Matrix2Xi& E,
	const Eigen::VectorXi& interVidx, const std::vector<int>& boundV, const VectorType& length)
{
	Eigen::Map<VectorType> vecV(V.data(), V.cols() * 3);
	int itc = 0;
	double dv = 0.0;
	do
	{
		VectorType d(3 * E.cols() + boundV.size() * 3);
		for (int i = 0; i < E.cols(); ++i)
		{
			const PosVector temp = (V.col(E(0, i)) - V.col(E(1, i))).normalized() * length(i);
			for (int j = 0; j < 3; ++j)
				d(i * 3 + j) = temp(j);
		}
		for (size_t i = 0; i < boundV.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				d(3 * E.cols() + i * 3 + j) = V(j, boundV[i]) * wp_;
			}
		}
		VectorType preV(vecV);
		vecV = solver.solve(At * d);
		dv = (vecV - preV).norm();
	} while (itc++ <= MAXIT && dv >= 1e-4);

	V = Eigen::Map<MatrixType>(vecV.data(), 3, V.cols());
}

void Solve(SolverType& solver, const SparseMatrixType& At, MatrixType& V, const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F,
	MatrixTypeConst& matAngles, const VectorType& vecAngles, const VectorType& areas, const Eigen::VectorXi& interVidx, const std::vector<int>& boundV)
{
	//计算系数
	VectorType s;
	compute_scale(V.cols(), E, F, matAngles, vecAngles, areas, interVidx, boundV, s);
	//根据系数计算目标边长
	VectorType length(E.cols());
	for (int i = 0; i < E.cols(); ++i)
	{
		length(i) = (V.col(E(0, i)) - V.col(E(1, i))).norm() * s(i);
	}
	//使用目标边长更新网格顶点
	update_vertices(solver, At, V, E, interVidx, boundV, length);
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

