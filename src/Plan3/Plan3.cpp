#include "Plan3.h"

#include "../tools/cal_angles_areas.h"
#include "../tools/cal_laplacian.h"
#include "../tools/cal_normals.h"
#include "../tools/AABBSearcher.h"

#define MAXIT 10000

int counter_ = 0;
double wl_ = 10.0;
double wp_ = 10.0;
double we_ = 100.0;

std::vector<int> interV_;
std::vector<int> boundV_;
std::vector<int> specialV_;

Eigen::VectorXi interVidx_;
Eigen::Matrix3Xi matF_;
Eigen::Matrix2Xi matE_;
MatrixType matV_;

VectorType orivecV_;
Surface_Mesh::AABBSearcher<MatrixType, Eigen::Matrix3Xi> ABTree_;

VectorType vecAngles_;
MatrixType matAngles_;
VectorType vecNormals_;
VectorType areas_;

double errsum_ = 0.0;

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (counter_ < MAXIT)
	{
		Solve(matV_, matE_, matF_, matAngles_, vecAngles_, areas_, interVidx_, boundV_);

		Zombie::cal_angles_and_areas(matV_, matF_, interVidx_, vecAngles_, areas_, matAngles_);
		double averE = cal_error(vecAngles_, interV_, 0);
		double maxE = cal_error(vecAngles_, interV_, 1);
		std::cout << "----------------------" << std::endl;
		std::cout << "第" << counter_++ << "次迭代，最大误差为： " << maxE << "，平均误差为： " << averE << std::endl;

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

	//specialV_.push_back(16);
	//specialV_.push_back(189);
	//interVidx_(16) = -2;
	//interVidx_(189) = -2;

	//网格初始信息收集
	mesh2matrix(mesh, matV_, matF_, matE_);
	Zombie::cal_angles_and_areas(matV_, matF_, interVidx_, vecAngles_, areas_, matAngles_);
	Zombie::cal_normal_per_vertex(matV_, matF_, interVidx_, vecNormals_);

	MatrixType oriV(matV_);
	VectorType oriA(vecAngles_);
	orivecV_ = Eigen::Map<VectorType>(oriV.data(), oriV.cols() * 3, 1);
	//使用初始顶点建立AABBTree
	ABTree_.build(matV_, matF_);

	//std::cout << "初始最大误差： " << cal_error(vecAngles_, interV_, 1) << std::endl;
	//std::cout << "初始平均误差： " << cal_error(vecAngles_, interV_, 0) << std::endl;

	//--------------测试---------------

	//Solve(matV_, matE_, matF_, matAngles_, vecAngles_, areas_, interVidx_, boundV_);
	//Zombie::cal_angles_and_areas(matV_, matF_, interVidx_, vecAngles_, areas_, matAngles_);

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
	interactor->CreateRepeatingTimer(100);

	auto timeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	timeCallback->SetCallback(CallbackFunction);
	timeCallback->SetClientData(renderer1->GetActors()->GetLastActor()->GetMapper()->GetInput());

	interactor->AddObserver(vtkCommand::TimerEvent, timeCallback);

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

void compute_scale(int Vnum, const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F, MatrixTypeConst& mAngles,
	const VectorType& vecAngles, const VectorType& areas, const Eigen::VectorXi& interVidx, const std::vector<int>& boundV,
	VectorType& s)
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

void update_vertices(MatrixType& V, const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F, MatrixTypeConst& matAngles,
	const VectorType& areas, const Eigen::VectorXi& interVidx, const std::vector<int>& boundV, const VectorType& s)
{
	const int Vnum = V.cols();
	const int Enum = E.cols();
	SparseMatrixType A;
	VectorType b;
	A.resize(Vnum * 6 + Enum, 3 * Vnum);
	b.setConstant(6 * Vnum + Enum, 0);
	//A.resize(Vnum * 4 + Enum, 3 * Vnum);
	//b.setConstant(4 * Vnum + Enum, 0);
	//A.resize(Vnum * 3 + Enum, 3 * Vnum);
	//b.setConstant(3 * Vnum + Enum, 0);
	std::vector<Tri> triple;

	//逼近目标边长
	for (int i = 0; i < Enum; ++i)
	{
		const Eigen::Vector2i& ev = E.col(i);
		const PosVector& v0 = V.col(ev(0));
		const PosVector& v1 = V.col(ev(1));
		const PosVector e01 = v1 - v0;
		const double l = e01.norm();
		const double tl = l * s(i);

		for (int j = 0; j < 3; ++j)
		{
			triple.push_back(Tri(i, ev(0) * 3 + j, -we_ * e01(j) / l));
			triple.push_back(Tri(i, ev(1) * 3 + j, we_ * e01(j) / l));
		}
		b(i) = we_ * e01.dot(e01) / l - l + tl;
	}

	//当前网格与初始网格的拉普拉斯坐标位置约束
	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::VectorXi& fv = F.col(i);
		const Eigen::VectorXd& ca = matAngles.col(i);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (interVidx(fv0) >= 0)
			{
				const double temp0 = wl_ * (1.0 / std::tan(ca[(vi + 1) % 3]) + 1.0 / std::tan(ca[(vi + 2) % 3])) / (2.0 * areas(fv0));
				const double temp1 = -wl_ * 1.0 / std::tan(ca[(vi + 2) % 3]) / (2.0 * areas(fv0));
				const double temp2 = -wl_ * 1.0 / std::tan(ca[(vi + 1) % 3]) / (2.0 * areas(fv0));
				for (int j = 0; j < 3; ++j)
				{
					triple.push_back(Tri(Enum + fv0 * 3 + j, fv0 * 3 + j, temp0));
					triple.push_back(Tri(Enum + fv0 * 3 + j, fv1 * 3 + j, temp1));
					triple.push_back(Tri(Enum + fv0 * 3 + j, fv2 * 3 + j, temp2));
					b(Enum + fv0 * 3 + j) += temp0 * orivecV_(fv0 * 3 + j);
					b(Enum + fv0 * 3 + j) += temp1 * orivecV_(fv1 * 3 + j);
					b(Enum + fv0 * 3 + j) += temp2 * orivecV_(fv2 * 3 + j);
				}
			}
			else
			{
				for (int j = 0; j < 3; ++j)
				{
					triple.push_back(Tri(Enum + fv0 * 3 + j, fv0 * 3 + j, wp_ * 100));
					b(Enum + fv0 * 3 + j) += wp_ * 100 * orivecV_(fv0 * 3 + j);
				}
			}
		}
	}

	////当前网格与初始网格的在顶点法向上的位置约束
	//for (int i = 0; i < Vnum; ++i)
	//{
	//	if (interVidx(i) != -1)
	//	{
	//		Eigen::Vector3d CP = ABTree_.closest_point((Eigen::Vector3d)V.col(i));
	//		for (int j = 0; j < 3; ++j)
	//		{
	//			triple.push_back(Tri(Enum + Vnum * 3 + i, i * 3 + j, wp_ * vecNormals_(i * 3 + j)));
	//			b(Enum + Vnum * 3 + i) += wp_ * CP(j) * vecNormals_(i * 3 + j);
	//		}
	//	}
	//}

	//当前网格与初始网格的位置约束
	for (int i = 0; i < Vnum; ++i)
	{
		if (interVidx(i) != -1)
		{
			Eigen::Vector3d CP = ABTree_.closest_point((Eigen::Vector3d)V.col(i));
			for (int j = 0; j < 3; ++j)
			{
				//triple.push_back(Tri(Enum + Vnum * 3 + i * 3 + j, i * 3 + j, wp_));
				//b(Enum + Vnum * 3 + i * 3 + j) += wp_ * CP(j);	
				triple.push_back(Tri(Enum + i * 3 + j, i * 3 + j, wp_));
				b(Enum + i * 3 + j) += wp_ * CP(j);
			}
		}
		else
		{
			for (int j = 0; j < 3; ++j)
			{
				triple.push_back(Tri(Enum + i * 3 + j, i * 3 + j, wp_ * 100));
				b(Enum + i * 3 + j) += wp_ * 100 * orivecV_(i * 3 + j);
			}
		}
	}

	A.setFromTriplets(triple.begin(), triple.end());

	//std::cout << A << std::endl;

	Eigen::SimplicialLLT<SparseMatrixType> solver;
	solver.compute(A.transpose() * A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Update Solve Failed !" << std::endl;
	}

	VectorType vecV = solver.solve(A.transpose() * b);
	V = Eigen::Map<MatrixType>(vecV.data(), 3, V.cols());
}

void Solve(MatrixType& V, const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F,
	MatrixTypeConst& matAngles, const VectorType& vecAngles, const VectorType& areas, const Eigen::VectorXi& interVidx, const std::vector<int>& boundV)
{
	//计算系数
	VectorType s;
	compute_scale(V.cols(), E, F, matAngles, vecAngles, areas, interVidx, boundV, s);

	//使用目标边长更新网格顶点
	update_vertices(V, E, F, matAngles, areas, interVidx, boundV, s);
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
