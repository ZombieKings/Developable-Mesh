#include "ADMM.h"

bool flag_ = false;
unsigned int counter_ = 0;

const double rho_ = 1.3;
double w_ = 0.1;

double mu_ = 0.5;
VectorType Y_;

std::vector<int> interV_;
std::vector<int> boundV_;
Eigen::VectorXi interVidx_;
Eigen::Matrix3Xi matF_;
MatrixType matV_;

VectorType vecAngles_;
MatrixType matAngles_;
VectorType areas_;
void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (counter_ < 20)
	{
		Eigen::Map<VectorType>vecV(matV_.data(), matV_.cols() * 3, 1);
		SparseMatrixType G;
		cal_gaussian_gradient(matV_, matF_, interVidx_, matAngles_, G);
		VectorType GV(G * vecV);
		VectorType theta(2.0 * M_PI - vecAngles_.array());
		VectorType b(theta + GV);
		VectorType C;
		Solve_C(GV, b, Y_, mu_, C);

		SparseMatrixType L;
		cal_cot_laplace(matF_, matAngles_, areas_, interVidx_, L);
		Solve_V(vecV, G, L, b, Y_, C, mu_, w_, matV_);

		Y_ += mu_ * (C - GV + b);
		mu_ *= rho_;

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

		//counter_++;
		std::cout << counter_++ << std::endl;
	}
	else if(!flag_)
	{
		flag_ = true;
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
	cal_angles_and_areas(matV_, matF_, interVidx_, matAngles_, vecAngles_, areas_);
	MatrixType oriV(matV_);
	VectorType oriA(vecAngles_);
	Y_.setConstant(matV_.cols(), 0);

	//--------------测试---------------
	Eigen::Map<VectorType>vecV(matV_.data(), matV_.cols() * 3, 1);
	SparseMatrixType G;
	cal_gaussian_gradient(matV_, matF_, interVidx_, matAngles_, G);
	VectorType GV(G * vecV);
	VectorType theta(2.0 * M_PI - vecAngles_.array());
	VectorType b(theta + GV);
	VectorType C;
	Solve_C(GV, b, Y_, mu_, C);

	SparseMatrixType L;
	cal_cot_laplace(matF_, matAngles_, areas_, interVidx_, L);
	Solve_V(vecV, G, L, b, Y_, C, mu_, w_, matV_);

	Y_ += mu_ * (C - GV + b);
	mu_ *= rho_;


	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer1, matV_, matF_, vecAngles_, interVidx_);
	renderer1->SetViewport(0.0, 0.0, 0.5, 1.0);

	// Setup the text and add it to the renderer
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

	// Setup the text and add it to the renderer
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
			areas(fv[vi]) += area;
			const PosVector& p0 = V.col(fv[vi]);
			const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
			const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			matAngles(vi, f) = angle;
			if (interVidx(fv(vi)) != -1)
				vecAngles(fv(vi)) += angle;
		}
	}
}

double cal_error(const VectorType& vecAngles, const std::vector<int>& interIdx, int flag)
{
	//计算最大误差或平均误差
	if (flag)
	{
		double max = 0;
		for (size_t i = 0; i < interIdx.size(); ++i)
		{
			const double e = abs(2.0 * M_PI - vecAngles(interIdx[i]));
			max = e > max ? e : max;
		}
		return max;
	}
	else
	{
		double averange = 0;
		for (size_t i = 0; i < interIdx.size(); ++i)
		{
			averange += vecAngles(interIdx[i]);
		}
		averange = 2.0 * M_PI - averange / interIdx.size();
		return averange;
	}
}

void cal_gaussian_gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, SparseMatrixType& mGradient)
{
	std::vector<Tri> triple;
	//高斯曲率1范数的梯度
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

		//对每个顶点计算相关系数
		for (int i = 0; i < 3; ++i)
		{
			const PosVector& p0 = V.col(fv[i]);
			const PosVector& p1 = V.col(fv[(i + 1) % 3]);
			const PosVector& p2 = V.col(fv[(i + 2) % 3]);

			//判断顶点fv是否为内部顶点，边界顶点不参与计算
			if (interVidx(fv[(i + 1) % 3]) != -1)
			{
				//对vp求偏微分的系数
				PosVector v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
				//对vq求偏微分的系数
				PosVector v10 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
				//系数项
				for (int j = 0; j < 3; ++j)
				{
					if (v11[j])
						triple.push_back(Tri(fv[(i + 1) % 3], fv[(i + 1) % 3] * 3 + j, v11[j]));
					if (v10[j] && interVidx(fv[i]) != -1)					
						triple.push_back(Tri(fv[(i + 1) % 3], fv[i] * 3 + j, v10[j]));					
				}

				if (interVidx(fv[(i + 2) % 3]) != -1)
				{
					//对vp求偏微分的系数
					PosVector v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
					//对vq求偏微分的系数
					PosVector v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
					//系数项
					for (int j = 0; j < 3; ++j)
					{
						if (v22[j])
							triple.push_back(Tri(fv[(i + 2) % 3], fv[(i + 2) % 3] * 3 + j, v22[j]));
						if (v20[j] && interVidx(fv[i]) != -1)
							triple.push_back(Tri(fv[(i + 2) % 3], fv[i] * 3 + j, v20[j]));
					}
				}
			}
		}
		mGradient.resize(V.cols(), V.cols() * 3);
		mGradient.setFromTriplets(triple.begin(), triple.end());
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
					triple.push_back(Tri(fv0 + j, fv0 + j, temp0));
					triple.push_back(Tri(fv0 + j, fv1 + j, temp1));
					triple.push_back(Tri(fv0 + j, fv2 + j, temp2));
				}
			}
		}
	}
	L.resize(areas.size() * 3, areas.size() * 3);
	L.setFromTriplets(triple.begin(), triple.end());
}

double shrink(double x, double tau)
{
	int sign = x > 0 ? 1 : -1;
	double temp = abs(x);
	return sign * std::max((temp - tau), 0.0);
}

void Solve_C(const VectorType& GV, const VectorType& b, const VectorType& Y, double mu, VectorType& C)
{
	VectorType TempV(GV - b - Y / mu);
	C.resize(TempV.size());
	for (int i = 0; i < C.size(); ++i)
	{
		C(i) = shrink(TempV(i), 1.0 / mu);
	}
}

void Solve_V(const VectorType& V, const SparseMatrixType& G, const SparseMatrixType& L, const VectorType& b, const VectorType& Y, const VectorType& C, double mu, double w, MatrixType& matV)
{
	SparseMatrixType Gt = G.transpose();
	SparseMatrixType wLtL(2 * w * L.transpose() * L);

	SparseMatrixType A(mu * Gt * G + wLtL);
	VectorType Rhs(Gt * (Y + C + b) + wLtL * V);

	//Eigen::SimplicialLDLT<SparseMatrixType> solver;
	Eigen::SparseQR<SparseMatrixType, Eigen::COLAMDOrdering<int>> solver;
	solver.compute(A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Solve Failed !" << std::endl;
	}
	VectorType temp(solver.solve(Rhs));
	matV = Eigen::Map<MatrixType>(temp.data(), 3, V.size() / 3);
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
	ctf->AddRGBPoint(0.0, 0, 0, 1);
	ctf->AddRGBPoint(0.25, 0, 1, 1);
	ctf->AddRGBPoint(0.5, 0, 1, 0);
	ctf->AddRGBPoint(0.75, 1, 1, 0);
	ctf->AddRGBPoint(1.0, 1, 0, 0);

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

	auto polyActor = vtkSmartPointer<vtkActor>::New();
	polyActor->SetMapper(polyMapper);
	polyActor->GetProperty()->SetDiffuseColor(1, 1, 1);
	Renderer->AddActor(polyActor);
	Renderer->AddActor2D(scalarBar);
}
