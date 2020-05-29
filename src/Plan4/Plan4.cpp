#include "Plan4.h"

#define MAXIT 100
#define RECORRIT 10

double DELTA1 = 0.40;

int counter_ = 0;
int cntsum = 0;
bool corr_fin_ = false;
bool cal_l_ = false;

double w1_ = 100.0;
double w2_ = 10.0;
double alpha_ = 0.70;
double beta_ = 0.30;

std::vector<int> interV_;
std::vector<int> boundV_;
std::vector<int> specialV_;

MatrixType matV_;
Eigen::Matrix2Xi matE_;
Eigen::Matrix3Xi matF_;
Eigen::Matrix3Xi matFE_;
Eigen::VectorXi Vtype_;

MatrixType oriV_;
MatrixType oriNormals_;
VectorType oriAngles_;
ABTreeType ABTree_;

VectorType tl_;
MatrixType corrV_;
MatrixType corrNormals_;

double averE_ = 0;
double maxE_ = 0;

void TimeCallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (counter_ < MAXIT && !corr_fin_)
	{
		VectorType vA;
		MatrixType mA;
		VectorType as;
		if (!cal_l_)
		{
			Zombie::cal_angles_and_areas_with_edges(matV_.cols(), matF_, matFE_, Vtype_, tl_, as, vA, mA);
			compute_length(matV_, matE_, matF_, mA, vA, Vtype_, tl_);
			cal_l_ = true;
		}
		else
		{
			MatrixType lastV(corrV_);
			double lastaverE(averE_);

			//使用目标边长更新网格顶点
			update_vertices(matV_, matE_, matF_, oriV_, Vtype_, tl_, corrV_, corrNormals_);
			Zombie::cal_angles_and_areas(matV_, matF_, Vtype_, vA, as, mA);

			averE_ = cal_error(vA, as, Vtype_, 0);
			maxE_ = cal_error(vA, as, Vtype_, 1);

			if ((matV_ - lastV).norm() < DELTA1 || averE_ > lastaverE || counter_ >= RECORRIT)
			{
				update_corr_vertices(matV_, matF_, oriNormals_, ABTree_, corrV_, corrNormals_);
				corr_fin_ = true;
				//std::cout << "updated corresponde vertices" << std::endl;
			}
		}

		counter_++;
		cntsum++;
		std::cout << "第" << cntsum << "次迭代，最大误差为： " << maxE_ << "，平均误差为： " << averE_ << std::endl;

		//--------------可视化更新---------------------
		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV_.cols());
		for (auto i = 0; i < vA.size(); ++i)
		{
			if (Vtype_(i) != -1)
				scalar->InsertTuple1(i, abs(2.0f * M_PI - vA(i)));
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

void KeypressCallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* vtkNotUsed(clientData), void* vtkNotUsed(callData))
{
	auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);
	if (*(iren->GetKeySym()) == 'n')
	{
		corr_fin_ = false;
		counter_ = 0;
	}

}

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Load failed!" << std::endl;
	}
	if (argc == 6)
	{
		w1_ = std::stod(argv[2]);
		w2_ = std::stod(argv[3]);
		alpha_ = std::stod(argv[4]);
		beta_ = std::stod(argv[5]);
	}

	//网格初始信息收集
	getMeshInfo(mesh, oriV_, matE_, matF_, matFE_, Vtype_, tl_);

	MatrixType mA;
	VectorType as;
	Zombie::cal_angles_and_areas(oriV_, matF_, Vtype_, oriAngles_, as, mA);
	Zombie::cal_normal_per_vertex(oriV_, matF_, Vtype_, oriNormals_);

	DELTA1 = oriV_.cols() * 0.001;
	matV_ = oriV_;

	//使用初始顶点建立AABBTree
	ABTree_.build(oriV_, matF_);
	update_corr_vertices(oriV_, matF_, oriNormals_, ABTree_, corrV_, corrNormals_);

	averE_ = cal_error(oriAngles_, as, Vtype_, 0);
	maxE_ = cal_error(oriAngles_, as, Vtype_, 1);
	std::cout << "初始最大误差为： " << maxE_ << "，平均误差为： " << averE_ << std::endl;

	//--------------测试---------------
	VectorType vA;
	Zombie::cal_angles_and_areas_with_edges(matV_.cols(), matF_, matFE_, Vtype_, tl_, as, vA, mA);
	compute_length(matV_, matE_, matF_, mA, vA, Vtype_, tl_);
	Zombie::cal_angles_and_areas_with_edges(matV_.cols(), matF_, matFE_, Vtype_, tl_, as, vA, mA);

	std::cout << "理想曲面的最大误差为： " << cal_error(vA, as, Vtype_, 1) << "，平均误差为： " << cal_error(vA, as, Vtype_, 0) << std::endl;

	////---------------可视化---------------
	////创建窗口
	//auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	//renderWindow->SetSize(1600, 800);

	//auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	//visualize_mesh(renderer1, matV_, matF_, oriAngles_, Vtype_);
	//renderer1->SetViewport(0.0, 0.0, 0.5, 1.0);

	////添加文本
	//auto textActor1 = vtkSmartPointer<vtkTextActor>::New();
	//textActor1->SetInput("Result Mesh");
	//textActor1->GetTextProperty()->SetFontSize(33);
	//textActor1->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	//renderer1->AddActor2D(textActor1);

	////视角设置
	//renderer1->ResetCamera();
	//renderWindow->AddRenderer(renderer1);

	//auto renderer2 = vtkSmartPointer<vtkRenderer>::New();
	//visualize_mesh(renderer2, oriV_, matF_, oriAngles_, Vtype_);
	//renderer2->SetViewport(0.5, 0.0, 1.0, 1.0);

	////添加文本
	//auto textActor2 = vtkSmartPointer<vtkTextActor>::New();
	//textActor2->SetInput("Original Mesh");
	//textActor2->GetTextProperty()->SetFontSize(33);
	//textActor2->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	//renderer2->AddActor2D(textActor2);

	////视角设置
	//renderer2->ResetCamera();
	//renderWindow->AddRenderer(renderer2);

	//auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	//interactor->SetRenderWindow(renderWindow);
	//auto style = vtkInteractorStyleTrackballCamera::New();
	//interactor->SetInteractorStyle(style);
	//interactor->Initialize();
	//interactor->CreateRepeatingTimer(100);

	//auto timeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	//timeCallback->SetCallback(TimeCallbackFunction);
	//timeCallback->SetClientData(renderer1->GetActors()->GetLastActor()->GetMapper()->GetInput());

	//interactor->AddObserver(vtkCommand::TimerEvent, timeCallback);

	//auto keypressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	//keypressCallback->SetCallback(KeypressCallbackFunction);
	//interactor->AddObserver(vtkCommand::KeyPressEvent, keypressCallback);

	////开始
	//renderWindow->Render();
	//interactor->Start();

	return EXIT_SUCCESS;
}

void getMeshInfo(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix2Xi& E, Eigen::Matrix3Xi& F, Eigen::Matrix3Xi& FE, Eigen::VectorXi& Vtype, VectorType& eLength)
{
	V.resize(3, mesh.n_vertices());
	E.resize(2, mesh.n_edges());
	F.resize(3, mesh.n_faces());
	FE.resize(3, mesh.n_faces());
	Vtype.setConstant(mesh.n_vertices(), 0);
	eLength.setConstant(mesh.n_edges(), 0);

	for (auto fit : mesh.faces())
	{
		int i = 0;
		for (auto fvit : mesh.vertices(fit))
		{
			//save faces informations
			F(i++, fit.idx()) = fvit.idx();
			//save vertices informations
			if (Vtype(fvit.idx()) == 0)
			{
				V.col(fvit.idx()) = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data()).cast<DataType>();
				//收集顶点信息
				//内部顶点为1、边界顶点为-1
				if (mesh.is_boundary(fvit))
					Vtype(fvit.idx()) = -1;
				else
					Vtype(fvit.idx()) = 1;
			}
		}
		int j = 0;
		for (auto feit : mesh.halfedges(fit))
		{
			FE(j++, fit.idx()) = mesh.edge(feit).idx();
		}
	}

	for (auto eit : mesh.edges())
	{
		E(0, eit.idx()) = mesh.vertex(eit, 0).idx();
		E(1, eit.idx()) = mesh.vertex(eit, 1).idx();
		eLength(eit.idx()) = mesh.edge_length(eit);
	}
}

double cal_error(const VectorType& vecAngles, const VectorType& areas, const Eigen::VectorXi& Vtype, int flag)
{
	//计算最大误差或平均误差
	if (flag)
	{
		size_t idx = 0;
		double max = 0;
		for (int i = 0; i < areas.size(); ++i)
		{
			if (Vtype(i) == 1)
			{
				const double e = abs(2.0 * M_PI - vecAngles(i)) / areas(i);
				//max = e > max ? e : max;
				if (e > max)
				{
					max = e;
					idx = i;
				}
			}
		}
		return max;
	}
	else
	{
		double averange = 0;
		double cnt = 0.0;
		for (int i = 0; i < areas.size(); ++i)
		{
			if (Vtype(i) == 1)
			{
				++cnt;
				const double e = abs(2.0 * M_PI - vecAngles(i)) / areas(i);
				averange += abs(2.0 * M_PI - vecAngles(i)) / areas(i);
			}
		}
		averange /= cnt;
		return averange;
	}
}

void update_corr_vertices(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& oriN,
	ABTreeType& ABTree, MatrixType& corrV, MatrixType& corrVnomal)
{
	const int Vnum = V.cols();
	MatrixType corrVcordinations;
	VectorType corrFid;
	corrV.resize(3, Vnum);
	corrVcordinations.resize(3, Vnum);
	corrFid.resize(Vnum);

	//查找当前曲面与原始曲面的对应点关系
	ABTree.closest_point(V, corrV, corrFid);
	ABTree.barycentric(corrV, corrFid, corrVcordinations);

	//对应顶点法向的重心坐标插值
	corrVnomal.resize(3, Vnum);
	for (int i = 0; i < Vnum; ++i)
	{
		PosVector tmpNomal(PosVector::Zero(3));
		for (int j = 0; j < 3; ++j)
		{
			tmpNomal += corrVcordinations(j, i) * oriN.col(F(j, corrFid[i]));
		}
		corrVnomal.col(i) = tmpNomal;
	}
}

void compute_length(MatrixTypeConst& V, const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F,
	MatrixTypeConst& matAngles, const VectorType& vecAngles, const Eigen::VectorXi& Vtype,
	VectorType& tl)
{
	const int Vnum = V.cols();
	SparseMatrixType A(Vnum, Vnum);
	VectorType b(vecAngles.array() - 2.0 * M_PI);

	std::vector<Tri> triple;
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		const PosVector& ca = matAngles.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (Vtype(fv0) == 1)
			{
				triple.push_back(Tri(fv0, fv0, (1.0 / std::tan(ca[(vi + 1) % 3]) + 1.0 / std::tan(ca[(vi + 2) % 3])) / 2.0));
				triple.push_back(Tri(fv0, fv1, -1.0 / std::tan(ca[(vi + 2) % 3]) / 2.0));
				triple.push_back(Tri(fv0, fv2, -1.0 / std::tan(ca[(vi + 1) % 3]) / 2.0));
			}
		}
	}

	for (int i = 0; i < Vtype.size(); ++i)
		if (Vtype(i) == -1)
		{
			triple.push_back(Tri(i, i, 1.0));
			b(i) = 0;
		}

	A.setFromTriplets(triple.begin(), triple.end());
	SparseMatrixType AT = A.transpose();
	Eigen::SimplicialLDLT<SparseMatrixType> solver;
	solver.compute(AT * A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Scales Solve Failed !" << std::endl;
	}
	VectorType phi = AT * solver.solve(b);

	//std::cout << phi << std::endl;
	//使用缩放因子计算出目标长度
	for (int i = 0; i < E.cols(); ++i)
	{
		const Eigen::Vector2i ev = E.col(i);
		const double p0 = phi(ev(0));
		const double p1 = phi(ev(1));
		double s(0);
		if (p0 == p1)
			s = exp(p0);
		else
			s = (exp(p0) - exp(p1)) / (p0 - p1);

		tl(i) *= s;
	}
}

void update_vertices(MatrixType& V, const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F, MatrixTypeConst& oriV, const Eigen::VectorXi& interVidx,
	const VectorType& tl, MatrixTypeConst& corrV, MatrixTypeConst& corrNormals)
{
	const int Vnum = V.cols();
	const int Enum = E.cols();
	SparseMatrixType A;
	VectorType b;
	A.resize(Vnum * 4 + Enum, 3 * Vnum);
	b.setConstant(4 * Vnum + Enum, 0);
	std::vector<Tri> triple;

	//逼近目标边长
	for (int i = 0; i < Enum; ++i)
	{
		const Eigen::Vector2i& ev = E.col(i);
		const PosVector& v0 = V.col(ev(0));
		const PosVector& v1 = V.col(ev(1));
		const PosVector e01 = v1 - v0;
		const double l = e01.norm();

		for (int j = 0; j < 3; ++j)
		{
			triple.push_back(Tri(i, ev(0) * 3 + j, -w1_ * e01(j) / l));
			triple.push_back(Tri(i, ev(1) * 3 + j, w1_ * e01(j) / l));
		}
		b(i) = w1_ * e01.dot(e01) / l - l + tl(i);
	}

	//当前网格与初始网格的位置约束
	for (int i = 0; i < Vnum; ++i)
	{
		if (interVidx(i) != -1)
		{
			for (int j = 0; j < 3; ++j)
			{
				triple.push_back(Tri(Enum + i * 3 + j, i * 3 + j, w2_ * alpha_));
				b(Enum + i * 3 + j) += w2_ * alpha_ * corrV(j, i);
			}
		}
		else
		{
			for (int j = 0; j < 3; ++j)
			{
				triple.push_back(Tri(Enum + i * 3 + j, i * 3 + j, w2_ * 100));
				b(Enum + i * 3 + j) += w2_ * 100 * oriV(j, i);
			}
		}
	}

	//当前网格与初始网格的在顶点法向上的位置约束
	for (int i = 0; i < Vnum; ++i)
	{
		//if (interVidx(i) != -1)
		{
			for (int j = 0; j < 3; ++j)
			{
				triple.push_back(Tri(Enum + Vnum * 3 + i, i * 3 + j, w2_ * beta_ * corrNormals(j, i)));
				b(Enum + Vnum * 3 + i) += w2_ * beta_ * corrV(j, i) * corrNormals(j, i);
			}
		}
	}

	A.setFromTriplets(triple.begin(), triple.end());
	Eigen::SimplicialLDLT<SparseMatrixType> solver;
	solver.compute(A.transpose() * A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Update Solve Failed !" << std::endl;
	}
	VectorType vecV = solver.solve(A.transpose() * b);
	V = Eigen::Map<MatrixType>(vecV.data(), 3, V.cols());
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
