#include "newtonMethod.h"

#define w1 1.0
#define w2 0.1
#define TAU 0.25

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read("test5.off"))
	{
		std::cout << "Laod failed!" << std::endl;
	}

	int boundary_num = 0;
	Eigen::VectorXi boundary_mark;
	std::vector<int> boundary_v;
	boundary_mark.resize(mesh.n_vertices());
	boundary_mark.setZero();
	for (auto vit : mesh.vertices())
	{
		if (mesh.is_boundary(vit))
		{
			boundary_v.push_back(vit.idx());
			boundary_mark(vit.idx()) = 1;
			++boundary_num;
		}
	}

	//-----------保存构造的网格-----------
	Eigen::Matrix3Xf matV;
	Eigen::Matrix3Xi matF;
	mesh2matrix(mesh, matV, matF);

	Eigen::Matrix3Xf angles;
	Eigen::VectorXi degrees;
	calAngles_Neigh(matV, matF, angles, degrees);

	Eigen::SparseMatrix<float> A;
	Eigen::VectorXf b;
	BuildrhsB(matV, matF, boundary_v, b);

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
	solver.compute((A.transpose() * A).eval());

	if (solver.info() != Eigen::Success)
	{
		std::cout << "solve fail" << std::endl;
	}

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1400, 800);

	////---------------原始网格及法向可视化---------------
	auto renderer = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer, matV, matF, degrees);
	renderer->SetViewport(0.0, 0.0, 0.5, 1.0);
	//视角设置
	renderer->GetActiveCamera()->SetPosition(-1, 0, 0);
	renderer->GetActiveCamera()->SetViewUp(0, 0, 1);
	renderer->ResetCamera();
	renderWindow->AddRenderer(renderer);

	auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(renderWindow);
	auto style = vtkInteractorStyleTrackballCamera::New();
	interactor->SetInteractorStyle(style);

	interactor->Initialize();
	interactor->CreateRepeatingTimer(1000);

	auto timeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	timeCallback->SetCallback(CallbackFunction);
	timeCallback->SetClientData(renderer->GetActors()->GetLastActor()->GetMapper()->GetInput());

	interactor->AddObserver(vtkCommand::TimerEvent, timeCallback);

	//开始s
	renderWindow->Render();
	interactor->Start();

	return EXIT_SUCCESS;
}

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	////update angle
	//auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
	//scalar->SetNumberOfComponents(1);
	//scalar->SetNumberOfTuples(matV.cols());
	//for (auto i = 0; i < angles.size(); ++i)
	//{
	//	scalar->InsertTuple1(i, abs(2.0 * M_PI - angles(i)));
	//}

	//auto polydata = static_cast<vtkPolyData*>(clientData);
	//auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);

	//auto points = vtkSmartPointer<vtkPoints>::New();
	//for (int i = 0; i < matV.cols(); ++i)
	//{
	//	points->InsertNextPoint(matV.col(i).data());
	//}
	//polydata->SetPoints(points);
	//polydata->GetPointData()->SetScalars(scalar);
	//polydata->Modified();;

	//iren->Render();

	//counter++;
}

void calAngles_Neigh(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xf& A, Eigen::VectorXi& degrees)
{
	A.resize(3, F.cols());
	Eigen::MatrixXi Adj(V.cols(), V.cols());
	Adj.setZero();
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const Eigen::VectorXf& p0 = V.col(fv[vi]);
			const Eigen::VectorXf& p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::VectorXf& p2 = V.col(fv[(vi + 2) % 3]);
			A(vi, j) = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));

			if (Adj(fv[vi], fv[(vi + 1) % 3]) == 0)
			{
				Adj(fv[vi], fv[(vi + 1) % 3]) = 1;
				Adj(fv[vi], fv[vi]) += 1;
			}
			if (Adj(fv[vi], fv[(vi + 2) % 3]) == 0)
			{
				Adj(fv[vi], fv[(vi + 2) % 3]) = 1;
				Adj(fv[vi], fv[vi]) += 1;
			}
		}
	}
	degrees = Adj.diagonal();
}

void BuildCoeffMatrix(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& angles, const Eigen::VectorXi& degrees, Eigen::SparseMatrix<float>& A)
{
	std::vector<Eigen::Triplet<float>> triA;
	for (int fi = 0; fi < F.cols(); ++fi)
	{
		const Eigen::Vector3i& fv = F.col(fi);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			triA.push_back(Eigen::Triplet<float>(fv[vi], fv[vi], w1 + w2));
			triA.push_back(Eigen::Triplet<float>(fv[vi], fv[(vi + 1) % 3], -0.5));
			triA.push_back(Eigen::Triplet<float>(fv[vi], fv[(vi + 2) % 3], -0.5));
		}
	}
}

void BuildrhsB(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const std::vector<int>& bv, Eigen::VectorXf& b)
{
	b.resize(12 * F.cols() + bv.size() * 3);
	b.setZero();
	for (int fi = 0; fi < F.cols(); ++fi)
	{
		srhs(b, (V.col(F(1, fi)) - V.col(F(0, fi))), F.cols() * 3 + fi * 9);
		srhs(b, (V.col(F(2, fi)) - V.col(F(1, fi))), F.cols() * 3 + fi * 9 + 3);
		srhs(b, (V.col(F(2, fi)) + V.col(F(1, fi)) + V.col(F(0, fi))) / 3.0f, F.cols() * 3 + fi * 9 + 6);
	}

	float ep = 0.001;
	for (size_t i = 1; i < bv.size(); ++i)
	{
		srhs(b, (V.col(bv[i]) / (ep * ep)), F.cols() * 12 + i * 3);
	}

	//for (int i = 0; i < V.cols(); ++i)
	//{
	//	int min = 0;
	//	double min_dist = (V.col(i) - V.col(bv[0])).norm();
	//	for (int j = 1; j < bv.size(); ++j)
	//	{
	//		double dist = (V.col(i) - V.col(bv[j])).norm();
	//		if (dist < min_dist)
	//		{
	//			min = j;
	//			min_dist = dist;
	//		}
	//	}
	//	srhs(b, (V.col(i) / (ep * ep)), F.cols() * 12 + i * 3);
	//}
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& vertices_mat, Eigen::Matrix3Xi& faces_mat)
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
				vertices_mat.col(fvit.idx()) = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data());
				flag(fvit.idx()) = 1;
			}
		}
	}
}

void matrix2vtk(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, vtkPolyData* P)
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

void MakeLUT(vtkFloatArray* Scalar, vtkLookupTable* LUT)
{
	auto ctf = vtkSmartPointer<vtkColorTransferFunction>::New();
	ctf->SetColorSpaceToHSV();
	ctf->AddRGBPoint(0.0, 0, 0, 1);
	ctf->AddRGBPoint(0.25, 0, 1, 1);
	ctf->AddRGBPoint(0.5, 0, 1, 0);
	ctf->AddRGBPoint(0.75, 1, 1, 0);
	ctf->AddRGBPoint(1.0, 1, 0, 0);

	//ctf->AddRGBPoint(0.0, 0, 0, 0);
	//ctf->AddRGBPoint(1.0, 1, 1, 1);

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

void MakeNormalGlyphs(vtkPolyData* src, vtkGlyph3D* glyph)
{
	auto centers = vtkSmartPointer<vtkCellCenters>::New();
	centers->SetInputData(src);
	auto arrow = vtkSmartPointer<vtkArrowSource>::New();
	arrow->SetTipLength(0.3);//参数
	arrow->SetTipRadius(.1);//参数
	arrow->SetShaftRadius(0.05);//参数
	arrow->Update();

	glyph->SetSourceConnection(arrow->GetOutputPort());
	glyph->SetInputConnection(centers->GetOutputPort());
	glyph->SetVectorModeToUseNormal();
	glyph->SetScaleFactor(2);//参数
	glyph->OrientOn();
	glyph->Update();
}

void visualize_vertices(vtkRenderer* Renderer, const Eigen::Matrix3Xf& V)
{
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < V.cols(); ++i)
	{
		points->InsertNextPoint(V.col(i).data());
	}
	auto polydata = vtkSmartPointer<vtkPolyData>::New();
	polydata->SetPoints(points);
	auto pointsFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	pointsFilter->SetInputData(polydata);
	pointsFilter->Update();

	auto pointsMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	pointsMapper->SetInputConnection(pointsFilter->GetOutputPort());
	auto pointsActor = vtkSmartPointer<vtkActor>::New();
	pointsActor->SetMapper(pointsMapper);
	pointsActor->GetProperty()->SetDiffuseColor(1.0, 0.0, 0.0);
	pointsActor->GetProperty()->SetPointSize(4.0);
	Renderer->AddActor(pointsActor);

	//视角设置
	Renderer->GetActiveCamera()->SetPosition(-3, 0, 0);
	Renderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
	Renderer->GetActiveCamera()->SetViewUp(0, 0, 1);
	Renderer->ResetCamera();
}

void visualize_mesh(vtkRenderer* Renderer, const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::VectorXf& angles)
{
	//生成网格
	auto P = vtkSmartPointer<vtkPolyData>::New();
	matrix2vtk(V, F, P);

	auto scalar = vtkSmartPointer<vtkFloatArray>::New();
	scalar->SetNumberOfComponents(1);
	scalar->SetNumberOfTuples(V.cols());
	for (auto i = 0; i < angles.size(); ++i)
	{
		scalar->InsertTuple1(i, abs(2.0f * M_PI - angles(i)));
	}
	P->GetPointData()->SetScalars(scalar);

	auto lut = vtkSmartPointer<vtkLookupTable>::New();
	MakeLUT(scalar, lut);
	//网格及法向渲染器
	auto polyMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	polyMapper->SetInputData(P);
	polyMapper->SetLookupTable(lut);
	polyMapper->SetScalarRange(scalar->GetValueRange()[0], scalar->GetValueRange()[1]);

	auto polyActor = vtkSmartPointer<vtkActor>::New();
	polyActor->SetMapper(polyMapper);
	polyActor->GetProperty()->SetDiffuseColor(1, 1, 1);
	Renderer->AddActor(polyActor);
}