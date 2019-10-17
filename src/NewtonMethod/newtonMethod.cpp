//------------------------------------------------------------------
//注：解法器使用LDLT比QR快，但是需要剔除边界点对应的0部分，保证矩阵满秩
//------------------------------------------------------------------
#include "newtonMethod.h"

#define w1 1.0f
#define w2 0.1f
#define TAU 0.5f

bool flag = false;
int it_conunter = 0;

std::vector<int> interV;
std::vector<int> boundV;
Eigen::VectorXi interVidx;

Eigen::Matrix3Xf matV;
Eigen::Matrix3Xi matF;
Eigen::Matrix3Xf oriV;

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Laod failed!" << std::endl;
	}

	interVidx.resize(mesh.n_vertices());
	memset(interVidx.data(), -1, sizeof(int) * interVidx.size());
	int count = 0;
	for (const auto& vit : mesh.vertices())
	{
		if (!mesh.is_boundary(vit))
		{
			interV.push_back(vit.idx());
			interVidx(vit.idx()) = count++;
		}
		else
		{
			boundV.push_back(vit.idx());
		}
	}
	//-----------保存构造的网格-----------
	mesh2matrix(mesh, matV, matF);
	oriV = matV;
	Eigen::Matrix3Xf angles;
	Eigen::VectorXf vangles;
	cal_angles(oriV, matF, boundV, angles, vangles);

	//Eigen::Matrix3Xf matV;
	//Eigen::Matrix3Xi matF;
	//mesh2matrix(mesh, matV, matF);
	//Eigen::Matrix3Xf oriV(matV);
	//
	//Eigen::Matrix3Xf angles;
	//Eigen::VectorXf vangles;
	//Eigen::VectorXf degrees;
	//Eigen::Matrix3Xf Lpos;
	//calLaplace_Angles_Neigh(matV, matF, angles, vangles, Lpos, degrees);
	//
	////std::cout << degrees << std::endl;
	//
	//Eigen::SparseMatrix<float> A;
	//BuildCoeffMatrix(matV, matF, angles, degrees, interVidx, A);
	//std::cout << A << std::endl;
	//
	//Eigen::VectorXf b;
	//BuildrhsB(matV, matF, Lpos, degrees, interVidx, oriV, vangles, b);
	//std::cout << b << std::endl;
	//
	//Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
	//solver.compute((A.transpose() * A).eval());
	//Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> solver;
	//solver.compute(A);
	//if (solver.info() != Eigen::Success)
	//{
	//	std::cout << "solve fail" << std::endl;
	//}
	//Eigen::VectorXf temp = solver.solve(b);
	//std::cout << temp << std::endl;
	//
	//Eigen::Matrix3Xf resV(matV);
	//for (size_t i = 0; i < interV.size(); ++i)
	//{
	//	for (int j = 0; j < 3; ++j)
	//	{
	//		resV(j, interV[i]) += TAU * temp(i * 3 + j);
	//	}
	//}
	//
	//Eigen::Matrix3Xf remA;
	//Eigen::VectorXf reA;
	//cal_angles(resV, matF, boundV, remA, reA);

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1400, 800);

	////---------------原始网格及法向可视化---------------
	auto renderer = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer, matV, matF, vangles);
	renderer->SetViewport(0.0, 0.0, 1.0, 1.0);
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
	interactor->CreateRepeatingTimer(0);

	auto timeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	timeCallback->SetCallback(CallbackFunction);
	timeCallback->SetClientData(renderer->GetActors()->GetLastActor()->GetMapper()->GetInput());

	interactor->AddObserver(vtkCommand::TimerEvent, timeCallback);

	//开始
	renderWindow->Render();
	interactor->Start();

	return EXIT_SUCCESS;
}

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (!flag)
	{
		Eigen::Matrix3Xf angles;
		Eigen::VectorXf vangles;
		Eigen::VectorXf degrees;
		Eigen::Matrix3Xf Lpos;
		calLaplace_Angles_Neigh(matV, matF, angles, vangles, Lpos, degrees);

		Eigen::SparseMatrix<float> A;
		BuildCoeffMatrix(matV, matF, angles, degrees, interVidx, A);

		Eigen::VectorXf b;
		BuildrhsB(matV, matF, Lpos, degrees, interVidx, oriV, vangles, b);

		Eigen::SimplicialLDLT< Eigen::SparseMatrix<float>> solver;
		solver.compute(A.transpose() * A);

		if (solver.info() != Eigen::Success) 
			std::cout << "solve fail" << std::endl;

		Eigen::VectorXf temp = solver.solve(A.transpose() * b);
		for (size_t i = 0; i < interV.size(); ++i)
			for (int j = 0; j < 3; ++j)
				matV(j, interV[i]) += TAU * temp(i * 3 + j);

		temp(temp.size() - 1) = 0;
		if (temp.norm() <= 1e-4)
			flag = true;

		Eigen::Matrix3Xf remA;
		Eigen::VectorXf reA;
		cal_angles(matV, matF, boundV, remA, reA);

		double error = 0.0f;
		for (size_t i = 0; i < interV.size(); ++i)
			error += abs(reA(interV[i]) - 2.0 * M_PI);

		//update angle
		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV.cols());
		for (auto i = 0; i < reA.size(); ++i)
			scalar->InsertTuple1(i, abs(2.0 * M_PI - reA(i)));

		auto polydata = static_cast<vtkPolyData*>(clientData);
		auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);

		auto points = vtkSmartPointer<vtkPoints>::New();
		for (int i = 0; i < matV.cols(); ++i)
			points->InsertNextPoint(matV.col(i).data());
		polydata->SetPoints(points);
		polydata->GetPointData()->SetScalars(scalar);
		polydata->Modified();; 

		iren->Render();

		std::cout << "第" << it_conunter++ << "次迭代，整体误差： " << error << std::endl;
	}
}

void cal_angles(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const std::vector<int>& boundIdx, Eigen::Matrix3Xf& matAngles, Eigen::VectorXf& vecAngles)
{
	matAngles.resize(3, F.cols());
	matAngles.setZero();
	vecAngles.resize(V.cols());
	vecAngles.setZero();
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const Eigen::Vector3f& p0 = V.col(fv[vi]);
			const Eigen::Vector3f& p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::Vector3f& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			vecAngles(F(vi, f)) += angle;
			matAngles(vi, f) = angle;
		}
	}
	for (size_t i = 0; i < boundIdx.size(); ++i)
		vecAngles(boundIdx[i]) = 2.0f * M_PI;
}

void calLaplace_Angles_Neigh(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xf& A, Eigen::VectorXf& vecA, Eigen::Matrix3Xf& Lpos, Eigen::VectorXf& degrees)
{
	A.resize(3, F.cols());
	vecA.resize(V.cols());
	vecA.setZero();
	std::vector<Eigen::Triplet<float>> tripleL;
	tripleL.reserve(F.cols() * 9);
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const Eigen::VectorXf& p0 = V.col(fv[vi]);
			const Eigen::VectorXf& p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::VectorXf& p2 = V.col(fv[(vi + 2) % 3]);
			const float angle = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			A(vi, j) = angle;
			vecA(fv[vi]) += angle;

			tripleL.push_back(Eigen::Triplet<float>(fv[vi], fv[vi], 1));
			tripleL.push_back(Eigen::Triplet<float>(fv[vi], fv[(vi + 1) % 3], -0.5f));
			tripleL.push_back(Eigen::Triplet<float>(fv[vi], fv[(vi + 2) % 3], -0.5f));
		}
	}
	Eigen::SparseMatrix<float> L;
	L.resize(V.cols(), V.cols());
	L.setFromTriplets(tripleL.begin(), tripleL.end());

	degrees = L.diagonal();
	for (int k = 0; k < L.outerSize(); ++k)
		for (Eigen::SparseMatrix<float>::InnerIterator it(L, k); it; ++it)
		{
			it.valueRef() /= degrees(it.row());
		}
	Lpos = (L * V.transpose()).transpose();
}

void calGradient(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& matAngles, const Eigen::VectorXi& interIdx, Eigen::VectorXf& Gradient)
{
	Gradient.resize(V.cols() * 3 + 1);
	Gradient.setZero();
	//高斯梯度
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//记录当前面信息
		const Eigen::Vector3i& fv = F.col(fit);
		const Eigen::Vector3f& ca = matAngles.col(fit);
		Eigen::Matrix3f p;
		for (int i = 0; i < 3; ++i)
			p.col(i) = V.col(fv[i]);

		//计算各角及各边长
		Eigen::Vector3f length;
		for (int i = 0; i < 3; ++i)
		{
			length(i) = (p.col((i + 1) % 3) - p.col(i)).norm();
		}

		//对每个顶点计算相关系数
		for (int i = 0; i < 3; ++i)
		{
			//Mix area
			const Eigen::Vector3f& p0 = p.col(i);
			const Eigen::Vector3f& p1 = p.col((i + 1) % 3);
			const Eigen::Vector3f& p2 = p.col((i + 2) % 3);

			//判断顶点fv是否为内部顶点，边界顶点不参与计算
			if (interIdx(fv[(i + 1) % 3]) != -1)
			{
				//对vp求偏微分的系数
				Eigen::Vector3f v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
				//对vq求偏微分的系数
				Eigen::Vector3f v10 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
				//系数项
				for (int j = 0; j < 3; ++j)
				{
					if (v11[j]) Gradient(fv[(i + 1) % 3] * 3 + j) += v11[j];
					if (v10[j]) Gradient(fv[i] * 3 + j) += v10[j];
				}
			}

			if (interIdx(fv[(i + 2) % 3]) != -1)
			{
				//对vp求偏微分的系数
				Eigen::Vector3f v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
				//对vq求偏微分的系数
				Eigen::Vector3f v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
				//系数项
				for (int j = 0; j < 3; ++j)
				{
					if (v22[j]) Gradient(fv[(i + 2) % 3] * 3 + j) += v22[j];
					if (v20[j]) Gradient(fv[i] * 3 + j) += v20[j];
				}
			}
		}
	}
}

void BuildCoeffMatrix(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& angles, const Eigen::VectorXf& degrees, const Eigen::VectorXi& interIdx, Eigen::SparseMatrix<float>& A)
{
	int interNum = interIdx.maxCoeff() + 1;
	std::vector<Eigen::Triplet<float>> triA;
	for (int fi = 0; fi < F.cols(); ++fi)
	{
		const Eigen::Vector3i& fv = F.col(fi);
		for (int vi = 0; vi < 3; ++vi)
		{
			if (interIdx(fv[vi]) != -1)
			{
				dcoeff(triA, interIdx(fv[vi]) * 3, interIdx(fv[vi]) * 3, w1 + w2);
				if (interIdx(fv[(vi + 1) % 3]) != -1)
					dcoeff(triA, interIdx(fv[vi]) * 3, interIdx(fv[(vi + 1) % 3]) * 3, -w1 / degrees(fv[vi]));
				if (interIdx(fv[(vi + 2) % 3]) != -1)
					dcoeff(triA, interIdx(fv[vi]) * 3, interIdx(fv[(vi + 2) % 3]) * 3, -w1 / degrees(fv[vi]));
			}
		}
	}

	Eigen::VectorXf Gradient;
	calGradient(V, F, angles, interIdx, Gradient);
	for (int i = 0; i < V.cols(); ++i)
	{
		if (interIdx(i) != -1)
		{
			for (int j = 0; j < 3; ++j)
			{
				triA.push_back(Eigen::Triplet<float>(interIdx(i) * 3 + j, interNum * 3, Gradient(i * 3 + j)));
				triA.push_back(Eigen::Triplet<float>(interNum * 3, interIdx(i) * 3 + j, Gradient(i * 3 + j)));
			}
		}
	}
	A.resize(interNum * 3 + 1, interNum * 3 + 1);
	A.setFromTriplets(triA.begin(), triA.end());
}

void BuildrhsB(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& Lpos, const Eigen::VectorXf& degrees, const Eigen::VectorXi& interIdx, const Eigen::Matrix3Xf& oriV, Eigen::VectorXf& vecA, Eigen::VectorXf& b)
{
	int interNum = interIdx.maxCoeff() + 1;
	b.resize(3 * interNum + 1);
	b.setZero();
	for (int i = 0; i < V.cols(); ++i)
	{
		if (interIdx(i) != -1)
		{
			Eigen::Vector3f tempv = -w1 * (Lpos.col(i) / degrees(i)) - w2 * (V.col(i) - oriV.col(i));
			srhs(b, tempv, interIdx(i) * 3);
			b(3 * interNum) += (2 * M_PI - vecA(i));
		}
	}
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

void visualize_vertices(vtkRenderer* Renderer, const Eigen::Matrix3Xf& V)
{
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < V.cols(); ++i)
		points->InsertNextPoint(V.col(i).data());

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
		scalar->InsertTuple1(i, abs(2.0f * M_PI - angles(i)));
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