#include "L0Inter.h"

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read("3D_p0.off"))
	{
		std::cout << "Load failed!" << std::endl;
	}

	//收集内部顶点下标
	std::vector<int> interV;
	std::vector<int> boundV;
	Eigen::VectorXi interVidx;
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
	int inter_num = interV.size();

	Eigen::Matrix3Xf matV;
	Eigen::Matrix3Xi matF;
	mesh2matrix(mesh, matV, matF);

	Eigen::VectorXf vAngles;
	Eigen::Matrix3Xf mAngles;
	Eigen::VectorXf areas;
	cal_angles_and_areas(matV, matF, boundV, mAngles, vAngles, areas);

	Eigen::VectorXf h;
	cal_H(vAngles, lambda / beta, h);

	std::vector<Eigen::Triplet<float>> triA;
	cal_Gaussian_tri(matV, matF, mAngles, interVidx, triA);
	cal_laplace_tri(matV, matF, mAngles, areas, inter_num, interVidx, triA);
	cal_interpolation_tri(boundV, inter_num * 4, triA);

	Eigen::SparseMatrix<float> A(interV.size() * 4 + boundV.size() * 3, matV.cols() * 3);
	A.setFromTriplets(triA.begin(), triA.end());
	Eigen::VectorXf b;
	cal_rhs(matV, A, interV, boundV, h, vAngles, b);

	//std::cout << b << std::endl;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
	solver.compute((A.transpose() * A).eval());

	if (solver.info() != Eigen::Success)
	{
		std::cout << "solve fail" << std::endl;
	}
	Eigen::VectorXf temp = solver.solve(A.transpose() * b);
	Eigen::Matrix3Xf resmatV = Eigen::Map<Eigen::Matrix3Xf>(temp.data(), 3, matV.cols());

	Eigen::VectorXf resAngles;
	Eigen::Matrix3Xf remAngles;
	Eigen::VectorXf reareas;
	cal_angles_and_areas(resmatV, matF, boundV, remAngles, resAngles, reareas);

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1400, 800);

	////---------------原始网格及法向可视化---------------
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer1, matV, matF, vAngles);
	renderer1->SetViewport(0.0, 0.0, 0.5, 1.0);
	////视角设置
	//renderer1->GetActiveCamera()->SetPosition(-1, 0, 0);
	//renderer1->GetActiveCamera()->SetViewUp(0, 0, 1);
	renderer1->ResetCamera();
	renderWindow->AddRenderer(renderer1);

	auto renderer2 = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer2, resmatV, matF, resAngles);
	renderer2->SetViewport(0.5, 0.0, 1.0, 1.0);
	////视角设置
	//renderer2->GetActiveCamera()->SetPosition(-1, 0, 0);
	//renderer2->GetActiveCamera()->SetViewUp(0, 0, 1);
	renderer2->ResetCamera();
	renderWindow->AddRenderer(renderer2);

	auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(renderWindow);
	auto style = vtkInteractorStyleTrackballCamera::New();
	interactor->SetInteractorStyle(style);
	interactor->Initialize();

	//开始
	renderWindow->Render();
	interactor->Start();

	return 1;
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& V, Eigen::Matrix3Xi& F)
{
	F.resize(3, mesh.n_faces());
	V.resize(3, mesh.n_vertices());

	Eigen::VectorXi flag;
	flag.resize(mesh.n_vertices());
	flag.setZero();
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
				V.col(fvit.idx()) = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data());
				flag(fvit.idx()) = 1;
			}
		}
	}
}

void cal_H(const Eigen::VectorXf& vAngles, float threshold, Eigen::VectorXf& h)
{
	h.resize(vAngles.size());
	for (int i = 0; i < vAngles.size(); ++i)
		h(i) = (2.0f * M_PI - vAngles(i)) <= threshold ? 0 : (2.0f * M_PI - vAngles(i));
}

void cal_angles_and_areas(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const std::vector<int>& boundIdx, Eigen::Matrix3Xf& mAngles, Eigen::VectorXf& vAngles, Eigen::VectorXf& areas)
{
	mAngles.resize(3, F.cols());
	vAngles.resize(V.cols());
	vAngles.setZero();
	areas.resize(V.cols());
	areas.setZero();
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);

		//Mix area
		float area = (V.col(fv[1]) - V.col(fv[0])).cross(V.col(fv[2]) - V.col(fv[0])).norm() / 6.0f;

		for (size_t vi = 0; vi < 3; ++vi)
		{
			areas(fv[vi]) += area;
			const Eigen::Vector3f& p0 = V.col(fv[vi]);
			const Eigen::Vector3f& p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::Vector3f& p2 = V.col(fv[(vi + 2) % 3]);
			const float angle = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			mAngles(vi, f) = angle;
			vAngles(F(vi, f)) += angle;
		}
	}

	for (size_t i = 0; i < boundIdx.size(); ++i)
	{
		vAngles(boundIdx[i]) = 2.0f * M_PI;
	}
}

void cal_Gaussian_tri(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& angles, const Eigen::VectorXi& interIdx, std::vector<Eigen::Triplet<float>>& triple)
{
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//记录当前面信息
		const Eigen::Vector3i& fv = F.col(fit);
		const Eigen::Vector3f& ca = angles.col(fit);
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
				for (int j = 0; j < 3; ++j)
				{
					if (v11[j]) triple.push_back(Eigen::Triplet<float>(interIdx(fv[(i + 1) % 3]), fv[(i + 1) % 3] * 3 + j, beta * WEIGHT1 * v11[j]));
					if (v10[j]) triple.push_back(Eigen::Triplet<float>(interIdx(fv[(i + 1) % 3]), fv[i] * 3 + j, beta * WEIGHT1 * v10[j]));
				}
			}

			if (interIdx(fv[(i + 2) % 3]) != -1)
			{
				//对vp求偏微分的系数
				Eigen::Vector3f v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
				//对vq求偏微分的系数
				Eigen::Vector3f v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
				for (int j = 0; j < 3; ++j)
				{
					if (v22[j]) triple.push_back(Eigen::Triplet<float>(interIdx(fv[(i + 2) % 3]), fv[(i + 2) % 3] * 3 + j, beta * WEIGHT1 * v22[j]));
					if (v20[j]) triple.push_back(Eigen::Triplet<float>(interIdx(fv[(i + 2) % 3]), fv[i] * 3 + j, beta * WEIGHT1 * v20[j]));
				}
			}
		}
	}
}

void cal_laplace_tri(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& angles, const Eigen::VectorXf& areas, int num, const Eigen::VectorXi& interIdx, std::vector<Eigen::Triplet<float>>& triple)
{
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		const Eigen::Vector3f& ca = angles.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (interIdx(fv0) != -1)
			{
				dcoeff(triple, num + interIdx(fv0) * 3, fv0 * 3, WEIGHT2 * (1.0f / std::tan(ca[(vi + 1) % 3]) + 1.0f / std::tan(ca[(vi + 2) % 3])) / (2.0f * areas(fv0)));
				dcoeff(triple, num + interIdx(fv0) * 3, fv1 * 3, -WEIGHT2 / std::tan(ca[(vi + 2) % 3]) / (2.0f * areas(fv0)));
				dcoeff(triple, num + interIdx(fv0) * 3, fv2 * 3, -WEIGHT2 / std::tan(ca[(vi + 1) % 3]) / (2.0f * areas(fv0)));
			}
		}
	}
}

void cal_interpolation_tri(std::vector<int>& anchor, int row, std::vector<Eigen::Triplet<float>>& triple)
{
	for (size_t i = 0; i < anchor.size(); ++i)
	{
		dcoeff(triple, row + i * 3, anchor[i] * 3, WEIGHT3);
	}
}

void cal_rhs(const Eigen::Matrix3Xf& V, const Eigen::MatrixXf& A, const std::vector<int>& interIdx, const std::vector<int>& boundIdx, const Eigen::VectorXf& h, const Eigen::VectorXf& vAngles, Eigen::VectorXf& rhb)
{
	rhb.resize(interIdx.size() * 4 + boundIdx.size() * 3);
	rhb.setZero();
	Eigen::VectorXf vecV(V.cols() * 3);
	memcpy(vecV.data(), V.data(), sizeof(float) * V.cols() * 3);
	for (size_t i = 0; i < interIdx.size(); ++i)
	{
		rhb(i) = A.row(i) * vecV + h(interIdx[i]) - (2.0f * M_PI - vAngles(interIdx[i]));
	}

	for (size_t i = 0; i < interIdx.size() * 3; ++i)
	{
		rhb(interIdx.size() + i) = A.row(interIdx.size() + i) * vecV;
	}

	for (size_t i = 0; i < boundIdx.size(); ++i)
	{
		srhs(rhb, WEIGHT3 * V.col(boundIdx[i]), interIdx.size() * 4 + i * 3);
	}
}

double cal_error(const Eigen::VectorXf& vAngles, const std::vector<int>& interIdx)
{
	double error = 0.0;
	for (int i = 0; i < interIdx.size(); ++i)
	{
		error += 2.0 * M_PI - vAngles(interIdx[i]);
	}
	return error;
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
