#include "least_norm.h"

bool flag = false;
unsigned int counter = 0;
std::vector<int> interV_;
std::vector<int> boundV_;
Eigen::VectorXi interVidx_;

Eigen::VectorXf angle_sum_;
Eigen::Matrix3Xf angle_mat;

Eigen::Matrix3Xf vertices_mat;
Eigen::Matrix3Xi faces_mat;
Eigen::VectorXf update_d_;

double epsilon = 0;
double dqn = 1;
double theta = 0;
double pretheta = 0;

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (abs(pretheta - theta) >= 0.001 && dqn >= epsilon && counter < 50)
	{
		//-------Least Norm----------
		Eigen::SparseMatrix<float> A;
		Eigen::VectorXf b;
		cal_least_norm(vertices_mat, faces_mat, angle_mat, A, b);

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
		const Eigen::SparseMatrix<float> tempAT = A.transpose();
		const Eigen::SparseMatrix<float> tempA = (A * tempAT).eval();
		solver.compute(tempA);
		if (solver.info() != Eigen::Success)
		{
			std::cout << "Least norm compute failed!!!" << std::endl;
		}
		update_d_.resize(vertices_mat.cols() * 3);
		update_d_.setZero();
		Eigen::VectorXf tempx = solver.solve(b);
		update_d_ = tempAT * tempx;

		if (!update_d_.allFinite())
		{
			std::cout << "Wrong result!" << std::endl;
		}
		dqn = update_d_.squaredNorm();
		pretheta = theta;
		theta = cal_error(vertices_mat, faces_mat, angle_mat, 1);
		double theta2 = cal_error(vertices_mat, faces_mat, angle_mat, 0);

		std::cout << "第" << counter << "次迭代，最大误差为： " << theta << "，平均误差为： " << theta2 << std::endl;
		//---------update mesh----------
		if (theta >= 0.001 || (counter <= 20 && (pretheta - theta) >= 0.01))
		{
			Eigen::SparseMatrix<float> LA;
			Eigen::MatrixX3f lb;
			cal_laplace(vertices_mat, faces_mat, angle_mat, LA, lb);

			//解方程组
			Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> lsolver;
			lsolver.compute(LA);

			if (lsolver.info() != Eigen::Success)
			{
				std::cout << "Update compute failed!!!" << std::endl;
			}

			vertices_mat = lsolver.solve(lb).transpose();
		}
		else
		{
			//直接使用update_d更新矩阵
			for (int r = 0; r < vertices_mat.cols(); ++r)
			{
				for (size_t i = 0; i < 3; ++i)
				{
					vertices_mat(i, r) += update_d_(r * 3 + i);
				}
			}
		}
		//update angle
		cal_angles(vertices_mat, faces_mat, angle_mat);

		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(vertices_mat.cols());
		for (auto i = 0; i < angle_sum_.size(); ++i)
		{
			scalar->InsertTuple1(i, abs(2.0 * M_PI - angle_sum_(i)));
		}

		auto polydata = static_cast<vtkPolyData*>(clientData);
		auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);

		auto points = vtkSmartPointer<vtkPoints>::New();
		for (int i = 0; i < vertices_mat.cols(); ++i)
		{
			points->InsertNextPoint(vertices_mat.col(i).data());
		}
		polydata->SetPoints(points);
		polydata->GetPointData()->SetScalars(scalar);
		polydata->Modified();;

		iren->Render();

		counter++;
	}
	else if(!flag)
	{
		double e1 = cal_error(vertices_mat, faces_mat, angle_mat, 1);
		double e2 = cal_error(vertices_mat, faces_mat, angle_mat, 0);
		std::cout << "共" << counter << "次迭代，优化结果最大误差为： " << e1 << "，平均误差为： " << e2 << std::endl;
		flag = true;
	}
}

int main(int argc, char** argv)
{
	Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Load failed!" << std::endl;
	}

	//收集内部顶点下标
	interV_.clear();
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

	epsilon = std::max(interV_.size() * pow(10, -8), pow(10, -5));

	//-----------保存构造的网格-----------
	mesh2matrix(mesh, vertices_mat, faces_mat);

	cal_angles(vertices_mat, faces_mat, angle_mat);
	theta = cal_error(vertices_mat, faces_mat, angle_mat, 1);
	std::cout << "初始最大误差： " << theta << std::endl;
	std::cout << "初始平均误差： " << cal_error(vertices_mat, faces_mat, angle_mat, 0) << std::endl;

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(800, 1000);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer1, vertices_mat, faces_mat, angle_sum_);
	renderer1->SetViewport(0.0, 0.0, 1.0, 1.0);
	////视角设置
	renderer1->ResetCamera();
	renderWindow->AddRenderer(renderer1);

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
				vertices_mat.col(fvit.idx()) = Eigen::Map<const Eigen::Vector3f>(mesh.position(Surface_mesh::Vertex(fvit.idx())).data());
				flag(fvit.idx()) = 1;
			}
		}
	}
}

void cal_angles(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xf& A)
{
	A.resize(3, F.cols());
	for (int f = 0; f < F.cols(); ++f) {
		const Eigen::Vector3i& fv = F.col(f);
		for (size_t vi = 0; vi < 3; ++vi) {
			const Eigen::VectorXf& p0 = V.col(fv[vi]);
			const Eigen::VectorXf& p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::VectorXf& p2 = V.col(fv[(vi + 2) % 3]);
			const float angle = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			A(vi, f) = angle;
		}
	}

	angle_sum_.resize(V.cols());
	angle_sum_.setZero();
	//计算内角和
	for (int j = 0; j < F.cols(); ++j)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			if (interVidx_(F(i, j)) != -1)
				angle_sum_(F(i, j)) += A(i, j);
			else
				angle_sum_(F(i, j)) = 2.0 * M_PI;
		}
	}
}

double cal_error(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& A, int flag)
{
	//计算最大误差或平均误差
	if (flag)
	{
		double max = 0;
		for (size_t i = 0; i < interV_.size(); ++i)
		{
			max = abs(2.0 * M_PI - angle_sum_(interV_[i])) > max ? abs(2.0 * M_PI - angle_sum_(interV_[i])) : max;
		}
		return max;
	}
	else
	{
		double averange = 0;
		for (size_t i = 0; i < interV_.size(); ++i)
		{
			averange += angle_sum_(interV_[i]);
		}
		averange = 2.0 * M_PI - averange / interV_.size();
		return averange;
	}
}

void cal_laplace(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& A, Eigen::SparseMatrix<float>& L, Eigen::MatrixX3f& b)
{
	//计算固定边界的拉普拉斯系数矩阵
	std::vector<Eigen::Triplet<float>> triple;

	Eigen::VectorXf areas;
	areas.resize(V.cols());
	areas.setZero();
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		const Eigen::Vector3f& ca = A.col(j);

		//Mix area
		const Eigen::Vector3f& p0 = V.col(fv[0]);
		const Eigen::Vector3f& p1 = V.col(fv[1]);
		const Eigen::Vector3f& p2 = V.col(fv[2]);
		float area = ((p1 - p0).cross(p2 - p0)).norm() / 6.0f;

		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];

			if (interVidx_(fv0) != -1)
			{
				areas(fv0) += area;
				triple.push_back(Eigen::Triplet<float>(fv0, fv0, 1.0f / std::tan(ca[(vi + 1) % 3]) + 1.0f / std::tan(ca[(vi + 2) % 3])));
				triple.push_back(Eigen::Triplet<float>(fv0, fv1, -1.0f / std::tan(ca[(vi + 2) % 3])));
				triple.push_back(Eigen::Triplet<float>(fv0, fv2, -1.0f / std::tan(ca[(vi + 1) % 3])));
			}
		}
	}
	for (size_t i = 0; i < boundV_.size(); ++i)
	{
		triple.push_back(Eigen::Triplet<float>(boundV_[i], boundV_[i], 100));
	}

	//下半部分单位矩阵
	for (int i = 0; i < V.cols(); ++i)
	{
		triple.push_back(Eigen::Triplet<float>(i + V.cols(), i, 1));
	}

	L.resize(V.cols() * 2, V.cols());
	L.setFromTriplets(triple.begin(), triple.end());

	float sum_area = areas.sum() / float(interV_.size());

	for (int r = 0; r < interV_.size(); ++r)
	{
		L.row(interV_[r]) *= sum_area / (2.0f * areas(interV_[r]));
	}

	b.resize(V.cols() * 2, 3);
	b.setZero();

	//固定边界
	for (size_t ib = 0; ib < boundV_.size(); ++ib)
	{
		b.row(boundV_[ib]) = V.col(boundV_[ib]).transpose() * 100;
	}
	//变形目标
	for (int r = 0; r < V.cols(); ++r)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			b(r + V.cols(), i) = V(i, r) + update_d_(r * 3 + i);
		}
	}
}

void cal_least_norm(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& A, Eigen::SparseMatrix<float>& N, Eigen::VectorXf& b)
{
	//---------------计算系数矩阵----------------
	std::vector<Eigen::Triplet<float>> triple;

	Eigen::VectorXf sum_angle;
	sum_angle.resize(V.cols());
	sum_angle.setZero();
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//记录当前面信息
		const Eigen::Vector3i& fv = F.col(fit);
		const Eigen::Vector3f& ca = A.col(fit);
		Eigen::Matrix3f p;
		for (int i = 0; i < 3; ++i)
		{
			p.col(i) = V.col(fv[i]);
		}

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

			sum_angle(fv[i]) += A(i, fit);
			//判断顶点fv是否为内部顶点，边界顶点不参与计算
			if (interVidx_(fv[(i + 1) % 3]) != -1)
			{
				//对vp求偏微分的系数
				Eigen::Vector3f v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
				//对vq求偏微分的系数
				Eigen::Vector3f v10 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
				for (int j = 0; j < 3; ++j)
				{
					if (v11[j]) triple.push_back(Eigen::Triplet<float>(interVidx_(fv[(i + 1) % 3]), fv[(i + 1) % 3] * 3 + j, v11[j]));
					if (v10[j]) triple.push_back(Eigen::Triplet<float>(interVidx_(fv[(i + 1) % 3]), fv[i] * 3 + j, v10[j]));
				}
			}

			if (interVidx_(fv[(i + 2) % 3]) != -1)
			{
				//对vp求偏微分的系数
				Eigen::Vector3f v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
				//对vq求偏微分的系数
				Eigen::Vector3f v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
				for (int j = 0; j < 3; ++j)
				{
					if (v22[j]) triple.push_back(Eigen::Triplet<float>(interVidx_(fv[(i + 2) % 3]), fv[(i + 2) % 3] * 3 + j, v22[j]));
					if (v20[j]) triple.push_back(Eigen::Triplet<float>(interVidx_(fv[(i + 2) % 3]), fv[i] * 3 + j, v20[j]));
				}
			}
		}
	}

	//rhs
	b.resize(interV_.size() + boundV_.size() * 3);
	b.setZero();
	//高斯曲率rhs
	for (size_t i = 0; i < interV_.size(); ++i)
	{
		b(i) = 2.0f * M_PI - sum_angle(interV_[i]);
	}
	//固定边界
	for (size_t i = 0; i < boundV_.size(); ++i)
	{
		for (size_t j = 0; j < 3; ++j)
		{
			triple.push_back(Eigen::Triplet<float>(interV_.size() + i * 3 + j, boundV_[i] * 3 + j, 1));
			b(interV_.size() + i * 3 + j) = 0;
		}
	}
	N.resize(interV_.size() + boundV_.size() * 3, V.cols() * 3);
	N.setFromTriplets(triple.begin(), triple.end());
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