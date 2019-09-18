#include "least_norm.h"

unsigned int counter = 0;
std::vector<int> interV;
std::vector<int> boundV;
Eigen::VectorXi interVidx;

Eigen::VectorXf angle_sum_;
Eigen::Matrix3Xf angle_mat;

Eigen::Matrix3Xf vertices_mat;
Eigen::Matrix3Xi faces_mat;
Eigen::VectorXf update_d_;

double epsilon = 0;
double dqn = 1;
double theta = 0;
double pretheta = 0;

vtkSmartPointer<vtkColorTransferFunction> ctf;
vtkSmartPointer<vtkLookupTable> lut;

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (dqn >= epsilon && counter < 50)
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
		std::cout << "第" << counter << "次迭代，最大误差为： " << theta << std::endl;
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
			for (size_t r = 0; r < vertices_mat.cols(); ++r)
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
		auto *iren = static_cast<vtkRenderWindowInteractor*>(caller);

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
}

int main(int argc, char** argv)
{
	Surface_mesh mesh;
	if (!mesh.read("3D_p1.off"))
	{
		std::cout << "Load failed!" << std::endl;
	}

	//收集内部顶点下标
	interV.clear();
	interVidx.resize(mesh.n_vertices());
	interVidx.setOnes();
	interVidx *= -1;
	int count = 0;
	for (const auto &vit : mesh.vertices())
	{
		if (!mesh.is_boundary(vit))
		{
			interV.push_back(vit.idx());
			inter_p_r_(vit.idx()) = count++;
		}
		else
		{
			boundV.push_back(vit.idx());
		}
	}

	epsilon = std::max(interV.size() * pow(10, -8), pow(10, -5));

	//-----------保存构造的网格-----------
	mesh2matrix(mesh, vertices_mat, faces_mat);

	cal_angles(vertices_mat, faces_mat, angle_mat);
	theta = cal_error(vertices_mat, faces_mat, angle_mat, 1);
	std::cout << "初始最大误差： " << theta << std::endl;
	std::cout << "初始平均误差： " << cal_error(vertices_mat, faces_mat, angle_mat, 0) << std::endl;

	// Input vertices and faces
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (auto vit : mesh.vertices())
		points->InsertNextPoint(mesh.position(vit).data());
	auto faces = vtkSmartPointer <vtkCellArray>::New();
	for (auto fit : mesh.faces())
	{
		auto triangle = vtkSmartPointer<vtkTriangle>::New();
		int idx = 0;
		for (auto fvit : mesh.vertices(fit))
			triangle->GetPointIds()->SetId(idx++, fvit.idx());
		faces->InsertNextCell(triangle);
	}

	//Assign the pieces to the vtkPolyData.
	auto polydata = vtkSmartPointer<vtkPolyData>::New();
	polydata->SetPoints(points);
	polydata->SetPolys(faces);

	//Build color look up table
	ctf = vtkSmartPointer<vtkColorTransferFunction>::New();
	ctf->SetColorSpaceToHSV();
	ctf->AddRGBPoint(0.0, 0, 0, 1);
	ctf->AddRGBPoint(0.25, 0, 1, 1);
	ctf->AddRGBPoint(0.5, 0, 1, 0);
	ctf->AddRGBPoint(0.75, 1, 1, 0);
	ctf->AddRGBPoint(1.0, 1, 0, 0);
	lut = vtkSmartPointer<vtkLookupTable>::New();
	lut->SetNumberOfColors(256);
	for (auto i = 0; i < lut->GetNumberOfColors(); ++i)
	{
		Eigen::Vector4d color;
		ctf->GetColor(double(i) / lut->GetNumberOfColors(), color.data());
		color(3) = 1.0;
		lut->SetTableValue(i, color.data());
	}
	lut->Build();

	//Color related setup
	auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
	scalar->SetNumberOfComponents(1);
	scalar->SetNumberOfTuples(vertices_mat.cols());
	for (auto i = 0; i < angle_sum_.size(); ++i)
	{
		scalar->InsertTuple1(i, abs(2.0 * M_PI - angle_sum_(i)));
	}

	polydata->GetPointData()->SetScalars(scalar);
	// Create a mapper and actor
	auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polydata);
	mapper->SetLookupTable(lut);
	mapper->SetScalarRange(scalar->GetValueRange());
	auto actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	// Create a renderer, render window, and interactor
	auto renderer = vtkSmartPointer<vtkRenderer>::New();
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	auto renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	// Initialize must be called prior to creating timer events.
	renderWindowInteractor->Initialize();
	renderWindowInteractor->CreateRepeatingTimer(1000);

	auto timeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	timeCallback->SetCallback(CallbackFunction);
	timeCallback->SetClientData(polydata);

	renderWindowInteractor->AddObserver(vtkCommand::TimerEvent, timeCallback);

	// Add the actor to the scene
	renderer->AddActor(actor);
	renderer->SetBackground(0, 0, 0); // Background color white

	// Render and interact
	renderWindow->Render();
	renderWindowInteractor->Start();

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

int LoadMatrix(const Eigen::Matrix3Xf &V, const Eigen::Matrix3Xi &F, vtkSmartPointer<vtkPolyData>& polydata)
{
	polydata = vtkSmartPointer<vtkPolyData>::New();

	// Input vertices and faces
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (auto i = 0; i < V.cols(); ++i)
	{
		points->InsertNextPoint(V.col(i).data());
	}

	auto faces = vtkSmartPointer < vtkCellArray>::New();
	for (auto i = 0; i < F.cols(); ++i)
	{
		auto triangle = vtkSmartPointer<vtkTriangle>::New();
		for (int j = 0; j < 3; ++j)
		{
			triangle->GetPointIds()->SetId(j, F(j, i));
		}
		faces->InsertNextCell(triangle);
	}

	//Assign the pieces to the vtkPolyData.
	polydata->SetPoints(points);
	polydata->SetPolys(faces);

	return EXIT_SUCCESS;
}

void cal_angles(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xf& A)
{
	A.resize(3, F.cols());
	for (int f = 0; f < F.cols(); ++f) {
		const Eigen::Vector3i &fv = F.col(f);
		for (size_t vi = 0; vi < 3; ++vi) {
			const Eigen::VectorXf &p0 = V.col(fv[vi]);
			const Eigen::VectorXf &p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::VectorXf &p2 = V.col(fv[(vi + 2) % 3]);
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
			if (inter_p_r_(F(i, j)) != -1)
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
		for (size_t i = 0; i < interV.size(); ++i)
		{
			max = abs(2.0 * M_PI - angle_sum_(interV[i])) > max ? abs(2.0 * M_PI - angle_sum_(interV[i])) : max;
		}
		return max;
	}
	else
	{
		double averange = 0;
		for (size_t i = 0; i < interV.size(); ++i)
		{
			averange += angle_sum_(interV[i]);
		}
		averange = 2.0 * M_PI - averange / interV.size();
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
		const Eigen::Vector3i &fv = F.col(j);
		const Eigen::Vector3f &ca = A.col(j);

		//Mix area
		const Eigen::Vector3f &p0 = V.col(fv[0]);
		const Eigen::Vector3f &p1 = V.col(fv[1]);
		const Eigen::Vector3f &p2 = V.col(fv[2]);
		float area = ((p1 - p0).cross(p2 - p0)).norm() / 6.0f;

		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];

			if (inter_p_r_(fv0) != -1)
			{
				areas(fv0) += area;
				triple.push_back(Eigen::Triplet<float>(fv0, fv0, 1.0f / std::tan(ca[(vi + 1) % 3]) + 1.0f / std::tan(ca[(vi + 2) % 3])));
				triple.push_back(Eigen::Triplet<float>(fv0, fv1, -1.0f / std::tan(ca[(vi + 2) % 3])));
				triple.push_back(Eigen::Triplet<float>(fv0, fv2, -1.0f / std::tan(ca[(vi + 1) % 3])));
			}
		}
	}
	for (size_t i = 0; i < boundV.size(); ++i)
	{
		triple.push_back(Eigen::Triplet<float>(boundV[i], boundV[i], 100));
	}

	//下半部分单位矩阵
	for (int i = 0; i < V.cols(); ++i)
	{
		triple.push_back(Eigen::Triplet<float>(i + V.cols(), i, 1));
	}

	L.resize(V.cols() * 2, V.cols());
	L.setFromTriplets(triple.begin(), triple.end());

	float sum_area = areas.sum() / float(interV.size());

	for (int r = 0; r < interV.size(); ++r)
	{
		L.row(interV[r]) *= sum_area / (2.0f * areas(interV[r]));
	}

	b.resize(V.cols() * 2, 3);
	b.setZero();

	//固定边界
	for (size_t ib = 0; ib < boundV.size(); ++ib)
	{
		b.row(boundV[ib]) = V.col(boundV[ib]).transpose() * 100;
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
		const Eigen::Vector3i &fv = F.col(fit);
		const Eigen::Vector3f &ca = A.col(fit);
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
			const Eigen::Vector3f &p0 = p.col(i);
			const Eigen::Vector3f &p1 = p.col((i + 1) % 3);
			const Eigen::Vector3f &p2 = p.col((i + 2) % 3);

			sum_angle(fv[i]) += A(i, fit);
			//判断顶点fv是否为内部顶点，边界顶点不参与计算
			if (inter_p_r_(fv[(i + 1) % 3]) != -1)
			{
				//对vp求偏微分的系数
				Eigen::Vector3f v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
				//对vq求偏微分的系数
				Eigen::Vector3f v10 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
				for (int j = 0; j < 3; ++j)
				{
					if (v11[j]) triple.push_back(Eigen::Triplet<float>(inter_p_r_(fv[(i + 1) % 3]), fv[(i + 1) % 3] * 3 + j, v11[j]));
					if (v10[j]) triple.push_back(Eigen::Triplet<float>(inter_p_r_(fv[(i + 1) % 3]), fv[i] * 3 + j, v10[j]));
				}
			}

			if (inter_p_r_(fv[(i + 2) % 3]) != -1)
			{
				//对vp求偏微分的系数
				Eigen::Vector3f v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
				//对vq求偏微分的系数
				Eigen::Vector3f v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
				for (int j = 0; j < 3; ++j)
				{
					if (v22[j]) triple.push_back(Eigen::Triplet<float>(inter_p_r_(fv[(i + 2) % 3]), fv[(i + 2) % 3] * 3 + j, v22[j]));
					if (v20[j]) triple.push_back(Eigen::Triplet<float>(inter_p_r_(fv[(i + 2) % 3]), fv[i] * 3 + j, v20[j]));
				}
			}
		}
	}

	//rhs
	b.resize(interV.size() + boundV.size() * 3);
	b.setZero();
	//高斯曲率rhs
	for (size_t i = 0; i < interV.size(); ++i)
	{
		b(i) = 2.0f * M_PI - sum_angle(interV[i]);
	}
	//固定边界
	for (size_t i = 0; i < boundV.size(); ++i)
	{
		for (size_t j = 0; j < 3; ++j)
		{
			triple.push_back(Eigen::Triplet<float>(interV.size() + i * 3 + j, boundV[i] * 3 + j, 1));
			b(interV.size() + i * 3 + j) = 0;
		}
	}
	N.resize(interV.size() + boundV.size() * 3, V.cols() * 3);
	N.setFromTriplets(triple.begin(), triple.end());
}

