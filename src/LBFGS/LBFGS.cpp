#include "LBFGS.h"

#define EPS 1e-6
#define M 5

double w1 = 1e1;
double w2 = 1e0;
double w3 = 1e3;
double w4 = 1e0;

int normtype = 0;

std::vector<int> interV;
std::vector<int> boundV;
Eigen::VectorXi interVidx;
Eigen::Matrix3Xi matF_;
MatrixType matV_;
Eigen::SparseMatrix<DataType> F2V_;

VectorType vecAngles_;
MatrixType matAngles_;
MatrixType matNormals_;
VectorType areas_;
Eigen::MatrixX3d oriLpos_;

VectorType oriGradX_;
VectorType oriGradY_;
VectorType oriGradZ_;

double maxE_ = 0.0;
double minE_ = 2.0 * M_PI;

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Load failed!" << std::endl;
	}

	if (argc == 3)
	{
		normtype = atoi(argv[2]);
	}

	//收集内部顶点下标
	interV.clear();
	interVidx.setConstant(mesh.n_vertices(), -1);
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
	mesh2matrix(mesh, matV_, matF_);
	MatrixType oriV(matV_);
	cal_topo(matF_, matV_.cols(), interVidx, F2V_);
	cal_angles_and_areas_and_normal(matV_, matF_, matAngles_, vecAngles_, areas_, matNormals_);
	VectorType oriA(vecAngles_);
	MatrixType orimA(matAngles_);
	VectorType oriare(areas_);
	for (auto i = 0; i < oriA.size(); ++i)
	{
		if (interVidx(i) != -1)
		{
			double e = abs(2.0 * M_PI - oriA(i));
			if (e > maxE_) maxE_ = e;
			if (e < minE_) minE_ = e;
		}
	}

	Eigen::SparseMatrix<double> L;
	cal_laplace(matF_, matAngles_, areas_, interVidx, L);
	oriLpos_ = L * (matV_.transpose());

	Eigen::SparseMatrix<DataType> Grad;
	cal_grad(matV_, matF_, F2V_, Grad);
	oriGradX_.resize(matV_.cols() * 3, 1);
	oriGradX_ = Grad * (matV_.row(0).transpose());
	oriGradY_.resize(matV_.cols() * 3, 1);
	oriGradY_ = Grad * (matV_.row(1).transpose());
	oriGradZ_.resize(matV_.cols() * 3, 1);
	oriGradZ_ = Grad * (matV_.row(2).transpose());

	real_1d_array x;
	//x.setcontent(matV_.cols() * 3, matV_.data());
	x.attach_to_ptr(matV_.cols() * 3, matV_.data());
	//bool flag = grad_function_test(x, grad_function, ori_function);

	double epsg = 0;
	double epsf = 0;
	double epsx = 0.0000000001;
	double stpmax = 0.1;
	ae_int_t maxits = 0;
	minlbfgsstate state;
	minlbfgsreport rep;

	// create and tune optimizer
	minlbfgscreate(M, x, state);
	minlbfgssetcond(state, epsg, epsf, epsx, maxits);
	minlbfgssetstpmax(state, stpmax);
	VectorType vs;
	vs.resize(matV_.cols() * 3);
	memset(vs.data(), 100, sizeof(double) * vs.size());
	real_1d_array scalar;
	scalar.setcontent(vs.size(), vs.data());
	minlbfgssetscale(state, scalar);

	//OptGuard is essential at the early prototyping stages.
	//minlbfgsoptguardsmoothness(state);
	//minlbfgsoptguardgradient(state, 1);

	// first run
	alglib::minlbfgsoptimize(state, grad_function);
	real_1d_array rex;
	minlbfgsresults(state, rex, rep);

	std::cout << "迭代次数 : " << rep.iterationscount << std::endl;
	std::cout << "梯度计算次数 : " << rep.nfev << std::endl;
	std::cout << "终止情况 : " << rep.terminationtype << std::endl;

	//optguardreport ogrep;
	//minlbfgsoptguardresults(state, ogrep);
	//printf("%s\n", ogrep.badgradsuspected ? "true" : "false"); // EXPECTED: false
	//printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
	//printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false

	MatrixType curV = Eigen::Map<MatrixType>(rex.getcontent(), 3, matV_.cols());
	VectorType reA;
	MatrixType remA;
	VectorType reare;
	cal_angles(curV, matF_, remA, reA);

	//for (int i = 0; i < x.length(); ++i)
	//{
	//	std::cout << x[i] - rex[i] << std::endl;
	//}

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	//visualize_mesh(renderer1, curV, matF_, reA);
	visualize_mesh(renderer1, curV, matF_, reA, interVidx);
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
	//visualize_mesh(renderer2, matV_, matF_, oriA);
	visualize_mesh(renderer2, matV_, matF_, oriA, interVidx);
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

	//开始
	renderWindow->Render();
	interactor->Start();

	return 0;
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F)
{
	F.resize(3, mesh.n_faces());
	V.resize(3, mesh.n_vertices());

	Eigen::VectorXi flag;
	flag.setConstant(mesh.n_vertices(), -1);
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

void cal_topo(const Eigen::Matrix3Xi& F, int Vnum, const Eigen::VectorXi& interVidx, SparseMatrixType& F2V)
{
	std::vector<Tri> tripleF;
	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::Vector3i& fv = F.col(i);
		for (int j = 0; j < 3; ++j)
		{
			if (interVidx(fv[j]) != -1)
			{
				for (int k = 0; k < 3; ++k)
				{
					tripleF.push_back(Tri(fv[j] * 3 + k, i * 3 + k, 1));
				}
			}
		}
	}
	F2V.resize(Vnum * 3, F.cols() * 3);
	F2V.setFromTriplets(tripleF.begin(), tripleF.end());
}

void cal_angles(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& matAngles, VectorType& vecAngles)
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
			const PosVector& p0 = V.col(fv[vi]);
			const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
			const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			vecAngles(F(vi, f)) += angle;
			matAngles(vi, f) = angle;
		}
	}
}

void cal_angles_and_areas_and_normal(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& matAngles, VectorType& vecAngles, VectorType& areas, MatrixType& matNormal)
{
	matAngles.resize(3, F.cols());
	matAngles.setZero();
	vecAngles.resize(V.cols());
	vecAngles.setZero();
	matNormal.resize(3, V.cols());
	matNormal.setZero();
	areas.resize(V.cols());
	areas.setZero();
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);

		const PosVector crosstemp = (V.col(fv[1]) - V.col(fv[0])).cross(V.col(fv[2]) - V.col(fv[0]));
		//Mix area
		double area = crosstemp.norm() / 6.0f;

		for (size_t vi = 0; vi < 3; ++vi)
		{
			areas(fv[vi]) += area;
			matNormal.col(fv[vi]) += crosstemp;
			const PosVector& p0 = V.col(fv[vi]);
			const PosVector& p1 = V.col(fv[(vi + 1) % 3]);
			const PosVector& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			matAngles(vi, f) = angle;
			vecAngles(F(vi, f)) += angle;
		}
	}
	for (int v = 0; v < matNormal.cols(); ++v)
	{
		matNormal.col(v).normalize();
	}
}

void cal_gaussian_gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, const VectorType& vAngles, SparseMatrixType& mGradient)
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
			//Mix area
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
						triple.push_back(Tri(fv[(i + 1) % 3] * 3 + j, (fv[(i + 1) % 3]), v11[j]));
					if (v10[j])
						triple.push_back(Tri(fv[i] * 3 + j, (fv[(i + 1) % 3]), v10[j]));
				}
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
						triple.push_back(Tri(fv[(i + 2) % 3] * 3 + j, (fv[(i + 2) % 3]), v22[j]));
					if (v20[j])
						triple.push_back(Tri(fv[i] * 3 + j, (fv[(i + 2) % 3]), v20[j]));
				}
			}
		}
	}
	mGradient.resize(V.cols() * 3, V.cols());
	mGradient.setFromTriplets(triple.begin(), triple.end());
}

void cal_laplace(const Eigen::Matrix3Xi& F, MatrixTypeConst& matAngles, const VectorType& areas, const Eigen::VectorXi& interIdx, SparseMatrixType& L)
{
	std::vector<Tri> triple;
	for (int j = 0; j < F.cols(); ++j)
	{
		const Eigen::Vector3i& fv = F.col(j);
		const Eigen::Vector3d& ca = matAngles.col(j);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const int fv0 = fv[vi];
			const int fv1 = fv[(vi + 1) % 3];
			const int fv2 = fv[(vi + 2) % 3];
			if (interIdx(fv0) != -1)
			{
				triple.push_back(Tri(interIdx(fv0), fv0, w2 * (1.0 / std::tan(ca[(vi + 1) % 3]) + 1.0 / std::tan(ca[(vi + 2) % 3])) / (2.0 * areas(fv0))));
				triple.push_back(Tri(interIdx(fv0), fv1, -w2 / std::tan(ca[(vi + 2) % 3]) / (2.0 * areas(fv0))));
				triple.push_back(Tri(interIdx(fv0), fv2, -w2 / std::tan(ca[(vi + 1) % 3]) / (2.0 * areas(fv0))));
			}
		}
	}
	L.resize(interV.size(), areas.size());
	L.setFromTriplets(triple.begin(), triple.end());
}

void cal_grad(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const SparseMatrixType& F2V, SparseMatrixType& G)
{
	std::vector<Tri> tripleFG;
	tripleFG.reserve(F.cols() * 3 * 4);
	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::Vector3i& fv = F.col(i);

		//三角形各边向量
		const PosVector v21 = V.col(fv[2]) - V.col(fv[1]);
		const PosVector v02 = V.col(fv[0]) - V.col(fv[2]);
		const PosVector v10 = V.col(fv[1]) - V.col(fv[0]);
		const PosVector n = v21.cross(v02);
		const double dblA = n.norm();
		PosVector u = n / dblA;

		PosVector B10 = u.cross(v10);
		B10.normalize();
		B10 *= v10.norm() / dblA;
		for (int j = 0; j < 3; ++j)
		{
			tripleFG.push_back(Tri(i * 3 + j, fv[1], B10(j)));
			tripleFG.push_back(Tri(i * 3 + j, fv[0], -B10(j)));
		}

		PosVector B02 = u.cross(v02);
		B02.normalize();
		B02 *= v02.norm() / dblA;
		for (int j = 0; j < 3; ++j)
		{
			tripleFG.push_back(Tri(i * 3 + j, fv[2], B02(j)));
			tripleFG.push_back(Tri(i * 3 + j, fv[0], -B02(j)));
		}
	}
	Eigen::SparseMatrix<DataType> FG;
	FG.resize(F.cols() * 3, V.cols());
	FG.setFromTriplets(tripleFG.begin(), tripleFG.end());
	G.resize(V.cols() * 3, V.cols());
	G = F2V * FG;
}

double cal_error(const VectorType& vAngles, const std::vector<int>& interIdx)
{
	double error = 0.0;
	for (size_t i = 0; i < interIdx.size(); ++i)
	{
		error += 2.0 * M_PI - vAngles(interIdx[i]);
	}
	return (error / double(interIdx.size()));
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

void visualize_mesh(vtkRenderer* Renderer, MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const VectorType& angles)
{
	//生成网格
	auto P = vtkSmartPointer<vtkPolyData>::New();
	matrix2vtk(V, F, P);

	auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
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
	//polyMapper->SetScalarRange(scalar->GetValueRange()[0], scalar->GetValueRange()[1]);
	polyMapper->SetScalarRange(minE_, maxE_);
	//polyMapper->SetScalarRange(0.01, 0.023);

	auto polyActor = vtkSmartPointer<vtkActor>::New();
	polyActor->SetMapper(polyMapper);
	polyActor->GetProperty()->SetDiffuseColor(1, 1, 1);
	Renderer->AddActor(polyActor);
	Renderer->AddActor2D(scalarBar);
}

void grad_function(const real_1d_array& x, double& func, real_1d_array& grad, void* ptr = nullptr)
{
	real_1d_array temp(x);
	Eigen::Map<MatrixType>curV(temp.getcontent(), 3, matV_.cols());

	VectorType anglesV;
	VectorType areasV;
	MatrixType anglesM;
	MatrixType normalM;
	cal_angles_and_areas_and_normal(curV, matF_, anglesM, anglesV, areasV, normalM);

	func = 0.0;
	VectorType Gradient(curV.cols() * 3);
	Gradient.setZero();

	//高斯曲率
	for (size_t i = 0; i < interV.size(); ++i)
	{
		const double K((2.0 * M_PI - anglesV(interV[i])) / areasV(interV[i]));
		const double temp = normtype ? sqrt(K * K + EPS) : K * K / (K * K + EPS);
		func += w1 * temp;
		//std::cout << w1 * temp << std::endl;
	}

	//高斯曲率的梯度
	Eigen::SparseMatrix<DataType> Gau_;
	cal_gaussian_gradient(curV, matF_, interVidx, anglesM, anglesV, Gau_);
	for (int i = 0; i < Gau_.outerSize(); ++i)
	{
		const double K((2.0 * M_PI - anglesV(i)) / areasV(i));
		const double K2e(K * K + EPS);
		const double temp = normtype ? K / sqrt(K2e) : 2 * K * EPS / (K2e * K2e);
		for (Eigen::SparseMatrix<DataType>::InnerIterator it(Gau_, i); it; ++it)
		{
			Gradient(it.row()) += - w1 * temp * it.value();
			//std::cout << -w1 * temp * it.value() << std::endl;
		}
	}

	//拉普拉斯坐标
	Eigen::SparseMatrix<double> L;
	cal_laplace(matF_, anglesM, areasV, interVidx, L);
	Eigen::MatrixX3d Lpos = L * (curV.transpose());
	Eigen::MatrixX3d DLpos = (Lpos - oriLpos_);
	func += w2 * DLpos.squaredNorm();
	//std::cout << w2 * DLpos.squaredNorm() << std::endl;
	//拉普拉斯坐标项梯度
	Eigen::SparseMatrix<double> Lt;
	Lt = L.transpose();
	Eigen::MatrixX3d GL = 2 * w2 * Lt * DLpos;
	//std::cout << 2 * w2 * Lt * DLpos << std::endl;
	Gradient += Eigen::Map<VectorType>(GL.data(), curV.cols() * 3, 1);

	////梯度坐标差的梯度
	//Eigen::SparseMatrix<DataType> Grad;
	//cal_grad(curV, matF_, F2V_, Grad);
	//Eigen::SparseMatrix<DataType> Gradt(Grad.transpose());
	//VectorType gradX(Grad * (curV.row(0).transpose()));
	//VectorType gradY(Grad * (curV.row(1).transpose()));
	//VectorType gradZ(Grad * (curV.row(2).transpose()));
	//VectorType dgradX(gradX - oriGradX_);
	//VectorType dgradY(gradY - oriGradY_);
	//VectorType dgradZ(gradZ - oriGradZ_);
	//for (int i = 0; i < curV.cols(); ++i)
	//{
	//	for (int j = 0; j < 3; ++j)
	//	{
	//		double a(areasV(i));
	//		double dx(dgradX(i * 3 + j)), dy(dgradY(i * 3 + j)), dz(dgradZ(i * 3 + j));
	//		func += w2 * a * dx * dx;
	//		func += w2 * a * dy * dy;
	//		func += w2 * a * dz * dz;
	//	}
	//}
	////梯度坐标差的梯度
	//VectorType tempX(Gradt * dgradX);
	//VectorType tempY(Gradt * dgradY);
	//VectorType tempZ(Gradt * dgradZ);
	//for (int i = 0; i < curV.cols(); ++i)
	//{
	//	double a(areasV(i));
	//	Gradient(i * 3) += 2.0 * a * w2 * tempX(i);
	//	Gradient(i * 3 + 1) += 2.0 * a * w2 * tempY(i);
	//	Gradient(i * 3 + 2) += 2.0 * a * w2 * tempZ(i);
	//}
	
	//边界固定项
	for (size_t i = 0; i < boundV.size(); ++i)
	{
		const PosVector d = curV.col(boundV[i]) - matV_.col(boundV[i]);
		func += w3 * d.squaredNorm();
		//std::cout << w3 * d.squaredNorm() << std::endl;
		for (int j = 0; j < 3; ++j)
		{
			Gradient(boundV[i] * 3 + j) += d(j) * 2 * w3;
			//std::cout << d(j) * 2 * w3 << std::endl;
		}
	}

	////拉普拉斯平滑项
	//Eigen::SparseMatrix<double> L;
	//cal_laplace(matF_, anglesM, areasV, interVidx, L);
	//Lpos = L * (curV.transpose());
	//func += w2 * Lpos.squaredNorm();
	////拉普拉斯平滑项梯度
	//Eigen::SparseMatrix<double> Lt;
	//Lt = L.transpose();
	//Eigen::MatrixX3d GL = 2 * w2 * Lt * Lpos;
	//Gradient += Eigen::Map<VectorType>(GL.data(), curV.cols() * 3, 1);
	//
	////切平面保形项
	//for (size_t i = 0; i < interV.size(); ++i)
	//{
	//	const double d = (curV.col(interV[i]) - matV_.col(interV[i])).dot(normalM.col(interV[i]));
	//	func += w4 * d * d;
	//	for (int j = 0; j < 3; ++j)
	//	{
	//		Gradient(interV[i] * 3 + j) += normalM.col(interV[i])(j) * 2 * w4 * d;
	//	}
	//}	

	std::cout << "Energy: " << func << std::endl;
	std::cout << "Gradient: " << Gradient.norm() << std::endl;
	std::cout << "---------------------------" << std::endl;
	for (int i = 0; i < grad.length(); ++i)
	{
		grad[i] = Gradient(i);
	}
}

void ori_function(const real_1d_array& x, real_1d_array& func, void* ptr)
{
	real_1d_array temp(x);
	MatrixType curV = Eigen::Map<MatrixType>(temp.getcontent(), 3, matV_.cols());

	VectorType anglesV;
	MatrixType anglesM;
	cal_angles(curV, matF_, anglesM, anglesV);

	const double eps = 1e-3;

	for (size_t i = 0; i < curV.cols(); ++i)
	{
		if (interVidx(i) != -1)
		{
			for (int j = 0; j < 3; ++j)
			{
				//std::cout << curV(j, i) << std::endl;
				curV(j, i) += eps;
				//std::cout << curV(j, i) << std::endl;
				VectorType dAnglesV;
				MatrixType dAnglesM;
				cal_angles(curV, matF_, dAnglesM, dAnglesV);
				func[i * 3 + j] = (sqrt((dAnglesV(i) - 2.0 * M_PI) * (dAnglesV(i) - 2.0 * M_PI) + EPS) - sqrt((anglesV(i) - 2.0 * M_PI) * (anglesV(i) - 2.0 * M_PI) + EPS)) / eps;
				curV(j, i) -= eps;
			}
		}
		else
		{
			func[i * 3] = 0;
			func[i * 3 + 1] = 0;
			func[i * 3 + 2] = 0;
		}
	}
}

bool grad_function_test(const real_1d_array& x, void (*gradf)(const real_1d_array& x, real_1d_array& grad, void* ptr), void (*func)(const real_1d_array& x, real_1d_array& func, void* ptr))
{
	real_1d_array agrad, afunc;
	agrad.setlength(x.length());
	afunc.setlength(x.length());

	gradf(x, agrad, NULL);
	func(x, afunc, NULL);
	double max = 0;
	for (int i = 0; i < agrad.length(); ++i)
	{
		//std::cout << agrad[i] << " " << afunc[i] << std::endl;
		if (abs(agrad[i] - afunc[i]) >= max)
			max = agrad[i] - afunc[i];
		//if (abs(agrad[i] - afunc[i]) >= 1e-4)
		//	return false;
	}
	std::cout << max;
	return true;
}