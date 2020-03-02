#include "QP.h"
#include "mosek_solver.h"

#define MAXIT 50

bool cvgflag_ = false;
unsigned int counter_ = 0;

double wl_ = 1.0;
double wn_ = 1.0;
double wp_ = 1.0;

std::vector<int> interV_;
std::vector<int> boundV_;
Eigen::VectorXi interVidx_;
Eigen::Matrix3Xi matF_;
VectorType orivecV_;
SparseMatrixType oriL_;
SparseMatrixType woriLtoriL_;
Eigen::MatrixXd woriNToriN_;
MatrixType matV_;

VectorType vecAngles_;
MatrixType matAngles_;
VectorType areas_;

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (counter_ < MAXIT && cvgflag_ == false)
	{
		//Eigen::Map<VectorType>vecV(matV_.data(), matV_.cols() * 3, 1);
		//MatrixType preV(matV_);
		//if (!Solve_with_Mosek_For4(matV_, matF_, interV_, interVidx_, matAngles_, vecAngles_, areas_, woriNToriN_, wn_, wp_))
		//	std::cout << "Solve failed !!!" << std::endl;
		if (!Solve_with_Mosek_For1(matV_, matF_, interV_, interVidx_, matAngles_, vecAngles_, areas_, wl_, wp_))
			std::cout << "Solve failed !!!" << std::endl;
		
		std::cout << "----------------------" << std::endl;
		std::cout << "第" << counter_++ << "次迭代，最大误差为： " << cal_error(vecAngles_, interV_, 1) << "，平均误差为： " << cal_error(vecAngles_, interV_, 0) << std::endl;
		
		//if ((preV - matV_).norm() <= MINDX)
		//{
		//	cvgflag_ = true;
		//}
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

	//网格初始信息收集
	mesh2matrix(mesh, matV_, matF_);
	cal_angles_and_areas(matV_, matF_, interVidx_, matAngles_, vecAngles_, areas_);
	MatrixType oriV(matV_);
	orivecV_ = Eigen::Map<VectorType>(oriV.data(), oriV.cols() * 3, 1);
	cal_cot_laplace(matF_, matAngles_, areas_, interVidx_, oriL_);
	//cal_uni_laplace(matF_, matV_.cols(), interVidx_, oriL_);
	woriLtoriL_ = wl_ * oriL_.transpose() * oriL_;
	VectorType oriA(vecAngles_);
	VectorType oriNormals;
	cal_normals(oriV, matF_, interVidx_, oriNormals);
	//MatrixType N = Eigen::Map<MatrixType>(oriNormals.data(), 3, matV_.cols());
	woriNToriN_ = wn_ * oriNormals * oriNormals.transpose();
	//Y_.setConstant(matV_.cols(), 0);

	std::cout << "初始最大误差： " << cal_error(vecAngles_, interV_, 1) << std::endl;
	std::cout << "初始平均误差： " << cal_error(vecAngles_, interV_, 0) << std::endl;

	////--------------测试---------------
	//Eigen::Map<VectorType>vecV(matV_.data(), matV_.cols() * 3, 1);
	//SparseMatrixType G;
	//cal_gaussian_gradient(matV_, matF_, interVidx_, matAngles_, G);
	//SparseMatrixType GdA(areas_.cwiseInverse().asDiagonal() * G);
	//VectorType GdAV(GdA * vecV);
	//VectorType theta(2.0 * M_PI - vecAngles_.array());
	//VectorType b(areas_.cwiseProduct(theta) + GdAV);
	//VectorType C;
	//Solve_C(GdAV, b, Y_, mu_, C);

	//SparseMatrixType L;
	//cal_cot_laplace(matF_, matAngles_, areas_, interVidx_, L);
	//Solve_V(vecV, GdA, L, b, Y_, C, mu_, wl_, wp_, matV_);

	//Y_ += mu_ * (C + GdAV - b);
	//mu_ *= rho_;

	////----------Mosek------------
	//int Vnum_ = matV_.cols();
	//int Vnum2_ = matV_.cols() * 2;
	//int Vnum3_ = matV_.cols() * 3;
	//int Vnum4_ = matV_.cols() * 4;
	//Eigen::Map<VectorType>vecV(matV_.data(), Vnum3_, 1);
	//SparseMatrixType G;
	//cal_gaussian_gradient(matV_, matF_, interVidx_, matAngles_, G);
	//Eigen::MatrixXd I;
	//I.setIdentity(Vnum3_, Vnum3_);
	//SparseMatrixType L;
	//cal_cot_laplace(matF_, matAngles_, areas_, interVidx_, L);
	//SparseMatrixType LTLI(wl_ * L.transpose() * L + wp_ * I);
	//SparseMatrixType Q(LTLI);
	//Q.conservativeResize(Vnum4_, Vnum4_);
	//for (int i = 0; i < Vnum3_; ++i)
	//{
	//	Q.coeffRef(i, i) *= 2.0;
	//}
	//SparseMatrixType QI(Q);
	//for (int i = 0; i < Vnum_; ++i)
	//{
	//	QI.coeffRef(Vnum3_ + i, Vnum3_ + i) = 1.0;
	//}
	//Q.makeCompressed();
	//QI.makeCompressed();
	//VectorType orivecV0;
	//orivecV0.setConstant(Vnum4_, 0);
	//orivecV0.topRows(orivecV_.size()) = orivecV_;
	//VectorType cl = 2.0 * orivecV0.transpose() * QI;
	//double cf = orivecV_.transpose() * LTLI * orivecV_;

	//std::vector<Tri> TriA;
	//for (int i = 0; i < G.outerSize(); ++i)
	//{
	//	for (Eigen::SparseMatrix<double>::InnerIterator it(G, i); it; ++it)
	//	{
	//		TriA.push_back(Tri(it.row(), it.col(), -it.value() / areas_(it.row())));
	//		TriA.push_back(Tri(it.row() + G.rows(), it.col(), it.value() / areas_(it.row())));
	//	}
	//}
	//for (int i = 0; i < Vnum_; ++i)
	//{
	//	TriA.push_back(Tri(i, G.cols() + i, 1));
	//	TriA.push_back(Tri(G.rows() + i, G.cols() + i, 1));
	//}
	//SparseMatrixType A;
	//A.resize(Vnum2_, Vnum4_);
	//A.setFromTriplets(TriA.begin(), TriA.end());

	//VectorType GV(G * vecV);
	//VectorType b;
	//b.setConstant(Vnum_, 0);
	//for (size_t i = 0; i < interV_.size(); ++i)
	//{
	//	b(interV_[i]) = (2.0 * M_PI - vecAngles_(interV_[i]) + GV(interV_[i])) / areas_[interV_[i]];
	//}
	//VectorType bb;
	//bb.resize(Vnum2_, 1);
	//bb.topRows(Vnum_) = b;
	//bb.bottomRows(Vnum_) = b;

	//mosek_solver solver(Q, A, cl, -bb, Vnum4_, cf);
	//VectorType result;
	//if (!solver.solve()) 
	//{
	//	result = Eigen::Map<VectorType>(solver.get_result(), orivecV_.size(), 1);
	//	//std::cout << result << std::endl;
	//}
	//matV_ = Eigen::Map<MatrixType>(result.data(), 3, Vnum_);
	//cal_angles(matV_, matF_, vecAngles_);

	int itn = 1;
	for (int i = 0; i < itn; ++i)
	{
		Solve_with_Mosek_For1(matV_, matF_, interV_, interVidx_, matAngles_, vecAngles_, areas_, wl_, wp_);
		//Solve_with_Mosek_For4(matV_, matF_, interV_, interVidx_, matAngles_, vecAngles_, areas_, woriNToriN_, wn_, wp_);
	}

	cal_angles_and_areas(matV_, matF_, interVidx_, matAngles_, vecAngles_, areas_);

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	visualize_mesh(renderer1, matV_, matF_, vecAngles_, interVidx_);
	//visualize_mesh_with_normal(renderer1, matV_, matF_, N);
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

void cal_normals(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, VectorType& Normals)
{
	MatrixType matTemp;
	matTemp.setConstant(3, V.cols(), 0);
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		const PosVector& p0 = V.col(fv[0]);
		const PosVector& p1 = V.col(fv[1]);
		const PosVector& p2 = V.col(fv[2]);
		const PosVector crosstemp = (p1 - p0).cross(p2 - p0);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			if (interVidx(fv[vi]) != -1)
				matTemp.col(fv[vi]) += crosstemp;
		}
	}

	for (int v = 0; v < matTemp.cols(); ++v)
	{
		if (interVidx(v) != -1)
			matTemp.col(v).normalize();
	}
	Normals = Eigen::Map<VectorType>(matTemp.data(), V.cols() * 3, 1);
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

double shrink(double x, double tau)
{
	int sign = x > 0 ? 1 : -1;
	double temp = abs(x);
	return sign * std::max((temp - tau), 0.0);
}

void build_tri_coeff(const Eigen::Matrix3Xi& F, int Vnum, SparseMatrixType& A1, SparseMatrixType& A2, SparseMatrixType& A3)
{
	std::vector<Tri> TriA1, TriA2, TriA3;
	TriA1.reserve(F.cols() * 2 * 3);
	TriA2.reserve(F.cols() * 2 * 3);
	TriA3.reserve(F.cols() * 3 * 3);

	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::Vector3i& fv = F.col(i);
		for (int j = 0; j < 3; ++j)
		{
			TriA1.push_back(Tri(i * 3 + j, fv[1] * 3 + j, 1.0));
			TriA1.push_back(Tri(i * 3 + j, fv[0] * 3 + j, -1.0));

			TriA2.push_back(Tri(i * 3 + j, fv[2] * 3 + j, 1.0));
			TriA2.push_back(Tri(i * 3 + j, fv[1] * 3 + j, -1.0));

			TriA3.push_back(Tri(i * 3 + j, fv[0] * 3 + j, 1.0 / 3.0));
			TriA3.push_back(Tri(i * 3 + j, fv[1] * 3 + j, 1.0 / 3.0));
			TriA3.push_back(Tri(i * 3 + j, fv[2] * 3 + j, 1.0 / 3.0));
		}
	}

	A1.resize(F.cols() * 3, Vnum * 3);
	A1.setFromTriplets(TriA1.begin(), TriA1.end());
	A2.resize(F.cols() * 3, Vnum * 3);
	A2.setFromTriplets(TriA2.begin(), TriA2.end());
	A3.resize(F.cols() * 3, Vnum * 3);
	A3.setFromTriplets(TriA3.begin(), TriA3.end());
}

void build_tri_coeff(const Eigen::Matrix3Xi& F, int Vnum, SparseMatrixType& A)
{
	std::vector<Tri> TriA;
	TriA.reserve(F.cols() * 7 * 3);

	for (int i = 0; i < F.cols(); ++i)
	{
		const Eigen::Vector3i& fv = F.col(i);
		for (int j = 0; j < 3; ++j)
		{
			TriA.push_back(Tri(i * 3 + j, fv[1] * 3 + j, 1.0));
			TriA.push_back(Tri(i * 3 + j, fv[0] * 3 + j, -1.0));

			TriA.push_back(Tri(i * 3 + j, fv[2] * 3 + j, 1.0));
			TriA.push_back(Tri(i * 3 + j, fv[1] * 3 + j, -1.0));

			TriA.push_back(Tri(i * 3 + j, fv[0] * 3 + j, 1.0 / 3.0));
			TriA.push_back(Tri(i * 3 + j, fv[1] * 3 + j, 1.0 / 3.0));
			TriA.push_back(Tri(i * 3 + j, fv[2] * 3 + j, 1.0 / 3.0));
		}
	}

	A.resize(F.cols() * 3, Vnum * 3);
	A.setFromTriplets(TriA.begin(), TriA.end());
}

void Solve_C(const VectorType& GdAV, const VectorType& b, const VectorType& Y, double mu, VectorType& C)
{
	VectorType TempV(-GdAV + b - Y / mu);
	C.resize(TempV.size());
	for (int i = 0; i < C.size(); ++i)
	{
		C(i) = shrink(TempV(i), 1.0 / mu);
	}
}

void Solve_V(const VectorType& V, const SparseMatrixType& GdA, const SparseMatrixType& L, const VectorType& b, const VectorType& Y, const VectorType& C, double mu, double wl, double wp, MatrixType& matV)
{
	SparseMatrixType GdAt = GdA.transpose();
	SparseMatrixType wLtL(2 * wl * L.transpose() * L);
	Eigen::MatrixXd I;
	I.setIdentity(V.size(), V.size());
	SparseMatrixType A(mu * GdAt * GdA + wLtL + 2 * wp * I);
	//VectorType Rhs(GdAt * (-Y + mu * (-C + b)) + wLtL * orivecV_ + 2 * wp * orivecV_);
	VectorType Rhs(GdAt * (-Y + mu * (-C + b)) + wLtL * V + 2 * wp * orivecV_);

	Eigen::SimplicialLDLT<SparseMatrixType> solver;
	//Eigen::SparseQR<SparseMatrixType, Eigen::COLAMDOrdering<int>> solver;
	solver.compute(A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Solve Failed !" << std::endl;
	}
	VectorType temp(solver.solve(Rhs));
	matV = Eigen::Map<MatrixType>(temp.data(), matV.rows(), matV.cols());
}

int Solve_with_Mosek_For1(MatrixType& V, const Eigen::Matrix3Xi& F, const std::vector<int> interV,
	const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, const VectorType& vAngles,
	const VectorType& areas, double wl, double wp)
{
	int Vnum = V.cols();
	int Vnum2 = Vnum * 2;
	int Vnum3 = Vnum * 3;
	int Vnum4 = Vnum * 4;
	Eigen::Map<VectorType>vecV(V.data(), Vnum3, 1);
	Eigen::MatrixXd I;
	I.setIdentity(Vnum3, Vnum3);
	SparseMatrixType LTLI(woriLtoriL_ + wp * I);
	SparseMatrixType Q(LTLI);
	Q.conservativeResize(Vnum4, Vnum4);
	for (int i = 0; i < Vnum3; ++i)
	{
		Q.coeffRef(i, i) *= 2.0;
	}
	Q.makeCompressed();

	VectorType v0QI((orivecV_.transpose() * LTLI).transpose());
	VectorType cl;
	cl.setConstant(Vnum4, 1);
	for (int i = 0; i < v0QI.size(); ++i)
	{
		cl(i) = -2.0 * v0QI(i);
	}

	double cf = v0QI.transpose() * orivecV_;

	//-------通用约束部分----------
	const size_t inVnum = interV.size();
	SparseMatrixType G;
	cal_gaussian_gradient(V, F, interVidx, mAngles, G);
	std::vector<Tri> TriA;
	for (int i = 0; i < G.outerSize(); ++i)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(G, i); it; ++it)
		{
			const int idx = interVidx(it.row());
			if (idx != -1)
			{
				TriA.push_back(Tri(idx, it.col(), -(-it.value())));
				TriA.push_back(Tri(idx + inVnum, it.col(), -it.value()));
			}
		}
	}
	VectorType b;
	b.setConstant(inVnum * 2, 0);
	VectorType GV(G * vecV);
	for (size_t i = 0; i < inVnum; ++i)
	{
		const int idx = interV[i];
		TriA.push_back(Tri(i, G.cols() + idx, 1));
		TriA.push_back(Tri(inVnum + i, G.cols() + idx, 1));

		const double tempv = 2.0 * M_PI - vAngles(interV[i]) + GV(interV[i]);
		b(i) = -tempv;
		b(i + inVnum) = tempv;
	}
	SparseMatrixType A;
	A.resize(inVnum * 2, Vnum4);
	A.setFromTriplets(TriA.begin(), TriA.end());

	mosek_solver solver(Q, A, cl, -b, Vnum4, cf);
	if (!solver.solve())
	{
		V = Eigen::Map<MatrixType>(solver.get_result(), 3, Vnum);
		return 1;
	}
	else
	{
		return 0;
	}
}

int Solve_with_Mosek_For4(MatrixType& V, const Eigen::Matrix3Xi& F, const std::vector<int> interV,
	const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, const VectorType& vAngles,
	const VectorType& areas, const Eigen::MatrixXd& wNNT, double wl, double wp)
{
	int Vnum = V.cols();
	int Vnum2 = Vnum * 2;
	int Vnum3 = Vnum * 3;
	int Vnum4 = Vnum * 4;
	Eigen::Map<VectorType>vecV(V.data(), Vnum3, 1);
	Eigen::MatrixXd I;
	I.setIdentity(Vnum3, Vnum3);

	SparseMatrixType T;
	build_tri_coeff(F, Vnum, T);
	SparseMatrixType Q(wNNT + wp * T.transpose() * T);

	VectorType v0Q((orivecV_.transpose() * Q).transpose());
	VectorType cl;
	cl.setConstant(Vnum4, 1);
	for (int i = 0; i < v0Q.size(); ++i)
	{
		cl(i) = -2.0 * v0Q(i);
	}
	double cf = v0Q.transpose() * orivecV_;

	Q.conservativeResize(Vnum4, Vnum4);
	for (int i = 0; i < Vnum3; ++i)
	{
		Q.coeffRef(i, i) *= 2.0;
	}
	Q.makeCompressed();

	//-------通用约束部分----------
	const size_t inVnum = interV.size();
	SparseMatrixType G;
	cal_gaussian_gradient(V, F, interVidx, mAngles, G);
	std::vector<Tri> TriA;
	for (int i = 0; i < G.outerSize(); ++i)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(G, i); it; ++it)
		{
			const int idx = interVidx(it.row());
			if (idx != -1)
			{
				TriA.push_back(Tri(idx, it.col(), -(-it.value())));
				TriA.push_back(Tri(idx + inVnum, it.col(), -it.value()));
			}
		}
	}
	VectorType b;
	b.setConstant(inVnum * 2, 0);
	VectorType GV(G * vecV);
	for (size_t i = 0; i < inVnum; ++i)
	{
		const int idx = interV[i];
		TriA.push_back(Tri(i, G.cols() + idx, 1));
		TriA.push_back(Tri(inVnum + i, G.cols() + idx, 1));

		const double tempv = 2.0 * M_PI - vAngles(interV[i]) + GV(interV[i]);
		b(i) = -tempv;
		b(i + inVnum) = tempv;
	}
	SparseMatrixType A;
	A.resize(inVnum * 2, Vnum4);
	A.setFromTriplets(TriA.begin(), TriA.end());

	mosek_solver solver(Q, A, cl, -b, Vnum4, cf);
	if (!solver.solve())
	{
		V = Eigen::Map<MatrixType>(solver.get_result(), 3, Vnum);
		return 1;
	}
	else
	{
		return 0;
	}
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

void matrix2vtk_normal(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& N, vtkPolyData* P)
{
	matrix2vtk(V, F, P);
	auto normaldata = vtkSmartPointer<vtkDoubleArray>::New();
	normaldata->SetNumberOfComponents(3);
	normaldata->SetNumberOfTuples(N.cols());
	for (auto i = 0; i < N.cols(); ++i)
		normaldata->InsertTuple(i, N.col(i).data());
	//P->GetCellData()->SetNormals(normaldata);
	P->GetPointData()->SetNormals(normaldata);
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

void MakeNormalGlyphs(vtkPolyData* src, vtkGlyph3D* glyph)
{
	auto arrow = vtkSmartPointer<vtkArrowSource>::New();
	arrow->SetTipLength(0.3);//参数
	arrow->SetTipRadius(.1);//参数
	arrow->SetShaftRadius(0.05);//参数
	arrow->Update();

	glyph->SetSourceConnection(arrow->GetOutputPort());
	glyph->SetInputData(src);
	glyph->SetVectorModeToUseNormal();
	glyph->SetScaleFactor(0.1);//参数
	glyph->OrientOn();
	glyph->Update();
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

void visualize_mesh_with_normal(vtkRenderer* Renderer, MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& N)
{
	//生成网格
	auto polydata = vtkSmartPointer<vtkPolyData>::New();
	matrix2vtk_normal(V, F, N, polydata);

	//网格及法向渲染器
	auto polyMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	polyMapper->SetInputData(polydata);
	auto polyActor = vtkSmartPointer<vtkActor>::New();
	polyActor->SetMapper(polyMapper);
	polyActor->GetProperty()->SetDiffuseColor(1, 1, 1);
	Renderer->AddActor(polyActor);

	//生成网格上的法向箭头
	auto glyphNormal = vtkSmartPointer<vtkGlyph3D>::New();
	MakeNormalGlyphs(polydata, glyphNormal);
	auto glyphNormalMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	glyphNormalMapper->SetInputConnection(glyphNormal->GetOutputPort());
	auto glyphNormalActor = vtkSmartPointer<vtkActor>::New();
	glyphNormalActor->SetMapper(glyphNormalMapper);
	glyphNormalActor->GetProperty()->SetDiffuseColor(1.0, 0.0, 0.0);
	Renderer->AddActor(glyphNormalActor);
}
