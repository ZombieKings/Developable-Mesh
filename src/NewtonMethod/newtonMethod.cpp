//------------------------------------------------------------------
//注：解法器使用LDLT比QR快，但是需要剔除边界点对应的0部分，保证矩阵满秩
//------------------------------------------------------------------
#include "newtonMethod.h"

#define w1_ 10.0
#define w2_ 1.0
#define TAU 0.25

bool flag_ = false;
int it_conunter = 0;

int innerNum_ = 0;
Eigen::VectorXi VType_;
Eigen::VectorXi vecD_;

MatrixType matV_;
Eigen::Matrix3Xi matF_;

MatrixType oriV_;
VectorType vecOriV_;

SparseMatrixType L_;

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Laod failed!" << std::endl;
	}
	//收集内部顶点下标
	VType_.setConstant(mesh.n_vertices(), 0);
	for (const auto& vit : mesh.vertices())
	{
		if (mesh.is_boundary(vit))
			VType_(vit.idx()) = -1;
		else
			VType_(vit.idx()) = innerNum_++;
	}
	//-----------保存构造的网格-----------
	mesh2matrix(mesh, matV_, matF_);
	oriV_ = matV_;
	vecOriV_ = Eigen::Map<VectorType>(oriV_.data(), 3 * oriV_.cols(), 1);
	VectorType oriA;
	Zombie::cal_angles(matV_, matF_, oriA);
	Zombie::cal_uni_laplace(matV_.cols(), matF_, 3, L_);
	vecD_ = L_.diagonal().cast<int>();
	for (int k = 0; k < L_.outerSize(); ++k)
		for (SparseMatrixType::InnerIterator it(L_, k); it; ++it)
			if (VType_(it.row() / 3) != -1)
				it.valueRef() /= vecD_(it.row());
			else
				it.valueRef() = 0;

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	Zombie::visualize_mesh(renderer1, matV_, matF_, oriA, VType_);
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
	Zombie::visualize_mesh(renderer2, oriV_, matF_, oriA, VType_);
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

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (!flag_)
	{
		Update(matV_, matF_, VType_, innerNum_, L_);

		assert(!matV_.hasNaN() && "have invalid vertices data");
			

		VectorType vecA;
		Zombie::cal_angles(matV_, matF_, vecA);
		double error = 0.0;

		//update angle
		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV_.cols());
		for (auto i = 0; i < VType_.rows(); ++i)
			if (VType_(i) != -1)
			{
				double tmp = abs(vecA(i) - 2.0 * M_PI);
				error += tmp;
				scalar->InsertTuple1(i, tmp);
			}
			else
				scalar->InsertTuple1(i, 0);
		auto polydata = static_cast<vtkPolyData*>(clientData);
		auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);

		auto points = vtkSmartPointer<vtkPoints>::New();
		for (int i = 0; i < matV_.cols(); ++i)
			points->InsertNextPoint(matV_.col(i).data());
		polydata->SetPoints(points);
		polydata->GetPointData()->SetScalars(scalar);
		polydata->Modified();;

		iren->Render();
		if (flag_)
			std::cout << "优化结束,共进行";
		else
			std::cout << "第";
		std::cout << it_conunter++ << "次迭代，误差为： " << error / innerNum_ << std::endl;
	}
}

//void calLaplace_Angles_Neigh(const MatrixType& V, const Eigen::Matrix3Xi& F, MatrixType& A, Eigen::VectorXf& vecA, MatrixType& Lpos, Eigen::VectorXi& degrees)
//{
//	A.resize(3, F.cols());
//	vecA.resize(V.cols());
//	vecA.setZero();
//	std::vector<Eigen::Triplet<float>> tripleL;
//	tripleL.reserve(F.cols() * 9);
//	for (int j = 0; j < F.cols(); ++j)
//	{
//		const Eigen::Vector3i& fv = F.col(j);
//		for (size_t vi = 0; vi < 3; ++vi)
//		{
//			const Eigen::VectorXf& p0 = V.col(fv[vi]);
//			const Eigen::VectorXf& p1 = V.col(fv[(vi + 1) % 3]);
//			const Eigen::VectorXf& p2 = V.col(fv[(vi + 2) % 3]);
//			const float angle = std::acos(std::max(-1.0f, std::min(1.0f, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
//			A(vi, j) = angle;
//			vecA(fv[vi]) += angle;
//
//			tripleL.push_back(Eigen::Triplet<float>(fv[vi], fv[vi], 1));
//			tripleL.push_back(Eigen::Triplet<float>(fv[vi], fv[(vi + 1) % 3], -0.5f));
//			tripleL.push_back(Eigen::Triplet<float>(fv[vi], fv[(vi + 2) % 3], -0.5f));
//		}
//	}
//	Eigen::SparseMatrix<float> L;
//	L.resize(V.cols(), V.cols());
//	L.setFromTriplets(tripleL.begin(), tripleL.end());
//
//	degrees = L.diagonal();
//	for (int k = 0; k < L.outerSize(); ++k)
//		for (Eigen::SparseMatrix<float>::InnerIterator it(L, k); it; ++it)
//		{
//			it.valueRef() /= degrees(it.row());
//		}
//	Lpos = (L * V.transpose()).transpose();
//}
//
//void calGradient(const MatrixType& V, const Eigen::Matrix3Xi& F, const MatrixType& matAngles, const Eigen::VectorXi& interIdx, Eigen::VectorXf& Gradient)
//{
//	Gradient.resize(V.cols() * 3 + 1);
//	Gradient.setZero();
//	//高斯梯度
//	for (int fit = 0; fit < F.cols(); ++fit)
//	{
//		//记录当前面信息
//		const Eigen::Vector3i& fv = F.col(fit);
//		const Eigen::Vector3f& ca = matAngles.col(fit);
//		Eigen::Matrix3f p;
//		for (int i = 0; i < 3; ++i)
//			p.col(i) = V.col(fv[i]);
//
//		//计算各角及各边长
//		Eigen::Vector3f length;
//		for (int i = 0; i < 3; ++i)
//		{
//			length(i) = (p.col((i + 1) % 3) - p.col(i)).norm();
//		}
//
//		//对每个顶点计算相关系数
//		for (int i = 0; i < 3; ++i)
//		{
//			//Mix area
//			const Eigen::Vector3f& p0 = p.col(i);
//			const Eigen::Vector3f& p1 = p.col((i + 1) % 3);
//			const Eigen::Vector3f& p2 = p.col((i + 2) % 3);
//
//			//判断顶点fv是否为内部顶点，边界顶点不参与计算
//			if (interIdx(fv[(i + 1) % 3]) != -1)
//			{
//				//对vp求偏微分的系数
//				Eigen::Vector3f v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
//				//对vq求偏微分的系数
//				Eigen::Vector3f v10 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
//				//系数项
//				for (int j = 0; j < 3; ++j)
//				{
//					if (v11[j]) Gradient(fv[(i + 1) % 3] * 3 + j) += v11[j];
//					if (v10[j]) Gradient(fv[i] * 3 + j) += v10[j];
//				}
//			}
//
//			if (interIdx(fv[(i + 2) % 3]) != -1)
//			{
//				//对vp求偏微分的系数
//				Eigen::Vector3f v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
//				//对vq求偏微分的系数
//				Eigen::Vector3f v20 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
//				//系数项
//				for (int j = 0; j < 3; ++j)
//				{
//					if (v22[j]) Gradient(fv[(i + 2) % 3] * 3 + j) += v22[j];
//					if (v20[j]) Gradient(fv[i] * 3 + j) += v20[j];
//				}
//			}
//		}
//	}
//}

void cal_Angle_Sum_Gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& Vtype, VectorType& G)
{
	MatrixType matA;
	VectorType vecA;
	Zombie::cal_angles(V, F, vecA, matA);
	G.setConstant(V.cols() * 3, 0);
	//高斯曲率的梯度
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//记录当前面信息
		const auto& fv = F.col(fit);
		const auto& ca = matA.col(fit);

		//计算各角及各边长
		PosVector length;
		for (int i = 0; i < 3; ++i)
			length(i) = (V.col(fv[(i + 1) % 3]) - V.col(fv[i])).norm();

		//对一个面片内每个顶点计算相关系数
		for (int i = 0; i < 3; ++i)
		{
			const auto& p0 = V.col(fv[i]);
			const auto& p1 = V.col(fv[(i + 1) % 3]);
			const auto& p2 = V.col(fv[(i + 2) % 3]);
			//判断顶点fv是否为内部顶点，边界顶点不参与计算
			PosVector v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
			PosVector v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));

			PosVector v01 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
			PosVector v02 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
			if (Vtype(fv[i]) != -1)
				for (int j = 0; j < 3; ++j)
				{
					if (v01[j]) G(fv[(i + 1) % 3] * 3 + j) += v01[j];
					if (v02[j]) G(fv[(i + 2) % 3] * 3 + j) += v02[j];
				}

			if (Vtype(fv[(i + 1) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
					if (v11[j]) G(fv[(i + 1) % 3] * 3 + j) += v11[j];

			if (Vtype(fv[(i + 2) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
					if (v22[j]) G(fv[(i + 2) % 3] * 3 + j) += v22[j];
		}
	}
}

void Update(MatrixType& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& Vtype, int innerNum, const SparseMatrixType& L)
{
	const int Vnum = V.cols();
	VectorType vecV = Eigen::Map<VectorType>(V.data(), 3 * Vnum, 1);
	MatrixType matA;
	VectorType vecA;
	Zombie::cal_angles(V, F, vecA, matA);

	VectorType B(- w1_ * L * vecV - w2_ * (vecV - vecOriV_));
	B.conservativeResize(Vnum * 3 + innerNum);

	SparseMatrixType H(L);
	H.conservativeResize(Vnum * 3 + innerNum, Vnum * 3 + innerNum);

	for (int i = 0; i < Vnum; ++i)
		for (int j = 0; j < 3; ++j)
			if (Vtype(i) != -1)
				H.coeffRef(i * 3 + j, i * 3 + j) += w2_;
			else
				H.coeffRef(i * 3 + j, i * 3 + j) += w2_ * 1000;

	// lambda
	//高斯曲率的梯度
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//记录当前面信息
		const auto& fv = F.col(fit);
		const auto& ca = matA.col(fit);

		//计算各角及各边长
		PosVector length;
		for (int i = 0; i < 3; ++i)
			length(i) = (V.col(fv[(i + 1) % 3]) - V.col(fv[i])).norm();

		//对一个面片内每个顶点计算相关系数
		for (int i = 0; i < 3; ++i)
		{
			const auto& p0 = V.col(fv[i]);
			const auto& p1 = V.col(fv[(i + 1) % 3]);
			const auto& p2 = V.col(fv[(i + 2) % 3]);
			//判断顶点fv是否为内部顶点，边界顶点不参与计算
			PosVector v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
			PosVector v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));

			PosVector v01 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
			PosVector v02 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;
			if (Vtype(fv[i]) != -1)
				for (int j = 0; j < 3; ++j)
				{
					H.coeffRef(Vnum * 3 + Vtype(fv[i]), fv[(i + 1) % 3] * 3 + j) += v01[j];
					H.coeffRef(Vnum * 3 + Vtype(fv[i]), fv[(i + 2) % 3] * 3 + j) += v02[j];
					H.coeffRef(fv[(i + 1) % 3] * 3 + j, Vnum * 3 + Vtype(fv[i])) += v01[j];
					H.coeffRef(fv[(i + 2) % 3] * 3 + j, Vnum * 3 + Vtype(fv[i])) += v02[j];
				}

			if (Vtype(fv[(i + 1) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
				{
					H.coeffRef(Vnum * 3 + Vtype(fv[(i + 1) % 3]), fv[(i + 1) % 3] * 3 + j) += v11[j];
					H.coeffRef(fv[(i + 1) % 3] * 3 + j, Vnum * 3 + Vtype(fv[(i + 1) % 3])) += v11[j];
				}

			if (Vtype(fv[(i + 2) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
				{
					H.coeffRef(Vnum * 3 + Vtype(fv[(i + 2) % 3]), fv[(i + 2) % 3] * 3 + j) += v22[j];
					H.coeffRef(fv[(i + 2) % 3] * 3 + j, Vnum * 3 + Vtype(fv[(i + 2) % 3])) += v22[j];
				}
		}
	}


	for (int i = 0; i < Vtype.size(); ++i)
	{
		if (Vtype[i] != -1)
		{
			for (int j = 0; j < 3; ++j)
			{
				//H.coeffRef(Vnum * 3 + cntIn, i * 3 + j) = G(i * 3 + j);
				//H.coeffRef(i * 3 + j, Vnum * 3 + cntIn) = G(i * 3 + j);
				B(Vnum * 3 + Vtype[i]) = 2.0 * M_PI - vecA(i);
			}
		}
	}
	H.makeCompressed();

	//std::cout << H << std::endl;


	Eigen::SimplicialLDLT<SparseMatrixType> solver;
	//Eigen::SparseQR<SparseMatrixType, Eigen::COLAMDOrdering<int>> solver;
	solver.compute(H);
	if (solver.info() != Eigen::Success)
		std::cout << "solve fail" << std::endl;
	VectorType temp = solver.solve(B);

	// X = X + /tau * /delta
	for (int i = 0; i < Vnum; ++i)
		for (int j = 0; j < 3; ++j)
		{
			V(j, i) += TAU * temp(i * 3 + j);
			//std::cout << TAU * temp(i * 3 + j) << std::endl;
		}

	if (temp.topRows(Vnum * 3).norm() <= 1e-5)
		flag_ = true;
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

