//------------------------------------------------------------------
// Paper: Toward flatten mesh surfaces
//------------------------------------------------------------------
#include "newtonMethod.h"

#define w1_ 10.0
#define w2_ 1.0
#define TAU 0.25

const int MAX_IT = 150;
bool flag_ = false;
int it_conunter = 0;

double theta_ = 0;
double pretheta_ = 0;

int innerNum_ = 0;
Eigen::VectorXi VType_;
Eigen::VectorXi vecD_;

MatrixType matV_;
Eigen::Matrix3Xi matF_;

MatrixType oriV_;
VectorType vecOriV_;

SparseMatrixType L_;
std::vector<Tri> basicH_;

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Laod failed!" << std::endl;
	}
	//�ռ��ڲ������±�
	VType_.setConstant(mesh.n_vertices(), 0);
	for (const auto& vit : mesh.vertices())
	{
		if (mesh.is_boundary(vit))
			VType_(vit.idx()) = -1;
		else
			VType_(vit.idx()) = innerNum_++;
	}
	//-----------���湹�������-----------
	Zombie::mesh2matrix(mesh, matV_, matF_);
	oriV_ = matV_;
	vecOriV_ = Eigen::Map<VectorType>(oriV_.data(), 3 * oriV_.cols(), 1);
	VectorType oriA;
	Zombie::cal_angles(matV_, matF_, oriA);
	Zombie::cal_uni_laplace(matV_.cols(), matF_, 3, L_);
	vecD_ = L_.diagonal().cast<int>();

	//Collect elements which is static in iteration.
	basicH_.reserve(matV_.cols() * 12);
	for (int k = 0; k < L_.outerSize(); ++k)
		for (SparseMatrixType::InnerIterator it(L_, k); it; ++it)
			if (VType_(it.row() / 3) != -1)
			{
				it.valueRef() /= vecD_(it.row());
				basicH_.push_back(Tri(it.row(), it.col(), it.value()));
			}
			else
				it.valueRef() = 0;

	for (int i = 0; i < matV_.cols(); ++i)
		for (int j = 0; j < 3; ++j)
			if (VType_(i) != -1)
				basicH_.push_back(Tri(i * 3 + j, i * 3 + j, w2_));
			else
				basicH_.push_back(Tri(i * 3 + j, i * 3 + j, w2_ * 1000));

	for (int i = 0; i < oriA.size(); ++i)
		if (VType_(i) == -1)
			oriA(i) = 0;
		else
			oriA(i) = abs(2. * M_PI - oriA(i));
	theta_ = cal_error(oriA, VType_, 1);
	std::cout << "��ʼ����� " << theta_ << std::endl;
	std::cout << "��ʼƽ���� " << cal_error(oriA, VType_, 0) << std::endl;

	//---------------���ӻ�---------------
	//��������
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	Zombie::visualize_mesh(renderer1, matV_, matF_, oriA, VType_);	
	renderer1->SetViewport(0.0, 0.0, 0.5, 1.0);

	//����ı�
	auto textActor1 = vtkSmartPointer<vtkTextActor>::New();
	textActor1->SetInput("Result Mesh");
	textActor1->GetTextProperty()->SetFontSize(33);
	textActor1->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	renderer1->AddActor2D(textActor1);

	//�ӽ�����
	renderer1->ResetCamera();
	renderWindow->AddRenderer(renderer1);

	auto renderer2 = vtkSmartPointer<vtkRenderer>::New();
	Zombie::visualize_mesh(renderer2, oriV_, matF_, oriA, VType_);
	renderer2->SetViewport(0.5, 0.0, 1.0, 1.0);

	//����ı�
	auto textActor2 = vtkSmartPointer<vtkTextActor>::New();
	textActor2->SetInput("Original Mesh");
	textActor2->GetTextProperty()->SetFontSize(33);
	textActor2->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	renderer2->AddActor2D(textActor2);

	//�ӽ�����
	renderer2->ResetCamera();
	renderWindow->AddRenderer(renderer2);

	auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(renderWindow);
	auto style = vtkInteractorStyleTrackballCamera::New();
	interactor->SetInteractorStyle(style);
	interactor->Initialize();
	interactor->CreateRepeatingTimer(100);

	auto timeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
	timeCallback->SetCallback(CallbackFunction);
	timeCallback->SetClientData(renderer1->GetActors()->GetLastActor()->GetMapper()->GetInput());

	interactor->AddObserver(vtkCommand::TimerEvent, timeCallback);

	//��ʼ
	renderWindow->Render();
	interactor->Start();

	return EXIT_SUCCESS;
}

double cal_error(const VectorType& vecAngles, const Eigen::VectorXi& VType, int flag)
{
	//�����������ƽ�����
	if (flag)
	{
		size_t idx = 0;
		double maxE = 0;
		for (int i = 0; i < VType_.size(); ++i)
			if (VType_(i) != -1)
				maxE = std::max(maxE, abs(2.0 * M_PI - vecAngles(i)));
		return maxE;
	}
	else
	{
		double averange = 0;
		int cnt = 0;
		for (size_t i = 0; i < VType_.size(); ++i)
			if (VType_(i) != -1)
			{
				averange += abs(2.0 * M_PI - vecAngles(i));
				++cnt;
			}
		averange /= double(cnt);
		return averange;
	}
}

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (!flag_ && it_conunter < MAX_IT)
	{
		Update(matV_, matF_, VType_, innerNum_, basicH_, L_);

		assert(!matV_.hasNaN() && "have invalid vertices data");

		VectorType vecA;
		Zombie::cal_angles(matV_, matF_, vecA);

		pretheta_ = theta_;
		theta_ = cal_error(vecA, VType_, 1);
		std::cout << "��" << it_conunter++ << "�ε�����������Ϊ�� " << theta_ << "��ƽ�����Ϊ�� " << cal_error(vecA, VType_, 0) << std::endl;

		//Update mesh
		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV_.cols());
		for (auto i = 0; i < VType_.rows(); ++i)
			if (VType_(i) != -1)
				scalar->InsertTuple1(i, abs(vecA(i) - 2.0 * M_PI));
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
	}
}

void Update(MatrixType& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& Vtype, int innerNum, std::vector<Tri> basicH, const SparseMatrixType& L)
{
	const int Vnum = V.cols();
	VectorType vecV = Eigen::Map<VectorType>(V.data(), 3 * Vnum, 1);
	MatrixType matA;
	VectorType vecA;
	Zombie::cal_angles(V, F, vecA, matA);

	//Calculate rhs B
	//	  /								\
	//B = |  - w1 * LV - w2 * (V - V0)  |
	//    |        2\pi - \theta(v)     |
	//    \								/
	VectorType B(-w1_ * L * vecV - w2_ * (vecV - vecOriV_));
	B.conservativeResize(Vnum * 3 + innerNum);
	for (int i = 0; i < Vtype.size(); ++i)
		if (Vtype[i] != -1)
			for (int j = 0; j < 3; ++j)
				B(Vnum * 3 + Vtype[i]) = 2.0 * M_PI - vecA(i);

	// Partial gradients via Lambda
	// �ڽǺ͵��ݶ�
	for (int fit = 0; fit < F.cols(); ++fit)
	{
		//��¼��ǰ����Ϣ
		const auto& fv = F.col(fit);
		const auto& ca = matA.col(fit);

		//������Ǽ����߳�
		PosVector length;
		for (int i = 0; i < 3; ++i)
			length(i) = (V.col(fv[(i + 1) % 3]) - V.col(fv[i])).norm();

		//��һ����Ƭ��ÿ������������ϵ��
		for (int i = 0; i < 3; ++i)
		{
			const auto& p0 = V.col(fv[i]);
			const auto& p1 = V.col(fv[(i + 1) % 3]);
			const auto& p2 = V.col(fv[(i + 2) % 3]);

			//��������������������v1��v2���ڽǶ����������ƫ΢���뵱ǰ��i��ص�ϵ��
			PosVector v11 = (p0 - p1) / (tan(ca[i]) * length(i) * length(i));
			PosVector v22 = (p0 - p2) / (tan(ca[i]) * length((i + 2) % 3) * length((i + 2) % 3));
			
			//����v0���ڽ�i����������������������v1��v2��ƫ΢��ϵ��
			PosVector v01 = (p0 - p2) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v11;
			PosVector v02 = (p0 - p1) / (sin(ca[i]) * length(i) * length((i + 2) % 3)) - v22;

			//�ж϶���i�Ƿ�Ϊ�ڲ����㣬�߽綥�㲻�������
			if (Vtype(fv[i]) != -1)
				for (int j = 0; j < 3; ++j)
				{
					basicH.push_back(Tri(Vnum * 3 + Vtype(fv[i]), fv[(i + 1) % 3] * 3 + j, v01[j]));
					basicH.push_back(Tri(Vnum * 3 + Vtype(fv[i]), fv[(i + 2) % 3] * 3 + j, v02[j]));
					basicH.push_back(Tri(fv[(i + 1) % 3] * 3 + j, Vnum * 3 + Vtype(fv[i]), v01[j]));
					basicH.push_back(Tri(fv[(i + 2) % 3] * 3 + j, Vnum * 3 + Vtype(fv[i]), v02[j]));
				}

			//�ж϶���i+1�Ƿ�Ϊ�ڲ����㣬�߽綥�㲻�������
			if (Vtype(fv[(i + 1) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
				{
					basicH.push_back(Tri(Vnum * 3 + Vtype(fv[(i + 1) % 3]), fv[(i + 1) % 3] * 3 + j, v11[j]));
					basicH.push_back(Tri(fv[(i + 1) % 3] * 3 + j, Vnum * 3 + Vtype(fv[(i + 1) % 3]), v11[j]));
				}

			//�ж϶���i+2�Ƿ�Ϊ�ڲ����㣬�߽綥�㲻�������
			if (Vtype(fv[(i + 2) % 3]) != -1)
				for (int j = 0; j < 3; ++j)
				{
					basicH.push_back(Tri(Vnum * 3 + Vtype(fv[(i + 2) % 3]), fv[(i + 2) % 3] * 3 + j, v22[j]));
					basicH.push_back(Tri(fv[(i + 2) % 3] * 3 + j, Vnum * 3 + Vtype(fv[(i + 2) % 3]), v22[j]));
				}
		}
	}
	SparseMatrixType H(Vnum * 3 + innerNum, Vnum * 3 + innerNum);
	H.setFromTriplets(basicH.begin(), basicH.end());

	//solve linear system
	Eigen::SimplicialLDLT<SparseMatrixType> solver;
	solver.compute(H);
	if (solver.info() != Eigen::Success)
		std::cout << "solve fail" << std::endl;
	VectorType temp = solver.solve(B);

	// move X with damping factor 0.25
	// X = X + /tau * /delta
	for (int i = 0; i < Vnum; ++i)
		for (int j = 0; j < 3; ++j)
			V(j, i) += TAU * temp(i * 3 + j);

	// check jump out conditions
	// ||\delta|| <= 10-5
	if (temp.topRows(Vnum * 3).norm() <= 1e-5)
		flag_ = true;
}

