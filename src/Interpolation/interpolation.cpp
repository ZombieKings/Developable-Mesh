//------------------------------------------------------------------
// Paper: Quasi - Developable Mesh Surface Interpolation via Mesh Deformation
//------------------------------------------------------------------

#include "interpolation.h"

bool terminal_ = false;
unsigned int counter_ = 0;

double w1_ = 100.0;
double w2_ = 0.7;
double w3_ = 0.3;

double epD_ = 1.0;
double epI_ = 1.0;
double epL_ = 1.0;

double deD_ = 1.0;
double deI_ = 1.0;
double deLu_ = 0.90;
double deLl_ = 0.06;

double ED_ = 0.0;
double EI_ = 0.0;
double EL_ = 0.0;

double preED_ = 0.0;
double preEI_ = 0.0;
double preEL_ = 0.0;

int innerNum_ = 0;
Eigen::VectorXi VType_;
Eigen::Matrix2Xi matE_;
Eigen::Matrix3Xi matF_;
MatrixType matV_;

MatrixType oriV_;
MatrixType preV_;
VectorType oriLength_;

std::vector<std::vector<int>> vvNeighbor_Vertices_;
std::vector<std::vector<int>> vvNeighbor_Faces_;

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	if (EI_ >= epI_ || EL_ >= epL_ || ED_ >= epD_ || counter_ <= 50)
	{
		preED_ = ED_;
		preEL_ = EL_;
		preEI_ = EI_;
		ED_ = 0.;
		EL_ = 0.;
		EI_ = 0.;
		preV_ = matV_;

		Update_Mesh(matV_, matE_, matF_, VType_, vvNeighbor_Faces_, innerNum_, oriLength_);

		Adjust_Weights();

		VectorType vA;
		Zombie::cal_angles(matV_, matF_, vA);
		std::cout << "��" << counter_++ << "�ε�����������Ϊ�� " << cal_error(vA, VType_, 1) << "��ƽ�����Ϊ�� " << cal_error(vA, VType_, 0) << std::endl;
		//std::cout << ED_ + EI_ + EL_ << std::endl;

		//--------------���ӻ�����---------------------
		auto polydata = static_cast<vtkPolyData*>(clientData);
		auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);

		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV_.cols());
		for (auto i = 0; i < vA.size(); ++i)
		{
			if (VType_(i) != -1)
				scalar->InsertTuple1(i, abs(2. * M_PI - vA(i)));
			else
				scalar->InsertTuple1(i, 0);
		}

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
	Zombie::mesh2matrix(mesh, matV_, matE_, matF_);
	oriV_ = matV_;

	Zombie::get_neighbor_faces(mesh, vvNeighbor_Faces_);
	Zombie::get_neighbor_vertices(mesh, vvNeighbor_Vertices_);

	MatrixType matA;
	VectorType oriA;
	VectorType vAreas;
	Zombie::cal_angles_and_areas(matV_, matF_, oriA, vAreas, matA);

	//����ÿ���ߵ�ԭʼ�߳�
	oriLength_.setConstant(matE_.cols(), 0);
	for (int i = 0; i < matE_.cols(); ++i)
		oriLength_(i) = (oriV_.col(matE_(1, i)) - oriV_.col(matE_(0, i))).norm();

	//�����ʼED
	for (int i = 0; i < VType_.size(); ++i)
		if (VType_(i) != -1)
		{
			double ed = (2. * M_PI - oriA(i)) / vAreas(i) / 3.;
			ED_ += ed * ed;
		}

	w1_ = 50. * ED_;
	w2_ = 50. * ED_;

	deD_ = matV_.cols() * 1e-3;
	deI_ = matV_.cols() * 1e-3;

	epD_ = matV_.cols() * 1e-5;
	epI_ = matV_.cols() * 1e-5;
	epL_ = matV_.cols() * 1e-5;
	//w1_ = 1.;
	//w2_ = 1.;

	std::cout << "��ʼ����� " << cal_error(oriA, VType_, 1) << std::endl;
	std::cout << "��ʼƽ���� " << cal_error(oriA, VType_, 0) << std::endl;

	for (int i = 0; i < oriA.size(); ++i)
		if (VType_(i) == -1)
			oriA(i) = 0;
		else
			oriA(i) = abs(2. * M_PI - oriA(i));

	////---------------����-----------------
	//Update_Mesh(matV_, matE_, matF_, VType_, innerNum_, oriLength_);

	////---------------���ӻ�---------------
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
	interactor->CreateRepeatingTimer(1000);

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
	double E = 0.0;
	switch (flag)
	{
	case 0:
	{
		int cnt = 0;
		for (size_t i = 0; i < VType.size(); ++i)
			if (VType(i) != -1)
			{
				E += abs(2.0 * M_PI - vecAngles(i));
				++cnt;
			}
		E /= double(cnt);
		break;
	}
	case 1:
	{
		for (int i = 0; i < VType.size(); ++i)
			if (VType(i) != -1)
				E = std::max(E, abs(2.0 * M_PI - vecAngles(i)));
		break;
	}
	}
	return E;
}

void cal_target_faces_angles(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const std::vector<int>& vNFi, MatrixType& matA)
{
	const int Fnum = vNFi.size();
	matA.setConstant(3, Fnum, 0);
	for (int f = 0; f < vNFi.size(); ++f)
	{
		const auto& fv = F.col(vNFi[f]);
		for (int vi = 0; vi < 3; ++vi)
		{
			const auto& p0 = V.col(fv[vi]);
			const auto& p1 = V.col(fv[(vi + 1) % 3]);
			const auto& p2 = V.col(fv[(vi + 2) % 3]);
			const auto angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			matA(vi, f) = angle;
		}
	}
}


void Update_Mesh(MatrixType& V, const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& Vtype, const std::vector<std::vector<int>>& vvNeiF, int innerNum, const VectorType& oriLength)
{
	const int Vnum = V.cols();
	const int Enum = E.cols();
	const int Fnum = F.cols();

	//SparseMatrixType L;
	//L.resize(innerNum + Enum + Vnum * 3, Vnum * 3);
	//VectorType b;
	//b.setConstant(innerNum + Enum + Vnum * 3, 0);

	SparseMatrixType L;
	L.resize(innerNum + Enum + (Vnum - innerNum) * 3, Vnum * 3);
	VectorType b;
	b.setConstant(innerNum + Enum + (Vnum - innerNum) * 3, 0);

	std::vector<Tri> triL;
	triL.reserve(Vnum * innerNum * 3 + Enum * 6 + Vnum * 3);

	//�����˹����
	{
		//���㵱ǰ����ĸ�˹����
		VectorType vG;
		VectorType vAreas;
		MatrixType matA;
		Zombie::cal_angles_and_areas(V, F, vG, vAreas, matA);
		vG = (2. * M_PI - vG.array());
		vG.array() /= vAreas.array() / 3;

		for (int i = 0; i < Vtype.size(); ++i)
		{
			if (Vtype(i) != -1)
			{
				b(Vtype(i)) = -w1_ * vG(i);
				//�ռ�ED
				ED_ += vG(i) * vG(i);
			}
		}

		//�����˹���ʵ���ֵ�ݶ�
		//  (G(delta) - G(0)) / delta
		const double diff_step = 1e-4;
		MatrixType tmpV(V);
		for (int i = 0; i < Vnum; ++i)
		{
			const std::vector<int>& vNFi = vvNeiF[i];
			for (int j = 0; j < 3; ++j)
			{
				double vstep = diff_step * tmpV(j, i);
				tmpV(j, i) += vstep;

				MatrixType tmpA;
				cal_target_faces_angles(tmpV, F, vNFi, tmpA);
				//VectorType vdG(vG);
				for (int nf = 0; nf < vNFi.size(); ++nf)
				{
					auto& nfv = matF_.col(vNFi[nf]);
					for (int nfvi = 0; nfvi < 3; ++nfvi)
					{
						if (Vtype(nfv[nfvi]) != -1)
						{
							double dG = (matA(nfvi, vNFi[nf]) - tmpA(nfvi, nf)) / vAreas(nfv[nfvi]) / 3.;
							//std::cout << matA(nfvi, vvNeighbor_Faces_[i][nf]) << " " << tmpA(nfvi, nf) << " " << vAreas(nfv[nfvi]) << std::endl;
							triL.push_back(Tri(Vtype(nfv[nfvi]), i * 3 + j, w1_ * dG / diff_step));
						}
					}
				}

				//for (int k = 0; k < Vtype.size(); ++k)
				//{
				//	if (Vtype(k) != -1)
				//	{
				//		triL.push_back(Tri(Vtype(k), i * 3 + j, w1_ * (vdG(k) - vG(k)) / diff_step));
				//		//std::cout << vdG(k) << " " << vG(k) << " " << (vdG(k) - vG(k)) << std::endl;
				//		//std::cout << w1_ * (vdG(k) - b(Vtype(k))) / diff_step << std::endl;
				//	}
				//}
				tmpV(j, i) -= vstep;
			}
		}
	}

	//����߳�����
	//	l(e) - l0(e)
	for (int i = 0; i < Enum; ++i)
	{
		auto& ev = E.col(i);
		auto& v0 = V.col(ev[0]);
		auto& v1 = V.col(ev[1]);
		const double l = (v1 - v0).norm();
		b(innerNum + i) = -w2_ * (l - oriLength(i));
		for (int j = 0; j < 3; ++j)
		{
			triL.push_back(Tri(innerNum + i, ev[0] * 3 + j, -w2_ * v0[j] / l));
			triL.push_back(Tri(innerNum + i, ev[1] * 3 + j, w2_ * v1[j] / l));
		}

		//�ռ�EL
		EL_ = (l - oriLength(i)) * (l - oriLength(i));
	}

	//�����ֵ����
	//	V - V0
	int bound_cnt = 0;
	for (int i = 0; i < Vnum; ++i)
	{
		if (Vtype(i) == -1)
		{
			for (int j = 0; j < 3; ++j)
			{
				triL.push_back(Tri(innerNum + Enum + bound_cnt * 3 + j, i * 3 + j, w3_ * V(j, i)));
				b(innerNum + Enum + bound_cnt * 3 + j) = w3_ * (oriV_(j, i) - V(j, i));
			}
			//�ռ�EI
			EI_ += (oriV_.col(i) - V.col(i)).squaredNorm();
			++bound_cnt;
		}
	}

	//for (int i = 0; i < Vnum; ++i)
	//{
	//	for (int j = 0; j < 3; ++j)
	//	{
	//		triL.push_back(Tri(innerNum + Enum + i * 3 + j, i * 3 + j, V(j, i)));
	//		b(innerNum + Enum + i * 3 + j) = (oriV_(j, i) - V(j, i));
	//	}
	//	//�ռ�EI
	//	EI_ += (oriV_.col(i) - V.col(i)).squaredNorm();
	//}

	L.setFromTriplets(triL.begin(), triL.end());

	SparseMatrixType LT = L.transpose();
	//Eigen::MatrixXd tss(LT * L);
	//VectorType tssv = tss.bdcSvd().singularValues();
	//std::cout << tssv << std::endl;
	//int cntsd = 0;
	//for (int i = 0; i < tssv.rows(); ++i)
	//	if (tssv(i) == 0)
	//		++cntsd;
	//std::cout << cntsd << std::endl;
	//std::cout << tss.determinant() << std::endl;
	//std::cout << LT * L << std::endl;
	//std::cout << L << std::endl;
	//std::cout << b << std::endl;


	//solve least square problam
	Eigen::SimplicialLLT<SparseMatrixType> solver(LT * L);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Scales Solve Failed !" << std::endl;
	}

	VectorType S = solver.solve(LT * b);
	//std::cout << S << std::endl;
	for (int i = 0; i < Vnum; ++i)
		for (int j = 0; j < 3; ++j)
			if (abs(S(i * 3 + j)) > 1e-7)
				V(j, i) *= (1. + S(i * 3 + j));

}

void Adjust_Weights()
{
	if ((EI_ - preEI_) < deI_ && EI_ > epI_)
	{
		w1_ /= 2.;
		w2_ /= 2.;
		std::cout << "EI high and reduce slow, increase w3." << std::endl;
	}
	else if ((abs(EL_ - preEL_) / preEL_) > deLu_)
	{
		w2_ /= 2.;
		matV_ = preV_;
		std::cout << "EL varies too quickly, reduce w2, restart." << std::endl;
	}
	else if ((abs(EL_ - preEL_) / preEL_) < deLl_ && EL_ > epL_)
	{
		w2_ *= 2.;
		std::cout << "EL high and reduce slow, increase w2." << std::endl;
	}
	else if ((ED_ - preED_) < deD_ && ED_ > epD_)
	{
		w1_ *= 2.;
		std::cout << "ED high and reduce slow, increase w1." << std::endl;
	}
}
