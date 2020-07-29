#include "DoTM.h"

#include <deque>

#include <fstream>
#include "../tools/vtk.h"

//A Few compiler flags that control the line searches
#define MAX_LINESEARCH_TRIES 100
#define ARMIJO_C1 1e-4
#define WOLFE_C2 0.99
#define N_LBFGS_VECTORS 8
#define MAX_T 100. //1.
#define MIN_T 1e-12

//Infinity
#define INFTY std::numeric_limits<double>::infinity()

bool switch_flag_ = true;
int itcnt_ = 0;
double averE_ = 0.0;
double maxE_ = 0.0;
double totalT = 0;

Eigen::VectorXi VType_;
Eigen::Matrix3Xi matF_;
MatrixType matV_;
std::vector<std::vector<Eigen::Vector2i>> adjF_;
VectorType areas_;
VectorType vecA_;
MatrixType matA_;
MatrixType matN_;

double t_ = 1e-5;
MatrixType p_;

double energy_;
MatrixType energyGrad_;

void TimeCallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData))
{
	auto polydata = static_cast<vtkPolyData*>(clientData);
	auto* iren = static_cast<vtkRenderWindowInteractor*>(caller);

	if (switch_flag_)
	{
		VectorType vA;
		VectorType as;
		MatrixType mA;
		int success = opt_solve(matV_, matF_, adjF_, t_, p_, energy_, energyGrad_);
		if (success < 0)
		{
			std::cout << "Line search failed with error code " << success << std::endl;
			switch_flag_ = false;
		}
		totalT += t_;
		Zombie::cal_angles_and_areas(matV_, matF_, vA, as, mA);

		//Update errors
		averE_ = cal_error(vA, as, VType_, 0);
		maxE_ = cal_error(vA, as, VType_, 1);

		itcnt_++;
		std::cout << "第" << itcnt_ << "次迭代，最大误差为： " << maxE_ << "，平均误差为： " << averE_ << std::endl;

		//--------------可视化更新---------------------
		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(matV_.cols());
		for (auto i = 0; i < vA.size(); ++i)
		{
			if (VType_(i) != -1)
				scalar->InsertTuple1(i, abs(2.0f * M_PI - vA(i)));
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
	//收集内部顶点下标
	VType_.setConstant(mesh.n_vertices(), 0);
	for (const auto& vit : mesh.vertices())
	{
		VType_(vit.idx()) = mesh.is_boundary(vit) ? -1 : 1;
	}

	//网格初始信息收集
	mesh2matrix(mesh, matV_, matF_);
	Zombie::cal_angles_and_areas(matV_, matF_, vecA_, areas_, matA_);
	Zombie::cal_normal_per_face(matV_, matF_, matN_);
	adj_face(matV_.cols(), matF_, adjF_);
	MatrixType oriV(matV_);
	VectorType oriA(vecA_);
	std::cout << "初始最大误差： " << cal_error(vecA_, areas_, VType_, 1) << std::endl;
	std::cout << "初始平均误差： " << cal_error(vecA_, areas_, VType_, 0) << std::endl;

	//--------------测试---------------

	//int success = opt_solve(matV_, matF_, adjF_, t_, p_, energy_, energyGrad_);
	//if (success < 0) {
	//	std::stringstream stream;
	//	//stream << "energy_at_step_" << t.totalSteps << ".mat";
	//	//write_energy(oldm.V, oldm.F, oldm.VF, oldm.VFi, oldm.isB, oldt.p, stream.str(), 1e-7, energyMode);
	//	std::cout << "Line search failed with error code " << success << std::endl;
	//	//animating = false;
	//}
	
	std::vector<double> FA;
	for (int i = 0; i < matF_.cols(); ++i)
		FA.push_back(i);

	std::ofstream os("angles.vtk");
	tri2vtk(os, matV_.data(), matV_.cols(), matF_.data(), matF_.cols());
	cell_data(os, FA.begin(), FA.size(), "angle");
	os.close();


	//Zombie::cal_angles(matV_, matF_, vecA_);
	//std::cout << "最终最大误差： " << cal_error(vecA_, VType_, areas_, 1) << std::endl;
	//std::cout << "最终平均误差： " << cal_error(vecA_, VType_, areas_, 0) << std::endl;
	//energy_test(matV_, matF_, adjF_);

	double e;
	MatrixType G;
	hinge_energy_and_grad(matV_, matF_, adjF_, e, G);

	//---------------可视化---------------
	//创建窗口
	auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(1600, 800);
	auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	Zombie::visualize_mesh(renderer1, matV_, matF_, vecA_, VType_);
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
	Zombie::visualize_mesh(renderer2, oriV, matF_, oriA, VType_);
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
	timeCallback->SetCallback(TimeCallbackFunction);
	timeCallback->SetClientData(renderer1->GetActors()->GetLastActor()->GetMapper()->GetInput());

	interactor->AddObserver(vtkCommand::TimerEvent, timeCallback);

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

void adj_face(int Vnum, const Eigen::Matrix3Xi& F, std::vector<std::vector<Eigen::Vector2i>>& adjF)
{
	adjF.resize(Vnum);
	for (int i = 0; i < F.cols(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			adjF[F(j, i)].push_back(Eigen::Vector2i(i, j));
		}
	}
}

double cal_error(const VectorType& vecAngles, const VectorType& vecAreas, const Eigen::VectorXi& VType, int flag)
{
	//计算最大误差或平均误差
	if (flag)
	{
		size_t idx = 0;
		double max = 0;
		for (int i = 0; i < VType_.size(); ++i)
		{
			if (VType_(i) != -1)
			{
				const double e = abs(2.0 * M_PI - vecAngles(i)) / vecAreas(i);
				//max = e > max ? e : max;
				if (e > max)
				{
					max = e;
					idx = i;
				}
			}
		}
		//std::cout << idx << std::endl;
		return max;
	}
	else
	{
		double averange = 0;
		int cnt = 0;
		for (size_t i = 0; i < VType_.size(); ++i)
		{
			if (VType_(i) != -1)
			{
				averange += abs(2.0 * M_PI - vecAngles(i)) / vecAreas(i);
				++cnt;
			}
		}
		averange /= double(cnt);
		return averange;
	}
}

void hinge_energy_and_grad(MatrixTypeConst V, const Eigen::Matrix3Xi& F, const std::vector<std::vector<Eigen::Vector2i>>& adjF, double& energy, MatrixType& energyGrad)
{
	const int Vnum = V.cols();
	energy = 0.0;
	energyGrad.setConstant(3, Vnum, 0);

	VectorType vecAngles;
	VectorType vecAreas;
	MatrixType matAngles;
	MatrixType matNormals;
	Zombie::cal_angles_and_areas(V, F, vecAngles, vecAreas, matAngles);
	Zombie::cal_normal_per_face(V, F, matNormals);

	MatrixType matX(3, Vnum);
	for (size_t i = 0; i < adjF.size(); ++i)
	{
		const auto& adjf = adjF[i];

		Eigen::Matrix3d NNT(Eigen::Matrix3d::Zero());
		for (auto it : adjf)
		{
			const PosVector& n = matNormals.col(it(0));
			NNT += matAngles(it(1), it(0)) * (n * n.transpose());
		}

		Eigen::EigenSolver<Eigen::Matrix3d> solver(NNT);
		VectorType v(solver.eigenvalues().real());
		int idx(0);
		energy += v.minCoeff(&idx);
		matX.col(i) = solver.eigenvectors().col(idx).real();
		//std::cout << "Target index: " << idx << std::endl;
		//std::cout << "Eigen value vector: " << v << std::endl;
		//std::cout << "Eigenvectors: " << solver.eigenvectors() << std::endl;

		//const PosVector& xx(matX.col(i));
		//for (auto it : adjf)
		//{
		//	const auto& n = matNormals.col(it(0));
		//	double xTn = xx.dot(n);
		//	const double& theta = matAngles(it(1), it(0));

		//	//Get the vertices nomenclature right
		//	const int& j = F((it(1) + 1) % 3, it(0));
		//	const int& k = F((it(1) + 2) % 3, it(0));

		//	PosVector ejk = V.col(k) - V.col(j);
		//	PosVector eki = V.col(i) - V.col(k);
		//	PosVector eij = V.col(j) - V.col(i);

		//	double A = eij.cross(eki).norm() / 2.;
		//	eigen_assert(A && "degenerate triangle area!");

		//	Eigen::Matrix3d dNdi = ejk.cross(n) * n.transpose() / A;
		//	Eigen::Matrix3d dNdj = eki.cross(n) * n.transpose() / A;
		//	Eigen::Matrix3d dNdk = eij.cross(n) * n.transpose() / A;

		//	eij.normalize(); eki.normalize();
		//	PosVector dThetadi = n.cross(eij + eki).transpose();
		//	PosVector dThetadj = -n.cross(eij).transpose();
		//	PosVector dThetadk = -n.cross(eki).transpose();

		//	//Putting it all together
		//	energyGrad.col(i) += xTn * xTn * dThetadi + 2 * theta * xTn * dNdi.transpose() * xx;
		//	energyGrad.col(j) += xTn * xTn * dThetadj + 2 * theta * xTn * dNdj.transpose() * xx;
		//	energyGrad.col(k) += xTn * xTn * dThetadk + 2 * theta * xTn * dNdk.transpose() * xx;
		//}
	}

	MatrixType energyGrad1;
	energyGrad1.setConstant(3, Vnum, 0);
	MatrixType energyGrad2;
	energyGrad2.setConstant(3, Vnum, 0);

	//梯度部分
	for (int i = 0; i < matF_.cols(); ++i)
	{
		const auto& fv = matF_.col(i);
		const auto& n = matNormals.col(i);
		const auto& p0 = V.col(fv[0]);
		const auto& p1 = V.col(fv[1]);
		const auto& p2 = V.col(fv[2]);

		Eigen::Matrix3d e;
		e.col(0) = p2 - p1;
		e.col(1) = p0 - p2;
		e.col(2) = p1 - p0;

		const double area = e.col(1).cross(e.col(2)).norm();
		eigen_assert(area && "degenerate triangle area!");

		for (int j = 0; j < 3; ++j)
		{
			const PosVector& xx(matX.col(fv[j]));
			double xTn = xx.dot(n);

			const PosVector c1 = xTn * xTn * n.cross(-e.col((j + 2) % 3).normalized());
			const PosVector c2 = xTn * xTn * n.cross(-e.col((j + 1) % 3).normalized());
			energyGrad1.col(fv[j]) += -(c1 + c2);
			energyGrad1.col(fv[(j + 1) % 3]) += c1;
			energyGrad1.col(fv[(j + 2) % 3]) += c2;

			Eigen::Matrix3d gradNdi((e.col(j).cross(n) * n.transpose()) / area);
			Eigen::Matrix3d gradNdj((e.col((j + 1) % 3).cross(n) * n.transpose()) / area);
			Eigen::Matrix3d gradNdk((e.col((j + 2) % 3).cross(n) * n.transpose()) / area);
			const PosVector c3i = 2.0 * matAngles(j, i) * xTn * gradNdi.transpose() * xx;
			const PosVector c3j = 2.0 * matAngles(j, i) * xTn * gradNdj.transpose() * xx;
			const PosVector c3k = 2.0 * matAngles(j, i) * xTn * gradNdk.transpose() * xx;
			energyGrad1.col(fv[j]) += c3i;
			energyGrad1.col(fv[(j + 1) % 3]) += c3j;
			energyGrad1.col(fv[(j + 2) % 3]) += c3k;
		}
	}

	for (int i = 0; i < Vnum; ++i)
	{
		const auto& adjf = adjF[i];
		const PosVector& xx(matX.col(i));

		for (auto it : adjf)
		{
			const auto& n = matNormals.col(i);
			double xTn = xx.dot(n);
			const double& theta = matAngles(it(1), it(0));

			//Get the vertices nomenclature right
			const int& j = F((it(1) + 1) % 3, it(0));
			const int& k = F((it(1) + 2) % 3, it(0));

			PosVector ejk = V.col(k) - V.col(j);
			PosVector eki = V.col(i) - V.col(k);
			PosVector eij = V.col(j) - V.col(i);

			double A = eij.cross(eki).norm() / 2.;
			eigen_assert(A && "degenerate triangle area!");

			Eigen::Matrix3d dNdi = ejk.cross(n) * n.transpose() / A;
			Eigen::Matrix3d dNdj = eki.cross(n) * n.transpose() / A;
			Eigen::Matrix3d dNdk = eij.cross(n) * n.transpose() / A;

			eij.normalize(); eki.normalize();
			PosVector dThetadi = n.cross(eij + eki).transpose();
			PosVector dThetadj = -n.cross(eij).transpose();
			PosVector dThetadk = -n.cross(eki).transpose();

			//Putting it all together
			energyGrad2.col(i) += xTn * xTn * dThetadi + 2 * theta * xTn * dNdi.transpose() * xx;
			energyGrad2.col(j) += xTn * xTn * dThetadj + 2 * theta * xTn * dNdj.transpose() * xx;
			energyGrad2.col(k) += xTn * xTn * dThetadk + 2 * theta * xTn * dNdk.transpose() * xx;
		}
	}

	assert(energy == energy && "There are nans in the energy");
	assert(energyGrad == energyGrad && "There are nans in the energyGrad");

	//std::cout << energyGrad1 - energyGrad2 << std::endl;

	//std::cout << "Energy: " << energy << std::endl;
	//std::cout << "---------------------------" << std::endl;
	////std::cout << "Gradient: " << Gradient.norm() << std::endl;
	//std::cout << "Gradient: " << energyGrad << std::endl;
}

void hinge_energy(MatrixTypeConst V, const Eigen::Matrix3Xi& F, const std::vector<std::vector<Eigen::Vector2i>>& adjF, double& energy)
{
	energy = 0.0;
	VectorType vecAngles;
	VectorType vecAreas;
	MatrixType matAngles;
	MatrixType matNormals;
	Zombie::cal_angles_and_areas(V, matF_, vecAngles, vecAreas, matAngles);
	Zombie::cal_normal_per_face(V, matF_, matNormals);

	for (size_t i = 0; i < adjF_.size(); ++i)
	{
		std::vector<Eigen::Vector2i>& adjf = adjF_[i];

		Eigen::Matrix3d NNT(Eigen::Matrix3d::Zero());
		for (auto it : adjf)
		{
			const PosVector& n = matNormals.col(it(0));
			NNT += matAngles(it(1), it(0)) * (n * n.transpose());
		}

		Eigen::EigenSolver<Eigen::Matrix3d> solver(NNT);
		energy += solver.eigenvalues().real().minCoeff();
	}
}

void energy_test(MatrixType V, const Eigen::Matrix3Xi& F, const std::vector<std::vector<Eigen::Vector2i>>& adjF)
{
	const double step = 1e-6;
	double e;
	MatrixType G;
	hinge_energy_and_grad(V, F, adjF, e, G);

	MatrixType numDG;
	numDG.setConstant(3, V.cols(), 0);
	for (int i = 0; i < V.cols(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			double temp;
			V(j, i) += step;
			hinge_energy(V, F, adjF, temp);
			numDG(j, i) = (temp - e) / step;
			V(j, i) -= step;
		}
	}
	std::cout << numDG - G << std::endl;
}

int opt_solve(MatrixType V, const Eigen::Matrix3Xi& F, const std::vector<std::vector<Eigen::Vector2i>>& adjF, double& t, MatrixType& p, double& energy, MatrixType& energyGrad)
{
	//The matrices-are-actually-vectors dot product. Will be used a lot.
	const auto dot = [](MatrixType A, MatrixType B) { return (A.array() * B.array()).sum(); };

	int retVal = 0;
	bool firstTime = false;

	if (p.cols() < 1) {
		//The search direction is not initialized or was just reset. Initialize it with the gradient.
		hinge_energy_and_grad(V, F, adjF, energy, energyGrad);
		p = -energyGrad;
		firstTime = true;
		std::cout << "Search direction initialized!" << std::endl;
	}

	//Cache old values
	MatrixType oldV = V;
	MatrixType oldGrad = energyGrad;
	double oldEnergy = energy;

	//Try to cautiously increase t
	t = std::min(2. * t, MAX_T);

	double g0dotp = dot(p, oldGrad);
	double tmin = 0, tmax = INFTY;
	int tries = 0;
	for (; tries < MAX_LINESEARCH_TRIES; ++tries)
	{
		V = oldV + t * p;
		hinge_energy_and_grad(V, F, adjF, energy, energyGrad);
		double totalEnergy = energy;

		if (totalEnergy > oldEnergy + ARMIJO_C1 * t * g0dotp)
		{
			tmax = t;
		}
		else if (dot(p, energyGrad) < WOLFE_C2 * g0dotp)
		{
			tmin = t;
		}
		else
		{
			break;
		}

		if (tmax < INFTY)
		{
			t = 0.5 * (tmin + tmax);
		}
		else
		{
			t = 2 * tmin;
		}
	}

	if (tries == MAX_LINESEARCH_TRIES)
		retVal = -1;

	if (t > MAX_T)
	{
		t = MAX_T;
		V = oldV + t * p;
		hinge_energy_and_grad(V, F, adjF, energy, energyGrad);
	}
	if (t < MIN_T)
	{
		t = MIN_T;
		V = oldV + t * p;
		hinge_energy_and_grad(V, F, adjF, energy, energyGrad);
	}

	std::cout << "Line search result step = " << t << std::endl;

	//Cache vectors
	static std::deque<MatrixType> y;
	static std::deque<MatrixType> s;
	if (firstTime)
	{
		s.clear();
		y.clear();
	}
	y.push_back(energyGrad - oldGrad);
	s.push_back(V - oldV);
	if (s.size() > N_LBFGS_VECTORS)
	{
		y.pop_front();
		s.pop_front();
	}
	int nvecs = s.size();

	//Compute next step
	MatrixType q = energyGrad;
	VectorType alpha(nvecs), rho(nvecs);
	for (int i = nvecs - 1; i >= 0; --i)
	{
		rho(i) = 1. / dot(y[i], s[i]);
		if (rho(i) != rho(i) || !std::isfinite(rho(i)))
			rho(i) = 0.;
		alpha(i) = rho(i) * dot(s[i], q);
		q -= alpha(i) * y[i];
	}
	double gammak = dot(s[nvecs - 1], y[nvecs - 1]) / dot(y[nvecs - 1], y[nvecs - 1]);
	q *= gammak;
	for (int i = 0; i < nvecs; ++i)
	{
		double beta = rho(i) * dot(y[i], q);
		q += s[i] * (alpha(i) - beta);
	}
	p = -q;
	if (p != p || retVal == -1)
	{
		s.clear();
		y.clear();
		p = MatrixType();
		firstTime = true;
	}

	if (V != V)
	{
		retVal = -2;
		assert(false && "There are nans in the vertices return value");
	}

	return retVal;
}

