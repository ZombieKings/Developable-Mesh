#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <windows.h>

#include "optimization.h"
#include "func_opt.h"
#include "newton_solver.h"

#include <surface_mesh/Surface_mesh.h>

#include <vtkMath.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkNamedColors.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCallbackCommand.h>

#include <vtkDoubleArray.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkCellCenters.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkLine.h>
#include <vtkTriangle.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkPolyDataMapper.h>

#include <vtkAxes.h>
#include <vtkAxesActor.h>
#include <vtkGlyph3D.h>
#include <vtkGlyph3DMapper.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkArrowSource.h>
#include <vtkSphereSource.h>
#include <vtkScalarBarActor.h>

#include <vtkLookupTable.h>
#include <vtkColorTransferFunction.h>

#include <vtkTextActor.h>
#include <vtkTextProperty.h>

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xd& vertices_mat, Eigen::Matrix3Xi& faces_mat);
void cal_angles(const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F, Eigen::VectorXd& vecAngles);

void matrix2vtk(const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F, vtkPolyData* P);
void MakeLUT(vtkDoubleArray* Scalar, vtkLookupTable* LUT);
void visualize_mesh(vtkRenderer* Renderer, const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXd& angles, const Eigen::VectorXi& interVidx);

int main(int argc, char** argv)
{
	surface_mesh::Surface_mesh mesh;
	if (!mesh.read(argv[1]))
	{
		std::cout << "Laod failed!" << std::endl;
	}
	std::cout << mesh.n_vertices() << std::endl;
	std::cout << mesh.n_faces() << std::endl;
	for (const auto& vit : mesh.vertices())
	{
		if (!mesh.is_boundary(vit))
		{
			mesh.compute_vertex_normal(vit);
		}
	}



	//Eigen::VectorXi interVidx;
	//interVidx.resize(mesh.n_vertices());
	//memset(interVidx.data(), -1, sizeof(int) * interVidx.size());
	//int count = 0;
	//for (const auto& vit : mesh.vertices())
	//{
	//	if (!mesh.is_boundary(vit))
	//	{
	//		interVidx(vit.idx()) = count++;
	//	}
	//}
	//Eigen::Matrix3Xd matV;
	//Eigen::Matrix3Xi matF;
	//mesh2matrix(mesh, matV, matF);
	//Eigen::Matrix3Xd oriV(matV);
	//Eigen::VectorXd oriA;
	//cal_angles(oriV, matF, oriA);

	//func_opt::my_function f(mesh, 0.01, 10.0, 1.0, 1.0);
	//opt_solver::newton_solver solver;
	//solver.set_f(f);
	////solver.solve_sqp(matV.data());
	//solver.solve(matV.data());

	//Eigen::VectorXd resA;
	//cal_angles(matV, matF, resA);

	////---------------可视化---------------
	////创建窗口
	//auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	//renderWindow->SetSize(1600, 800);
	//auto renderer1 = vtkSmartPointer<vtkRenderer>::New();
	////visualize_mesh(renderer1, curV, matF_, reA);
	//visualize_mesh(renderer1, matV, matF, resA, interVidx);
	//renderer1->SetViewport(0.0, 0.0, 0.5, 1.0);

	//// Setup the text and add it to the renderer
	//auto textActor1 = vtkSmartPointer<vtkTextActor>::New();
	//textActor1->SetInput("Result Mesh");
	//textActor1->GetTextProperty()->SetFontSize(33);
	//textActor1->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	//renderer1->AddActor2D(textActor1);

	////视角设置
	//renderer1->ResetCamera();
	//renderWindow->AddRenderer(renderer1);

	//auto renderer2 = vtkSmartPointer<vtkRenderer>::New();
	////visualize_mesh(renderer2, matV_, matF_, oriA);
	//visualize_mesh(renderer2, oriV, matF, oriA, interVidx);
	//renderer2->SetViewport(0.5, 0.0, 1.0, 1.0);

	//// Setup the text and add it to the renderer
	//auto textActor2 = vtkSmartPointer<vtkTextActor>::New();
	//textActor2->SetInput("Original Mesh");
	//textActor2->GetTextProperty()->SetFontSize(33);
	//textActor2->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	//renderer2->AddActor2D(textActor2);

	////视角设置
	//renderer2->ResetCamera();
	//renderWindow->AddRenderer(renderer2);

	//auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	//interactor->SetRenderWindow(renderWindow);
	//auto style = vtkInteractorStyleTrackballCamera::New();
	//interactor->SetInteractorStyle(style);
	//interactor->Initialize();

	////开始
	//renderWindow->Render();
	//interactor->Start();

	//Eigen::SparseMatrix<double> M;
	//std::vector<Eigen::Triplet<double>> t;
	//t.push_back(Eigen::Triplet<double>(0, 0, 1));
	//t.push_back(Eigen::Triplet<double>(1, 1, 0));
	//t.push_back(Eigen::Triplet<double>(2, 2, 1));
	//M.resize(3, 3);
	//M.setFromTriplets(t.begin(), t.end());
	//for (int k = 0, i = 0; k < M.outerSize(); ++k)
	//{
	//	for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it)
	//	{
	//		std::cout << it.row() << "," << it.col() << std::endl;
	//	}
	//}
	//std::cout << M << std::endl;
	//std::cout << M.nonZeros() << std::endl;

	//Eigen::SparseMatrix<double> M;
	//int n = 3500;
	//double wp = 0.01;
	//std::vector<Eigen::Triplet<double>> t;
	//for (int i = 0; i < n; ++i)
	//{
	//	t.push_back(Eigen::Triplet<double>(i, i, i));
	//}
	//M.resize(n, n);
	//M.setFromTriplets(t.begin(), t.end());

	//clock_t stime1 = clock();
	//Eigen::MatrixXd I;
	//I.setIdentity(n, n);
	//Eigen::SparseMatrix<double> A(M + 2 * wp * I);
	//clock_t etime1 = clock();
	//std::cout << (etime1 - stime1) << std::endl;

	//clock_t stime = clock();
	//Eigen::SparseMatrix<double> B(M);
	//for (int i = 0; i < n; ++i)
	//{
	//	B.coeffRef(i, i) += 2 * wp;
	//}
	//B.makeCompressed();
	//clock_t etime = clock();
	//std::cout << (etime - stime) << std::endl;

	//Eigen::VectorXd d;
	//d.setConstant(n, 2);
	//clock_t stime = clock();
	//Eigen::SparseMatrix<double> GdA(d.cwiseInverse().asDiagonal() * M);
	//clock_t etime = clock();
	//std::cout << (etime - stime) << std::endl;

	//Eigen::VectorXd d1;
	//d1.setConstant(n, 2);
	//clock_t stime1 = clock();
	//Eigen::SparseMatrix<double> B(M);
	//for (int i = 0; i < n; ++i)
	//{
	//	B.coeffRef(i, i) *= d1(i);
	//}
	//B.makeCompressed();
	//clock_t etime1 = clock();
	//std::cout << (etime1 - stime1) << std::endl;

	//std::cout << M << std::endl;
	//Eigen::Vector3d d(1,1,1);
	//Eigen::Vector3d c(1,2,3);
	//std::cout << d.array() - 2.0 << std::endl;
	//std::cout << d<< std::endl;
	//std::cout << d.cwiseProduct(c).squaredNorm() << std::endl;
	//int n = 3;
	//Eigen::MatrixXd x;
	//x.setIdentity(3 * n, 3 * n);
	//Eigen::VectorXd v;
	//v.setConstant(81, 1.4);
	//x = Eigen::Map<Eigen::MatrixXd>(v.data(), x.rows(), x.cols());
	//std::cout << x << std::endl;
	//Eigen::VectorXd v;
	//Eigen::VectorXd w;
	//v.setConstant(9, 3);
	//w.setConstant(9, 0.5);
	//v.cwiseProduct(w);
	//std::cout << v.cwiseProduct(w) + v << std::endl;

return 1;
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xd& vertices_mat, Eigen::Matrix3Xi& faces_mat)
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
				const Eigen::Vector3f& temp = Eigen::Map<const Eigen::Vector3f>(mesh.position(surface_mesh::Surface_mesh::Vertex(fvit.idx())).data());
				vertices_mat.col(fvit.idx()) = temp.cast<double>();
				flag(fvit.idx()) = 1;
			}
		}
	}
}

void cal_angles(const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F, Eigen::VectorXd& vecAngles)
{
	vecAngles.resize(V.cols());
	vecAngles.setZero();
	for (int f = 0; f < F.cols(); ++f)
	{
		const Eigen::Vector3i& fv = F.col(f);
		for (size_t vi = 0; vi < 3; ++vi)
		{
			const Eigen::Vector3d& p0 = V.col(fv[vi]);
			const Eigen::Vector3d& p1 = V.col(fv[(vi + 1) % 3]);
			const Eigen::Vector3d& p2 = V.col(fv[(vi + 2) % 3]);
			const double angle = std::acos(std::max(-1.0, std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
			vecAngles(F(vi, f)) += angle;
		}
	}
}

void matrix2vtk(const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F, vtkPolyData* P)
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

void visualize_mesh(vtkRenderer* Renderer, const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXd& angles, const Eigen::VectorXi& interVidx)
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

	auto polyActor = vtkSmartPointer<vtkActor>::New();
	polyActor->SetMapper(polyMapper);
	polyActor->GetProperty()->SetDiffuseColor(1, 1, 1);
	Renderer->AddActor(polyActor);
	Renderer->AddActor2D(scalarBar);
}