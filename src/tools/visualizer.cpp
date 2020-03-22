#include "visualizer.h"

Zombie::visualizer::visualizer()
{
}

int Zombie::visualizer::LoadFile(const std::string& fileName)
{
	polydata_ = vtkSmartPointer<vtkPolyData>::New();

	std::string extension = vtksys::SystemTools::GetFilenameLastExtension(fileName);
	if (extension == ".ply")
	{
		auto reader = vtkSmartPointer<vtkPLYReader>::New();
		reader->SetFileName(fileName.c_str());
		reader->Update();
		polydata_ = reader->GetOutput();
	}
	else if (extension == ".obj")
	{
		auto reader = vtkSmartPointer<vtkOBJReader>::New();
		reader->SetFileName(fileName.c_str());
		reader->Update();
		polydata_ = reader->GetOutput();
	}
	else if (extension == ".vtk")
	{
		auto reader = vtkSmartPointer<vtkPolyDataReader>::New();
		reader->SetFileName(fileName.c_str());
		reader->Update();
		polydata_ = reader->GetOutput();
	}
	else
	{
		std::cout << "Wrong extension!!!" << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

int Zombie::visualizer::LoadMesh(const surface_mesh::Surface_mesh& mesh)
{
	polydata_ = vtkSmartPointer<vtkPolyData>::New();

	// Input vertices and faces
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (auto vit : mesh.vertices())
	{
		points->InsertNextPoint(mesh.position(vit).data());
	}

	auto faces = vtkSmartPointer <vtkCellArray>::New();
	for (auto fit : mesh.faces())
	{
		auto triangle = vtkSmartPointer<vtkTriangle>::New();
		int idx = 0;
		for (auto fvit : mesh.vertices(fit))
		{
			triangle->GetPointIds()->SetId(idx++, fvit.idx());
		}
		faces->InsertNextCell(triangle);
	}

	//Assign the pieces to the vtkPolyData.
	polydata_->SetPoints(points);
	polydata_->SetPolys(faces);

	return EXIT_SUCCESS;
}

int Zombie::visualizer::LoadMatrix(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F)
{
	polydata_ = vtkSmartPointer<vtkPolyData>::New();

	// Input vertices and faces
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (auto i = 0; i < V.cols(); ++i)
	{
		points->InsertNextPoint(V.col(i).data());
	}

	auto faces = vtkSmartPointer <vtkCellArray>::New();
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
	polydata_->SetPoints(points);
	polydata_->SetPolys(faces);

	return EXIT_SUCCESS;
}

int Zombie::visualizer::InputColor(const Eigen::VectorXd& Scalar)
{
	if (Scalar.size() != polydata_->GetNumberOfPoints())
	{
		std::cout << "Invalid color data!!" << std::endl;
		return EXIT_FAILURE;
	}
	scalar_ = vtkSmartPointer<vtkDoubleArray>::New();
	scalar_->SetNumberOfComponents(1);
	scalar_->SetNumberOfTuples(Scalar.size());
	for (auto i = 0; i < Scalar.size(); ++i)
	{
		scalar_->InsertTuple1(i, Scalar(i));
	}

	//Build look up table using color transfer funciton
	ctf_ = vtkSmartPointer<vtkColorTransferFunction>::New();
	ctf_->SetColorSpaceToHSV();
	ctf_->AddRGBPoint(0.0, 0.1, 0.3, 1);
	ctf_->AddRGBPoint(0.25, 0.55, 0.65, 1);
	ctf_->AddRGBPoint(0.5, 1, 1, 1);
	ctf_->AddRGBPoint(0.75, 1, 0.65, 0.55);
	ctf_->AddRGBPoint(1.0, 1, 0.3, 0.1);

	lut_ = vtkSmartPointer<vtkLookupTable>::New();
	lut_->SetNumberOfColors(256);
	for (auto i = 0; i < lut_->GetNumberOfColors(); ++i)
	{
		Eigen::Vector4d color;
		ctf_->GetColor(double(i) / lut_->GetNumberOfColors(), color.data());
		color(3) = 1.0;
		lut_->SetTableValue(i, color.data());
	}
	lut_->Build();

	color_flag_ = true;
	return EXIT_SUCCESS;
}

int Zombie::visualizer::Initialize()
{
	//valid check
	if (polydata_->GetNumberOfCells() == EXIT_FAILURE)
	{
		std::cout << "NO DATA!!!" << std::endl;
		return EXIT_FAILURE;
	}

	//initialize rendering related variaties
	window_ = vtkSmartPointer<vtkRenderWindow>::New();
	renderer_ = vtkSmartPointer<vtkRenderer>::New();
	window_->AddRenderer(renderer_);
	interactor_ = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor_->SetRenderWindow(window_);

	//rendering properties setup
	renderer_->SetBackground(0, 0, 0);
	auto style = vtkInteractorStyleTrackballCamera::New();
	interactor_->SetInteractorStyle(style);

	//fisrt rendering
	auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polydata_);

	//Color related setup
	if (color_flag_)
	{
		polydata_->GetPointData()->SetScalars(scalar_);
		mapper->SetLookupTable(lut_);
		mapper->SetScalarRange(scalar_->GetValueRange());
	}

	auto actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	renderer_->AddActor(actor);

	return EXIT_SUCCESS;
}

int Zombie::visualizer::Run()
{
	if (Initialize() == EXIT_FAILURE)
	{
		std::cout << "Initialize failed!!" << std::endl;
		return EXIT_FAILURE;
	}
	window_->Render();
	interactor_->Start();

	return EXIT_SUCCESS;
}
