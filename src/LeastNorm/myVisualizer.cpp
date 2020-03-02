#include "myVisualizer.h"

myVisualizer::myVisualizer()
{
}

int myVisualizer::LoadFile(const std::string& fileName)
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

int myVisualizer::LoadMesh(const surface_mesh::Surface_mesh& mesh)
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

int myVisualizer::LoadMatrix(const Eigen::Matrix3Xf &V, const Eigen::Matrix3Xi &F)
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

int myVisualizer::InputColor(const Eigen::VectorXd &Color)
{
	if (Color.size() != polydata_->GetNumberOfPoints())
	{
		std::cout << "Invalid color data!!" << std::endl;
		return EXIT_FAILURE;
	}
	scalar_ = vtkSmartPointer<vtkDoubleArray>::New();
	scalar_->SetNumberOfComponents(1);
	scalar_->SetNumberOfTuples(Color.size());
	for (auto i = 0; i < Color.size(); ++i)
	{
		scalar_->InsertTuple1(i, Color(i));
	}

	//Build look up table using color transfer funciton
	ctf_ = vtkSmartPointer<vtkColorTransferFunction>::New();
	ctf_->SetColorSpaceToDiverging();
	ctf_->AddRGBPoint(0.0, 0, 0, 1);
	ctf_->AddRGBPoint(1.0, 1, 0, 0);

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

int myVisualizer::Initialize()
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

int myVisualizer::Run()
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
