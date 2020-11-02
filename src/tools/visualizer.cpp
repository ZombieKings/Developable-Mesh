#include "visualizer.h"

template<typename DerivedV, typename DerivedF>
void Zombie::matrix2vtk(const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	vtkPolyData* P)
{
	const int DIM = F.rows();
	assert(DIM == 3 && "Only for triangle mesh!");

	const int Fnum = F.cols();
	const int Vnum = V.cols();
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < Vnum; ++i)
		points->InsertNextPoint(V.col(i).data());

	auto faces = vtkSmartPointer <vtkCellArray>::New();
	for (int i = 0; i < Fnum; ++i)
	{
		auto triangle = vtkSmartPointer<vtkTriangle>::New();
		for (int j = 0; j < DIM; ++j)
			triangle->GetPointIds()->SetId(j, F(j, i));
		faces->InsertNextCell(triangle);
	}
	P->SetPoints(points);
	P->SetPolys(faces);
}

void Zombie::MakeLUT(vtkLookupTable* LUT)
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
		double color[4];
		ctf->GetColor((double)i / LUT->GetNumberOfColors(), color);
		color[3] = 1.0;
		LUT->SetTableValue(i, color);
	}
	LUT->Build();
}

void Zombie::add_scalarbar(
	vtkRenderer* Renderer,
	vtkLookupTable* LUT,
	std::string str /*= " "*/,
	int numLabels /*= 4 */)
{
	auto scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
	scalarBar->SetLookupTable(LUT);
	scalarBar->SetTitle(str.c_str());
	scalarBar->SetNumberOfLabels(numLabels);
	Renderer->AddActor2D(scalarBar);
}

void Zombie::MakeArrow(
	vtkPolyData* P,
	bool forCell,
	vtkActor* glyphNormalActor)
{
	auto arrow = vtkSmartPointer<vtkArrowSource>::New();
	arrow->SetTipResolution(16);
	arrow->SetTipLength(.3);//参数
	arrow->SetTipRadius(.05);//参数
	arrow->SetShaftRadius(0.025);//参数
	arrow->Update();

	auto glyph = vtkSmartPointer<vtkGlyph3D>::New();
	glyph->SetSourceConnection(arrow->GetOutputPort());
	if (forCell)
	{
		auto CellCenters = vtkSmartPointer<vtkCellCenters>::New();
		CellCenters->SetInputData(P);
		CellCenters->Update();
		glyph->SetInputConnection(CellCenters->GetOutputPort());
	}
	else
	{
		glyph->SetInputData(P);
	}

	glyph->SetVectorModeToUseNormal();
	glyph->SetColorModeToColorByVector();
	glyph->SetScaleModeToScaleByVector();
	glyph->SetScaleFactor(4.);//大小参数
	glyph->OrientOn();
	glyph->Update();

	auto glyphNormalMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	glyphNormalMapper->SetInputConnection(glyph->GetOutputPort());

	glyphNormalActor->SetMapper(glyphNormalMapper);
	glyphNormalActor->GetProperty()->SetDiffuseColor(1.0, 0.0, 0.0);
}

template<typename DerivedV, typename DerivedF, typename DerivedType>
void Zombie::visualize_mesh(vtkRenderer* Renderer,
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::PlainObjectBase<DerivedType>& Vtype,
	const double range_min,
	const double range_max)
{
	Eigen::Matrix3d _;
	visualize_mesh(Renderer, V, F, _, Vtype, range_min, range_max);
}

template<typename DerivedV, typename DerivedF, typename DerivedScalar, typename DerivedType>
void Zombie::visualize_mesh(vtkRenderer* Renderer,
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::PlainObjectBase<DerivedScalar>& Scalar,
	const Eigen::PlainObjectBase<DerivedType>& Vtype,
	const double range_min,
	const double range_max)
{
	Eigen::Matrix3d _;
	visualize_mesh(Renderer, V, F, Scalar, _, Vtype, range_min, range_max);
}

template<typename DerivedV, typename DerivedF, typename DerivedScalar, typename DerivedData, typename DerivedType>
void Zombie::visualize_mesh(
	vtkRenderer* Renderer,
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	const Eigen::PlainObjectBase<DerivedScalar>& Scalar,
	const Eigen::MatrixBase<DerivedData>& Data,
	const Eigen::PlainObjectBase<DerivedType>& Vtype,
	const double range_min,
	const double range_max)
{
	const int Fnum = F.cols();
	const int Vnum = V.cols();
	//生成网格
	auto P = vtkSmartPointer<vtkPolyData>::New();
	matrix2vtk(V, F, P);

	auto polyMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	polyMapper->SetInputData(P);

	const int Snum = Scalar.rows();
	if (Snum == Vnum)
	{
		auto scalar = vtkSmartPointer<vtkDoubleArray>::New();
		scalar->SetNumberOfComponents(1);
		scalar->SetNumberOfTuples(Vnum);
		for (auto i = 0; i < Snum; ++i)
		{
			if (Vtype(i) != -1)
				scalar->InsertTuple1(i, Scalar(i));
			else
				scalar->InsertTuple1(i, 0);
		}
		P->GetPointData()->SetScalars(scalar);

		auto lut = vtkSmartPointer<vtkLookupTable>::New();
		MakeLUT(lut);

		polyMapper->SetLookupTable(lut);
		if (range_min == 0. && range_max == 0.)
			polyMapper->SetScalarRange(scalar->GetValueRange()[0], scalar->GetValueRange()[1]);
		else
			polyMapper->SetScalarRange(range_min, range_max);

		add_scalarbar(Renderer, lut, "Curvature Error", 4);
	}

	const int Dnum = Data.cols();
	if (Dnum == Vnum || Dnum == Fnum)
	{
		auto normaldata = vtkSmartPointer<vtkDoubleArray>::New();
		normaldata->SetNumberOfComponents(3);
		normaldata->SetNumberOfTuples(Dnum);
		for (auto i = 0; i < Dnum; ++i)
			normaldata->InsertTuple(i, Data.col(i).data());

		auto glyphNormalActor = vtkSmartPointer<vtkActor>::New();
		if (Dnum == Fnum)
		{
			P->GetCellData()->SetNormals(normaldata);
			MakeArrow(P, true, glyphNormalActor);
		}
		else
		{
			P->GetPointData()->SetNormals(normaldata);
			MakeArrow(P, false, glyphNormalActor);
		}
		Renderer->AddActor(glyphNormalActor);
	}

	auto polyActor = vtkSmartPointer<vtkActor>::New();
	polyActor->SetMapper(polyMapper);
	polyActor->GetProperty()->SetDiffuseColor(1, 1, 1);
	Renderer->AddActor(polyActor);
}


void Zombie::add_string(
	vtkRenderer* Renderer,
	const std::string& str,
	const double size /*= 33*/,
	const double R /*= 1.0*/,
	const double G /*= 1.0*/,
	const double B /*= 1.0*/)
{
	auto textActor = vtkSmartPointer<vtkTextActor>::New();
	textActor->SetInput(str.c_str());
	textActor->GetTextProperty()->SetFontSize(size);
	textActor->GetTextProperty()->SetColor(R, G, B);
	Renderer->AddActor2D(textActor);
}

template<typename DerivedV>
void Zombie::visualize_vertices(vtkRenderer* Renderer, 
	const Eigen::MatrixBase<DerivedV>& V,
	const double size,
	const double R,
	const double G,
	const double B)
{
	auto points = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < V.cols(); ++i)
	{
		points->InsertNextPoint(V.col(i).data());
	}
	auto polydata = vtkSmartPointer<vtkPolyData>::New();
	polydata->SetPoints(points);
	auto pointsFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	pointsFilter->SetInputData(polydata);
	pointsFilter->Update();

	auto pointsMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	pointsMapper->SetInputConnection(pointsFilter->GetOutputPort());
	auto pointsActor = vtkSmartPointer<vtkActor>::New();
	pointsActor->SetMapper(pointsMapper);
	pointsActor->GetProperty()->SetDiffuseColor(R, G, B);
	pointsActor->GetProperty()->SetPointSize(size);
	Renderer->AddActor(pointsActor);
}