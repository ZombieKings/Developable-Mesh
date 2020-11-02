#ifndef ZOMBIE_VISUALIZER_H
#define ZOMBIE_VISUALIZER_H

#include <Eigen/Core>

#include <vtkMath.h>
#include <vtkCamera.h>
#include <vtkSmartPointer.h>
#include <vtkNamedColors.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRendererCollection.h>
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

#include <vtkCellPicker.h>
#include <vtkDataSetMapper.h>
#include <vtkIdTypeArray.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkExtractSelection.h>

namespace Zombie
{
	// Construct a polydata object with input vertices and faces datas 
	//
	// Inputs:
	//   @V  3 by #V list of vertices position
	//
	//   @F  3 by #F list of mesh faces (must be triangles)
	//
	// Outputs:
	//   @P  vtkPolydata object
	template<typename DerivedV, typename DerivedF>
	void matrix2vtk(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		vtkPolyData* P);

	// Build a look up table for vertices colors with input scalar
	// Input/Output:
	//   @LUT  256 list of color, a vtkLookupTable object
	void MakeLUT(vtkLookupTable* LUT);

	void MakeArrow(
		vtkPolyData* P,
		bool forCell,
		vtkActor* glyphNormalActor);

	void add_scalarbar(
		vtkRenderer* Renderer,
		vtkLookupTable* LUT,
		std::string str = " ",
		int numLabels = 4
	);

	// Visualize a surface mesh with input information
	//
	// Inputs:
	//	Renderer the vtkRenderer where the mesh will displays.
	//
	//  @V  3 by #V list of vertices position
	//
	//  @F  3 by #F list of mesh faces (must be triangles)
	//
	//	@Scalar #V a list of scalars to visualized; 
	//
	//	@Vtype #V list of vertex type, -1 for boundary vertices, 1 for internal vertices, 2 for special vertices.
	template<typename DerivedV, typename DerivedF, typename DerivedScalar, typename DerivedData, typename DerivedType>
	void visualize_mesh(
		vtkRenderer* Renderer,
		const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		const Eigen::PlainObjectBase<DerivedScalar>& Scalar,
		const Eigen::MatrixBase<DerivedData>& Data,
		const Eigen::PlainObjectBase<DerivedType>& Vtype,
		const double range_min = 0.,
		const double range_max = 0.);

	template<typename DerivedV, typename DerivedF, typename DerivedScalar, typename DerivedType>
	void visualize_mesh(
		vtkRenderer* Renderer,
		const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		const Eigen::PlainObjectBase<DerivedScalar>& Scalar,
		const Eigen::PlainObjectBase<DerivedType>& Vtype,
		const double range_min = 0.,
		const double range_max = 0.);

	template<typename DerivedV, typename DerivedF, typename DerivedType>
	void visualize_mesh(
		vtkRenderer* Renderer,
		const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		const Eigen::PlainObjectBase<DerivedType>& Vtype,
		const double range_min = 0.,
		const double range_max = 0.);

	// Visualize a set of vertices
	//
	// Inputs:
	//	Renderer the vtkRenderer where the mesh will displays.
	//
	//  @V  3 by #V list of vertices position
	//
	//  @size size of the points
	//
	//  @R @G @B colors of points, stand for red, green and blue, respectively.
	template<typename DerivedV>
	void visualize_vertices(
		vtkRenderer* Renderer,
		const Eigen::MatrixBase<DerivedV>& V,
		const double size = 4.0,
		const double R = 1.0,
		const double G = 0.0,
		const double B = 0.0);

	void add_string(
		vtkRenderer* Renderer,
		const std::string& str,
		const double size = 33,
		const double R = 1.0,
		const double G = 1.0,
		const double B = 1.0);
};
#include "visualizer.cpp"

#endif 