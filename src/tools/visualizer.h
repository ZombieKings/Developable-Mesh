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
	void TimeCallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData));
	void KeypressCallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* vtkNotUsed(clientData), void* vtkNotUsed(callData));

	// Construct a polydata object with input vertices and faces datas 
	//
	// Inputs:
	//   V  3 by #V list of vertices position
	//
	//   F  3 by #F list of mesh faces (must be triangles)
	//
	// Outputs:
	//   P  vtkPolydata object
	template<typename DerivedV, typename DerivedF>
	void matrix2vtk(const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		vtkPolyData* P);

	// Build a look up table for vertices colors with input scalar
	// Inputs:
	//   Scalar  #V list of vertices position
	//
	// Outputs:
	//   LUT  256 list of color, a vtkLookupTable object
	void MakeLUT(vtkDoubleArray* Scalar, vtkLookupTable* LUT);

	// Visualize a surface mesh with input information
	//
	// Inputs:
	//	Renderer the vtkRenderer where the mesh will displays.
	//
	//  V  3 by #V list of vertices position
	//
	//  F  3 by #F list of mesh faces (must be triangles)
	//
	//	vecAngles #V list of vertices internal angles for visualize errors.
	//
	//	Vtype #V list of vertex type, -1 for boundary vertices, 1 for internal vertices, 2 for special vertices.
	template<typename DerivedV, typename DerivedF, typename DerivedA, typename DerivedIDX>
	void visualize_mesh(vtkRenderer* Renderer,
		const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		const Eigen::PlainObjectBase<DerivedA>& vecAngles,
		const Eigen::PlainObjectBase<DerivedIDX>& Vtype);
};
#include "visualizer.cpp"

#endif 