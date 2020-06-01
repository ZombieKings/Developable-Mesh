#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

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

#include "../tools/cal_angles_areas.h"
#include "../tools/cal_laplacian.h"
#include "../tools/cal_normals.h"
#include "../tools/AABBSearcher.h"
#include "../tools/cal_edge_length.h"

typedef double DataType;
typedef Eigen::Triplet<DataType> Tri;
typedef Eigen::Vector3d PosVector;
typedef Eigen::VectorXd VectorType;
typedef Eigen::Matrix3Xd MatrixType;
typedef Eigen::SparseMatrix<DataType> SparseMatrixType;
typedef const Eigen::Matrix3Xd MatrixTypeConst;
typedef Eigen::SimplicialLDLT<SparseMatrixType> SolverType;
typedef Surface_Mesh::AABBSearcher<MatrixType, Eigen::Matrix3Xi> ABTreeType;

inline bool scoeff(std::vector<Tri>& target_matrix, const PosVector& input_vector, size_t row_, size_t col_)
{
	for (size_t i = 0; i < 3; ++i)
		if (input_vector[i])
			target_matrix.push_back(Eigen::Triplet<double>(row_, col_ + i, input_vector[i]));
	return true;
}
inline bool dcoeff(std::vector<Tri>& target_matrix, size_t row_, size_t col_, DataType val)
{
	for (size_t i = 0; i < 3; ++i)
		target_matrix.push_back(Eigen::Triplet<double>(row_ + i, col_ + i, val));
	return true;
}
inline bool srhs(VectorType& b, const PosVector& input_vector, size_t idx)
{
	for (size_t i = 0; i < 3; ++i)
		b(idx + i) += input_vector[i];
	return true;
}

void getMeshInfo(const surface_mesh::Surface_mesh& mesh,
	MatrixType& V,
	Eigen::Matrix2Xi& E,
	Eigen::Matrix3Xi& F,
	Eigen::Matrix3Xi& FE,
	Eigen::VectorXi& Vtype, 
	VectorType& eLength);

void compute_length(MatrixTypeConst& V,
	const Eigen::Matrix2Xi& E,
	const Eigen::Matrix3Xi& F,
	MatrixTypeConst& matAngles,
	const VectorType& vecAngles,
	const VectorType& vecAreas,
	const Eigen::VectorXi& Vtype,
	VectorType& tl);

void update_corr_vertices(MatrixTypeConst& V,
	const Eigen::Matrix3Xi& F,
	MatrixTypeConst& oriN,
	ABTreeType& ABTree,
	MatrixType& corrV,
	MatrixType& corrVnomal);

void update_vertices(MatrixType& V,
	const Eigen::Matrix2Xi& E,
	const Eigen::Matrix3Xi& F,
	MatrixTypeConst& oriV,
	const Eigen::VectorXi& interVidx,
	const VectorType& tl,
	MatrixTypeConst& corrV,
	MatrixTypeConst& corrNormals);

double cal_error(const VectorType& vAngles, const VectorType& areas, const Eigen::VectorXi& Vtype, int flag);

void TimeCallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData));
void matrix2vtk(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, vtkPolyData* P);
void MakeLUT(vtkDoubleArray* Scalar, vtkLookupTable* LUT);
void visualize_mesh(vtkRenderer* Renderer, MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const VectorType& angles, const Eigen::VectorXi& interVidx);

