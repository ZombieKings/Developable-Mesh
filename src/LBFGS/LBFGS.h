#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include "optimization.h"

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

using namespace alglib;

typedef double DataType;
typedef Eigen::Triplet<DataType> Tri;
typedef Eigen::Vector3d PosVector;
typedef Eigen::VectorXd VectorType;
typedef Eigen::Matrix3Xd MatrixType;
typedef Eigen::SparseMatrix<DataType> SparseMatrixType;
typedef const Eigen::Matrix3Xd MatrixTypeConst;

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

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& vertices_mat, Eigen::Matrix3Xi& faces_mat);

void cal_angles(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& matAngles, VectorType& vecAngles);
void cal_angles_and_areas_and_normal(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& matAngles, VectorType& vecAngles, VectorType& areas, MatrixType& matNormal);
void cal_gaussian_gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles, const VectorType& vAngles, SparseMatrixType& mGradient);
void cal_laplace(const Eigen::Matrix3Xi& F, MatrixTypeConst& angles, const VectorType& areas, const Eigen::VectorXi& interIdx, SparseMatrixType& L);
void cal_topo(const Eigen::Matrix3Xi& F, int Vnum, const Eigen::VectorXi& interVidx, SparseMatrixType& F2V);
void cal_grad(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const SparseMatrixType& F2V, SparseMatrixType& G);

double cal_error(const VectorType& vAngles, const std::vector<int>& interIdx);

void matrix2vtk(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, vtkPolyData* P);
void MakeLUT(vtkDoubleArray* Scalar, vtkLookupTable* LUT);
void visualize_mesh(vtkRenderer* Renderer, MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const VectorType& angles);
void visualize_mesh(vtkRenderer* Renderer, MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const VectorType& angles, const Eigen::VectorXi& interVidx);

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData));
void grad_function(const real_1d_array& x, double& func, real_1d_array& grad, void* ptr);

void ori_function(const real_1d_array& x, real_1d_array& grad, void* ptr);
bool grad_function_test(const real_1d_array& x,
	void (*grad)(const real_1d_array& x, double& func, real_1d_array& grad, void* ptr),
	void (*func)(const real_1d_array& x, real_1d_array& func, void* ptr));