#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCallbackCommand.h>

#include <vtkSmartPointer.h>
#include <vtkProperty.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkTriangle.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>

#include <vtksys/SystemTools.hxx>

#include <vtkLookupTable.h>
#include <vtkProgrammableFilter.h>
#include <vtkColorTransferFunction.h>


//=============================================================================
using namespace surface_mesh;
//=============================================================================

int LoadMatrix(const Eigen::Matrix3Xf &V, const Eigen::Matrix3Xi &F, vtkSmartPointer<vtkPolyData>& polydata);
void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& vertices_mat, Eigen::Matrix3Xi& faces_mat);

void cal_angles(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xf& A);
double cal_error(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& A, int flag);

void cal_laplace(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& A, Eigen::SparseMatrix<float>& L, Eigen::MatrixX3f& b);
void cal_least_norm(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& A, Eigen::SparseMatrix<float>& N, Eigen::VectorXf& b);

void matrix2vtk(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, vtkPolyData* P);
void MakeLUT(vtkFloatArray* Scalar, vtkLookupTable* LUT);
void visualize_mesh(vtkRenderer* Renderer, const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::VectorXf& angles);
