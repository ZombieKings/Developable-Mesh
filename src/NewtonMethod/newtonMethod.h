#define _USE_MATH_DEFINES
#include <math.h>

#include <surface_mesh/Surface_mesh.h>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseQR"

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

#include <vtkFloatArray.h>
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

#include <vtkLookupTable.h>
#include <vtkColorTransferFunction.h>

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& vertices_mat, Eigen::Matrix3Xi& faces_mat);

void calAngles_Neigh(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xf& A, Eigen::VectorXi& degrees);
void BuildCoeffMatrix(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& angles, const Eigen::VectorXi& degrees, Eigen::SparseMatrix<float>& A);
void BuildrhsB(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const std::vector<int>& bv, Eigen::VectorXf& b);

void MakeLUT(vtkFloatArray* Scalar, vtkLookupTable* LUT);
void MakeNormalGlyphs(vtkPolyData* src, vtkGlyph3D* glyph);
void visualize_vertices(vtkRenderer* Renderer, const Eigen::Matrix3Xf& V);
void visualize_mesh(vtkRenderer* Renderer, const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::VectorXf& angles);
void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData));

//将得到的系数向量导入系数矩阵中
bool dcoeff(std::vector<Eigen::Triplet<float>>& target_matrix, const Eigen::Vector3f& input_vector, size_t row_, size_t col_)
{
	for (size_t i = 0; i < 3; ++i)
		if (input_vector[i])
			target_matrix.push_back(Eigen::Triplet <float>(row_, col_ + i, input_vector[i]));
	return true;
}

bool scoeff(std::vector<Eigen::Triplet<float>>& target_matrix, size_t row_, size_t col_, float val)
{
	for (size_t i = 0; i < 3; ++i)
		target_matrix.push_back(Eigen::Triplet <float>(row_ + i, col_ + i, val));
	return true;
}

bool srhs(Eigen::VectorXf& b, const Eigen::Vector3f& input_vector, size_t idx)
{
	for (size_t i = 0; i < 3; ++i)
		b(idx + i) = input_vector[i];
	return true;
}