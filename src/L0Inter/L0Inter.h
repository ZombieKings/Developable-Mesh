#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

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

#define KAP 2
#define MAXBETA 1e5

float w1 = 10.0f;
float w2 = 1.0f;
float w3 = 1.0f;

float lambda = 0.0000015;
float beta = 0.000003;

std::vector<int> interV;
std::vector<int> boundV;
Eigen::VectorXi interVidx;
Eigen::Matrix3Xi matF;

Eigen::Matrix3Xf matV;
Eigen::VectorXf vAngles;
Eigen::Matrix3Xf mAngles;
Eigen::VectorXf areas;

int counter = 0;

inline bool scoeff(std::vector<Eigen::Triplet<float>>& target_matrix, const Eigen::Vector3f& input_vector, size_t row_, size_t col_)
{
	for (size_t i = 0; i < 3; ++i)
		if (input_vector[i])
			target_matrix.push_back(Eigen::Triplet <float>(row_, col_ + i, input_vector[i]));
	return true;
}
inline bool dcoeff(std::vector<Eigen::Triplet<float>>& target_matrix, size_t row_, size_t col_, float val)
{
	for (size_t i = 0; i < 3; ++i)
		target_matrix.push_back(Eigen::Triplet <float>(row_ + i, col_ + i, val));
	return true;

}
inline bool srhs(Eigen::VectorXf& b, const Eigen::Vector3f& input_vector, size_t idx)
{
	for (size_t i = 0; i < 3; ++i)
		b(idx + i) += input_vector[i];
	return true;

}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& vertices_mat, Eigen::Matrix3Xi& faces_mat);

void cal_H(const Eigen::VectorXf& vAngles, float threshold, Eigen::VectorXf& h);
void cal_angles_and_areas(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const std::vector<int>& boundIdx, Eigen::Matrix3Xf& matAngles, Eigen::VectorXf& vecAngles, Eigen::VectorXf& areas);
void cal_Gaussian_tri(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& angles, const Eigen::VectorXi& interIdx, std::vector<Eigen::Triplet<float>>& triple);
void cal_laplace_tri(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, const Eigen::Matrix3Xf& angles, const Eigen::VectorXf& areas, int num, const Eigen::VectorXi& interIdx, std::vector<Eigen::Triplet<float>>& triple);
void cal_interpolation_tri(std::vector<int>& anchor, int row, std::vector<Eigen::Triplet<float>>& triple);
void cal_rhs(const Eigen::Matrix3Xf& V, const Eigen::MatrixXf& A, const std::vector<int>& interIdx, const std::vector<int>& boundIdx, const Eigen::VectorXf& h, Eigen::VectorXf& rhb);
double cal_error(const Eigen::VectorXf& vAngles, const std::vector<int>& interIdx);

void matrix2vtk(const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, vtkPolyData* P);
void MakeLUT(vtkFloatArray* Scalar, vtkLookupTable* LUT);
void visualize_mesh(vtkRenderer* Renderer, const Eigen::Matrix3Xf& V, const Eigen::Matrix3Xi& F, Eigen::VectorXf& angles);

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData));
