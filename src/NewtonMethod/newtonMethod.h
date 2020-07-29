#define _USE_MATH_DEFINES
#include <math.h>

#include <surface_mesh/Surface_mesh.h>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

#include "../tools/cal_angles_areas.h"
#include "../tools/cal_laplacian.h"
#include "../tools/cal_normals.h"
#include "../tools/cal_edge_length.h"
#include "../tools/visualizer.h"
#include "../tools/mesh_io.h"

typedef double DataType;
typedef Eigen::Triplet<DataType> Tri;
typedef Eigen::Vector3d PosVector;
typedef Eigen::VectorXd VectorType;
typedef Eigen::Matrix3Xd MatrixType;
typedef Eigen::SparseMatrix<DataType> SparseMatrixType;
typedef const Eigen::Matrix3Xd MatrixTypeConst;

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F);

/**
 * @brief Main processing function
 * 
 * @param V			geometrical information of mesh
 * @param F			topological information of mesh
 * @param Vtype		vertices type mask
 * @param innerNum  number of internal vertex
 * @param basicH	vecter of non change elements in H matrix
 * @param L			uniform Laplacian operator matrix
 */
void Update(MatrixType& V,
	const Eigen::Matrix3Xi& F, 
	const Eigen::VectorXi& Vtype, 
	int innerNum,
	std::vector<Tri> basicH,
	const SparseMatrixType& L);

void CallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData));
