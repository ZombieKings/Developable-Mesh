#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include "../tools/cal_angles_areas.h"
#include "../tools/visualizer.h"
#include "../tools/mesh_io.h"
#include "../tools/get_vertex_neighbor.h"

//=============================================================================
typedef double DataType;
typedef Eigen::Triplet<DataType> Tri;
typedef Eigen::Vector3d PosVector;
typedef Eigen::VectorXd VectorType;
typedef Eigen::Matrix3Xd MatrixType;
typedef Eigen::SparseMatrix<DataType> SparseMatrixType;
typedef const Eigen::Matrix3Xd MatrixTypeConst;
//=============================================================================

double cal_error(const VectorType& vecAngles, const Eigen::VectorXi& VType, int flag);

void cal_target_faces_angles(MatrixTypeConst& V,
	const Eigen::Matrix3Xi& F,
	const std::vector<int>& vNFi,
	MatrixType& matA);

void Adjust_Weights();

void Update_Mesh(MatrixType& V,
	const Eigen::Matrix2Xi& E,
	const Eigen::Matrix3Xi& F,
	const Eigen::VectorXi& Vtype, 
	const std::vector<std::vector<int>>& vvNeiF, 
	int innerNum, const VectorType& oriLength);
