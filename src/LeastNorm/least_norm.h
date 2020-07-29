#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../tools/cal_angles_areas.h"
#include "../tools/visualizer.h"
#include "../tools/mesh_io.h"

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

/**
 * @brief compute update vector d via least norm
 *
 * @param[in] V			geometrical information of mesh
 * @param[in] F			topological information of mesh
 * @param[in] Vtype		vertices type mask
 * @param[in] innerNum  number of internal vertex
 *
 * @param[out] update_d	update vector d
 */
void compute_update_vector(MatrixTypeConst& V, 
	const Eigen::Matrix3Xi& F, 
	const Eigen::VectorXi& Vtype,
	int innerNum, 
	VectorType& update_d);

/**
 * @brief update mesh with input update vector vis Laplacian 
 *
 * @param[in & out] V	geometrical information of mesh
 * @param[in] F			topological information of mesh
 * @param[in] Vtype		vertices type mask
 * @param[in] innerNum  number of internal vertex
 * @param[in] update_d	update vector d
 */
void update_points(MatrixType& V, 
	const Eigen::Matrix3Xi& F, 
	const Eigen::VectorXi& Vtype,
	int innerNum,	
	const VectorType& update_d);

