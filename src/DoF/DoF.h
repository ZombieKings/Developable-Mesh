#include <iostream>

#define _USE_MATH_DEFINES

#include <math.h>
#include <algorithm>
#include <assert.h>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "optimization.h"

#include "../tools/cal_angles_areas.h"
#include "../tools/cal_laplacian.h"
#include "../tools/cal_normals.h"
#include "../tools/cal_edge_length.h"
#include "../tools/visualizer.h"

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
		b(idx * 3 + i) += input_vector[i];
	return true;
}

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F);

void adj_face(int Vnum, const Eigen::Matrix3Xi& F, std::vector<std::vector<Eigen::Vector2i>>& adjF);

void grad_function(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr = nullptr);

double cal_error(const VectorType& vecAngles, const Eigen::VectorXi & VType_, int flag);
