#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <surface_mesh/Surface_mesh.h>

typedef float DataType;
typedef Eigen::Triplet<DataType> Tri;
typedef Eigen::Vector3f VectorType;
typedef Eigen::Matrix3Xf MatrixType;
typedef const Eigen::Matrix3Xf MatrixTypeConst;

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& vertices_mat, Eigen::Matrix3Xi& faces_mat);

void cal_angles(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& A);
void cal_cot_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& A, Eigen::SparseMatrix<DataType>& L);
void cal_uni_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& L);
void cal_grad_pos(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& gradX, MatrixType& gradY, MatrixType& gradZ);
