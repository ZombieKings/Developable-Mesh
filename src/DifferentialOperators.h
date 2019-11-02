// [10/24/2019]
// 实现了几种用到的微分算子

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

//离散梯度算子，G为梯度算子离散后的矩阵
void cal_face_grad(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& G);
void cal_grad_pos(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& gradX, MatrixType& gradY, MatrixType& gradZ);

//拉普拉斯算子，L为拉普拉斯系数矩阵
//注:为剔除边界顶点对应的行，在实际使用中根据需求做相应处理
void cal_angles(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& A);
void cal_cot_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& A, const Eigen::VectorXi& interVidx, Eigen::SparseMatrix<DataType>& L);
void cal_uni_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& L);
