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

#include "interpolation.h"

//=============================================================================
using namespace surface_mesh;
//=============================================================================

class Dev_LN
{
public:

	//Constructor
	Dev_LN() {};
	//Deconstructor
	~Dev_LN() {};

private:
	//------Mesh Data------
	//Original mesh
	Surface_mesh ori_mesh_;

	Eigen::Matrix3Xf ori_mesh_mat_;
	Eigen::Matrix3Xi face_mat_;

	//Number of vertices
	size_t vnum_ = 0;

	//Internal Vertex indexyoa
	std::vector<int> inter_p_;
	std::vector<int> boundary_p_;
	Eigen::VectorXi inter_p_r_;

	//Current mesh
	Surface_mesh cur_mesh_;
	Eigen::Matrix3Xf cur_mesh_mat_;

public:
	//creat a mesh with mesh_size height and weight
	int Load_Mesh(const std::string &filename);

	int SetCondition(double delta, size_t times);

	int Deformation();

	const Surface_mesh& Dev_LN::Get_Result() const;
private:
	//------Equation Datas------
	Eigen::SparseMatrix<float> coeff_A_;

	Eigen::VectorXf result_x_;

	Eigen::VectorXf right_b_;

	//------Condition Datas--------
	int it_count_ = 50;

	double epsilon_ = 0;

	size_t dense_ = 0;
private:
	//Build the linear system equation
	int Build_Equation();

	//Solve the linear system
	int Solve_Problem();

private:
	//Use S update mesh
	int Update_Mesh();

	//calculate developable error
	//flag == 1:return maximum error 
	//flag == 0:return averange error 
	double Cal_Error(const Eigen::Matrix3Xf &V, int flag);


	void cal_angles(const Eigen::Matrix3Xf &V, const Eigen::Matrix3Xi &F, Eigen::Matrix3Xf &angles);

	int cal_cot_laplace(const Surface_mesh& mesh, Eigen::SparseMatrix<float> &L);

	void mesh2matrix(const surface_mesh::Surface_mesh& mesh, Eigen::Matrix3Xf& vertices_mat, Eigen::Matrix3Xi& faces_mat);


private:
	//---------Temporary Data------------
	std::vector<Eigen::Triplet<float>> tri_Coeff_;

};

//将得到的系数向量导入系数矩阵中
inline bool vec2mat(std::vector<Eigen::Triplet<float>>& target_matrix, const Vec3& input_vector, size_t row_, size_t col_)
{
	for (size_t i = 0; i < 3; ++i)
		if (input_vector[i])
			target_matrix.push_back(Eigen::Triplet <float>(row_, col_ * 3 + i, input_vector[i]));

	return true;
}