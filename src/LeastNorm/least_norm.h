#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <pcl/point_types.h>
#include <pcl/io/obj_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

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
	//Original vertices
	Eigen::Matrix3Xd input_vertice_;

	//Original edges index
	std::vector<std::vector<int>> input_edges_;

	//Number of vertices
	size_t vnum_ = 0;
	//Line Datas
	Eigen::MatrixX3d U, D, L, R;

	//Original mesh
	Surface_mesh ori_mesh_;
	Eigen::Matrix3Xd ori_mesh_mat_;
	Eigen::Matrix3Xi face_mat_;

	//Internal Vertex index
	std::vector<int> inter_p_;
	//Boundary Vertex index
	std::vector<int> bound_p_;

	//Current mesh
	Surface_mesh cur_mesh_;
	Eigen::Matrix3Xd cur_mesh_mat_;

public:
	//Read line datas from file
	int Read_File(const std::string &filename);

	//creat a mesh with mesh_size height and weight
	int CreatMesh(size_t dense);

	int SetCondition(double delta, size_t times);

	int Deformation();

	const Surface_mesh& Dev_LN::Get_Result() const;
private:
	//------Equation Datas------
	Eigen::SparseMatrix<double> coeff_A_;

	Eigen::VectorXd result_x_;

	Eigen::VectorXd right_b_;

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
	//Fit contours
	Eigen::MatrixX3d Fit_Line(const std::vector<int>& l_idx, int dense);

	//Use S update mesh
	int Update_Mesh();

	//calculate developable error
	//flag == 1:return maximum error 
	//flag == 0:return averange error 
	double Cal_Error(const Eigen::Matrix3Xd &V, int flag);

	int cal_topo_laplace(const Eigen::MatrixXd &V, const Eigen::Matrix3Xi &F, Eigen::SparseMatrix<double> &L);
	
	void Dev_LN::cal_angles(const Eigen::Matrix3Xd &V, const Eigen::Matrix3Xi &F, Eigen::Matrix3Xd &angles);

	int Dev_LN::cal_cot_laplace(const Eigen::MatrixXd &V, const Eigen::Matrix3Xi &F, Eigen::SparseMatrix<double> &L);
private:
	//---------Temporary Data------------
	std::vector<Eigen::Triplet<double>> tri_Coeff_;

};

//将得到的系数向量导入系数矩阵中
inline bool vec2mat(std::vector<Eigen::Triplet<double>>& target_matrix, const Vec3& input_vector, size_t row_, size_t col_)
{
	for (size_t i = 0; i < 3; ++i)
		if (input_vector[i])
			target_matrix.push_back(Eigen::Triplet <double>(row_, col_ * 3 + i, input_vector[i]));

	return true;
}