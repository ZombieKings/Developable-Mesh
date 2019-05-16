#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//=============================================================================
using namespace surface_mesh;
//=============================================================================

typedef Eigen::Triplet<float> Tri;

class Dev_Inter
{
public:

	//Constructor
	Dev_Inter(Surface_mesh input_mesh, std::vector<Point> input_anchor, std::vector<unsigned int> input_anchor_idx);
	//Deconstructor
	~Dev_Inter() {};

private:
	//------Mesh Data------
	//Original mesh
	Surface_mesh ori_mesh_;

	//Anchor points
	std::vector<Point> anchor_position_;
	std::vector<unsigned int> anchor_idx_;

	//Internal Vertex index
	std::vector<int> inter_p_;

	//Current mesh
	Surface_mesh cur_mesh_;

	//Previous mesh
	Surface_mesh pre_mesh_;

public:

	//Deformation
	int Deformation();

	//Get deformation result
	const Surface_mesh& Get_Result() const;

	//Set all conditions
	int SetConditions(const float& D, const float& I, const float& L, const float& dD, const float& dI, const float& duL, const float& ddL);

private:
	//------Equation Datas------
	Eigen::SparseMatrix<double> coeff_A_;

	Eigen::VectorXf right_b_;

	Eigen::VectorXf scale_s_;

	//------Condition Datas--------
	//Terminal conditions
	float epD_ = 0, epL_ = 0, epI_ = 0;
	float deD_ = 0, udeL_ = 0, ddeL_ = 0, deI_ = 0;

	//Weights of different errors
	float w1_ = 1, w2_ = 1;

	//Errors of developable,length preservation and interpolation, respectively
	double ED_ = 0, EL_ = 0, EI_ = 0;

	//Errors of previous iteration
	double preED_ = 0, preEL_ = 0, preEI_ = 0;
private:
	//Build the coefficient matrix of linear system
	int BuildMetrix();

	//Solve the linear system
	int SolveProblem();

	//creat a mesh with mesh_size height and weight
	Surface_mesh CreatMesh(size_t mesh_size);

private:
	//Calculate Gaussian curvature constraint at vertex v
	int Cal_CurvatureCoeff(const Surface_mesh::Vertex& v, size_t num);

	//Calculate length preservation of e
	float Cal_LengthCoeff(const Surface_mesh::Edge& e, size_t num);

	//Calculate interpolation error of v
	float Cal_InterCoeff(size_t idx, size_t num);

	//Calculate error of current deformation
	void Cal_Error();

	//Dynamically update weights in processing
	int Adjust_Weights();

	//Use S update mesh
	int Update_Mesh();

	//Matrix visualizator
	void Matrix_Visualization(const Eigen::MatrixXf& iMatrix, double first, double second);

	int Matrix_Rank(const Eigen::SparseMatrix<float>& inMatrix);

	float Cal_Guassion_Curvature(const Surface_mesh::Vertex& v);

private:
	//---------Temporary Data------------
	std::vector<Tri> tri_Coeff_;

};

//将得到的系数向量导入系数矩阵中
inline bool vec2mat(std::vector<Eigen::Triplet<float>>& target_matrix, const Vec3& input_vector, size_t row_, size_t col_)
{
	for (size_t i = 0; i < 3; ++i)
		if (input_vector[i])
			target_matrix.push_back(Eigen::Triplet <float>(row_, col_ * 3 + i, input_vector[i]));

	return true;
}