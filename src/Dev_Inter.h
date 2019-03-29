#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

//=============================================================================
using namespace surface_mesh;
//=============================================================================

class Dev_Inter
{
public:

	//Constructor
	Dev_Inter(Surface_mesh input_mesh, Surface_mesh input_anchor) :ori_mesh_(input_mesh), anchor_p_(input_anchor) {};
	//Deconstructor
	~Dev_Inter() {};

private:
	//------Mesh Data------
	//Original mesh
	Surface_mesh ori_mesh_;

	//Anchor points
	Surface_mesh anchor_p_;

	//Internal Vertex index
	Eigen::VectorXd inter_p_;

	//Current mesh
	Surface_mesh cur_mesh_;

public:

	//Deformation
	int Deformation();

	//Get deformation result
	const Surface_mesh& Get_Result() const;

private:
	//------Equation Datas------
	Eigen::SparseMatrix<float> coeff_A_;

	Eigen::VectorXf right_b_;

	unsigned int w1_, w2_;
private:
	//Build the coefficient matrix of linear system
	int BuildMetrix();

	//Solve the linear system
	int SolveProblem();

private:
	//Calculate Gaussian Curvature at vertex v
	float Cal_Curvature(const Surface_mesh::Vertex& v);

	//Calculate Developable Error of v
	float Cal_EInter(const Surface_mesh::Vertex& v);

	//Calculate Length Error of v
	float Cal_Elength(const Surface_mesh::Edge& v);

	//Dynamically Update Weights in Processing
	int Cal_Weights(Surface_mesh X0, double errD, double errL, double errI, const Surface_mesh& X0r, double errDr, double errLr, double errIr);

};

