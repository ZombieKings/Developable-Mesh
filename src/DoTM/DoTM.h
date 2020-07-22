#include <iostream>

#define _USE_MATH_DEFINES

#include <math.h>
#include <algorithm>
#include <assert.h>

#include <surface_mesh/Surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

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

void TimeCallbackFunction(vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* clientData, void* vtkNotUsed(callData));

void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F);

void adj_face(int Vnum, const Eigen::Matrix3Xi& F, std::vector<std::vector<Eigen::Vector2i>>& adjF);

void hinge_energy_and_grad(MatrixTypeConst V,
	const Eigen::Matrix3Xi& F,
	const std::vector<std::vector<Eigen::Vector2i>>& adjF,
	double& energy,
	MatrixType& energyGrad
);

void hinge_energy(MatrixTypeConst V,
	const Eigen::Matrix3Xi& F,
	const std::vector<std::vector<Eigen::Vector2i>>& adjF,
	double& energy
);

int opt_solve(MatrixType V,
	const Eigen::Matrix3Xi& F,
	const std::vector<std::vector<Eigen::Vector2i>>& adjF,
	double& t,
	MatrixType& p,
	double& energy,
	MatrixType& energyGrad);

void energy_test(MatrixType V, const Eigen::Matrix3Xi& F, const std::vector<std::vector<Eigen::Vector2i>>& adjF);

double cal_error(const VectorType& vecAngles, const VectorType& vecAreas, const Eigen::VectorXi& VType_, int flag);
