#ifndef FUNC_OPT_H_
#define FUNC_OPT_H_

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <vector>
#include <set>

#include <boost/property_tree/ptree.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <surface_mesh/Surface_mesh.h>
#include "residual_func.h"

namespace func_opt {


	//! @brief: R^n -> R
	class function
	{
	public:
		virtual ~function() {}

		virtual size_t dim(void) const = 0;
		//NOTE: add to v
		virtual int val(const double* x, double& v) = 0;

		virtual int gra(const double* x, double* g) = 0;

		virtual int val_gra(const double* x, double& v, double* g) = 0;
		//NOTE: add to hes
		//! @param h,ptr,idx == 0 means query nnz
		//! @param h == 0, ptr,idx != 0 means query patten
		//! @param h,ptr,idx != 0 means accumulate
		virtual int hes(const double* x, Eigen::SparseMatrix<double>& h) = 0;
	};

	double gra_err(function& f, double* x);
	double hes_err(function& f, double* x);

	class my_function : public function
	{
		typedef double DataType;
		typedef Eigen::Triplet<DataType> Tri;
		typedef Eigen::Vector3d PosVector;
		typedef Eigen::VectorXd VectorType;
		typedef Eigen::Matrix3Xd MatrixType;
		typedef const Eigen::Matrix3Xd MatrixTypeConst;

	public:
		my_function(const surface_mesh::Surface_mesh& mesh, double eps, double w1, double w2, double w3);
		virtual size_t dim(void) const;
		virtual int val(const double* x, double& v);
		virtual int gra(const double* x, double* g);
		virtual int val_gra(const double* x, double& v, double* g);
		virtual int hes(const double* x, Eigen::SparseMatrix<DataType>& h);
	protected:
		int Vnum_ = 0;
		std::vector<int> interV_;
		std::vector<int> boundV_;
		Eigen::VectorXi interVidx_;
		Eigen::SparseMatrix<DataType> F2V_, F2Vt_;

		MatrixType V_;
		Eigen::Matrix3Xi F_;

		MatrixType mAngles_;
		VectorType vAngles_;
		VectorType areas_;

		int normtype_ = 0;

		double eps_ = 0.0;
		double w1_ = 1.0;
		double w2_ = 1.0;
		double w3_ = 1.0;

		VectorType preGradX_;
		VectorType preGradY_;
		VectorType preGradZ_;

		Eigen::SparseMatrix<DataType> L_;
		Eigen::SparseMatrix<DataType> Grad_, Gradt_;
		Eigen::SparseMatrix<DataType> Gau_;
	protected:
		void mesh2matrix(const surface_mesh::Surface_mesh& mesh, MatrixType& V, Eigen::Matrix3Xi& F);
		void cal_topo(const Eigen::Matrix3Xi& F, int Vnum, Eigen::SparseMatrix<DataType>& F2V);
		void cal_angles_and_areas(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const std::vector<int>& boundIdx, MatrixType& matAngles, VectorType& vecAngles, VectorType& areas);
		void cal_grad(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& G);
		void cal_grad_pos(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixType& gradXpos, MatrixType& gradYpos, MatrixType& gradZpos);
		
		//计算出高斯曲率的梯度矩阵,每列对应每个变量
		void cal_gaussian_gradient(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, const Eigen::VectorXi& interVidx, MatrixTypeConst& mAngles,const VectorType& vAngles, Eigen::SparseMatrix<DataType>& mGradient);

		//计算均值laplace矩阵，边界顶点部分未剔除
		void cal_cot_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, MatrixTypeConst& A, const Eigen::VectorXi& interVidx, Eigen::SparseMatrix<DataType>& L);
		void cal_uni_laplace(MatrixTypeConst& V, const Eigen::Matrix3Xi& F, Eigen::SparseMatrix<DataType>& L);

	};
}

#endif
