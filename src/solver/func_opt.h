#ifndef FUNC_OPT_H_
#define FUNC_OPT_H_

#include <iostream>
#include <vector>
#include <set>

#include <boost/property_tree/ptree.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

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

	class sum_function : public function
	{
	public:
		typedef std::vector<boost::shared_ptr<function> > container;
		sum_function(const container& children);
		virtual size_t dim(void) const;
		virtual int val(const double* x, double& v);
		virtual int gra(const double* x, double* g);
		virtual int val_gra(const double* x, double& v, double* g);
		virtual int hes(const double* x, Eigen::SparseMatrix<double>& h);

		size_t size() const;
	protected:
		container children_;
		std::vector<std::set<int32_t> > pattern_;
		size_t nnz_;
	};
}

#endif
