#include "func_opt.h"

#include <limits>
#include <iostream>
#include <set>

using namespace std;

namespace func_opt {

	sum_function::sum_function(const container& children)
		:children_(children), nnz_(0)
	{
		const size_t fn = children_.size();
		assert(fn);
		for (size_t i = 1; i < fn; ++i) {
			if (children_[i]->dim() != children_[0]->dim()) {
				cerr << "incompatible functions." << children_[0]->dim()
					<< " " << children_[i]->dim() << endl;
			}
		}
	}

	size_t sum_function::size() const
	{
		return children_.size();
	}

	size_t sum_function::dim(void) const
	{
		return children_[0]->dim();
	}

	int sum_function::val(const double* x, double& v)
	{
		const size_t fn = children_.size();
		for (size_t i = 0; i < fn; ++i) {
			children_[i]->val(x, v);
		}
		return 0;
	}

	int sum_function::gra(const double* x, double* g)
	{
		const size_t fn = children_.size();
		for (size_t i = 0; i < fn; ++i) {
			children_[i]->gra(x, g);
		}
		return 0;
	}

	int sum_function::val_gra(const double* x, double& v, double* g)
	{
		const size_t fn = children_.size();
		for (size_t i = 0; i < fn; ++i) {
			children_[i]->val_gra(x, v, g);
		}
		return 0;
	}

	int sum_function::hes(const double* x, Eigen::SparseMatrix<double>& h)
	{
		/*
		if(h == 0 && ptr == 0 && idx == 0) {// query nnz
			pattern_.resize(dim());
			for(size_t fi = 0; fi < children_.size(); ++fi) {
				size_t nnz0;
				if(children_[fi]->hes(x, nnz0, 0, 0, 0))
					return __LINE__;
				pair<vector<int32_t>, vector<int32_t> > ptr_idx;
				ptr_idx.first.resize(dim()+1);
				ptr_idx.first[0] = 0;
				ptr_idx.second.resize(nnz0);
				if(children_[fi]->hes(x, nnz0, 0, &ptr_idx.first[0], &ptr_idx.second[0]))
					return __LINE__;
				for(size_t ci = 0; ci < dim(); ++ci) {
					for(size_t nzi = ptr_idx.first[ci]; nzi < ptr_idx.first[ci+1]; ++nzi) {
						pattern_[ci].insert(ptr_idx.second[nzi]);
					}
				}
			}
			nnz = 0;
			for(size_t xi = 0; xi < dim(); ++xi)
				nnz += pattern_[xi].size();
			nnz_ = nnz;
			return 0;
		}
		if(h == 0 && ptr != 0 && idx != 0) {// query patten
			if(nnz < nnz_) {
				cerr << "incorrect input at query pattern: " << nnz << " " << nnz_;
				return __LINE__;
			}
			for(size_t xi = 0; xi < dim(); ++xi) {
				ptr[xi+1] = ptr[xi] + pattern_[xi].size();
				size_t nzi = ptr[xi];
				for(set<int32_t>::const_iterator i = pattern_[xi].begin();
					i != pattern_[xi].end(); ++i, ++nzi) {
					idx[nzi] = *i;
				}
			}
			std::vector<std::set<int32_t> > tmp;
			std::swap(pattern_, tmp);
			return 0;
		}
		if(h != 0 && ptr != 0 && idx != 0) {// accumulate
			if(nnz < nnz_ && nnz_ != -1) { // when nnz_ == -1, client know the pattern already
				cerr << "incorrect input at accumulate: " << nnz << " " << nnz_;
				return __LINE__;
			}
			for(size_t fi = 0; fi < children_.size(); ++fi)
				if(children_[fi]->hes(x, nnz, h, ptr, idx, alpha))
					return __LINE__;
			return 0;
		}
		*/
		return __LINE__;
	}

	double gra_err(function& f, double* x)
	{
		const size_t dim = f.dim();
		double val = 0;
		matrix<double> g = zeros<double>(dim, 1);
		f.val_gra(x, val, &g[0]);
		cerr << "max g: " << max(fabs(g)) << endl;
		const double eps = 1e-6;
		for (size_t xi = 0; xi < dim; ++xi) {
			const double save = x[xi];
			double v[2] = { 0, 0 };
			x[xi] = save - eps;
			f.val(x, v[0]);
			x[xi] = save + eps;
			f.val(x, v[1]);
			g[xi] -= (v[1] - v[0]) / (2 * eps);
			x[xi] = save;
		}
		return max(fabs(g));
	}

	// assume grad is accurate
	double hes_err(function& f, double* x)
	{
		const size_t dim = f.dim();
		Eigen::VectorXd g0(dim);
		g0.setZero();
		double v;
		f.val_gra(x, v, &g0[0]);
		Eigen::SparseMatrix<double> H;
		f.hes(x, H);
		std::fill(H.valuePtr(), H.valuePtr() + H.nonZeros(), 0);
		f.hes(x, H);

		Eigen::MatrixXd hes(dim, dim);
		hes.setZero();
		for (size_t ci = 0; ci < dim; ++ci) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(H, ci); it; ++it) {
				hes(ci, it.row()) = it.value();
			}
		}
		cout << "max hes: " << hes.cwiseAbs().maxCoeff() << endl;

		//cout<<hes<<endl;
		const double eps = 1e-6;
		Eigen::VectorXd ga(dim), gb(dim);
		for (size_t xi = 0; xi < dim; ++xi) {
			const double x0 = x[xi];

			x[xi] = x0 + eps;
			ga.setZero();
			f.gra(x, &ga[0]);

			x[xi] = x0 - eps;
			gb.setZero();
			f.gra(x, &gb[0]);

			hes.col(xi) -= (ga - gb) / (2 * eps);

			x[xi] = x0;
		}

		return hes.cwiseAbs().maxCoeff();
	}

}
