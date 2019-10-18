#ifndef NONLINEAROPT_H
#define NONLINEAROPT_H

#include <Eigen/Sparse>
#include "func_opt.h"

namespace opt_solver{
class newton_solver
{
public:
    newton_solver();

public:
    int set_ptree(boost::property_tree::ptree  &spt);
    int set_f(func_opt::function &f);

    int solve(double *x);
    double get_mu() const{
        return mu_;
    }
    int solve_sqp(double *x);
 protected:
    int set_hess_diag(std::vector<double*>  &diag_H);
    int add_to_diag(const double *D);
    int sub_from_diag(const double *D);
    int compute_D(double *D);
private:
protected:
    size_t dim_x_;
    double mu_;
    func_opt::function           *f_;

    Eigen::SparseMatrix<double> H_;
    Eigen::VectorXd  grad_, step_;
    boost::property_tree::ptree      *spt_;

    std::vector<double*>  diag_H_;
    //hj::sparse::csc<double, int32_t> H_;

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> slv_;
};


}

#endif // NONLINEAROPT_H
