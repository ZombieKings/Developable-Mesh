#include "newton_solver.h"

#include <vector>
#include <fstream>
#include <iomanip>
//#include <zjucad/matrix/io.h>
//#include <zjucad/matrix/itr_matrix.h>
//#include <hjlib/sparse/operation.h>
#include <Eigen/Dense>

namespace opt_solver{
static double adjust_mu(double mu, double ratio)
{
    const double min_max_mu[] = {1e-5, 1e6};
    if(ratio < 0) {
        mu *= 4;
        if(mu < min_max_mu[0])
            mu = min_max_mu[0];
    }
    else if(ratio < 0.25) {
        mu *= 4;
        if(mu < min_max_mu[0])
            mu = min_max_mu[0];
    }
    else
        mu /= sqrt(ratio*4); // better than /2
    if(mu > min_max_mu[1])
        mu = min_max_mu[1];
    return mu;
}

newton_solver::newton_solver()
    :spt_(NULL),f_(NULL),mu_(1e-6)
{

}

int newton_solver::set_ptree(boost::property_tree::ptree &spt)
{
    spt_ = &spt;
}

int newton_solver::set_f(func_opt::function &f)
{
    f_ = &f;
    dim_x_ = f_->dim();
    diag_H_.resize(f_->dim());
    grad_.resize(dim_x_);
    step_.resize(dim_x_);
}

int newton_solver::set_hess_diag(std::vector<double*>  &diag_H)
{
    const size_t num_diag = H_.rows();
    diag_H.resize(num_diag);
    for(size_t i = 0; i < num_diag; ++i) {
        diag_H[i] = &H_.coeffRef(i, i);
    }
    return 0;
}

int newton_solver::add_to_diag(const double *D)
{
    const size_t num_diag = diag_H_.size();
    for(size_t i = 0;i < num_diag;++i){
        *diag_H_[i] += D[i];
    }
    return 1;
}

int newton_solver::sub_from_diag(const double *D)
{
    const size_t num_diag = diag_H_.size();
    for(size_t i = 0;i < num_diag;++i){
        *diag_H_[i] -= D[i];
    }
    return 1;
}


int newton_solver::compute_D(double *D)
{
    const size_t num_diag = diag_H_.size();
    for(size_t i = 0;i < num_diag;++i){
        double d = *diag_H_[i] < 0 ? 0 : sqrt(*diag_H_[i]);
        if(D[i] < d){
            D[i] = d;
        }
    }
    return 1;
}

int newton_solver::solve(double *x)
{
    double norm2_of_f_g[2] = {std::numeric_limits<double>::infinity()};
    const double eps[3] = {1e-6,1e-8,1e-8};

    const size_t dim_x = f_->dim();

    size_t max_iter = 1000;//spt_->get<int>("iter.value");
    double sigma = 1.0,cur_mu = 0; // mu = 1e-1
    Eigen::Map<Eigen::VectorXd> x0(x, dim_x_);
    const double min_mu = 1e-6,max_mu = 10e16,min_sigma = 1e-21;
    double &mu = mu_ = min_mu;
    int res = 0;
    Eigen::VectorXd D(dim_x), xold;
    size_t iter;
    for(iter = 1; iter < max_iter; ++iter) {
        std::cerr<<"iter: "<<iter<<std::endl;
        norm2_of_f_g[0] = 0;
        grad_.setZero();
        f_->val_gra(x,norm2_of_f_g[0],&grad_[0]);
        norm2_of_f_g[1] = grad_.squaredNorm();

        xold = x0;
        //double old_fun_val = norm2_of_f_g[0];
        std::cerr <<std::setprecision(12) << "\t" << norm2_of_f_g[0] << "\t" << norm2_of_f_g[1] << "\t" << mu << "\n";
        if(norm2_of_f_g[1] < eps[0]){
            std::cout<<"gradient convergence!"<<std::endl;
            res = 1;
            break;
        }

        if(H_.nonZeros() == 0 ) { // the first run
            f_->hes(x, H_);
            slv_.analyzePattern(H_);
            set_hess_diag(diag_H_);
        }
        std::fill(H_.valuePtr(), H_.valuePtr() + H_.nonZeros(), 0);
        f_->hes(x, H_);

        Eigen::VectorXd ori_x = x0;
        slv_.factorize(H_);
        if(slv_.info() == Eigen::Success) {
            mu = min_mu;
            cur_mu = 0;
        }else{
            int run = 0;
            bool success = false;
            while(!success && mu < max_mu){
                D.setConstant(mu);
                add_to_diag(&D[0]);
                slv_.factorize(H_);
                sub_from_diag(&D[0]);

                if(slv_.info() == Eigen::Success) {
                    if(cur_mu != mu){
                        sigma = 1;
                    }
                    cur_mu = mu;
                    if(run == 0 && mu > min_mu){
                        mu *= 0.1;
                    }
                    success = true;
                }else{
                    if(mu < max_mu){
                        mu *= 10;
                    }
                }
                ++run;
            }
        }

        if(mu < max_mu){
            // compute newton step direction
            step_ = -slv_.solve(grad_);
        }
        else{
            // gradient descent
            cur_mu = -1;
            mu *= 1e-1;
            step_ = -grad_;
        }

        double new_val;
        int run = 0;
        bool repeat = true;
        sigma = std::min(sigma,1.0);
        while(repeat) {
            x0 = ori_x + sigma*step_;
            new_val = 0;
            f_->val(&x0[0], new_val);
            if(norm2_of_f_g[0] > new_val ||
                    sigma < min_sigma){
                repeat = false;
                if(run == 0 && sigma < 1.0){
                    sigma *= 2;
                }
            }
            else{
                sigma *= 0.5;
            }
            ++run;
        }

        if((x0 - ori_x).cwiseAbs().maxCoeff() < eps[1]){
            std::cerr << "\nstep converge." << std::endl;
            break;
        }
    }
    std::cerr << "\noptimization over." << std::endl;
    return iter;
}

int newton_solver::solve_sqp(double *x)
{

    const double eps[3] = {1e-6,1e-6,1e-20 };
    double norm2_of_f_g[2], mu = 1e-6, radius= 1e4;

    const size_t dim = f_->dim();

    Eigen::VectorXd D(dim);
    Eigen::Map<Eigen::VectorXd> x0(x, dim);
    const int iter_num = 1000;

    for(int i = 0; i < iter_num; ++i) {
        std::cerr << i;
        norm2_of_f_g[0] = 0;
        f_->val_gra(x,norm2_of_f_g[0],&grad_[0]);
        norm2_of_f_g[1] = grad_.squaredNorm();

        std::cerr << "\t" << norm2_of_f_g[0] << "\t" << norm2_of_f_g[1] << "\t" << mu << "\t" << radius;
        if(norm2_of_f_g[1] < eps[0]) {
            std::cerr << "\ngradient converge." << std::endl;
            break;
        }

        if(H_.nonZeros() == 0 ) { // the first run
            f_->hes(x, H_);
            slv_.analyzePattern(H_);
            set_hess_diag(diag_H_);
        }
        std::fill(H_.valuePtr(), H_.valuePtr() + H_.nonZeros(), 0);
        f_->hes(x, H_);

        bool at_border = true;
        int step_type = -1; // 0 TC, 1 NP, 2 DL
        // Cauchy point
        Eigen::VectorXd tmp = H_ * grad_;
        //mv(false, H_, g_, tmp); // corrected_H?
        const double Cauchy_len = norm2_of_f_g[1]/tmp.dot(grad_), norm_g = sqrt(norm2_of_f_g[1]);
        if(Cauchy_len*norm_g > radius) {
            step_ = -grad_*radius/norm_g; // truncate
            step_type = 0;
            std::cerr << "\tTC";
        }
        else { // Cauchy point is inside, evaluate Newton point
            compute_D(&D[0]);
            D *= mu;
            add_to_diag(&D[0]);
            slv_.factorize(H_);
            sub_from_diag(&D[0]);

            if(slv_.info() == Eigen::Success){
                step_ = -slv_.solve(grad_);

                if((step_).norm() < radius) { // Newton point is inside of the radius
                    step_type = 1;
                    std::cerr << "\tNP";
                    at_border = false;
                }
                else { // intersect with radius
                    const Eigen::VectorXd CP = -grad_*Cauchy_len;
                    const double gamma = CP.squaredNorm()/step_.squaredNorm();
                    Eigen::VectorXd C2N = (0.8*gamma+0.2)*step_-CP;
                    const double quadric_eq[3] = {C2N.squaredNorm(), 2*CP.dot(C2N), CP.squaredNorm()-radius*radius};
                    double Delta = quadric_eq[1]*quadric_eq[1]-4*quadric_eq[0]*quadric_eq[2];
                    assert(Delta > 0);
                    Delta = sqrt(Delta);
                    if(quadric_eq[0] < 0) {
                        std::cerr << "\tbad dog leg: " << quadric_eq[0] << std::endl;
                    }
                    const double roots[2] = {
                        (-quadric_eq[1]+Delta)/(2*quadric_eq[0]),
                        (-quadric_eq[1]-Delta)/(2*quadric_eq[0])
                    };
                    // it's highly possible that dot(CP, C2N) > 0
                    if(roots[0] >= 0 && roots[0] <= 1) {
                        if(roots[1] >= 0 && roots[1] <= 1)
                            std::cerr << "two intersection?" << std::endl;
                        step_ = CP+roots[0]*C2N;
                    }
                    else if(roots[1] >= 0 && roots[1] <= 1) {
                        std::cerr << "strange why the second root?" << std::endl;
                        step_ = CP+roots[1]*C2N;
                    }
                    else {
                        if(roots[0] > 0) {
                            step_ = CP+roots[0]*C2N;
                        }
                        if(roots[1] > 0) {
                            step_ = CP+roots[1]*C2N;
                        }
                        if(roots[0]*roots[1] > 0)
                            std::cerr << "\nunbelievable that no root is OK: " << roots[0] << " " << roots[1] << std::endl;
                    }
                    step_type = 2;
                    std::cerr << "\tDL";
                }
            }
            else { // H is not SPD
                mu *= 16;
                continue;
            }
        }
        Eigen::VectorXd ori_x = x0;
        if(step_.cwiseAbs().maxCoeff() < eps[2]) {
            std::cerr << "\nstep converge." << std::endl;
            break;
        }
        x0 += step_;
        // update mu and radius
        double new_val = 0, est_val;
        f_->val(x, new_val);
        //tmp.setZero();
        //zjucad::matrix::mv(false, H_, s_, tmp_);
        tmp = H_ * step_;
        est_val = norm2_of_f_g[0] + step_.dot(grad_) + step_.dot(tmp)/2;
        double ratio = -1;
        if(new_val < norm2_of_f_g[0]) {
            const double delta_f = new_val-norm2_of_f_g[0];
            double t = est_val-norm2_of_f_g[0];
            if(fabs(t) < 1e-10)
                t = 1e-5;
            ratio = delta_f/t;
        }
        //    cerr << "\t" << est_val << "\t" << new_val << "\t" << norm2_of_f_g[0] << "\t" << ratio;
        if(ratio < 0 && step_type == 0) { // accept bad step for NP and DL
            x0 = ori_x;
        }
        mu = adjust_mu(mu, ratio);
        if(ratio < 0.25) {
            radius /= 4;
        }
        else if(ratio > 0.75 || at_border) {
            radius *= 2;
        }
        std::cerr << std::endl;
    }
    std::cerr << "\noptimization over." << std::endl;
    return 0;
}

}
