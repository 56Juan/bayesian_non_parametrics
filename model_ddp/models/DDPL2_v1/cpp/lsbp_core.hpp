#ifndef LSBP_CORE_HPP
#define LSBP_CORE_HPP

#include <vector>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

class LSBPCore {
public:
    LSBPCore(unsigned int seed = 42);
    ~LSBPCore() = default;

    // Slice sampling para truncamiento adaptativo
    std::vector<double> sample_slice_variables(
        const std::vector<int>& z,
        const std::vector<double>& w,
        int n
    );

    // Actualización de asignaciones con slice constraint
    std::vector<int> update_assignments(
        const MatrixXd& y_normalized,
        const MatrixXd& design_matrix,
        const MatrixXd& lambda_h,
        const MatrixXd& xi_h,
        const std::vector<double>& w,
        const std::vector<double>& u,
        int n,
        int H,
        int K
    );

    // Actualización de coeficientes lambda (weighted least squares)
    MatrixXd update_lambda(
        const MatrixXd& y_normalized,
        const MatrixXd& design_matrix,
        const std::vector<int>& z,
        const MatrixXd& xi_h,
        const VectorXd& mu_lambda,
        const MatrixXd& Sigma_lambda,
        int n,
        int H,
        int K
    );

    // Actualización de coeficientes xi (Metropolis-Hastings)
    MatrixXd update_xi(
        const MatrixXd& y_normalized,
        const MatrixXd& design_matrix,
        const MatrixXd& lambda_h,
        const MatrixXd& xi_h,
        const std::vector<int>& z,
        const VectorXd& mu_xi,
        const MatrixXd& Sigma_xi,
        double scale,
        int n,
        int H,
        int K
    );

private:
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;

    // Funciones auxiliares
    double compute_log_likelihood_point(
        double y_val,
        const VectorXd& d_row,
        const VectorXd& lambda_vec,
        const VectorXd& xi_vec
    );

    double log_normal_pdf(double x, double mean, double sd);
    
    VectorXd sample_multivariate_normal(
        const VectorXd& mean,
        const MatrixXd& cov
    );
};

#endif // LSBP_CORE_HPP