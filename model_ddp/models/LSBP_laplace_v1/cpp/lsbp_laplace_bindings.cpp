// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "lsbp_laplace_core.hpp"

namespace py = pybind11;

// ============================================================================
// Funciones auxiliares de conversión NumPy <-> C++
// ============================================================================

template<typename T>
std::vector<std::vector<T>> numpy_to_vector_2d(py::array_t<T> arr) {
    auto buf = arr.unchecked<2>();
    std::vector<std::vector<T>> result(buf.shape(0));
    for (py::ssize_t i = 0; i < buf.shape(0); i++) {
        result[i].resize(buf.shape(1));
        for (py::ssize_t j = 0; j < buf.shape(1); j++) {
            result[i][j] = buf(i, j);
        }
    }
    return result;
}

template<typename T>
std::vector<T> numpy_to_vector_1d(py::array_t<T> arr) {
    auto buf = arr.unchecked<1>();
    std::vector<T> result(buf.shape(0));
    for (py::ssize_t i = 0; i < buf.shape(0); i++) {
        result[i] = buf(i);
    }
    return result;
}

// ============================================================================
// Módulo Python
// ============================================================================

PYBIND11_MODULE(lsbp_laplace_cpp, m) {
    m.doc() = "LSBP Laplace C++ accelerated functions - Version 0.1.0\n\n"
              "Optimized implementations for Laplace kernel:\n"
              "  - compute_eta: Calculate η_h(x)\n"
              "  - compute_weights: Calculate weights via logit stick-breaking\n"
              "  - update_lambda_latent: Update latent mixing variables λ_ih\n"
              "  - update_assignments: Update cluster assignments z_i\n"
              "  - update_atoms: Update atoms θ_h = (μ_h, b_h)\n"
              "  - update_alpha: Update α_h with Metropolis-Hastings\n"
              "  - update_psi: Update ψ_{hj} with Metropolis-Hastings\n"
              "  - update_ell: Update ℓ_{hj} with discrete sampling";
    
    // ========================================================================
    // 1. compute_eta
    // ========================================================================
    m.def("compute_eta", 
        [](py::array_t<double> X_batch,
           py::array_t<double> alpha,
           py::array_t<double> psi,
           py::array_t<int> ell,
           py::array_t<double> ell_grid) {
            
            return lsbp_laplace::compute_eta(
                numpy_to_vector_2d<double>(X_batch),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid)
            );
        },
        py::arg("X_batch"),
        py::arg("alpha"),
        py::arg("psi"),
        py::arg("ell"),
        py::arg("ell_grid"),
        "Calculate η_h(x) = α_h - Σ_j ψ_{hj}|x_j - ℓ_{hj}|"
    );
    
    // ========================================================================
    // 2. compute_weights
    // ========================================================================
    m.def("compute_weights", 
        [](py::array_t<double> X_normalized,
           py::array_t<double> alpha,
           py::array_t<double> psi,
           py::array_t<int> ell,
           py::array_t<double> ell_grid) {
            
            return lsbp_laplace::compute_weights(
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid)
            );
        },
        py::arg("X_normalized"),
        py::arg("alpha"),
        py::arg("psi"),
        py::arg("ell"),
        py::arg("ell_grid"),
        "Calculate mixture weights w_h(x) using logit stick-breaking"
    );
    
    // ========================================================================
    // 3. update_lambda_latent
    // ========================================================================
    m.def("update_lambda_latent",
        [](py::array_t<int> z,
           py::array_t<double> y_normalized,
           py::array_t<double> theta_mu,
           py::array_t<double> theta_b,
           py::array_t<double> lambda_current,
           int H) {
            
            return lsbp_laplace::update_lambda_latent(
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_1d<double>(y_normalized),
                numpy_to_vector_1d<double>(theta_mu),
                numpy_to_vector_1d<double>(theta_b),
                numpy_to_vector_2d<double>(lambda_current),
                H
            );
        },
        py::arg("z"),
        py::arg("y_normalized"),
        py::arg("theta_mu"),
        py::arg("theta_b"),
        py::arg("lambda_current"),
        py::arg("H"),
        "Update latent mixing variables λ_ih ~ InverseGaussian"
    );
    
    // ========================================================================
    // 4. update_assignments
    // ========================================================================
    m.def("update_assignments",
        [](py::array_t<double> u,
           py::array_t<double> w,
           py::array_t<double> y_normalized,
           py::array_t<double> theta_mu,
           py::array_t<double> theta_b,
           py::array_t<int> z_current) {
            
            return lsbp_laplace::update_assignments(
                numpy_to_vector_1d<double>(u),
                numpy_to_vector_2d<double>(w),
                numpy_to_vector_1d<double>(y_normalized),
                numpy_to_vector_1d<double>(theta_mu),
                numpy_to_vector_1d<double>(theta_b),
                numpy_to_vector_1d<int>(z_current)
            );
        },
        py::arg("u"),
        py::arg("w"),
        py::arg("y_normalized"),
        py::arg("theta_mu"),
        py::arg("theta_b"),
        py::arg("z_current"),
        "Update cluster assignments z_i given slice variables u_i"
    );
    
    // ========================================================================
    // 5. update_atoms
    // ========================================================================
    m.def("update_atoms",
        [](py::array_t<int> z,
           py::array_t<double> y_normalized,
           py::array_t<double> lambda_latent,
           py::array_t<double> theta_mu_current,
           py::array_t<double> theta_b_current,
           double mu0,
           double tau0,
           double a0,
           double beta0,
           int H) {
            
            return lsbp_laplace::update_atoms(
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_1d<double>(y_normalized),
                numpy_to_vector_2d<double>(lambda_latent),
                numpy_to_vector_1d<double>(theta_mu_current),
                numpy_to_vector_1d<double>(theta_b_current),
                mu0,
                tau0,
                a0,
                beta0,
                H
            );
        },
        py::arg("z"),
        py::arg("y_normalized"),
        py::arg("lambda_latent"),
        py::arg("theta_mu_current"),
        py::arg("theta_b_current"),
        py::arg("mu0"),
        py::arg("tau0"),
        py::arg("a0"),
        py::arg("beta0"),
        py::arg("H"),
        "Update atoms θ_h = (μ_h, b_h) using Gibbs sampling"
    );
    
    // ========================================================================
    // 6. update_alpha
    // ========================================================================
    m.def("update_alpha",
        [](py::array_t<double> alpha_current,
           py::array_t<int> z,
           py::array_t<double> X_normalized,
           py::array_t<double> psi,
           py::array_t<int> ell,
           py::array_t<double> ell_grid,
           double mu,
           double mh_scale) {
            
            return lsbp_laplace::update_alpha(
                numpy_to_vector_1d<double>(alpha_current),
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid),
                mu,
                mh_scale
            );
        },
        py::arg("alpha_current"),
        py::arg("z"),
        py::arg("X_normalized"),
        py::arg("psi"),
        py::arg("ell"),
        py::arg("ell_grid"),
        py::arg("mu"),
        py::arg("mh_scale"),
        "Update α_h with Metropolis-Hastings"
    );
    
    // ========================================================================
    // 7. update_psi
    // ========================================================================
    m.def("update_psi",
        [](py::array_t<double> psi_current,
           py::array_t<int> z,
           py::array_t<double> alpha,
           py::array_t<double> X_normalized,
           py::array_t<int> ell,
           py::array_t<double> ell_grid,
           double mu_psi,
           double tau_psi_inv,
           double mh_scale) {
            
            return lsbp_laplace::update_psi(
                numpy_to_vector_2d<double>(psi_current),
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid),
                mu_psi,
                tau_psi_inv,
                mh_scale
            );
        },
        py::arg("psi_current"),
        py::arg("z"),
        py::arg("alpha"),
        py::arg("X_normalized"),
        py::arg("ell"),
        py::arg("ell_grid"),
        py::arg("mu_psi"),
        py::arg("tau_psi_inv"),
        py::arg("mh_scale"),
        "Update ψ_{hj} with Metropolis-Hastings"
    );
    
    // ========================================================================
    // 8. update_ell
    // ========================================================================
    m.def("update_ell",
        [](py::array_t<int> ell_current,
           py::array_t<int> z,
           py::array_t<double> alpha,
           py::array_t<double> psi,
           py::array_t<double> X_normalized,
           py::array_t<double> ell_grid,
           int n_grid) {
            
            return lsbp_laplace::update_ell(
                numpy_to_vector_2d<int>(ell_current),
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_2d<double>(ell_grid),
                n_grid
            );
        },
        py::arg("ell_current"),
        py::arg("z"),
        py::arg("alpha"),
        py::arg("psi"),
        py::arg("X_normalized"),
        py::arg("ell_grid"),
        py::arg("n_grid"),
        "Update ℓ_{hj} with discrete sampling"
    );
    
    // ========================================================================
    // Exponer estructuras de resultado
    // ========================================================================
    py::class_<lsbp_laplace::EtaResult>(m, "EtaResult")
        .def_readonly("eta", &lsbp_laplace::EtaResult::eta)
        .def_readonly("n_batch", &lsbp_laplace::EtaResult::n_batch)
        .def_readonly("H", &lsbp_laplace::EtaResult::H);
    
    py::class_<lsbp_laplace::WeightsResult>(m, "WeightsResult")
        .def_readonly("weights", &lsbp_laplace::WeightsResult::weights)
        .def_readonly("n", &lsbp_laplace::WeightsResult::n)
        .def_readonly("H", &lsbp_laplace::WeightsResult::H);
    
    py::class_<lsbp_laplace::LambdaResult>(m, "LambdaResult")
        .def_readonly("lambda_latent", &lsbp_laplace::LambdaResult::lambda_latent)
        .def_readonly("n", &lsbp_laplace::LambdaResult::n)
        .def_readonly("H", &lsbp_laplace::LambdaResult::H);
    
    py::class_<lsbp_laplace::AtomsResult>(m, "AtomsResult")
        .def_readonly("theta_mu", &lsbp_laplace::AtomsResult::theta_mu)
        .def_readonly("theta_b", &lsbp_laplace::AtomsResult::theta_b)
        .def_readonly("H", &lsbp_laplace::AtomsResult::H);
    
    py::class_<lsbp_laplace::AlphaUpdateResult>(m, "AlphaUpdateResult")
        .def_readonly("alpha", &lsbp_laplace::AlphaUpdateResult::alpha)
        .def_readonly("acceptance", &lsbp_laplace::AlphaUpdateResult::acceptance)
        .def_readonly("H", &lsbp_laplace::AlphaUpdateResult::H);
    
    py::class_<lsbp_laplace::PsiUpdateResult>(m, "PsiUpdateResult")
        .def_readonly("psi", &lsbp_laplace::PsiUpdateResult::psi)
        .def_readonly("acceptance", &lsbp_laplace::PsiUpdateResult::acceptance)
        .def_readonly("H", &lsbp_laplace::PsiUpdateResult::H)
        .def_readonly("p", &lsbp_laplace::PsiUpdateResult::p);
    
    py::class_<lsbp_laplace::EllUpdateResult>(m, "EllUpdateResult")
        .def_readonly("ell", &lsbp_laplace::EllUpdateResult::ell)
        .def_readonly("H", &lsbp_laplace::EllUpdateResult::H)
        .def_readonly("p", &lsbp_laplace::EllUpdateResult::p);
    
    // ========================================================================
    // Información de versión
    // ========================================================================
    m.attr("__version__") = "0.1.0";
    m.attr("__author__") = "Juan Ceballos";
    m.attr("__kernel__") = "Laplace";
    m.attr("__functions__") = py::make_tuple(
        "compute_eta", "compute_weights", "update_lambda_latent",
        "update_assignments", "update_atoms", "update_alpha",
        "update_psi", "update_ell"
    );
}