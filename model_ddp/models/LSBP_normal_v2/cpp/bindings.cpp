// model_ddp/models/LSBP_normal_v2/cpp/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "lsbp_core.hpp"

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

PYBIND11_MODULE(lsbp_cpp, m) {
    m.doc() = "LSBP C++ accelerated functions - Version 0.2.0\n\n"
              "Optimized implementations of:\n"
              "  - compute_eta: Calculate η_h(x)\n"
              "  - compute_weights: Calculate weights via logit stick-breaking\n"
              "  - update_assignments: Update cluster assignments z_i\n"
              "  - update_atoms: Update atoms θ_h = (μ_h, σ²_h)";
    
    // ========================================================================
    // 1. compute_eta
    // ========================================================================
    m.def("compute_eta", 
        [](py::array_t<double> X_batch,
           py::array_t<double> alpha,
           py::array_t<double> psi,
           py::array_t<int> ell,
           py::array_t<double> ell_grid) {
            
            return lsbp::compute_eta(
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
        R"pbdoc(
        Calculate η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}| for all clusters.
        
        Parameters
        ----------
        X_batch : ndarray, shape (n_batch, p)
            Covariate matrix (normalized)
        alpha : ndarray, shape (H,)
            Intercept parameters
        psi : ndarray, shape (H, p)
            Decay parameters
        ell : ndarray, shape (H, p)
            Grid indices for cluster centers
        ell_grid : ndarray, shape (p, n_grid)
            Grid of possible center values
            
        Returns
        -------
        EtaResult
            Object with eta matrix (n_batch, H)
        )pbdoc"
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
            
            return lsbp::compute_weights(
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
        R"pbdoc(
        Calculate mixture weights w_h(x) using logit stick-breaking.
        
        w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
        v_h(x) = expit(η_h(x))
        
        Parameters
        ----------
        X_normalized : ndarray, shape (n, p)
            Normalized covariate matrix
        alpha : ndarray, shape (H,)
            Intercept parameters
        psi : ndarray, shape (H, p)
            Decay parameters
        ell : ndarray, shape (H, p)
            Grid indices for cluster centers
        ell_grid : ndarray, shape (p, n_grid)
            Grid of possible center values
            
        Returns
        -------
        WeightsResult
            Object with weights matrix (n, H)
        )pbdoc"
    );
    
    // ========================================================================
    // 3. update_assignments
    // ========================================================================
    m.def("update_assignments",
        [](py::array_t<double> u,
           py::array_t<double> w,
           py::array_t<double> y_normalized,
           py::array_t<double> theta_mu,
           py::array_t<double> theta_sigma2,
           py::array_t<int> z_current) {
            
            return lsbp::update_assignments(
                numpy_to_vector_1d<double>(u),
                numpy_to_vector_2d<double>(w),
                numpy_to_vector_1d<double>(y_normalized),
                numpy_to_vector_1d<double>(theta_mu),
                numpy_to_vector_1d<double>(theta_sigma2),
                numpy_to_vector_1d<int>(z_current)
            );
        },
        py::arg("u"),
        py::arg("w"),
        py::arg("y_normalized"),
        py::arg("theta_mu"),
        py::arg("theta_sigma2"),
        py::arg("z_current"),
        R"pbdoc(
        Update cluster assignments z_i given slice variables u_i.
        
        For each observation i:
        - Find active clusters: {h : w_h(x_i) > u_i}
        - Sample z_i ~ Categorical based on likelihood
        
        Parameters
        ----------
        u : ndarray, shape (n,)
            Slice variables
        w : ndarray, shape (n, H)
            Mixture weights
        y_normalized : ndarray, shape (n,)
            Normalized response
        theta_mu : ndarray, shape (H,)
            Cluster means
        theta_sigma2 : ndarray, shape (H,)
            Cluster variances
        z_current : ndarray, shape (n,)
            Current assignments (not used, for API consistency)
            
        Returns
        -------
        z_new : list of int
            New cluster assignments
        )pbdoc"
    );
    
    // ========================================================================
    // 4. update_atoms
    // ========================================================================
    m.def("update_atoms",
        [](py::array_t<int> z,
           py::array_t<double> y_normalized,
           py::array_t<double> theta_mu_current,
           py::array_t<double> theta_sigma2_current,
           double mu0,
           double kappa0,
           double a0,
           double b0,
           int H) {
            
            return lsbp::update_atoms(
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_1d<double>(y_normalized),
                numpy_to_vector_1d<double>(theta_mu_current),
                numpy_to_vector_1d<double>(theta_sigma2_current),
                mu0,
                kappa0,
                a0,
                b0,
                H
            );
        },
        py::arg("z"),
        py::arg("y_normalized"),
        py::arg("theta_mu_current"),
        py::arg("theta_sigma2_current"),
        py::arg("mu0"),
        py::arg("kappa0"),
        py::arg("a0"),
        py::arg("b0"),
        py::arg("H"),
        R"pbdoc(
        Update atoms θ_h = (μ_h, σ²_h) using posterior or prior.
        
        For non-empty clusters: Use Normal-Inverse-Gamma posterior
        For empty clusters: Sample from prior G₀
        
        Parameters
        ----------
        z : ndarray, shape (n,)
            Cluster assignments
        y_normalized : ndarray, shape (n,)
            Normalized response
        theta_mu_current : ndarray, shape (H,)
            Current cluster means (not used)
        theta_sigma2_current : ndarray, shape (H,)
            Current cluster variances (not used)
        mu0 : float
            Prior mean
        kappa0 : float
            Prior precision parameter
        a0 : float
            Prior shape parameter for σ²
        b0 : float
            Prior scale parameter for σ²
        H : int
            Number of clusters
            
        Returns
        -------
        AtomsResult
            Object with updated theta_mu and theta_sigma2
        )pbdoc"
    );
    
    // ========================================================================
    // Exponer estructuras de resultado
    // ========================================================================
    py::class_<lsbp::EtaResult>(m, "EtaResult",
        "Result structure for compute_eta")
        .def_readonly("eta", &lsbp::EtaResult::eta,
                     "Linear predictor matrix (n_batch, H)")
        .def_readonly("n_batch", &lsbp::EtaResult::n_batch,
                     "Number of observations")
        .def_readonly("H", &lsbp::EtaResult::H,
                     "Number of clusters")
        .def("__repr__", [](const lsbp::EtaResult &r) {
            return "<EtaResult: n_batch=" + std::to_string(r.n_batch) + 
                   ", H=" + std::to_string(r.H) + ">";
        });
    
    py::class_<lsbp::WeightsResult>(m, "WeightsResult",
        "Result structure for compute_weights")
        .def_readonly("weights", &lsbp::WeightsResult::weights,
                     "Mixture weights matrix (n, H)")
        .def_readonly("n", &lsbp::WeightsResult::n,
                     "Number of observations")
        .def_readonly("H", &lsbp::WeightsResult::H,
                     "Number of clusters")
        .def("__repr__", [](const lsbp::WeightsResult &r) {
            return "<WeightsResult: n=" + std::to_string(r.n) + 
                   ", H=" + std::to_string(r.H) + ">";
        });
    
    py::class_<lsbp::AtomsResult>(m, "AtomsResult",
        "Result structure for update_atoms")
        .def_readonly("theta_mu", &lsbp::AtomsResult::theta_mu,
                     "Cluster means (H,)")
        .def_readonly("theta_sigma2", &lsbp::AtomsResult::theta_sigma2,
                     "Cluster variances (H,)")
        .def_readonly("H", &lsbp::AtomsResult::H,
                     "Number of clusters")
        .def("__repr__", [](const lsbp::AtomsResult &r) {
            return "<AtomsResult: H=" + std::to_string(r.H) + ">";
        });
    
    // ========================================================================
    // Información de versión
    // ========================================================================
    m.attr("__version__") = "0.2.0";
    m.attr("__author__") = "Juan Ceballos";
    m.attr("__functions__") = py::make_tuple(
        "compute_eta", "compute_weights", 
        "update_assignments", "update_atoms"
    );
}