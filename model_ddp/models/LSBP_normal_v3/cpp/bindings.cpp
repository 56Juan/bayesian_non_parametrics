// model_ddp/models/LSBP_normal_v3/cpp/bindings.cpp
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
    m.doc() = "LSBP C++ accelerated functions - Version 0.3.0\n\n"
              "Optimized implementations of:\n"
              "  - compute_eta: Calculate η_h(x)\n"
              "  - compute_weights: Calculate weights via logit stick-breaking\n"
              "  - update_assignments: Update cluster assignments z_i\n"
              "  - update_atoms: Update atoms θ_h = (μ_h, σ²_h)\n"
              "  - update_ell: Update kernel location parameters\n"
              "  - update_psi: Update dependency parameters (M-H)\n"
              "  - update_alpha: Update intercept parameters (M-H)";
    
    // ========================================================================
    // 1. compute_eta (EXISTENTE)
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
        )pbdoc"
    );
    
    // ========================================================================
    // 2. compute_weights (EXISTENTE)
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
        )pbdoc"
    );
    
    // ========================================================================
    // 3. update_assignments (EXISTENTE)
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
        )pbdoc"
    );
    
    // ========================================================================
    // 4. update_atoms (EXISTENTE)
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
        )pbdoc"
    );
    
    // ========================================================================
    // 5. update_ell (NUEVO)
    // ========================================================================
    m.def("update_ell",
        [](py::array_t<int> ell_current,
           py::array_t<int> z,
           py::array_t<double> X_normalized,
           py::array_t<double> alpha,
           py::array_t<double> psi,
           py::array_t<double> ell_grid,
           int H,
           int n_grid) {
            
            return lsbp::update_ell(
                numpy_to_vector_2d<int>(ell_current),
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<double>(ell_grid),
                H,
                n_grid
            );
        },
        py::arg("ell_current"),
        py::arg("z"),
        py::arg("X_normalized"),
        py::arg("alpha"),
        py::arg("psi"),
        py::arg("ell_grid"),
        py::arg("H"),
        py::arg("n_grid"),
        R"pbdoc(
        Update kernel location parameters ℓ_hj using discrete sampling.
        
        For each (h, j):
        - Evaluate log-likelihood for all grid positions
        - Sample from categorical distribution
        
        Parameters
        ----------
        ell_current : ndarray, shape (H, p)
            Current grid indices
        z : ndarray, shape (n,)
            Cluster assignments
        X_normalized : ndarray, shape (n, p)
            Normalized covariates
        alpha : ndarray, shape (H,)
            Intercepts
        psi : ndarray, shape (H, p)
            Decay parameters
        ell_grid : ndarray, shape (p, n_grid)
            Grid of possible locations
        H : int
            Number of clusters
        n_grid : int
            Grid size
            
        Returns
        -------
        UpdateEllResult
            New ell matrix
        )pbdoc"
    );
    
    // ========================================================================
    // 6. update_psi (NUEVO)
    // ========================================================================
    m.def("update_psi",
        [](py::array_t<double> psi_current,
           py::array_t<int> z,
           py::array_t<double> X_normalized,
           py::array_t<double> alpha,
           py::array_t<int> ell,
           py::array_t<double> ell_grid,
           double mu_psi,
           double tau_psi_inv,
           double mh_scale,
           int H) {
            
            return lsbp::update_psi(
                numpy_to_vector_2d<double>(psi_current),
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid),
                mu_psi,
                tau_psi_inv,
                mh_scale,
                H
            );
        },
        py::arg("psi_current"),
        py::arg("z"),
        py::arg("X_normalized"),
        py::arg("alpha"),
        py::arg("ell"),
        py::arg("ell_grid"),
        py::arg("mu_psi"),
        py::arg("tau_psi_inv"),
        py::arg("mh_scale"),
        py::arg("H"),
        R"pbdoc(
        Update dependency parameters ψ_hj using Metropolis-Hastings.
        
        For each (h, j):
        - Propose ψ' ~ N(ψ, mh_scale²)
        - Compute acceptance ratio
        - Update if accepted
        
        Parameters
        ----------
        psi_current : ndarray, shape (H, p)
            Current values
        z : ndarray, shape (n,)
            Cluster assignments
        X_normalized : ndarray, shape (n, p)
            Normalized covariates
        alpha : ndarray, shape (H,)
            Intercepts
        ell : ndarray, shape (H, p)
            Location indices
        ell_grid : ndarray, shape (p, n_grid)
            Location grid
        mu_psi : float
            Prior mean
        tau_psi_inv : float
            Prior variance
        mh_scale : float
            Proposal standard deviation
        H : int
            Number of clusters
            
        Returns
        -------
        UpdatePsiResult
            New psi matrix and acceptance indicators
        )pbdoc"
    );
    
    // ========================================================================
    // 7. update_alpha (NUEVO)
    // ========================================================================
    m.def("update_alpha",
        [](py::array_t<double> alpha_current,
           py::array_t<int> z,
           py::array_t<double> X_normalized,
           py::array_t<double> psi,
           py::array_t<int> ell,
           py::array_t<double> ell_grid,
           double mu,
           double mh_scale,
           int H) {
            
            return lsbp::update_alpha(
                numpy_to_vector_1d<double>(alpha_current),
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid),
                mu,
                mh_scale,
                H
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
        py::arg("H"),
        R"pbdoc(
        Update intercept parameters α_h using Metropolis-Hastings.
        
        For each h:
        - Propose α' ~ N(α, mh_scale²)
        - Compute acceptance ratio
        - Update if accepted
        
        Parameters
        ----------
        alpha_current : ndarray, shape (H,)
            Current values
        z : ndarray, shape (n,)
            Cluster assignments
        X_normalized : ndarray, shape (n, p)
            Normalized covariates
        psi : ndarray, shape (H, p)
            Decay parameters
        ell : ndarray, shape (H, p)
            Location indices
        ell_grid : ndarray, shape (p, n_grid)
            Location grid
        mu : float
            Prior mean
        mh_scale : float
            Proposal standard deviation
        H : int
            Number of clusters
            
        Returns
        -------
        UpdateAlphaResult
            New alpha vector and acceptance indicators
        )pbdoc"
    );
    
    // ========================================================================
    // Exponer estructuras de resultado EXISTENTES
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
    // Exponer estructuras de resultado NUEVAS
    // ========================================================================
    py::class_<lsbp::UpdateEllResult>(m, "UpdateEllResult",
        "Result structure for update_ell")
        .def_readonly("ell", &lsbp::UpdateEllResult::ell,
                     "Updated location indices (H, p)")
        .def_readonly("H", &lsbp::UpdateEllResult::H,
                     "Number of clusters")
        .def_readonly("p", &lsbp::UpdateEllResult::p,
                     "Number of covariates")
        .def("__repr__", [](const lsbp::UpdateEllResult &r) {
            return "<UpdateEllResult: H=" + std::to_string(r.H) + 
                   ", p=" + std::to_string(r.p) + ">";
        });
    
    py::class_<lsbp::UpdatePsiResult>(m, "UpdatePsiResult",
        "Result structure for update_psi")
        .def_readonly("psi", &lsbp::UpdatePsiResult::psi,
                     "Updated psi matrix (H, p)")
        .def_readonly("acceptances", &lsbp::UpdatePsiResult::acceptances,
                     "Acceptance indicators")
        .def_readonly("H", &lsbp::UpdatePsiResult::H,
                     "Number of clusters")
        .def_readonly("p", &lsbp::UpdatePsiResult::p,
                     "Number of covariates")
        .def("__repr__", [](const lsbp::UpdatePsiResult &r) {
            return "<UpdatePsiResult: H=" + std::to_string(r.H) + 
                   ", p=" + std::to_string(r.p) + ">";
        });
    
    py::class_<lsbp::UpdateAlphaResult>(m, "UpdateAlphaResult",
        "Result structure for update_alpha")
        .def_readonly("alpha", &lsbp::UpdateAlphaResult::alpha,
                     "Updated alpha vector (H,)")
        .def_readonly("acceptances", &lsbp::UpdateAlphaResult::acceptances,
                     "Acceptance indicators")
        .def_readonly("H", &lsbp::UpdateAlphaResult::H,
                     "Number of clusters")
        .def("__repr__", [](const lsbp::UpdateAlphaResult &r) {
            return "<UpdateAlphaResult: H=" + std::to_string(r.H) + ">";
        });
    
    // ========================================================================
    // Información de versión y metadatos
    // ========================================================================
    m.attr("__version__") = "0.3.0";
    m.attr("__author__") = "Juan Ceballos";
    m.attr("__functions__") = py::make_tuple(
        "compute_eta", "compute_weights", 
        "update_assignments", "update_atoms",
        "update_ell", "update_psi", "update_alpha"
    );
}