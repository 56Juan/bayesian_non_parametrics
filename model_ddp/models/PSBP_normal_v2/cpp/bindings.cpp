// PSBP_normal/cpp/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "psbp_core.hpp"

namespace py = pybind11;

// ============================================================================
// Funciones auxiliares de conversión NumPy <-> C++ (SEGURO)
// ============================================================================

template <typename T>
std::vector<std::vector<T>>
numpy_to_vector_2d(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    auto buf = arr.template unchecked<2>();
    std::vector<std::vector<T>> result(buf.shape(0),
                                       std::vector<T>(buf.shape(1)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        for (py::ssize_t j = 0; j < buf.shape(1); ++j) {
            result[i][j] = buf(i, j);
        }
    }
    return result;
}

template <typename T>
std::vector<T>
numpy_to_vector_1d(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    auto buf = arr.template unchecked<1>();
    std::vector<T> result(buf.shape(0));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        result[i] = buf(i);
    }
    return result;
}

// ============================================================================
// Módulo Python para PSBP
// ============================================================================

PYBIND11_MODULE(psbp_cpp, m) {
    m.doc() =
        "PSBP C++ accelerated functions - Probit Stick-Breaking Process\n"
        "Optimized implementations for PSBP with Normal kernel";

    // ========================================================================
    // 1. compute_eta
    // ========================================================================

    m.def(
        "compute_eta",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& X_batch,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& alpha,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& psi,
           const py::array_t<int,    py::array::c_style | py::array::forcecast>& gamma,
           const py::array_t<int,    py::array::c_style | py::array::forcecast>& ell,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& ell_grid) {

            return psbp::compute_eta(
                numpy_to_vector_2d<double>(X_batch),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(gamma),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid)
            );
        }
    );

    // ========================================================================
    // 2. compute_weights_probit
    // ========================================================================

    m.def(
        "compute_weights_probit",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& X,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& alpha,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& psi,
           const py::array_t<int,    py::array::c_style | py::array::forcecast>& gamma,
           const py::array_t<int,    py::array::c_style | py::array::forcecast>& ell,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& ell_grid) {

            return psbp::compute_weights_probit(
                numpy_to_vector_2d<double>(X),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(gamma),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid)
            );
        }
    );

    // ========================================================================
    // 3. update_u_latent
    // ========================================================================

    m.def(
        "update_u_latent",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& u_latent,
           const py::array_t<int,    py::array::c_style | py::array::forcecast>& z,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& X,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& alpha,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& psi,
           const py::array_t<int,    py::array::c_style | py::array::forcecast>& gamma,
           const py::array_t<int,    py::array::c_style | py::array::forcecast>& ell,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& ell_grid) {

            return psbp::update_u_latent(
                numpy_to_vector_2d<double>(u_latent),
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_2d<double>(X),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(gamma),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid)
            );
        }
    );

    // ========================================================================
    // 4. update_assignments (slice)
    // ========================================================================

    m.def(
        "update_assignments",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& u_slice,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& w,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& y,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& mu,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& sigma2,
           const py::array_t<int,    py::array::c_style | py::array::forcecast>& z) {

            return psbp::update_assignments_slice(
                numpy_to_vector_1d<double>(u_slice),
                numpy_to_vector_2d<double>(w),
                numpy_to_vector_1d<double>(y),
                numpy_to_vector_1d<double>(mu),
                numpy_to_vector_1d<double>(sigma2),
                numpy_to_vector_1d<int>(z)
            );
        }
    );
    
    // ========================================================================
    // 5. update_atoms (SIN CAMBIOS)
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
            
            return psbp::update_atoms(
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
        
        Returns
        -------
        AtomsResult
            Updated means and variances
        )pbdoc"
    );
    
    // ========================================================================
    // 6. update_ell (MODIFICADO PARA PSBP)
    // ========================================================================
    m.def("update_ell",
        [](py::array_t<int> ell_current,
           py::array_t<int> z,
           py::array_t<double> X_normalized,
           py::array_t<double> alpha,
           py::array_t<double> psi,
           py::array_t<int> gamma,
           py::array_t<double> u_latent,
           py::array_t<double> ell_grid,
           int H,
           int n_grid) {
            
            return psbp::update_ell_probit(
                numpy_to_vector_2d<int>(ell_current),
                numpy_to_vector_1d<int>(z),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(gamma),
                numpy_to_vector_2d<double>(u_latent),
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
        py::arg("gamma"),
        py::arg("u_latent"),
        py::arg("ell_grid"),
        py::arg("H"),
        py::arg("n_grid"),
        R"pbdoc(
        Update kernel location parameters ℓ_hj for probit model.
        
        Uses latent variables u_{ih} instead of assignments z_i directly.
        
        Returns
        -------
        UpdateEllResult
            Updated ell matrix and acceptance info
        )pbdoc"
    );
    
// 7. update_psi_gamma
    m.def("update_psi_gamma",
        [](py::array_t<double> psi_current,
           py::array_t<int> gamma_current,
           py::array_t<double> mu_psi,
           py::array_t<double> tau_psi,
           py::array_t<double> kappa,
           py::array_t<double> u_latent,
           py::array_t<double> X_normalized,
           py::array_t<double> alpha,
           py::array_t<int> ell,
           py::array_t<double> ell_grid,
           py::array_t<int> z,  //  AGREGAR
           double mh_scale_psi,
           bool psi_positive,
           int H) {

            return psbp::update_psi_gamma(
                numpy_to_vector_2d<double>(psi_current),
                numpy_to_vector_2d<int>(gamma_current),
                numpy_to_vector_1d<double>(mu_psi),
                numpy_to_vector_1d<double>(tau_psi),
                numpy_to_vector_1d<double>(kappa),
                numpy_to_vector_2d<double>(u_latent),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid),
                numpy_to_vector_1d<int>(z),  //  AGREGAR
                mh_scale_psi,
                psi_positive,
                H
            );
        },
        //  AGREGAR ARGUMENTO
        py::arg("psi_current"), py::arg("gamma_current"), py::arg("mu_psi"),
        py::arg("tau_psi"), py::arg("kappa"), py::arg("u_latent"),
        py::arg("X_normalized"), py::arg("alpha"), py::arg("ell"),
        py::arg("ell_grid"), py::arg("z"), py::arg("mh_scale_psi"),
        py::arg("psi_positive"), py::arg("H")
    );

    // 8. update_alpha_probit
    m.def("update_alpha_probit",
        [](py::array_t<double> alpha_current,
           py::array_t<double> u_latent,
           py::array_t<double> X_normalized,
           py::array_t<double> psi,
           py::array_t<int> gamma,
           py::array_t<int> ell,
           py::array_t<double> ell_grid,
           py::array_t<int> z,  //  AGREGAR
           double mu_alpha,
           double mh_scale,
           int H) {

            return psbp::update_alpha_probit(
                numpy_to_vector_1d<double>(alpha_current),
                numpy_to_vector_2d<double>(u_latent),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_2d<double>(psi),
                numpy_to_vector_2d<int>(gamma),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid),
                numpy_to_vector_1d<int>(z),  //  AGREGAR
                mu_alpha,
                mh_scale,
                H
            );
        },
        //  AGREGAR ARGUMENTO
        py::arg("alpha_current"), py::arg("u_latent"), py::arg("X_normalized"),
        py::arg("psi"), py::arg("gamma"), py::arg("ell"), py::arg("ell_grid"),
        py::arg("z"), py::arg("mu_alpha"), py::arg("mh_scale"), py::arg("H")
    );
    
    // ========================================================================
    // 9. update_gamma_psi (ALTERNATIVA - GIBBS PARA ψ)
    // ========================================================================
    // 9. update_gamma_psi
    m.def("update_gamma_psi",
        [](py::array_t<double> psi_current,
           py::array_t<int> gamma_current,
           py::array_t<double> u_latent,
           py::array_t<double> X_normalized,
           py::array_t<double> alpha,
           py::array_t<int> ell,
           py::array_t<double> ell_grid,
           py::array_t<int> z,  // ✅ AGREGAR
           py::array_t<double> mu_psi,
           py::array_t<double> tau_psi,
           py::array_t<double> kappa,
           bool psi_positive,
           int H) {
            
            return psbp::update_gamma_psi_gibbs(
                numpy_to_vector_2d<double>(psi_current),
                numpy_to_vector_2d<int>(gamma_current),
                numpy_to_vector_2d<double>(u_latent),
                numpy_to_vector_2d<double>(X_normalized),
                numpy_to_vector_1d<double>(alpha),
                numpy_to_vector_2d<int>(ell),
                numpy_to_vector_2d<double>(ell_grid),
                numpy_to_vector_1d<int>(z),  // ✅ AGREGAR
                numpy_to_vector_1d<double>(mu_psi),
                numpy_to_vector_1d<double>(tau_psi),
                numpy_to_vector_1d<double>(kappa),
                psi_positive,
                H
            );
        },
        py::arg("psi_current"),
        py::arg("gamma_current"),
        py::arg("u_latent"),
        py::arg("X_normalized"),
        py::arg("alpha"),
        py::arg("ell"),
        py::arg("ell_grid"),
        py::arg("z"),  // ✅ AGREGAR
        py::arg("mu_psi"),
        py::arg("tau_psi"),
        py::arg("kappa"),
        py::arg("psi_positive"),
        py::arg("H"),
        R"pbdoc(
        Update γ and ψ using Gibbs sampling (truncated normal posterior).
        
        This is an alternative to update_psi_gamma that uses exact Gibbs
        sampling for ψ instead of Metropolis-Hastings.
        
        Returns
        -------
        PsiGammaResult
            Updated psi, gamma
        )pbdoc"
    );
    
    // ========================================================================
    // EXPONER ESTRUCTURAS DE RESULTADO
    // ========================================================================
    
    // Estructuras existentes
    py::class_<psbp::EtaResult>(m, "EtaResult")
        .def_readonly("eta", &psbp::EtaResult::eta)
        .def_readonly("n_batch", &psbp::EtaResult::n_batch)
        .def_readonly("H", &psbp::EtaResult::H);
    
    py::class_<psbp::WeightsResult>(m, "WeightsResult")
        .def_readonly("weights", &psbp::WeightsResult::weights)
        .def_readonly("n", &psbp::WeightsResult::n)
        .def_readonly("H", &psbp::WeightsResult::H);
    
    py::class_<psbp::AtomsResult>(m, "AtomsResult")
        .def_readonly("theta_mu", &psbp::AtomsResult::theta_mu)
        .def_readonly("theta_sigma2", &psbp::AtomsResult::theta_sigma2)
        .def_readonly("H", &psbp::AtomsResult::H);
    
    // Nuevas estructuras para PSBP
    py::class_<psbp::ULatentResult>(m, "ULatentResult")
        .def_readonly("u_latent", &psbp::ULatentResult::u_latent)
        .def_readonly("n", &psbp::ULatentResult::n)
        .def_readonly("H", &psbp::ULatentResult::H);
    
    py::class_<psbp::UpdateEllResult>(m, "UpdateEllResult")
        .def_readonly("ell", &psbp::UpdateEllResult::ell)
        .def_readonly("H", &psbp::UpdateEllResult::H)
        .def_readonly("p", &psbp::UpdateEllResult::p);
    
    py::class_<psbp::UpdateAlphaResult>(m, "UpdateAlphaResult")
        .def_readonly("alpha", &psbp::UpdateAlphaResult::alpha)
        .def_readonly("acceptances", &psbp::UpdateAlphaResult::acceptances)
        .def_readonly("H", &psbp::UpdateAlphaResult::H);
    
    py::class_<psbp::PsiGammaResult>(m, "PsiGammaResult")
        .def_readonly("psi", &psbp::PsiGammaResult::psi)
        .def_readonly("gamma", &psbp::PsiGammaResult::gamma)
        .def_readonly("acceptances_psi", &psbp::PsiGammaResult::acceptances_psi)
        .def_readonly("H", &psbp::PsiGammaResult::H)
        .def_readonly("p", &psbp::PsiGammaResult::p);
    
    // ========================================================================
    // INFORMACIÓN DE VERSIÓN
    // ========================================================================
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Adapted for PSBP Normal Model";
    m.attr("__functions__") = py::make_tuple(
        "compute_eta", "compute_weights_probit", "update_u_latent",
        "update_assignments", "update_atoms", "update_ell",
        "update_psi_gamma", "update_alpha_probit", "update_gamma_psi"
    );
}