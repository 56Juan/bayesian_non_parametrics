#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include "lsbp_core.hpp"

namespace py = pybind11;

// IMPORTANTE: El nombre debe ser ddp2_cpp para que coincida con tu import
PYBIND11_MODULE(ddp2_cpp, m) {
    m.doc() = "C++ accelerated core functions for DDP Linear Spline model";

    py::class_<LSBPCore>(m, "LSBPCore")
        .def(py::init<unsigned int>(), 
             py::arg("seed") = 42,
             "Initialize LSBPCore with random seed")
        
        .def("sample_slice_variables",
             &LSBPCore::sample_slice_variables,
             py::arg("z"),
             py::arg("w"),
             py::arg("n"),
             R"pbdoc(
                Sample slice variables for truncation.
                
                Parameters
                ----------
                z : list of int
                    Current cluster assignments
                w : list of float
                    Stick-breaking weights
                n : int
                    Number of observations
                
                Returns
                -------
                list of float
                    Slice variables u_i
             )pbdoc")
        
        .def("update_assignments",
             &LSBPCore::update_assignments,
             py::arg("y_normalized"),
             py::arg("design_matrix"),
             py::arg("lambda_h"),
             py::arg("xi_h"),
             py::arg("w"),
             py::arg("u"),
             py::arg("n"),
             py::arg("H"),
             py::arg("K"),
             R"pbdoc(
                Update cluster assignments with slice constraint.
                
                Parameters
                ----------
                y_normalized : ndarray (n, 1)
                    Normalized response variable
                design_matrix : ndarray (n, K)
                    Design matrix d(X)
                lambda_h : ndarray (H, K)
                    Mean coefficients
                xi_h : ndarray (H, K)
                    Log-variance coefficients
                w : list of float
                    Stick-breaking weights
                u : list of float
                    Slice variables
                n : int
                    Number of observations
                H : int
                    Number of clusters
                K : int
                    Dimension of coefficients
                
                Returns
                -------
                list of int
                    New cluster assignments
             )pbdoc")
        
        .def("update_lambda",
             &LSBPCore::update_lambda,
             py::arg("y_normalized"),
             py::arg("design_matrix"),
             py::arg("z"),
             py::arg("xi_h"),
             py::arg("mu_lambda"),
             py::arg("Sigma_lambda"),
             py::arg("n"),
             py::arg("H"),
             py::arg("K"),
             R"pbdoc(
                Update lambda coefficients using weighted least squares.
                
                Parameters
                ----------
                y_normalized : ndarray (n, 1)
                    Normalized response variable
                design_matrix : ndarray (n, K)
                    Design matrix d(X)
                z : list of int
                    Cluster assignments
                xi_h : ndarray (H, K)
                    Log-variance coefficients
                mu_lambda : ndarray (K,)
                    Prior mean for lambda
                Sigma_lambda : ndarray (K, K)
                    Prior covariance for lambda
                n : int
                    Number of observations
                H : int
                    Number of clusters
                K : int
                    Dimension of coefficients
                
                Returns
                -------
                ndarray (H, K)
                    Updated lambda coefficients
             )pbdoc")
        
        .def("update_xi",
             &LSBPCore::update_xi,
             py::arg("y_normalized"),
             py::arg("design_matrix"),
             py::arg("lambda_h"),
             py::arg("xi_h"),
             py::arg("z"),
             py::arg("mu_xi"),
             py::arg("Sigma_xi"),
             py::arg("scale"),
             py::arg("n"),
             py::arg("H"),
             py::arg("K"),
             R"pbdoc(
                Update xi coefficients using Metropolis-Hastings.
                
                Parameters
                ----------
                y_normalized : ndarray (n, 1)
                    Normalized response variable
                design_matrix : ndarray (n, K)
                    Design matrix d(X)
                lambda_h : ndarray (H, K)
                    Mean coefficients
                xi_h : ndarray (H, K)
                    Current log-variance coefficients
                z : list of int
                    Cluster assignments
                mu_xi : ndarray (K,)
                    Prior mean for xi
                Sigma_xi : ndarray (K, K)
                    Prior covariance for xi
                scale : float
                    Proposal scale
                n : int
                    Number of observations
                H : int
                    Number of clusters
                K : int
                    Dimension of coefficients
                
                Returns
                -------
                ndarray (H, K)
                    Updated xi coefficients
             )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}