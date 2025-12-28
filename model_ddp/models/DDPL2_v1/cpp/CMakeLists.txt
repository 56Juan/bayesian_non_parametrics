cmake_minimum_required(VERSION 3.15)
project(ddp2_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar pybind11
find_package(pybind11 REQUIRED)

# Eigen (header-only)
find_package(Eigen3 CONFIG REQUIRED)

# Encontrar OpenMP (opcional, para paralelización)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found - enabling parallel computation")
else()
    message(STATUS "OpenMP not found - using sequential computation")
endif()

# IMPORTANTE: El nombre debe ser ddp2_cpp para coincidir con el PYBIND11_MODULE
pybind11_add_module(ddp2_cpp 
    bindings.cpp
    lsbp_core.cpp
)

target_link_libraries(ddp2_cpp
    PRIVATE
        Eigen3::Eigen
)

# Optimizaciones
target_compile_options(ddp2_cpp PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Link OpenMP si está disponible
if(OpenMP_CXX_FOUND)
    target_link_libraries(ddp2_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Propiedades adicionales
set_target_properties(ddp2_cpp PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    INTERPROCEDURAL_OPTIMIZATION TRUE
)