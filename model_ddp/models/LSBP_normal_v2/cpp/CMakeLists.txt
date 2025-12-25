# model_ddp/models/LSBP_normal_v2/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(lsbp_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar pybind11
find_package(pybind11 REQUIRED)

# Encontrar OpenMP (opcional, para paralelizacion)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found - enabling parallel computation")
else()
    message(STATUS "OpenMP not found - using sequential computation")
endif()

# Crear módulo Python
pybind11_add_module(lsbp_cpp 
    bindings.cpp
    lsbp_core.cpp
)

# Optimizaciones
target_compile_options(lsbp_cpp PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Link OpenMP si está disponible
if(OpenMP_CXX_FOUND)
    target_link_libraries(lsbp_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Propiedades adicionales
set_target_properties(lsbp_cpp PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    INTERPROCEDURAL_OPTIMIZATION TRUE
)