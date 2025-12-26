# PSBP_normal/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(psbp_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar pybind11
find_package(pybind11 REQUIRED)

# Encontrar OpenMP (opcional, para paralelización)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found - enabling parallel computation")
else()
    message(STATUS "OpenMP not found - using sequential computation")
endif()

# Encontrar biblioteca de estadísticas (para funciones de normal)
find_library(M_LIBRARY m)
if(M_LIBRARY)
    message(STATUS "Math library found")
endif()

# Crear módulo Python
pybind11_add_module(psbp_cpp 
    bindings.cpp
    psbp_core.cpp
)

# Optimizaciones
target_compile_options(psbp_cpp PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math -funroll-loops>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Incluir directorios
target_include_directories(psbp_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link OpenMP si está disponible
if(OpenMP_CXX_FOUND)
    target_link_libraries(psbp_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link math library
if(M_LIBRARY)
    target_link_libraries(psbp_cpp PRIVATE ${M_LIBRARY})
endif()

# Propiedades adicionales
set_target_properties(psbp_cpp PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# Instalación
install(TARGETS psbp_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../)