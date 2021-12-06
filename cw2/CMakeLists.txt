project(set10108-cw02 LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

#### OPEN MP PACKAGES ####
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


add_executable(NQueens-Serial nqueens-serial.cpp)
add_executable(NQueens-OpenMP nqueens-openmp.cpp)
add_executable(NQueens-GPU nqueens-gpu.cu)
set_property(TARGET NQueens-GPU PROPERTY CUDA_SEPARABLE_COMPILATION ON)