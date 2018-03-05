
cmake_minimum_required(VERSION 2.8)
project(matrixmatrix CXX C)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -march=native -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "-fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffp-contract=fast")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

add_executable(blas_1 src/blas_1.cc)
add_executable(blas_2 src/blas_2.cc)
add_executable(blas_3 src/blas_3.cc)

