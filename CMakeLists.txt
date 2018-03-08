
cmake_minimum_required(VERSION 2.8)
project(matrixmatrix CXX C)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -march=haswell -fopenmp ")
set(CMAKE_EXE_LINKER_FLAGS "-fopenmp -march=haswell")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffp-contract=fast")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

add_executable(blas_1 src/blas_1.cc)
add_executable(blas_2 src/blas_2.cc)


add_executable(blas_3_v0 src/blas_3.cc)

target_compile_definitions(blas_3_v0 PRIVATE BL_TEST=0)


add_executable(blas_3_v1 src/blas_3.cc 
    includes/blislab/kernels/bl_dgemm_asm_12x4.cpp 
    includes/blislab/kernels/bl_dgemm_asm_4x12.cpp 
    includes/blislab/kernels/bl_dgemm_asm_6x8.cpp
    includes/blislab/kernels/bl_dgemm_asm_8x6.cpp 
    includes/blislab/kernels/bl_dgemm_int_8x4.cpp 
    includes/blislab/kernels/bl_dgemm_ukr.cpp 
)

target_compile_definitions(blas_3_v1 PRIVATE BL_TEST=1)

add_executable(blas_3_v2 src/blas_3.cc 
    includes/blislab/kernels/bl_dgemm_asm_12x4.cpp 
    includes/blislab/kernels/bl_dgemm_asm_4x12.cpp 
    includes/blislab/kernels/bl_dgemm_asm_6x8.cpp
    includes/blislab/kernels/bl_dgemm_asm_8x6.cpp 
    includes/blislab/kernels/bl_dgemm_int_8x4.cpp 
    includes/blislab/kernels/bl_dgemm_ukr.cpp 
)

target_compile_definitions(blas_3_v2 PRIVATE BL_TEST=2)

