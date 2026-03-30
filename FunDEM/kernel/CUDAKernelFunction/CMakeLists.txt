# kernel/CUDAKernelFunction/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)

set(CUDAK_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(UTIL_DIR  ${CUDAK_DIR}/myUtility)

# ------------------------------------------------------------------
# Collect compilation units under CUDAKernelFunction/
# ------------------------------------------------------------------
file(GLOB_RECURSE KERNEL_SRCS CONFIGURE_DEPENDS
    ${CUDAK_DIR}/*.cu
    ${CUDAK_DIR}/*.cpp
    ${CUDAK_DIR}/*.cxx
)

# Track headers for IDE (optional)
file(GLOB_RECURSE KERNEL_HEADERS CONFIGURE_DEPENDS
    ${CUDAK_DIR}/*.h
    ${CUDAK_DIR}/*.hpp
    ${CUDAK_DIR}/*.cuh
)

add_library(FunDEM_kernels STATIC
    ${KERNEL_SRCS}
    ${KERNEL_HEADERS}
)

# ------------------------------------------------------------------
# CUDA runtime headers/libs (cuda_runtime.h etc.)
# ------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
target_link_libraries(FunDEM_kernels PUBLIC CUDA::cudart)

# ------------------------------------------------------------------
# Export include directories to anything linking FunDEM_kernels
# ------------------------------------------------------------------
target_include_directories(FunDEM_kernels PUBLIC
    ${FUNDEM_ROOT}   # project root (provided by top-level)
    ${CUDAK_DIR}     # kernel/CUDAKernelFunction
    ${UTIL_DIR}      # kernel/CUDAKernelFunction/myUtility
)

# ------------------------------------------------------------------
# Build properties
# ------------------------------------------------------------------
set_target_properties(FunDEM_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${FUNDEM_ENABLE_RDC}
    POSITION_INDEPENDENT_CODE  ${FUNDEM_ENABLE_PIC}
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)