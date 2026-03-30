# kernel/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)

# ------------------------------------------------------------------
# This directory contains:
#   - CUDAKernelFunction/   (compilable CUDA/C++ sources)
#   - extra headers under kernel/ (helpers / configs / etc.)
# We build the actual library in CUDAKernelFunction and only attach
# the extra headers here for IDE convenience.
# ------------------------------------------------------------------

set(KERNEL_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Add the real kernel implementation subdir (creates target: FunDEM_kernels)
add_subdirectory(CUDAKernelFunction)

# Track headers directly under kernel/ (not compiled, only for IDE/project view)
file(GLOB_RECURSE KERNEL_EXTRA_HEADERS CONFIGURE_DEPENDS
    ${KERNEL_ROOT}/*.h
    ${KERNEL_ROOT}/*.hpp
    ${KERNEL_ROOT}/*.cuh
)

# Attach those headers to the target so IDE can see them
target_sources(FunDEM_kernels PRIVATE ${KERNEL_EXTRA_HEADERS})

# Make sure apps can include headers under kernel/
target_include_directories(FunDEM_kernels PUBLIC ${KERNEL_ROOT})