cmake_minimum_required(VERSION 3.22)

project(FunDEM LANGUAGES CXX CUDA)

# ============================================================
# Standards
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Options
# ============================================================
option(FUNDEM_ENABLE_RDC "Enable CUDA separable compilation (RDC)" ON)
option(FUNDEM_ENABLE_PIC "Build position independent code (PIC)" ON)

# ============================================================
# Project root
# ============================================================
set(FUNDEM_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# Bring in kernel/ (defines target: FunDEM_kernels)
# ============================================================
add_subdirectory(kernel)

# ============================================================
# Bring in apps/ (each subdir defines its own executable)
# ============================================================
add_subdirectory(apps)