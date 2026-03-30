# apps/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)

# ------------------------------------------------------------
# Auto-discover apps (any subdir) and build one executable per main()
# This file assumes the root CMakeLists already defines/creates:
#   - target: FunDEM_kernels   (your kernel lib)
#   - options: FUNDEM_ENABLE_PIC, FUNDEM_ENABLE_RDC (optional)
# ------------------------------------------------------------

set(APPS_ROOT ${CMAKE_CURRENT_LIST_DIR})

# Collect all app translation units under apps/
file(GLOB_RECURSE APP_SRCS CONFIGURE_DEPENDS
    ${APPS_ROOT}/*.cpp
    ${APPS_ROOT}/*.cxx
    ${APPS_ROOT}/*.cc
    ${APPS_ROOT}/*.cu
)

set(APP_MAIN_SRCS "")
set(APP_COMMON_SRCS "")

foreach(src ${APP_SRCS})
    # Heuristic: detect "main(" (not a full C++ parser, but works well in practice)
    file(STRINGS "${src}" _main_hits
        REGEX "[^A-Za-z0-9_]main[ \t\r\n]*\\("
        LIMIT_COUNT 2000
    )
    if (_main_hits)
        list(APPEND APP_MAIN_SRCS "${src}")
    else()
        list(APPEND APP_COMMON_SRCS "${src}")
    endif()
endforeach()

# Optional: common library for non-main app sources
if (APP_COMMON_SRCS)
    add_library(FunDEM_apps_common STATIC ${APP_COMMON_SRCS})
    target_link_libraries(FunDEM_apps_common PUBLIC FunDEM_kernels)

    # In case apps use headers in apps/ itself
    target_include_directories(FunDEM_apps_common PUBLIC ${APPS_ROOT})

    set_target_properties(FunDEM_apps_common PROPERTIES
        POSITION_INDEPENDENT_CODE ${FUNDEM_ENABLE_PIC}
        CUDA_SEPARABLE_COMPILATION ${FUNDEM_ENABLE_RDC}
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
endif()

if (NOT APP_MAIN_SRCS)
    message(WARNING "No main() found under ${APPS_ROOT}")
endif()

# Create one executable per main file
foreach(main_src ${APP_MAIN_SRCS})
    # Make a stable unique target name from relative path
    file(RELATIVE_PATH rel "${APPS_ROOT}" "${main_src}")
    get_filename_component(exe_base "${rel}" NAME_WE)

    # Replace path separators to avoid name collisions: foo/bar/main.cpp -> foo__bar__main
    string(REPLACE "/" "__" exe_name "${rel}")
    string(REPLACE "\\" "__" exe_name "${exe_name}")
    string(REPLACE ".cpp" "" exe_name "${exe_name}")
    string(REPLACE ".cxx" "" exe_name "${exe_name}")
    string(REPLACE ".cc"  "" exe_name "${exe_name}")
    string(REPLACE ".cu"  "" exe_name "${exe_name}")

    add_executable(${exe_name} ${main_src})

    # Inherit includes/flags from kernel lib
    target_link_libraries(${exe_name} PRIVATE FunDEM_kernels)

    if (TARGET FunDEM_apps_common)
        target_link_libraries(${exe_name} PRIVATE FunDEM_apps_common)
    endif()

    # Optional: let apps include headers in apps/ (e.g., #include "foo.h")
    target_include_directories(${exe_name} PRIVATE ${APPS_ROOT})

    set_target_properties(${exe_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ${FUNDEM_ENABLE_PIC}
        CUDA_SEPARABLE_COMPILATION ${FUNDEM_ENABLE_RDC}
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
endforeach()