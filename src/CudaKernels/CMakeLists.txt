cmake_minimum_required(VERSION 3.27)
project(G)

set(CMAKE_PREFIX_PATH "/home/oak/dev/gemm/.venv/lib/python3.14/site-packages/nanobind/cmake" ${CMAKE_PREFIX_PATH})

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Python 3.14 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "~/dev/gemm/.venv/bin/python3" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(modules gemm_01 gemm_02 gemm_03 gemm_04 gemm_05 gemm_06 cublas test)

foreach(mod IN LISTS modules)
    nanobind_add_module("${mod}" "${mod}.cu")

    target_link_libraries("${mod}" PRIVATE CUDA::cublas CUDA::cudart)

    # This runs AFTER the .so file is successfully created
    add_custom_command(
        TARGET "${mod}" POST_BUILD
        COMMAND python -m nanobind.stubgen -m "${mod}" -o "${CMAKE_CURRENT_SOURCE_DIR}/build/${mod}.pyi"
        COMMENT "Generating Python stubs (.pyi) for ${mod}..."
        VERBATIM
    )
endforeach()