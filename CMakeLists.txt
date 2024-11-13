cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/cmake-modules")
#include(GetGitRevisionDescription)

######################################################################

project(mlplifter LANGUAGES CXX CUDA)
enable_language(CUDA)

add_executable(mlplifter
    src/tools/mlplifter/Main.cu
    src/tools/mlplifter/CudaUtils.cuh
    src/core/Includes.h
    src/core/io/ImageIO.cpp
    src/core/io/ImageIO.h
    src/core/io/FilesystemUtils.h
    src/core/math/MathUtils.h
    src/core/Codec.cpp
)

target_compile_features(mlplifter PUBLIC cxx_std_17)

set_target_properties( mlplifter
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(mlplifter PRIVATE "${CMAKE_SOURCE_DIR}/src")

set_property(TARGET mlplifter PROPERTY CXX_STANDARD 17)