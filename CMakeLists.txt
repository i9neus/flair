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
    src/tools/mlplifter/LiftingCodec.cu
    src/tools/mlplifter/LiftingCodec.cuh   
    src/tools/mlplifter/FeatureHeuristic.cuh
    src/core/utils/cuda/CudaObject.cuh
    src/core/utils/cuda/CudaVector.cuh
    src/core/utils/TemplateUtils.h
    src/core/nn/ParameterInitialiser.cuh
    src/core/nn/TensorOps.cuh
    src/core/nn/Tensor2D.cuh
    src/core/nn/Tensor1D.cuh
    src/core/nn/Activation.cuh
    src/core/nn/Loss.cuh
    src/core/nn/ContinuousRandomVariable.cuh
    src/core/nn/Permute.cuh
    src/core/nn/MLP/LinearSequential.cuh
    src/core/nn/MLP/LinearSequentialCPU.cuh
    src/core/nn/MLP/MLP.cuh
    src/core/nn/MLP/Ctx.cuh
    src/core/nn/MLP/Training.cuh
    src/core/nn/MLP/TrainingCPU.cuh
    src/core/nn/MLP/Inference.cuh
    src/core/nn/MLP/Optimiser.cuh
    src/core/nn/MLP/Propagate.cuh
    src/core/Includes.h
    src/core/io/ImageIO.cpp
    src/core/io/ImageIO.h
    src/core/io/FilesystemUtils.h
    src/core/image/ImageOps.h
    src/core/math/MathUtils.h
    src/core/math/wavelets/1d/LiftingDWT1.h
    src/core/math/wavelets/2d/StaticDWT2.h
    src/tests/cuda/TensorTests.cuh
    src/tests/cuda/TensorTests.cu 
)

target_compile_features(mlplifter PUBLIC cxx_std_17)

set_target_properties( mlplifter
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(mlplifter PRIVATE "${CMAKE_SOURCE_DIR}/src")

set_property(TARGET mlplifter PROPERTY CXX_STANDARD 17)

######################################################################

project(converter LANGUAGES CXX)

add_executable(converter
    src/tools/converter/Main.cpp
    src/core/Includes.h
    src/core/io/ImageIO.cpp
    src/core/io/ImageIO.h
    src/core/io/FilesystemUtils.h
    src/core/math/MathUtils.h
    src/core/Codec.cpp
)

target_compile_features(converter PUBLIC cxx_std_17)

target_include_directories(converter PRIVATE "${CMAKE_SOURCE_DIR}/src")

set_property(TARGET converter PROPERTY CXX_STANDARD 17)