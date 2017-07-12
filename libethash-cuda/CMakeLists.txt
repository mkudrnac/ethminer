find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -use_fast_math")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v,-dlcm=ca")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --disable-warnings")

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.h" "*.cuh")

cuda_add_library(ethash-cuda STATIC ${sources} ${headers})
target_link_libraries(ethash-cuda ethash)
target_include_directories(ethash-cuda PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(ethash-cuda PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})

