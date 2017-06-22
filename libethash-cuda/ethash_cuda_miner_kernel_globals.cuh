#ifndef _ETHASH_CUDA_MINER_KERNEL_GLOBALS_H_
#define _ETHASH_CUDA_MINER_KERNEL_GLOBALS_H_

__device__ __constant__ uint32_t d_dag_size;
__device__ __constant__ hash128_t* d_dag;
__device__ __constant__ uint32_t d_light_size;
__device__ __constant__ hash64_t* d_light;
__device__ __constant__ hash32_t d_header;
__device__ __constant__ uint64_t d_target;

#endif
