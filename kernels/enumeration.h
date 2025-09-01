#ifndef KERNELS_ENUMERATION
#define KERNELS_ENUMERATION

#include <cstdint>

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "graph/match_gpu.h"

__global__ void write_initial_partial_results(
    RelationsGPU global_index_gpu,
    const uint8_t idx,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_new_res_size
);


__global__ void gamma_write_initial_partial_results(
    RelationsGPU local_index_gpu,  // In fact, local index are passed in.
    const uint8_t edge_idx,
    const uint8_t edge_list_idx,
    const uint8_t num_query_vertices,
    unsigned long long int new_res_start,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
);

__global__ void extendDFS(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const RelationsGPU index,
    const uint8_t oi
);

__global__ void extendBFSRegTwo(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth_arg,
    const bool write_res
);

__global__ void extendBFSShareAll(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth_arg,
    const bool write_res
);

__global__ void extendBFSDFSShareZero(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

__global__ void extendBFSDFSRegTwo(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

__global__ void extendBFSDFSShareTwo(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

__global__ void extendBFSDFSShareAll(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

#define DynMemSize(start_depth, end_depth)              \
    start_depth * NUM_WARP_PER_BLOCK * sizeof(uint32_t) +  \
    (end_depth - start_depth) * NUM_WARP_PER_BLOCK * (     \
        sizeof(uint32_t) * WARP_SIZE                    \
        + sizeof(uint8_t) * 3 + sizeof(uint32_t)        \
        + sizeof(bool) + sizeof(uint8_t)                \
    ) + NUM_WARP_PER_BLOCK * WARP_SIZE + NUM_WARP_PER_BLOCK

__global__ void extendBFSDFSDynamicShared(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

#endif