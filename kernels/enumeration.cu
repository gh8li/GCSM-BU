#include <cstdint>

#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/match_gpu.h"

// #define DEBUGGING_ENUMERATION

namespace {
__device__ void warp_print(const uint8_t lane_id, const uint32_t warp_id, const unsigned long long global_warp_id, const char *content) {
    if(lane_id == 0)
    {
        printf("warp_id: %d, %lld, %s", warp_id, global_warp_id, content);
        // printf(content);
    }
    __syncwarp();
}
}  // namespace

__global__ void write_initial_partial_results(
        RelationsGPU global_index_gpu,  // In fact, local index are passed in.
        const uint8_t idx,
        unsigned long long int new_res,
        unsigned long long int *new_res_size,
        const unsigned long long int h_max_new_res_size) {
    __shared__ unsigned long long int write_pos[NUM_WARP_PER_BLOCK];
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t v = gwarp_id; v < C_DV_COUNT; v+= num_warps) {

#ifdef DEBUGGING_ENUMERATION
        if (warp_id == 0 && lane_id == 0) {
            warp_print(lane_id, warp_id, gwarp_id, "before atomicAdd.\n");
            printf("warp_id: %d, %lld, before atomicAdd, new_res_size: %llu \n", warp_id, gwarp_id, *new_res_size);
//             printf("warp_id: %d, %lld, before atomicAdd, idx: %u, v: %u \n", warp_id, gwarp_id, (uint32_t)idx, v);
            printf("warp_id: %d, %lld, before atomicAdd, idx: %u, global_index_gpu.sizes[idx]: %p \n", warp_id, gwarp_id, (uint32_t)idx, global_index_gpu.sizes_[idx]);
            printf("warp_id: %d, %lld, before atomicAdd, global_index_gpu.sizes[idx][v]: %u \n", warp_id, gwarp_id, global_index_gpu.sizes_[idx][v]);
//             printf("warp_id: %d, %lld, before atomicAdd, global_index_gpu.sizes[idx][v]: %u \n", warp_id, gwarp_id, global_index_gpu.sizes_[idx][v]);
        }
#endif  // DEBUGGING_ENUMERATION

        if (lane_id == 0) write_pos[warp_id] = atomicAdd(new_res_size, (unsigned long long)global_index_gpu.sizes_[idx][v]);
        __syncwarp();

        //if (*new_res_size >= h_max_new_res_size) return;

// #ifdef DEBUGGING_ENUMERATION
//         if (warp_id == 0 && lane_id == 0) {
//             printf("warp_id: %d, %lld, global_index_gpu.sizes_[idx][v]: %u, new_res_size: %llu \n", warp_id, gwarp_id, global_index_gpu.sizes_[idx][v], *new_res_size);
//         }
// #endif  // DEBUGGING_ENUMERATION

        for (uint32_t j = lane_id; j < global_index_gpu.sizes_[idx][v]; j += WARP_SIZE) {
            C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + j) * 2] = v;
            C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + j) * 2 + 1] = global_index_gpu.nbrs_[idx][v][j];
        }

// #ifdef DEBUGGING_ENUMERATION
//        if (warp_id == 0 && lane_id == 0) {
//            warp_print(lane_id, warp_id, gwarp_id, "After array writing.\n");
//        }
// #endif  // DEBUGGING_ENUMERATION
    
    }

// #ifdef DEBUGGING_ENUMERATION
//         if (warp_id == 0 && lane_id == 0) {
//             warp_print(lane_id, warp_id, gwarp_id, "Finish!\n");
//         }
// #endif  // DEBUGGING_ENUMERATION

}

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
) {
    __shared__ uint32_t result_queue[NUM_WARP_PER_BLOCK][MAX_QV_COUNT][WARP_SIZE];
    __shared__ uint8_t queue_pos[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];
    __shared__ uint8_t queue_size[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];

    __shared__ uint8_t end_v[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];
    __shared__ uint32_t end_nbr[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];

    __shared__ bool intersection_continue[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];

    __shared__ uint8_t rem_nbr_count[NUM_WARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t depth[NUM_WARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;

    if (gwarp_id >= res_size)
    {
        return;
    }
    if (lane_id < start_depth)
    {
        result_queue[warp_id][lane_id][0] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + lane_id) % C_RES_QUEUE.capacity_];
    }
    __syncwarp();
#ifdef DEBUG
    if(lane_id == 0)
    {
        printf("%d %d\n", v0, v1);
    }
    __syncwarp();
#endif

    if (lane_id < C_QV_COUNT)
    {
        queue_pos[warp_id][lane_id] = 0u;
        queue_size[warp_id][lane_id] = (lane_id < start_depth) ? 1u: 0u;
        end_v[warp_id][lane_id] = 0u;
        end_nbr[warp_id][lane_id] = 0u;
        intersection_continue[warp_id][lane_id] = false;
    }
    if (lane_id == 0)
    {
        depth[warp_id] = start_depth;
    }
    __syncwarp();

    while (depth[warp_id] >= start_depth)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[depth[warp_id]]];
        const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
#ifdef DEBUG
        if(lane_id == 0)
        {
            printf("enter while loop depth=%d, pre_qv_idx=%d, pre_qe_idx=%d\n", (uint32_t)depth[warp_id], (uint32_t)pre_qv_idx, (uint32_t)pre_qe_idx);
        }
        __syncwarp();
#endif

        // check if all local candidates of this level are consumed,
        if (queue_pos[warp_id][depth[warp_id]] >= queue_size[warp_id][depth[warp_id]])
        {
#ifdef DEBUG
            if (lane_id == 0)
            {
                printf("depth=%d all results processed, (%d, %d)\n", (uint32_t)depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id][depth[warp_id] - 2], pre_qv_idx < 2 ? 0 :queue_pos[warp_id][pre_qv_idx - 2]);
            }
            __syncwarp();
#endif
            // check if there is no remaining intersection workload for the current level
            if (
                intersection_continue[warp_id][depth[warp_id]] && 
                end_v[warp_id][depth[warp_id]] > queue_pos[warp_id][pre_qv_idx]
            ) {
                // no more intersection to do, go back to the previous level
                if (lane_id == 0)
                {
#ifdef DEBUG
                    printf("backtrack from level %d (%d, %d)\n", depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id][depth[warp_id] - 2], pre_qv_idx < 2 ? 0 :queue_pos[warp_id][pre_qv_idx - 2]);
#endif
                    queue_pos[warp_id][depth[warp_id]] = 0u;
                    queue_size[warp_id][depth[warp_id]] = 0u;

                    intersection_continue[warp_id][depth[warp_id]] = false;

                    depth[warp_id]--;

                    queue_pos[warp_id][depth[warp_id]]++;
                }
                __syncwarp();
            }
            else
            {
                if (!intersection_continue[warp_id][depth[warp_id]])
                {
                    if (lane_id == 0)
                    {
#ifdef DEBUG
                        printf("START intersection\n");
#endif
                        end_v[warp_id][depth[warp_id]] = queue_pos[warp_id][pre_qv_idx];
                        end_nbr[warp_id][depth[warp_id]] = 0u;
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("perform intersection at level %d\n", depth[warp_id]);
                }
                __syncwarp();
#endif
                // need to start/continue the intersection
                
                intersection_continue[warp_id][depth[warp_id]] = true;

                const uint8_t rem_pre_dv_count = 1u;
                uint32_t pre_dv = UINT32_MAX;
                pre_dv = lane_id < rem_pre_dv_count ? result_queue[warp_id][pre_qv_idx][queue_pos[warp_id][pre_qv_idx]] : UINT32_MAX;
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("lane 0 have a pre_dv=%d\n", pre_dv);
                }
                __syncwarp();
#endif

                // 2. retrieve the number of neighbors of each vertex
                // end_v[warp_id][depth[warp_id] - 2] should be equal to queue_pos[warp_id][pre_qv_idx - 2]
                rem_nbr_count[warp_id][lane_id] = lane_id < rem_pre_dv_count
                    ? min(index.sizes_[pre_qe_idx][pre_dv] - (lane_id == 0 ? end_nbr[warp_id][depth[warp_id]] : 0u), 64u)
                    : 0u;
                __syncwarp();
                if (lane_id > 0) rem_nbr_count[warp_id][lane_id] = rem_nbr_count[warp_id][0];
                __syncwarp();
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("rem_nbr_count[%d][31]=%d\n", warp_id, rem_nbr_count[warp_id][31]);
                }
                __syncwarp();
#endif
                // each lane found a vertex temp_nbr
                uint32_t temp_nbr = UINT32_MAX;
                if (lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1])
                {
                    // 3. each lane gets a neighbor of a vertex in the previous level
                    temp_nbr = index.nbrs_[pre_qe_idx][
                        result_queue[warp_id][pre_qv_idx][queue_pos[warp_id][pre_qv_idx]]
                    ][end_nbr[warp_id][depth[warp_id]] + lane_id];
#ifdef DEBUG
                    printf("lane %d found process the vertex %d\n", lane_id, temp_nbr);
#endif
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // update start_v/nbr and end_v/nbr by the active lane with the greatest id
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("before update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - 2], start_nbr[warp_id][depth[warp_id] - 2], end_v[warp_id][depth[warp_id] - 2], end_nbr[warp_id][depth[warp_id] - 2]);
                }
                __syncwarp();
#endif
                uint8_t num_active_lanes = __popc(__ballot_sync(0xffffffff, lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1]));
                if (num_active_lanes == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id][depth[warp_id]] += rem_pre_dv_count;
                        end_nbr[warp_id][depth[warp_id]] = 0;
                    }
                    __syncwarp();
                }
                else
                {
                    if (lane_id == num_active_lanes - 1)
                    {
                        if (rem_nbr_count[warp_id][0] > lane_id + 1)
                        {
                            // not all neighbors of the last vertex are processed
                            end_nbr[warp_id][depth[warp_id]] += lane_id + 1;
                        }
                        else
                        {
                            end_nbr[warp_id][depth[warp_id]] = 0u;
                            end_v[warp_id][depth[warp_id]] += 1u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("after update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - 2], start_nbr[warp_id][depth[warp_id] - 2], end_v[warp_id][depth[warp_id] - 2], end_nbr[warp_id][depth[warp_id] - 2]);
                }
                __syncwarp();
#endif

                // 5. the lane search for the vertex on other nbr arrays
                bool found = lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1];
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id]] = 0u;
                    queue_size[warp_id][depth[warp_id]] = 0u;
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found1 %d temp_nbr=%d\n", lane_id, temp_nbr);
#endif
                    for (uint8_t i = 0u; i < depth[warp_id]; i++)
                    {
                        if (result_queue[warp_id][i][queue_pos[warp_id][i]] == temp_nbr)
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found2 %d\n", lane_id);
#endif
                    for (uint8_t off = C_ORDERS[oi].bni_offs_[depth[warp_id]] + 1; off < C_ORDERS[oi].bni_offs_[depth[warp_id] + 1]; off++)
                    {
                        const uint8_t& bni = C_ORDERS[oi].bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
                        const uint32_t& pre_pre_v = result_queue[warp_id][bni][queue_pos[warp_id][bni]];
#ifdef DEBUG
                        printf("bni=%d, pre_pre_qe_idx=%d, pre_pre_v=%d\n", bni, pre_pre_qe_idx, pre_pre_v);
#endif
                        const uint32_t res = lower_bound(index.nbrs_[pre_pre_qe_idx][pre_pre_v], index.sizes_[pre_pre_qe_idx][pre_pre_v], temp_nbr);
                        if (res == index.sizes_[pre_pre_qe_idx][pre_pre_v] || index.nbrs_[pre_pre_qe_idx][pre_pre_v][res] != temp_nbr)
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // 6. write the local candidates to result_queue and their group id to group_id
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (depth[warp_id] < end_depth - 1)
                {
#ifdef DEBUG
                    printf("found3 %d\n", lane_id);
#endif
                    // do not need to store results at the least level
                    if (found) result_queue[warp_id][depth[warp_id]][rank] = temp_nbr;
                }
                else
                {
#ifdef DEBUG
                    printf("found a match %d %d %d %d %d\n", v0, v1, result_queue[warp_id][0][queue_pos[warp_id][0]], result_queue[warp_id][1][queue_pos[warp_id][1]], temp_nbr);
#endif
                    if (write_res)
                    {
                        if (found_mask)
                        {
                            unsigned long long int write_pos;
                            if (lane_id == 0) write_pos = atomicAdd(new_res_size, __popc(found_mask));
                            write_pos = __shfl_sync(0xffffffff, write_pos, 0, 64);
                            if (write_pos + __popc(found_mask) > h_max_new_res_size_) return;
                            if (found)
                            {
                                write_pos += rank;
                                for (uint8_t j = 0u; j < end_depth - 1; j++)
                                {
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capacity_] = result_queue[warp_id][j][queue_pos[warp_id][j]];
                                }
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capacity_] = temp_nbr;
                            }
                        }
                    }
                    else
                    {
                        if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
                    }
                }
                __syncwarp();
                if (found && rank == 0)
                {
                    queue_pos[warp_id][depth[warp_id]] = (depth[warp_id] < end_depth - 1) ? 0u : __popc(found_mask);
                    queue_size[warp_id][depth[warp_id]] = __popc(found_mask);
                }
                __syncwarp();
            }
        }
        else // go to the next level
        {
            if (lane_id == 0 && depth[warp_id] < end_depth - 1)
            {
#ifdef DEBUG
                printf("increase from depth %d\n", depth[warp_id]);
#endif
                depth[warp_id] ++;
            }
            __syncwarp();
        }
    }
}

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
) {
    __shared__ uint32_t result_queue[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2][WARP_SIZE];
    __shared__ uint8_t queue_pos[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];
    __shared__ uint8_t queue_size[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ uint8_t end_v[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];
    __shared__ uint32_t end_nbr[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ bool intersection_continue[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ uint8_t rem_nbr_count[NUM_WARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t depth[NUM_WARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;

    if (gwarp_id >= res_size)
    {
        return;
    }
    uint32_t v0 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth) % C_RES_QUEUE.capacity_];
    uint32_t v1 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 1) % C_RES_QUEUE.capacity_];
    if (lane_id < start_depth - 2)
    {
        result_queue[warp_id][lane_id][0] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 2 + lane_id) % C_RES_QUEUE.capacity_];
    }
    __syncwarp();
#ifdef DEBUG
    if(lane_id == 0)
    {
        printf("%d %d\n", v0, v1);
    }
    __syncwarp();
#endif

    if (lane_id < C_QV_COUNT - 2)
    {
        queue_pos[warp_id][lane_id] = 0u;
        queue_size[warp_id][lane_id] = (lane_id < start_depth - 2) ? 1u: 0u;
        end_v[warp_id][lane_id] = 0u;
        end_nbr[warp_id][lane_id] = 0u;
        intersection_continue[warp_id][lane_id] = false;
    }
    if (lane_id == 0)
    {
        depth[warp_id] = start_depth;
    }
    __syncwarp();

    while (depth[warp_id] >= start_depth)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[depth[warp_id]]];
        const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
#ifdef DEBUG
        if(lane_id == 0)
        {
            printf("enter while loop depth=%d, pre_qv_idx=%d, pre_qe_idx=%d\n", (uint32_t)depth[warp_id], (uint32_t)pre_qv_idx, (uint32_t)pre_qe_idx);
        }
        __syncwarp();
#endif

        // check if all local candidates of this level are consumed,
        if (queue_pos[warp_id][depth[warp_id] - 2] >= queue_size[warp_id][depth[warp_id] - 2])
        {
#ifdef DEBUG
            if (lane_id == 0)
            {
                printf("depth=%d all results processed, (%d, %d)\n", (uint32_t)depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id][depth[warp_id] - 2], pre_qv_idx < 2 ? 0 :queue_pos[warp_id][pre_qv_idx - 2]);
            }
            __syncwarp();
#endif
            // check if there is no remaining intersection workload for the current level
            if (
                intersection_continue[warp_id][depth[warp_id] - 2] && 
                ((pre_qv_idx < 2 && end_v[warp_id][depth[warp_id] - 2] > 0) ||
                (pre_qv_idx >= 2 && end_v[warp_id][depth[warp_id] - 2] > queue_pos[warp_id][pre_qv_idx - 2]))
            ) {
                // no more intersection to do, go back to the previous level
                if (lane_id == 0)
                {
#ifdef DEBUG
                    printf("backtrack from level %d (%d, %d)\n", depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id][depth[warp_id] - 2], pre_qv_idx < 2 ? 0 :queue_pos[warp_id][pre_qv_idx - 2]);
#endif
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;

                    intersection_continue[warp_id][depth[warp_id] - 2] = false;

                    depth[warp_id]--;

                    if (depth[warp_id] >= 2)
                    {
                        queue_pos[warp_id][depth[warp_id] - 2]++;
                    }
                }
                __syncwarp();
            }
            else
            {
                if (!intersection_continue[warp_id][depth[warp_id] - 2])
                {
                    if (lane_id == 0)
                    {
#ifdef DEBUG
                        printf("START intersection\n");
#endif
                        if (pre_qv_idx < 2)
                        {
                            end_v[warp_id][depth[warp_id] - 2] = 0u;
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                        }
                        else
                        {
                            end_v[warp_id][depth[warp_id] - 2] = queue_pos[warp_id][pre_qv_idx - 2];
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("perform intersection at level %d\n", depth[warp_id]);
                }
                __syncwarp();
#endif
                // need to start/continue the intersection
                
                intersection_continue[warp_id][depth[warp_id] - 2] = true;

                const uint8_t rem_pre_dv_count = 1u;
                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx == 0u)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? v0 : UINT32_MAX;
                }
                else if (pre_qv_idx == 1u)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? v1 : UINT32_MAX;
                }
                else
                {
                    pre_dv = lane_id < rem_pre_dv_count ? result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]] : UINT32_MAX;
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("lane 0 have a pre_dv=%d\n", pre_dv);
                }
                __syncwarp();
#endif

                // 2. retrieve the number of neighbors of each vertex
                // end_v[warp_id][depth[warp_id] - 2] should be equal to queue_pos[warp_id][pre_qv_idx - 2]
                // Doubt: only effective when lane_id == 0; 
                // for lane_id > 0, rem_nbr_count[warp_id][lane_id] will be written as rem_nbr_count[warp_id][0]
                // Question: Why min(..., 64)? Why not min(..., 32)? (may be harmless)
                rem_nbr_count[warp_id][lane_id] = lane_id < rem_pre_dv_count
                    ? min(index.sizes_[pre_qe_idx][pre_dv] - (lane_id == 0 ? end_nbr[warp_id][depth[warp_id] - 2] : 0u), 64u)
                    : 0u;
                __syncwarp();
                if (lane_id > 0) rem_nbr_count[warp_id][lane_id] = rem_nbr_count[warp_id][0];
                __syncwarp();
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("rem_nbr_count[%d][31]=%d\n", warp_id, rem_nbr_count[warp_id][31]);
                }
                __syncwarp();
#endif
                // each lane found a vertex temp_nbr
                uint32_t temp_nbr = UINT32_MAX;
                if (lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1])
                {
                    // 3. each lane gets a neighbor of a vertex in the previous level
                    if (pre_qv_idx == 0)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][v0][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                    else if (pre_qv_idx == 1)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][v1][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                    else
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][
                            result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]]
                        ][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
#ifdef DEBUG
                    printf("lane %d found process the vertex %d\n", lane_id, temp_nbr);
#endif
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // update start_v/nbr and end_v/nbr by the active lane with the greatest id
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("before update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - 2], start_nbr[warp_id][depth[warp_id] - 2], end_v[warp_id][depth[warp_id] - 2], end_nbr[warp_id][depth[warp_id] - 2]);
                }
                __syncwarp();
#endif
                uint8_t num_active_lanes = __popc(__ballot_sync(0xffffffff, lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1]));
                if (num_active_lanes == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id][depth[warp_id] - 2] += rem_pre_dv_count;
                        end_nbr[warp_id][depth[warp_id] - 2] = 0;
                    }
                    __syncwarp();
                }
                else
                {
                    if (lane_id == num_active_lanes - 1)
                    {
                        if (rem_nbr_count[warp_id][0] > lane_id + 1)
                        {
                            // not all neighbors of the last vertex are processed
                            end_nbr[warp_id][depth[warp_id] - 2] += lane_id + 1;
                        }
                        else
                        {
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            end_v[warp_id][depth[warp_id] - 2] += 1u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("after update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - 2], start_nbr[warp_id][depth[warp_id] - 2], end_v[warp_id][depth[warp_id] - 2], end_nbr[warp_id][depth[warp_id] - 2]);
                }
                __syncwarp();
#endif

                // 5. the lane search for the vertex on other nbr arrays
                bool found = lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1];
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found1 %d temp_nbr=%d\n", lane_id, temp_nbr);
#endif
                    if (temp_nbr == v0 || temp_nbr == v1)
                    {
                        found = false;
                    }
                    else
                    {
                        for (uint8_t i = 2u; i < depth[warp_id]; i++)
                        {
                            if (result_queue[warp_id][i - 2u][queue_pos[warp_id][i - 2]] == temp_nbr)
                            {
                                found = false;
                                break;
                            }
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found2 %d\n", lane_id);
#endif
                    for (uint8_t off = C_ORDERS[oi].bni_offs_[depth[warp_id]] + 1; off < C_ORDERS[oi].bni_offs_[depth[warp_id] + 1]; off++)
                    {
                        const uint8_t& bni = C_ORDERS[oi].bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
                        const uint32_t& pre_pre_v = bni == 0 ? v0 : (bni == 1 ? v1 : result_queue[warp_id][bni - 2][queue_pos[warp_id][bni - 2]]);
#ifdef DEBUG
                        printf("bni=%d, pre_pre_qe_idx=%d, pre_pre_v=%d\n", bni, pre_pre_qe_idx, pre_pre_v);
#endif
                        const uint32_t res = lower_bound(index.nbrs_[pre_pre_qe_idx][pre_pre_v], index.sizes_[pre_pre_qe_idx][pre_pre_v], temp_nbr);
                        if (res == index.sizes_[pre_pre_qe_idx][pre_pre_v] || index.nbrs_[pre_pre_qe_idx][pre_pre_v][res] != temp_nbr)
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // 6. write the local candidates to result_queue and their group id to group_id
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (depth[warp_id] < end_depth - 1)
                {
#ifdef DEBUG
                    printf("found3 %d\n", lane_id);
#endif
                    // do not need to store results at the least level
                    if (found) result_queue[warp_id][depth[warp_id] - 2][rank] = temp_nbr;
                }
                else
                {
#ifdef DEBUG
                    printf("found a match %d %d %d %d %d\n", v0, v1, result_queue[warp_id][0][queue_pos[warp_id][0]], result_queue[warp_id][1][queue_pos[warp_id][1]], temp_nbr);
#endif
                    if (write_res)
                    {
                        if (found_mask)
                        {
                            unsigned long long int write_pos;
                            if (lane_id == 0) write_pos = atomicAdd(new_res_size, __popc(found_mask));
                            // Question: why the last parameter, width, is 64 ? (64 is larger than the warp size, 32, which will cause undefined behavior)
                            write_pos = __shfl_sync(0xffffffff, write_pos, 0, 64);
                            if (write_pos + __popc(found_mask) > h_max_new_res_size_) return;
                            if (found)
                            {
                                write_pos += rank;
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth) % C_RES_QUEUE.capacity_] = v0;
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 1) % C_RES_QUEUE.capacity_] = v1;

                                for (uint8_t j = 2u; j < end_depth - 1; j++)
                                {
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capacity_] = result_queue[warp_id][j - 2][queue_pos[warp_id][j - 2]];
                                }
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capacity_] = temp_nbr;
                            }
                        }
                    }
                    else
                    {
                        if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
                    }
                }
                __syncwarp();
                if (found && rank == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = (depth[warp_id] < end_depth - 1) ? 0u : __popc(found_mask);
                    queue_size[warp_id][depth[warp_id] - 2] = __popc(found_mask);
                }
                __syncwarp();
            }
        }
        else // go to the next level
        {
            if (lane_id == 0 && depth[warp_id] < end_depth - 1)
            {
#ifdef DEBUG
                printf("increase from depth %d\n", depth[warp_id]);
#endif
                depth[warp_id] ++;
            }
            __syncwarp();
        }
    }
}

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
) {
    __shared__ uint32_t partial[NUM_WARP_PER_BLOCK][2];
    __shared__ uint32_t result_queue[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2][WARP_SIZE];
    __shared__ uint8_t queue_pos[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];
    __shared__ uint8_t queue_size[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ uint8_t end_v[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];
    __shared__ uint32_t end_nbr[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ bool intersection_continue[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ uint8_t rem_nbr_count[NUM_WARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t depth[NUM_WARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;

    if (gwarp_id >= res_size)
    {
        return;
    }
    partial[warp_id][0] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth) % C_RES_QUEUE.capacity_];
    partial[warp_id][1] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 1) % C_RES_QUEUE.capacity_];
    if (lane_id < start_depth - 2)
    {
        result_queue[warp_id][lane_id][0] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 2 + lane_id) % C_RES_QUEUE.capacity_];
    }
    __syncwarp();
#ifdef DEBUG
    if(lane_id == 0)
    {
        printf("%d %d\n", v0, v1);
    }
    __syncwarp();
#endif

    if (lane_id < C_QV_COUNT - 2)
    {
        queue_pos[warp_id][lane_id] = 0u;
        queue_size[warp_id][lane_id] = (lane_id < start_depth - 2) ? 1u: 0u;
        end_v[warp_id][lane_id] = 0u;
        end_nbr[warp_id][lane_id] = 0u;
        intersection_continue[warp_id][lane_id] = false;
    }
    if (lane_id == 0)
    {
        depth[warp_id] = start_depth;
    }
    __syncwarp();

    while (depth[warp_id] >= start_depth)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[depth[warp_id]]];
        const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
#ifdef DEBUG
        if(lane_id == 0)
        {
            printf("enter while loop depth=%d, pre_qv_idx=%d, pre_qe_idx=%d\n", (uint32_t)depth[warp_id], (uint32_t)pre_qv_idx, (uint32_t)pre_qe_idx);
        }
        __syncwarp();
#endif

        // check if all local candidates of this level are consumed,
        if (queue_pos[warp_id][depth[warp_id] - 2] >= queue_size[warp_id][depth[warp_id] - 2])
        {
#ifdef DEBUG
            if (lane_id == 0)
            {
                printf("depth=%d all results processed, (%d, %d)\n", (uint32_t)depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id][depth[warp_id] - 2], pre_qv_idx < 2 ? 0 :queue_pos[warp_id][pre_qv_idx - 2]);
            }
            __syncwarp();
#endif
            // check if there is no remaining intersection workload for the current level
            if (
                intersection_continue[warp_id][depth[warp_id] - 2] && 
                ((pre_qv_idx < 2 && end_v[warp_id][depth[warp_id] - 2] > 0) ||
                (pre_qv_idx >= 2 && end_v[warp_id][depth[warp_id] - 2] > queue_pos[warp_id][pre_qv_idx - 2]))
            ) {
                // no more intersection to do, go back to the previous level
                if (lane_id == 0)
                {
#ifdef DEBUG
                    printf("backtrack from level %d (%d, %d)\n", depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id][depth[warp_id] - 2], pre_qv_idx < 2 ? 0 :queue_pos[warp_id][pre_qv_idx - 2]);
#endif
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;

                    intersection_continue[warp_id][depth[warp_id] - 2] = false;

                    depth[warp_id]--;

                    if (depth[warp_id] >= 2)
                    {
                        queue_pos[warp_id][depth[warp_id] - 2]++;
                    }
                }
                __syncwarp();
            }
            else
            {
                if (!intersection_continue[warp_id][depth[warp_id] - 2])
                {
                    if (lane_id == 0)
                    {
#ifdef DEBUG
                        printf("START intersection\n");
#endif
                        if (pre_qv_idx < 2)
                        {
                            end_v[warp_id][depth[warp_id] - 2] = 0u;
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                        }
                        else
                        {
                            end_v[warp_id][depth[warp_id] - 2] = queue_pos[warp_id][pre_qv_idx - 2];
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("perform intersection at level %d\n", depth[warp_id]);
                }
                __syncwarp();
#endif
                // need to start/continue the intersection
                
                intersection_continue[warp_id][depth[warp_id] - 2] = true;

                const uint8_t rem_pre_dv_count = 1u;
                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx == 0u)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? partial[warp_id][0] : UINT32_MAX;
                }
                else if (pre_qv_idx == 1u)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? partial[warp_id][1] : UINT32_MAX;
                }
                else
                {
                    pre_dv = lane_id < rem_pre_dv_count ? result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]] : UINT32_MAX;
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("lane 0 have a pre_dv=%d\n", pre_dv);
                }
                __syncwarp();
#endif

                // 2. retrieve the number of neighbors of each vertex
                // end_v[warp_id][depth[warp_id] - 2] should be equal to queue_pos[warp_id][pre_qv_idx - 2]
                rem_nbr_count[warp_id][lane_id] = lane_id < rem_pre_dv_count
                    ? min(index.sizes_[pre_qe_idx][pre_dv] - (lane_id == 0 ? end_nbr[warp_id][depth[warp_id] - 2] : 0u), 64u)
                    : 0u;
                __syncwarp();
                if (lane_id > 0) rem_nbr_count[warp_id][lane_id] = rem_nbr_count[warp_id][0];
                __syncwarp();
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("rem_nbr_count[%d][31]=%d\n", warp_id, rem_nbr_count[warp_id][31]);
                }
                __syncwarp();
#endif
                // each lane found a vertex temp_nbr
                uint32_t temp_nbr = UINT32_MAX;
                if (lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1])
                {
                    // 3. each lane gets a neighbor of a vertex in the previous level
                    if (pre_qv_idx == 0)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][partial[warp_id][0]][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                    else if (pre_qv_idx == 1)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][partial[warp_id][1]][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                    else
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][
                            result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]]
                        ][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
#ifdef DEBUG
                    printf("lane %d found process the vertex %d\n", lane_id, temp_nbr);
#endif
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // update start_v/nbr and end_v/nbr by the active lane with the greatest id
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("before update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - 2], start_nbr[warp_id][depth[warp_id] - 2], end_v[warp_id][depth[warp_id] - 2], end_nbr[warp_id][depth[warp_id] - 2]);
                }
                __syncwarp();
#endif
                uint8_t num_active_lanes = __popc(__ballot_sync(0xffffffff, lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1]));
                if (num_active_lanes == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id][depth[warp_id] - 2] += rem_pre_dv_count;
                        end_nbr[warp_id][depth[warp_id] - 2] = 0;
                    }
                    __syncwarp();
                }
                else
                {
                    if (lane_id == num_active_lanes - 1)
                    {
                        if (rem_nbr_count[warp_id][0] > lane_id + 1)
                        {
                            // not all neighbors of the last vertex are processed
                            end_nbr[warp_id][depth[warp_id] - 2] += lane_id + 1;
                        }
                        else
                        {
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            end_v[warp_id][depth[warp_id] - 2] += 1u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("after update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - 2], start_nbr[warp_id][depth[warp_id] - 2], end_v[warp_id][depth[warp_id] - 2], end_nbr[warp_id][depth[warp_id] - 2]);
                }
                __syncwarp();
#endif

                // 5. the lane search for the vertex on other nbr arrays
                bool found = lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1];
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found1 %d temp_nbr=%d\n", lane_id, temp_nbr);
#endif
                    if (temp_nbr == partial[warp_id][0] || temp_nbr == partial[warp_id][1])
                    {
                        found = false;
                    }
                    else
                    {
                        for (uint8_t i = 2u; i < depth[warp_id]; i++)
                        {
                            if (result_queue[warp_id][i - 2u][queue_pos[warp_id][i - 2]] == temp_nbr)
                            {
                                found = false;
                                break;
                            }
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found2 %d\n", lane_id);
#endif
                    for (uint8_t off = C_ORDERS[oi].bni_offs_[depth[warp_id]] + 1; off < C_ORDERS[oi].bni_offs_[depth[warp_id] + 1]; off++)
                    {
                        const uint8_t& bni = C_ORDERS[oi].bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
                        const uint32_t& pre_pre_v = bni == 0 ? partial[warp_id][0] : (bni == 1 ? partial[warp_id][1] : result_queue[warp_id][bni - 2][queue_pos[warp_id][bni - 2]]);
#ifdef DEBUG
                        printf("bni=%d, pre_pre_qe_idx=%d, pre_pre_v=%d\n", bni, pre_pre_qe_idx, pre_pre_v);
#endif
                        const uint32_t res = lower_bound(index.nbrs_[pre_pre_qe_idx][pre_pre_v], index.sizes_[pre_pre_qe_idx][pre_pre_v], temp_nbr);
                        if (res == index.sizes_[pre_pre_qe_idx][pre_pre_v] || index.nbrs_[pre_pre_qe_idx][pre_pre_v][res] != temp_nbr)
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // 6. write the local candidates to result_queue and their group id to group_id
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (depth[warp_id] < end_depth - 1)
                {
#ifdef DEBUG
                    printf("found3 %d\n", lane_id);
#endif
                    // do not need to store results at the least level
                    if (found) result_queue[warp_id][depth[warp_id] - 2][rank] = temp_nbr;
                }
                else
                {
#ifdef DEBUG
                    printf("found a match %d %d %d %d %d\n", v0, v1, result_queue[warp_id][0][queue_pos[warp_id][0]], result_queue[warp_id][1][queue_pos[warp_id][1]], temp_nbr);
#endif
                    if (write_res)
                    {
                        if (found_mask)
                        {
                            unsigned long long int write_pos;
                            if (lane_id == 0) write_pos = atomicAdd(new_res_size, __popc(found_mask));
                            write_pos = __shfl_sync(0xffffffff, write_pos, 0, 64);
                            if (write_pos + __popc(found_mask) > h_max_new_res_size_) return;
                            if (found)
                            {
                                write_pos += rank;
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth) % C_RES_QUEUE.capacity_] = partial[warp_id][0];
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 1) % C_RES_QUEUE.capacity_] = partial[warp_id][1];

                                for (uint8_t j = 2u; j < end_depth - 1; j++)
                                {
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capacity_] = result_queue[warp_id][j - 2][queue_pos[warp_id][j - 2]];
                                }
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capacity_] = temp_nbr;
                            }
                        }
                    }
                    else
                    {
                        if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
                    }
                }
                __syncwarp();
                if (found && rank == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = (depth[warp_id] < end_depth - 1) ? 0u : __popc(found_mask);
                    queue_size[warp_id][depth[warp_id] - 2] = __popc(found_mask);
                }
                __syncwarp();
            }
        }
        else // go to the next level
        {
            if (lane_id == 0 && depth[warp_id] < end_depth - 1)
            {
#ifdef DEBUG
                printf("increase from depth %d\n", depth[warp_id]);
#endif
                depth[warp_id] ++;
            }
            __syncwarp();
        }
    }
}

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
) {
    __shared__ uint32_t partial[NUM_WARP_PER_BLOCK][MAX_QV_COUNT];
    __shared__ uint32_t result_queue[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2][WARP_SIZE];
    __shared__ uint8_t queue_pos[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];
    __shared__ uint8_t queue_size[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ uint8_t end_v[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];
    __shared__ uint32_t end_nbr[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ bool intersection_continue[NUM_WARP_PER_BLOCK][MAX_QV_COUNT - 2];

    __shared__ uint8_t rem_nbr_count[NUM_WARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t depth[NUM_WARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;

    if (gwarp_id >= res_size)
    {
        return;
    }
    if (lane_id < start_depth)
    {
        partial[warp_id][lane_id] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + lane_id) % C_RES_QUEUE.capacity_];
    }
    __syncwarp();
#ifdef DEBUG
    if(lane_id == 0)
    {
        printf("%d %d\n", v0, v1);
    }
    __syncwarp();
#endif

    if (lane_id < C_QV_COUNT - 2)
    {
        queue_pos[warp_id][lane_id] = 0u;
        queue_size[warp_id][lane_id] = 0u;
        end_v[warp_id][lane_id] = 0u;
        end_nbr[warp_id][lane_id] = 0u;
        intersection_continue[warp_id][lane_id] = false;
    }
    if (lane_id == 0)
    {
        depth[warp_id] = start_depth;
    }
    __syncwarp();

    while (depth[warp_id] >= start_depth)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[depth[warp_id]]];
        const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
#ifdef DEBUG
        if(lane_id == 0)
        {
            printf("enter while loop depth=%d, pre_qv_idx=%d, pre_qe_idx=%d\n", (uint32_t)depth[warp_id], (uint32_t)pre_qv_idx, (uint32_t)pre_qe_idx);
        }
        __syncwarp();
#endif

        // check if all local candidates of this level are consumed,
        if (queue_pos[warp_id][depth[warp_id] - start_depth] >= queue_size[warp_id][depth[warp_id] - start_depth])
        {
#ifdef DEBUG
            if (lane_id == 0)
            {
                printf("depth=%d all results processed, (%d, %d)\n", (uint32_t)depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id][depth[warp_id] - start_depth], pre_qv_idx < 2 ? 0 :queue_pos[warp_id][pre_qv_idx - start_depth]);
            }
            __syncwarp();
#endif
            // check if there is no remaining intersection workload for the current level
            if (
                intersection_continue[warp_id][depth[warp_id] - start_depth] && 
                ((pre_qv_idx < start_depth && end_v[warp_id][depth[warp_id] - start_depth] > 0) ||
                (pre_qv_idx >= start_depth && end_v[warp_id][depth[warp_id] - start_depth] > queue_pos[warp_id][pre_qv_idx - start_depth]))
            ) {
                // no more intersection to do, go back to the previous level
                if (lane_id == 0)
                {
#ifdef DEBUG
                    printf("backtrack from level %d (%d, %d)\n", depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id][depth[warp_id] - start_depth], pre_qv_idx < 2 ? 0 :queue_pos[warp_id][pre_qv_idx - start_depth]);
#endif
                    queue_pos[warp_id][depth[warp_id] - start_depth] = 0u;
                    queue_size[warp_id][depth[warp_id] - start_depth] = 0u;

                    intersection_continue[warp_id][depth[warp_id] - start_depth] = false;

                    depth[warp_id]--;

                    if (depth[warp_id] >= start_depth)
                    {
                        queue_pos[warp_id][depth[warp_id] - start_depth]++;
                    }
                }
                __syncwarp();
            }
            else
            {
                if (!intersection_continue[warp_id][depth[warp_id] - start_depth])
                {
                    if (lane_id == 0)
                    {
#ifdef DEBUG
                        printf("START intersection\n");
#endif
                        if (pre_qv_idx < start_depth)
                        {
                            end_v[warp_id][depth[warp_id] - start_depth] = 0u;
                            end_nbr[warp_id][depth[warp_id] - start_depth] = 0u;
                        }
                        else
                        {
                            end_v[warp_id][depth[warp_id] - start_depth] = queue_pos[warp_id][pre_qv_idx - start_depth];
                            end_nbr[warp_id][depth[warp_id] - start_depth] = 0u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("perform intersection at level %d\n", depth[warp_id]);
                }
                __syncwarp();
#endif
                // need to start/continue the intersection
                
                intersection_continue[warp_id][depth[warp_id] - start_depth] = true;

                const uint8_t rem_pre_dv_count = 1u;
                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx < start_depth)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? partial[warp_id][pre_qv_idx] : UINT32_MAX;
                }
                else
                {
                    pre_dv = lane_id < rem_pre_dv_count ? result_queue[warp_id][pre_qv_idx - start_depth][queue_pos[warp_id][pre_qv_idx - start_depth]] : UINT32_MAX;
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("lane 0 have a pre_dv=%d\n", pre_dv);
                }
                __syncwarp();
#endif

                // 2. retrieve the number of neighbors of each vertex
                // end_v[warp_id][depth[warp_id] - start_depth] should be equal to queue_pos[warp_id][pre_qv_idx - start_depth]
                rem_nbr_count[warp_id][lane_id] = lane_id < rem_pre_dv_count
                    ? min(index.sizes_[pre_qe_idx][pre_dv] - (lane_id == 0 ? end_nbr[warp_id][depth[warp_id] - start_depth] : 0u), 64u)
                    : 0u;
                __syncwarp();
                if (lane_id > 0) rem_nbr_count[warp_id][lane_id] = rem_nbr_count[warp_id][0];
                __syncwarp();
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("rem_nbr_count[%d][31]=%d\n", warp_id, rem_nbr_count[warp_id][31]);
                }
                __syncwarp();
#endif
                // each lane found a vertex temp_nbr
                uint32_t temp_nbr = UINT32_MAX;
                if (lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1])
                {
                    // 3. each lane gets a neighbor of a vertex in the previous level
                    if (pre_qv_idx < start_depth)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][partial[warp_id][pre_qv_idx]][end_nbr[warp_id][depth[warp_id] - start_depth] + lane_id];
                    }
                    else
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][
                            result_queue[warp_id][pre_qv_idx - start_depth][queue_pos[warp_id][pre_qv_idx - start_depth]]
                        ][end_nbr[warp_id][depth[warp_id] - start_depth] + lane_id];
                    }
#ifdef DEBUG
                    printf("lane %d found process the vertex %d\n", lane_id, temp_nbr);
#endif
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // update start_v/nbr and end_v/nbr by the active lane with the greatest id
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("before update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - start_depth], start_nbr[warp_id][depth[warp_id] - start_depth], end_v[warp_id][depth[warp_id] - start_depth], end_nbr[warp_id][depth[warp_id] - start_depth]);
                }
                __syncwarp();
#endif
                uint8_t num_active_lanes = __popc(__ballot_sync(0xffffffff, lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1]));
                if (num_active_lanes == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id][depth[warp_id] - start_depth] += rem_pre_dv_count;
                        end_nbr[warp_id][depth[warp_id] - start_depth] = 0;
                    }
                    __syncwarp();
                }
                else
                {
                    if (lane_id == num_active_lanes - 1)
                    {
                        if (rem_nbr_count[warp_id][0] > lane_id + 1)
                        {
                            // not all neighbors of the last vertex are processed
                            end_nbr[warp_id][depth[warp_id] - start_depth] += lane_id + 1;
                        }
                        else
                        {
                            end_nbr[warp_id][depth[warp_id] - start_depth] = 0u;
                            end_v[warp_id][depth[warp_id] - start_depth] += 1u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("after update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - start_depth], start_nbr[warp_id][depth[warp_id] - start_depth], end_v[warp_id][depth[warp_id] - start_depth], end_nbr[warp_id][depth[warp_id] - start_depth]);
                }
                __syncwarp();
#endif

                // 5. the lane search for the vertex on other nbr arrays
                bool found = lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1];
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - start_depth] = 0u;
                    queue_size[warp_id][depth[warp_id] - start_depth] = 0u;
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found1 %d temp_nbr=%d\n", lane_id, temp_nbr);
#endif
                    for (uint8_t i = 0u; i < depth[warp_id]; i++)
                    {
                        if (i < start_depth && partial[warp_id][i] == temp_nbr)
                        {
                            found = false;
                            break;
                        }
                        else if (i >= start_depth && result_queue[warp_id][i - start_depth][queue_pos[warp_id][i - start_depth]] == temp_nbr)
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found2 %d\n", lane_id);
#endif
                    for (uint8_t off = C_ORDERS[oi].bni_offs_[depth[warp_id]] + 1; off < C_ORDERS[oi].bni_offs_[depth[warp_id] + 1]; off++)
                    {
                        const uint8_t& bni = C_ORDERS[oi].bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
                        const uint32_t& pre_pre_v = (bni < start_depth) ? partial[warp_id][bni] : result_queue[warp_id][bni - start_depth][queue_pos[warp_id][bni - start_depth]];
#ifdef DEBUG
                        printf("bni=%d, pre_pre_qe_idx=%d, pre_pre_v=%d\n", bni, pre_pre_qe_idx, pre_pre_v);
#endif
                        const uint32_t res = lower_bound(index.nbrs_[pre_pre_qe_idx][pre_pre_v], index.sizes_[pre_pre_qe_idx][pre_pre_v], temp_nbr);
                        if (res == index.sizes_[pre_pre_qe_idx][pre_pre_v] || index.nbrs_[pre_pre_qe_idx][pre_pre_v][res] != temp_nbr)
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // 6. write the local candidates to result_queue and their group id to group_id
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (depth[warp_id] < end_depth - 1)
                {
#ifdef DEBUG
                    printf("found3 %d\n", lane_id);
#endif
                    // do not need to store results at the least level
                    if (found) result_queue[warp_id][depth[warp_id] - start_depth][rank] = temp_nbr;
                }
                else
                {
#ifdef DEBUG
                    printf("found a match %d %d %d %d %d\n", v0, v1, result_queue[warp_id][0][queue_pos[warp_id][0]], result_queue[warp_id][1][queue_pos[warp_id][1]], temp_nbr);
#endif
                    if (write_res)
                    {
                        if (found_mask)
                        {
                            unsigned long long int write_pos;
                            if (lane_id == 0) write_pos = atomicAdd(new_res_size, __popc(found_mask));
                            write_pos = __shfl_sync(0xffffffff, write_pos, 0, 64);
                            if (write_pos + __popc(found_mask) > h_max_new_res_size_) return;
                            if (found)
                            {
                                write_pos += rank;
                                for (uint8_t j = 0u; j < end_depth - 1; j++)
                                {
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capacity_] = 
                                        (j < start_depth) ? partial[warp_id][j] : result_queue[warp_id][j - start_depth][queue_pos[warp_id][j - start_depth]];
                                }
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capacity_] = temp_nbr;
                            }
                        }
                    }
                    else
                    {
                        if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
                    }
                }
                __syncwarp();
                if (found && rank == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - start_depth] = (depth[warp_id] < end_depth - 1) ? 0u : __popc(found_mask);
                    queue_size[warp_id][depth[warp_id] - start_depth] = __popc(found_mask);
                }
                __syncwarp();
            }
        }
        else // go to the next level
        {
            if (lane_id == 0 && depth[warp_id] < end_depth - 1)
            {
#ifdef DEBUG
                printf("increase from depth %d\n", depth[warp_id]);
#endif
                depth[warp_id] ++;
            }
            __syncwarp();
        }
    }
}

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
) {
    extern __shared__ uint32_t _s[];
    uint32_t *partial = _s;
    uint32_t *result_queue = partial + NUM_WARP_PER_BLOCK * start_depth;
    uint8_t *queue_pos = (uint8_t*) (result_queue + NUM_WARP_PER_BLOCK * (end_depth - start_depth) * WARP_SIZE);
    uint8_t *queue_size = queue_pos + NUM_WARP_PER_BLOCK * (end_depth - start_depth);

    uint8_t *end_v = queue_size + NUM_WARP_PER_BLOCK * (end_depth - start_depth);
    uint32_t *end_nbr = (uint32_t*)(end_v + NUM_WARP_PER_BLOCK * (end_depth - start_depth));

    bool *intersection_continue = (bool*)(end_nbr + NUM_WARP_PER_BLOCK * (end_depth - start_depth));

    uint8_t *rem_nbr_count = (uint8_t*) (intersection_continue + NUM_WARP_PER_BLOCK * (end_depth - start_depth));
    uint8_t *depth = rem_nbr_count + NUM_WARP_PER_BLOCK * WARP_SIZE;

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;

    if (gwarp_id >= res_size)
    {
        return;
    }
    if (lane_id < start_depth)
    {
        partial[warp_id * start_depth + lane_id] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + lane_id) % C_RES_QUEUE.capacity_];
    }
    __syncwarp();
#ifdef DEBUG
    if(lane_id == 0)
    {
        printf("%d %d\n", v0, v1);
    }
    __syncwarp();
#endif

    if (lane_id < end_depth - start_depth)
    {
        queue_pos[warp_id * (end_depth - start_depth) + lane_id] = 0u;
        queue_size[warp_id * (end_depth - start_depth) + lane_id] = 0u;
        end_v[warp_id * (end_depth - start_depth) + lane_id] = 0u;
        end_nbr[warp_id * (end_depth - start_depth) + lane_id] = 0u;
        intersection_continue[warp_id * (end_depth - start_depth) + lane_id] = false;
    }
    if (lane_id == 0)
    {
        depth[warp_id] = start_depth;
    }
    __syncwarp();

    while (depth[warp_id] >= start_depth)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[depth[warp_id]]];
        const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
#ifdef DEBUG
        if(lane_id == 0)
        {
            printf("enter while loop depth=%d, pre_qv_idx=%d, pre_qe_idx=%d\n", (uint32_t)depth[warp_id], (uint32_t)pre_qv_idx, (uint32_t)pre_qe_idx);
        }
        __syncwarp();
#endif

        // check if all local candidates of this level are consumed,
        if (queue_pos[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] >= queue_size[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth])
        {
#ifdef DEBUG
            if (lane_id == 0)
            {
                printf("depth=%d all results processed, (%d, %d)\n", (uint32_t)depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth], pre_qv_idx < 2 ? 0 :queue_pos[warp_id * (end_depth - start_depth) + pre_qv_idx - start_depth]);
            }
            __syncwarp();
#endif
            // check if there is no remaining intersection workload for the current level
            if (
                intersection_continue[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] && 
                ((pre_qv_idx < start_depth && end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] > 0) ||
                (pre_qv_idx >= start_depth && end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] > queue_pos[warp_id * (end_depth - start_depth) + pre_qv_idx - start_depth]))
            ) {
                // no more intersection to do, go back to the previous level
                if (lane_id == 0)
                {
#ifdef DEBUG
                    printf("backtrack from level %d (%d, %d)\n", depth[warp_id], pre_qv_idx < 2 ? 0 :end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth], pre_qv_idx < 2 ? 0 :queue_pos[warp_id * (end_depth - start_depth) + pre_qv_idx - start_depth]);
#endif
                    queue_pos[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0u;
                    queue_size[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0u;

                    intersection_continue[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = false;

                    depth[warp_id]--;

                    if (depth[warp_id] >= start_depth)
                    {
                        queue_pos[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth]++;
                    }
                }
                __syncwarp();
            }
            else
            {
                if (!intersection_continue[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth])
                {
                    if (lane_id == 0)
                    {
#ifdef DEBUG
                        printf("START intersection\n");
#endif
                        if (pre_qv_idx < start_depth)
                        {
                            end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0u;
                            end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0u;
                        }
                        else
                        {
                            end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = queue_pos[warp_id * (end_depth - start_depth) + pre_qv_idx - start_depth];
                            end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("perform intersection at level %d\n", depth[warp_id]);
                }
                __syncwarp();
#endif
                // need to start/continue the intersection
                
                intersection_continue[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = true;

                const uint8_t rem_pre_dv_count = 1u;
                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx < start_depth)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? partial[warp_id * start_depth + pre_qv_idx] : UINT32_MAX;
                }
                else
                {
                    pre_dv = lane_id < rem_pre_dv_count ? result_queue[warp_id * (end_depth - start_depth) * WARP_SIZE + (pre_qv_idx - start_depth) * WARP_SIZE + queue_pos[warp_id * (end_depth - start_depth) + pre_qv_idx - start_depth]] : UINT32_MAX;
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("lane 0 have a pre_dv=%d\n", pre_dv);
                }
                __syncwarp();
#endif

                // 2. retrieve the number of neighbors of each vertex
                // end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] should be equal to queue_pos[warp_id * (end_depth - start_depth) + pre_qv_idx - start_depth]
                rem_nbr_count[warp_id * WARP_SIZE + lane_id] = lane_id < rem_pre_dv_count
                    ? min(index.sizes_[pre_qe_idx][pre_dv] - (lane_id == 0 ? end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] : 0u), 64u)
                    : 0u;
                __syncwarp();
                if (lane_id > 0) rem_nbr_count[warp_id * WARP_SIZE + lane_id] = rem_nbr_count[warp_id * WARP_SIZE + 0];
                __syncwarp();
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("rem_nbr_count[%d][31]=%d\n", warp_id, rem_nbr_count[warp_id][31]);
                }
                __syncwarp();
#endif
                // each lane found a vertex temp_nbr
                uint32_t temp_nbr = UINT32_MAX;
                if (lane_id < rem_nbr_count[warp_id * WARP_SIZE + WARP_SIZE - 1])
                {
                    // 3. each lane gets a neighbor of a vertex in the previous level
                    if (pre_qv_idx < start_depth)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][partial[warp_id * start_depth + pre_qv_idx]][end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] + lane_id];
                    }
                    else
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][
                            result_queue[warp_id * (end_depth - start_depth) * WARP_SIZE + (pre_qv_idx - start_depth) * WARP_SIZE + queue_pos[warp_id * (end_depth - start_depth) + pre_qv_idx - start_depth]]
                        ][end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] + lane_id];
                    }
#ifdef DEBUG
                    printf("lane %d found process the vertex %d\n", lane_id, temp_nbr);
#endif
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // update start_v/nbr and end_v/nbr by the active lane with the greatest id
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("before update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id][depth[warp_id] - start_depth], start_nbr[warp_id][depth[warp_id] - start_depth], end_v[warp_id][depth[warp_id] - start_depth], end_nbr[warp_id][depth[warp_id] - start_depth]);
                }
                __syncwarp();
#endif
                uint8_t num_active_lanes = __popc(__ballot_sync(0xffffffff, lane_id < rem_nbr_count[warp_id * WARP_SIZE + WARP_SIZE - 1]));
                if (num_active_lanes == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] += rem_pre_dv_count;
                        end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0;
                    }
                    __syncwarp();
                }
                else
                {
                    if (lane_id == num_active_lanes - 1)
                    {
                        if (rem_nbr_count[warp_id * WARP_SIZE + 0] > lane_id + 1)
                        {
                            // not all neighbors of the last vertex are processed
                            end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] += lane_id + 1;
                        }
                        else
                        {
                            end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0u;
                            end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] += 1u;
                        }
                    }
                    __syncwarp();
                }
#ifdef DEBUG
                if (lane_id == 0)
                {
                    printf("after update start_v=%d, start_nbr=%d, end_v=%d, end_nbr=%d\n", start_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth], start_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth], end_v[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth], end_nbr[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth]);
                }
                __syncwarp();
#endif

                // 5. the lane search for the vertex on other nbr arrays
                bool found = lane_id < rem_nbr_count[warp_id * WARP_SIZE + WARP_SIZE - 1];
                if (lane_id == 0)
                {
                    queue_pos[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0u;
                    queue_size[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = 0u;
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found1 %d temp_nbr=%d\n", lane_id, temp_nbr);
#endif
                    for (uint8_t i = 0u; i < depth[warp_id]; i++)
                    {
                        if (i < start_depth && partial[warp_id * start_depth + i] == temp_nbr)
                        {
                            found = false;
                            break;
                        }
                        else if (i >= start_depth && result_queue[warp_id * (end_depth - start_depth) * WARP_SIZE + (i - start_depth) * WARP_SIZE + queue_pos[warp_id * (end_depth - start_depth) + i - start_depth]] == temp_nbr)
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
#ifdef DEBUG
                    printf("found2 %d\n", lane_id);
#endif
                    for (uint8_t off = C_ORDERS[oi].bni_offs_[depth[warp_id]] + 1; off < C_ORDERS[oi].bni_offs_[depth[warp_id] + 1]; off++)
                    {
                        uint8_t bni = C_ORDERS[oi].bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
                        const uint32_t& pre_pre_v = (bni < start_depth) ? partial[warp_id * start_depth + bni] : result_queue[warp_id * (end_depth - start_depth) * WARP_SIZE + (bni - start_depth) * WARP_SIZE + queue_pos[warp_id * (end_depth - start_depth) + bni - start_depth]];
#ifdef DEBUG
                        printf("bni=%d, pre_pre_qe_idx=%d, pre_pre_v=%d\n", bni, pre_pre_qe_idx, pre_pre_v);
#endif
                        uint32_t res = lower_bound(index.nbrs_[pre_pre_qe_idx][pre_pre_v], index.sizes_[pre_pre_qe_idx][pre_pre_v], temp_nbr);
                        if (res == index.sizes_[pre_pre_qe_idx][pre_pre_v] || index.nbrs_[pre_pre_qe_idx][pre_pre_v][res] != temp_nbr)
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // 6. write the local candidates to result_queue and their group id to group_id
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (depth[warp_id] < end_depth - 1)
                {
#ifdef DEBUG
                    printf("found3 %d\n", lane_id);
#endif
                    // do not need to store results at the least level
                    if (found) result_queue[warp_id * (end_depth - start_depth) * WARP_SIZE + (depth[warp_id] - start_depth) * WARP_SIZE + rank] = temp_nbr;
                }
                else
                {
#ifdef DEBUG
                    printf("found a match %d %d %d %d %d\n", v0, v1, result_queue[warp_id][0][queue_pos[warp_id][0]], result_queue[warp_id][1][queue_pos[warp_id][1]], temp_nbr);
#endif
                    if (write_res)
                    {
                        if (found_mask)
                        {
                            unsigned long long int write_pos;
                            if (lane_id == 0) write_pos = atomicAdd(new_res_size, __popc(found_mask));
                            write_pos = __shfl_sync(0xffffffff, write_pos, 0, 64);
                            if (write_pos + __popc(found_mask) > h_max_new_res_size_) return;
                            if (found)
                            {
                                write_pos += rank;
                                for (uint8_t j = 0u; j < end_depth - 1; j++)
                                {
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capacity_] = 
                                        (j < start_depth) ? partial[warp_id * start_depth + j] : result_queue[warp_id * (end_depth - start_depth) * WARP_SIZE + (j - start_depth) * WARP_SIZE + queue_pos[warp_id * (end_depth - start_depth) + j - start_depth]];
                                }
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capacity_] = temp_nbr;
                            }
                        }
                    }
                    else
                    {
                        if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
                    }
                }
                __syncwarp();
                if (found && rank == 0)
                {
                    queue_pos[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = (depth[warp_id] < end_depth - 1) ? 0u : __popc(found_mask);
                    queue_size[warp_id * (end_depth - start_depth) + depth[warp_id] - start_depth] = __popc(found_mask);
                }
                __syncwarp();
            }
        }
        else // go to the next level
        {
            if (lane_id == 0 && depth[warp_id] < end_depth - 1)
            {
#ifdef DEBUG
                printf("increase from depth %d\n", depth[warp_id]);
#endif
                depth[warp_id] ++;
            }
            __syncwarp();
        }
    }
}
