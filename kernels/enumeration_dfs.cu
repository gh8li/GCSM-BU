#include <cstdint>

#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/match_gpu.h"


__global__ void extendDFS(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const RelationsGPU index,
    const uint8_t oi
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
    uint32_t v0 = C_RES_QUEUE.array_[(res + gwarp_id * 2) % C_RES_QUEUE.capacity_];
    uint32_t v1 = C_RES_QUEUE.array_[(res + gwarp_id * 2 + 1) % C_RES_QUEUE.capacity_];
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
        depth[warp_id] = 2u;
    }
    __syncwarp();

    while (depth[warp_id] >= 2)
    {
        __syncwarp();
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

                uint8_t rem_pre_dv_count = 1u;
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
                rem_nbr_count[warp_id][lane_id] = lane_id < rem_pre_dv_count
                    ? min(index.sizes_[pre_qe_idx][pre_dv] - (lane_id == 0 ? end_nbr[warp_id][depth[warp_id] - 2] : 0u), 64u)
                    : 0u;
                
                // prefix sum
                /*for (uint8_t i = 1; i < WARP_SIZE; i *= 2)
                {
                    uint8_t temp_rem_nbr_count = 0u;
                    if (lane_id >= i)
                    {
                        temp_rem_nbr_count = rem_nbr_count[warp_id][lane_id - i];
                    }
                    __syncwarp();
                    if (lane_id >= i)
                    {
                        rem_nbr_count[warp_id][lane_id] += temp_rem_nbr_count;
                        rem_nbr_count[warp_id][lane_id] = min(rem_nbr_count[warp_id][lane_id], 64u);
                    }
                    __syncwarp();
                }*/
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

                // 6. write the local candidates to result_queue and their group id to group_id
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                if (found)
                {
#ifdef DEBUG
                    printf("found3 %d\n", lane_id);
#endif
                    const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                    // do not need to store results at the least level
                    if (depth[warp_id] < C_QV_COUNT - 1)
                    {
                        result_queue[warp_id][depth[warp_id] - 2][rank] = temp_nbr;
                    }
                    else
                    {
                        if (rank == 0) atomicAdd(new_res_size, __popc(found_mask));
#ifdef DEBUG
                        printf("found a match %d %d %d %d %d\n", v0, v1, result_queue[warp_id][0][queue_pos[warp_id][0]], result_queue[warp_id][1][queue_pos[warp_id][1]], temp_nbr);
#endif
                    }
                    if (rank == 0)
                    {
                        queue_pos[warp_id][depth[warp_id] - 2] = depth[warp_id] < C_QV_COUNT - 1 ? 0u : __popc(found_mask);
                        queue_size[warp_id][depth[warp_id] - 2] = __popc(found_mask);
                    }
                }
                __syncwarp();
            }
        }
        else // go to the next level
        {
            if (lane_id == 0 && depth[warp_id] < C_QV_COUNT - 1)
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
