#include <iostream>
#include "cub/cub.cuh"
// #include <cub/cub.cuh>

#include "graph/graph.h"
#include "graph/match_gpu.h"
#include "kernels/cartesian_product.h"
#include "kernels/enumeration.h"
#include "kernels/gamma_enumeration.h"
#include "kernels/indexing.h"
#include "kernels/ours_enumeration.h"
#include "utils/config.h"
#include "utils/constants.h"
#include "utils/cuda_helpers.h"
#include "utils/globals.h"
#include "utils/types.h"

// #define IS_DEBUGGING_MATCH_GPU

__constant__ OrderPerEdge C_INDEXING_ORDERS[MAX_QE_COUNT];
__constant__ OrderPerEdge C_ORDERS[MAX_QE_COUNT];
__constant__ uint8_t C_EQUIV_EDGE_COUNT[MAX_QE_COUNT];
__constant__ uint8_t C_NUM_EQUIV_GROUPS;
__constant__ uint8_t C_NON_TAIL_LEAF_DEPTHS[MAX_QE_COUNT];

RelationsGPU::RelationsGPU(): nbrs_(), nbrs_is_update_(), capacity_(), sizes_() {}

__global__ void AccessdCandCount(uint32_t *d_new_cand_count[]) {
    uint32_t uint1 = *(d_new_cand_count[0]);
    uint32_t uint2 = *(d_new_cand_count[1]);
    printf("d_new_cand_count_[1]: %d\n", uint1);
    printf("d_new_cand_count_[0]: %d\n", uint2);
}

// RelationsGPU& RelationsGPU::CopyFrom(const RelationsGPU &other) {

//     for (auto edge_list_idx = 0u; edge_list_idx < QE_COUNT; edge_list_idx++) {
//         cudaErrorCheck(cudaMemcpy(this->nbrs_[edge_list_idx], other.nbrs_[edge_list_idx], 
//                                   sizeof(uint32_t*) * (DV_COUNT + 1), cudaMemcpyDeviceToDevice));
//         cudaErrorCheck(cudaMemcpy(this->capacity_[edge_list_idx], other.capacity_[edge_list_idx], 
//                                   sizeof(uint32_t) * (DV_COUNT + 1), cudaMemcpyDeviceToDevice));
//         cudaErrorCheck(cudaMemcpy(this->sizes_[edge_list_idx], other.sizes_[edge_list_idx], 
//                                   sizeof(uint32_t) * (DV_COUNT + 1), cudaMemcpyDeviceToDevice));
//     }

//     if (build_graph)
//     {
//         cudaErrorCheck(cudaMemcpy(data_graph_gpu.capacity_[idx], global_index_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1), cudaMemcpyDeviceToDevice));
//     }

//     CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, global_index_gpu.capacity_[idx], capacity_prefix_sum, DV_COUNT + 1));
    
//     // d. set nbrs
//     uint32_t total_size;
//     cudaErrorCheck(cudaMemcpy(&total_size, &capacity_prefix_sum[DV_COUNT], sizeof(uint32_t), cudaMemcpyDeviceToHost));

//     cudaErrorCheck(cudaMalloc(&index_all_nbrs, sizeof(uint32_t) * total_size));
//     cudaErrorCheck(cudaMemset(index_all_nbrs, 0u, sizeof(uint32_t) * total_size));
//     cudaErrorCheck(cudaDeviceSynchronize());
//     setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(index_all_nbrs, capacity_prefix_sum, DV_COUNT, global_index_gpu.nbrs_[idx]);

//     return *self;
// }

RelationsGPUAllVersions::RelationsGPUAllVersions() : array_data_graphs_() {}

CandidatesGPUAllVersions::CandidatesGPUAllVersions(): array_candidates_gpu_() {}

// __host__ void RelationsGPUAllVersions::AllocateAllRelations(const QueryGraph &query_) {
//     uint32_t num_query_edges = query_.getEdgeCount();
//     for (uint8_t i = 0u; i < num_query_edges; i++) {
//         for (const uint8_t& idx: {query_.getEdgeIdxByEdgeListIdx(i), query_.getTheOtherEdgeIdxByEdgeListIdx(i)}) {
//             cudaErrorCheck(cudaMalloc(&local_index_base_gpu.sizes_[idx], sizeof(uint32_t) * (DV_COUNT + 1u)));
//             cudaErrorCheck(cudaMalloc(&local_index_base_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1u)));
//             cudaErrorCheck(cudaMalloc(&local_index_base_gpu.nbrs_[idx], sizeof(uint32_t*) * (DV_COUNT)));
//         }
//     }
// }

CandidatesGPU::CandidatesGPU()
: candidate_bits_()
{}

MatchGPU::MatchGPU(
    const GraphFileIoManager& graph_file_io, 
    const QueryGraph& query,
    const PlanManager& plan)
: query_(query)
, plan_(plan)
, data_graph_all_nbrs_{NULL}
, index_all_nbrs_{NULL}

, d_temp_storage_(NULL)
, temp_storage_bytes_(0ul)
, temp_storage_capacity_(0ul)
, cand_flag_(NULL)
, cand_flag_capacity_(0u)
, d_new_cand_count_{NULL, NULL}
, temp_tries_()
, temp_tries_capacity_{{0u,0u,0u}, {0u,0u,0u}}
, helper_relation_{NULL, NULL}
, helper_relation_capacity_{0u}
, local_nbr_()
, local_nbr_capacity_{0u}
, cum_bn_()

, res_(0ul)
, res_size_(0ul)
, new_res_(0ul)
, new_res_size_(NULL)
, h_new_res_size_(0ul)
, h_max_new_res_size_(0ul)
, cur_depth_(0u)
, new_depth_(0u)

, nbr_mem_pool_()
, res_queue_()
, res_size_cartesian_product_(NULL)
, max_res_size_cartesian_product_(NULL)
{
    nbr_mem_pool_.Alloc(NBR_SPACE);
    cudaErrorCheck(cudaMalloc(&max_res_size_cartesian_product_, sizeof(unsigned long)));

    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[0], sizeof(uint32_t)));
    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[1], sizeof(uint32_t)));

    cudaErrorCheck(cudaMalloc(&new_res_size_, sizeof(unsigned long long int)));
}

MatchGPU::MatchGPU(
    const GraphFileIoManager& graph_file_io, 
    const QueryGraph& query,
    const PlanManager& plan,
    const AutomorphismManager *am_ptr)
: query_(query)
, plan_(plan)
, am_ptr_(am_ptr)
, data_graph_all_nbrs_{NULL}
, index_all_nbrs_{NULL}

, d_temp_storage_(NULL)
, temp_storage_bytes_(0ul)
, temp_storage_capacity_(0ul)
, cand_flag_(NULL)
, cand_flag_capacity_(0u)
, d_new_cand_count_{NULL, NULL}
, temp_tries_()
, temp_tries_capacity_{{0u,0u,0u}, {0u,0u,0u}}
, helper_relation_{NULL, NULL}
, helper_relation_capacity_{0u}
, local_nbr_()
, local_nbr_capacity_{0u}
, cum_bn_()

, res_(0ul)
, res_size_(0ul)
, new_res_(0ul)
, new_res_size_(NULL)
, h_new_res_size_(0ul)
, h_max_new_res_size_(0ul)
, cur_depth_(0u)
, new_depth_(0u)

, nbr_mem_pool_()
, res_queue_()
, res_size_cartesian_product_(NULL)
, max_res_size_cartesian_product_(NULL)
{
    nbr_mem_pool_.Alloc(NBR_SPACE);
    cudaErrorCheck(cudaMalloc(&max_res_size_cartesian_product_, sizeof(unsigned long)));

    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[0], sizeof(uint32_t)));
    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[1], sizeof(uint32_t)));

    cudaErrorCheck(cudaMalloc(&new_res_size_, sizeof(unsigned long long int)));
}

MatchGPU::~MatchGPU()
{
    nbr_mem_pool_.Free();

    cudaErrorCheck(cudaFree(new_res_size_));

    cudaErrorCheck(cudaFree(d_new_cand_count_[0]));
    cudaErrorCheck(cudaFree(d_new_cand_count_[1]));

    if (temp_storage_capacity_ > 0u) cudaErrorCheck(cudaFree(d_temp_storage_));
    if (cand_flag_capacity_ > 0u) cudaErrorCheck(cudaFree(cand_flag_));

    for (auto i = 0u; i < 2u; i++)
    {
        if (temp_tries_capacity_[i].vs_capacity_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].vs_));
        if (temp_tries_capacity_[i].off_capacity_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].offs_));
        if (temp_tries_capacity_[i].es_capacity_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].nbrs_));

        if (helper_relation_capacity_[i] > 0u) cudaErrorCheck(cudaFree(helper_relation_[i]));
    }

    for (auto i = 0u; i < QE_COUNT; i++)
    {
        if (local_nbr_capacity_[query_.qe_eidx_[i].first] > 0u) cudaErrorCheck(cudaFree(local_nbr_[query_.qe_eidx_[i].first]));
        if (local_nbr_capacity_[query_.qe_eidx_[i].second] > 0u) cudaErrorCheck(cudaFree(local_nbr_[query_.qe_eidx_[i].second]));
    }
}

void MatchGPU::LoadQuery()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_QV_COUNT, &QV_COUNT, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_QE_COUNT, &QE_COUNT, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_QV_OFFS, query_.qv_offs_.data(), sizeof(uint8_t) * (QV_COUNT + 1u)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NLF, query_.NLF_.data(), sizeof(uint8_t) * QE_COUNT * 2u));
    cudaErrorCheck(cudaMemcpyToSymbol(C_EIDX, query_.eidx_.data(), sizeof(uint8_t) * QV_COUNT * QV_COUNT));
}

void MatchGPU::LoadPlan()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_ORDERS, plan_.orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
    cudaErrorCheck(cudaMemcpyToSymbol(C_INDEXING_ORDERS, plan_.indexing_orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
}

void MatchGPU::OursLoadPlan()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_ORDERS, plan_.orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
    cudaErrorCheck(cudaMemcpyToSymbol(C_INDEXING_ORDERS, plan_.indexing_orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
    
    std::vector<uint8_t> non_tail_leaf_depths = plan_.getNonTailLeafDepths();
    // std::vector<uint8_t> non_tail_leaf_depths = std::vector<uint8_t>(query_.getEdgeCount(), 3u);

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "C_NON_TAIL_LEAF_DEPTHS: ";
    for (int i = 0; i < non_tail_leaf_depths.size(); i++) {
        std::cout << " " << (uint32_t)non_tail_leaf_depths[i] << " ";
    }
    std::cout << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaMemcpyToSymbol(C_NON_TAIL_LEAF_DEPTHS, non_tail_leaf_depths.data(), sizeof(uint8_t) * non_tail_leaf_depths.size()));
}

void MatchGPU::GammaLoadPlan()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_ORDERS, plan_.orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
    cudaErrorCheck(cudaMemcpyToSymbol(C_INDEXING_ORDERS, plan_.indexing_orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "C_EQUIV_EDGE_COUNT: ";
    const uint8_t *equiv_edge_count_array_hptr = am_ptr_->getEquivEdgeCountArrayPointer();
    for (int i = 0; i < MAX_QE_COUNT; i++) {
        std::cout << " " << (uint32_t)equiv_edge_count_array_hptr[i] << " ";
    }
    std::cout << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaMemcpyToSymbol(C_EQUIV_EDGE_COUNT, am_ptr_->getEquivEdgeCountArrayPointer(), sizeof(uint8_t) * MAX_QE_COUNT));
    
    uint8_t num_groups = am_ptr_->getNumEquivGroups();
    cudaErrorCheck(cudaMemcpyToSymbol(C_NUM_EQUIV_GROUPS, &num_groups, sizeof(uint8_t)));
    
    std::vector<uint8_t> non_tail_leaf_depths = plan_.getNonTailLeafDepths();
    // std::vector<uint8_t> non_tail_leaf_depths = std::vector<uint8_t>(query_.getEdgeCount(), 3u);

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "C_NON_TAIL_LEAF_DEPTHS: ";
    for (int i = 0; i < non_tail_leaf_depths.size(); i++) {
        std::cout << " " << (uint32_t)non_tail_leaf_depths[i] << " ";
    }
    std::cout << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaMemcpyToSymbol(C_NON_TAIL_LEAF_DEPTHS, non_tail_leaf_depths.data(), sizeof(uint8_t) * non_tail_leaf_depths.size()));
}

void MatchGPU::CorrectGammaLoadPlan()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_ORDERS, plan_.orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));
    cudaErrorCheck(cudaMemcpyToSymbol(C_INDEXING_ORDERS, plan_.indexing_orders_, sizeof(OrderPerEdge) * MAX_QE_COUNT));

    // uint8_t equiv_edge_count[MAX_QE_COUNT] = {1u};
    uint8_t equiv_edge_count[MAX_QE_COUNT];
    for (int i = 0; i < MAX_QE_COUNT; i++) {
        equiv_edge_count[i] = 1u;
    }
    cudaErrorCheck(cudaMemcpyToSymbol(C_EQUIV_EDGE_COUNT, equiv_edge_count, sizeof(uint8_t) * MAX_QE_COUNT));
    
    uint8_t num_groups = query_.getEdgeCount();
    cudaErrorCheck(cudaMemcpyToSymbol(C_NUM_EQUIV_GROUPS, &num_groups, sizeof(uint8_t)));
    
    std::vector<uint8_t> non_tail_leaf_depths = plan_.getNonTailLeafDepths();
    // std::vector<uint8_t> non_tail_leaf_depths = std::vector<uint8_t>(query_.getEdgeCount(), 3u);

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "C_NON_TAIL_LEAF_DEPTHS: ";
    for (int i = 0; i < non_tail_leaf_depths.size(); i++) {
        std::cout << " " << (uint32_t)non_tail_leaf_depths[i] << " ";
    }
    std::cout << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaMemcpyToSymbol(C_NON_TAIL_LEAF_DEPTHS, non_tail_leaf_depths.data(), sizeof(uint8_t) * non_tail_leaf_depths.size()));
}

void MatchGPU::BuildTries(const EdgeBatch& edge_lists, CSR_GPU csr_gpu[], const uint8_t i)
{
    // build csr_gpu based on the initial edge lists

    // uint32_t temp;  // 250216
    // cudaErrorCheck(cudaMemcpy(&temp, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
    // cout << "inside BuildTries - before BuildTriesFromEdgeList, d_new_cand_count_[0]: " << d_new_cand_count_[0] << endl;  // 250216
    // cout << "inside BuildTries - before BuildTriesFromEdgeList, d_new_cand_count_[1]: " << d_new_cand_count_[1] << endl;  // 250216
    
    BuildTriesFromEdgeList(
        edge_lists[query_.qe_eidx_[i].first], 
        csr_gpu[query_.qe_eidx_[i].first], csr_gpu[query_.qe_eidx_[i].second]);
        
    // cout << "inside BuildTries - after BuildTriesFromEdgeList, d_new_cand_count_[0]: " << d_new_cand_count_[0] << endl;  // 250216
    // cout << "inside BuildTries - after BuildTriesFromEdgeList, d_new_cand_count_[1]: " << d_new_cand_count_[1] << endl;  // 250216
    // uint32_t temp2;  // 250216
    // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
    // cout << "temp2: " << temp2 << endl;  // 250216
    // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
    // cout << "temp2: " << temp2 << endl;  // 250216
}

void MatchGPU::DeallocTries(CSR_GPU csr_gpu[])
{
    for (auto i = 0u; i < QE_COUNT * 2u; i++)
    {
        csr_gpu[i].vs_size_ = 0u;
        csr_gpu[i].es_size_ = 0u;
        cudaErrorCheck(cudaFree(csr_gpu[i].vs_));
        cudaErrorCheck(cudaFree(csr_gpu[i].offs_));
        cudaErrorCheck(cudaFree(csr_gpu[i].nbrs_));
    }
}

void MatchGPU::AllocRelations(
    const DataGraphManager& data_graph, const CSR_GPU csr_gpu[],
    RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
    CandidatesGPU& global_bitmap_gpu
) {
    cudaErrorCheck(cudaMemcpyToSymbol(C_DV_COUNT, &DV_COUNT, sizeof(uint32_t)));
    uint32_t *dvlabels, *capacity_prefix_sum;

    // copy the vertex label array to gpu
    cudaErrorCheck(cudaMalloc(&dvlabels, sizeof(uint32_t) * DV_COUNT));
    cudaErrorCheck(cudaMemcpy(dvlabels, data_graph.vlabels_.data(), sizeof(uint32_t) * DV_COUNT, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMalloc(&capacity_prefix_sum, sizeof(uint32_t) * (DV_COUNT + 1u)));

    // allocate memory for each relation in the data graph and the index
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        AllocRelation(query_.qe_eidx_[i].first, csr_gpu[query_.qe_eidx_[i].first], 
            query_.qe_list_[i].first, dvlabels, 
            data_graph_gpu, global_index_gpu, capacity_prefix_sum,
            data_graph_all_nbrs_[query_.qe_eidx_[i].first],
            index_all_nbrs_[query_.qe_eidx_[i].first],
            query_.first_NL_[query_.qe_eidx_[i].first] == query_.qe_eidx_[i].first ||
            query_.last_NL_[query_.qe_eidx_[i].first] == query_.qe_eidx_[i].first);
        AllocRelation(query_.qe_eidx_[i].second, csr_gpu[query_.qe_eidx_[i].second], 
            query_.qe_list_[i].second, dvlabels, 
            data_graph_gpu, global_index_gpu, capacity_prefix_sum,
            data_graph_all_nbrs_[query_.qe_eidx_[i].second],
            index_all_nbrs_[query_.qe_eidx_[i].second],
            query_.first_NL_[query_.qe_eidx_[i].second] == query_.qe_eidx_[i].second ||
            query_.last_NL_[query_.qe_eidx_[i].second] == query_.qe_eidx_[i].second);
    }

    cudaErrorCheck(cudaFree(dvlabels));
    cudaErrorCheck(cudaFree(capacity_prefix_sum));

    // allocate the candidate bits arrays
    for (auto i = 0u; i < QV_COUNT; i++)
    {
        cudaErrorCheck(cudaMalloc(&global_bitmap_gpu.candidate_bits_[i], sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
        cudaErrorCheck(cudaMemset(global_bitmap_gpu.candidate_bits_[i], 0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
    }
}

void MatchGPU::DeallocRelations(
    RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
    CandidatesGPU& global_bitmap_gpu
) {
    for (auto i = 0u; i < QV_COUNT; i++)
    {
        cudaErrorCheck(cudaFree(global_bitmap_gpu.candidate_bits_[i]));
    }
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        for (const auto& idx: {query_.qe_eidx_[i].first, query_.qe_eidx_[i].second})
        {
            if (query_.first_NL_[idx] == idx || query_.last_NL_[idx] == idx)
            {
                cudaErrorCheck(cudaFree(data_graph_all_nbrs_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.nbrs_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.capacity_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.sizes_[idx]));
            }

            cudaErrorCheck(cudaFree(index_all_nbrs_[idx]));
            cudaErrorCheck(cudaFree(global_index_gpu.nbrs_[idx]));
            cudaErrorCheck(cudaFree(global_index_gpu.capacity_[idx]));
            cudaErrorCheck(cudaFree(global_index_gpu.sizes_[idx]));
        }
    }
}

void MatchGPU::OursAllocRelations(const DataGraphManager& data_graph, 
                                  const CSR_GPU csr_gpu[],
                                  RelationsGPU& data_graph_gpu,
                                  RelationsGPUAllVersions& global_index_gpu_all_versions, 
                                  CandidatesGPUAllVersions& global_bitmap_gpu_all_versions) {
    cudaErrorCheck(cudaMemcpyToSymbol(C_DV_COUNT, &DV_COUNT, sizeof(uint32_t)));
    uint32_t *dvlabels, *capacity_prefix_sum;

    // copy the vertex label array to gpu
    cudaErrorCheck(cudaMalloc(&dvlabels, sizeof(uint32_t) * DV_COUNT));
    cudaErrorCheck(cudaMemcpy(dvlabels, data_graph.vlabels_.data(), sizeof(uint32_t) * DV_COUNT, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMalloc(&capacity_prefix_sum, sizeof(uint32_t) * (DV_COUNT + 1u)));

    // allocate memory for each relation in the data graph and the index
    for (auto version_idx = 0u; version_idx < QE_COUNT; version_idx++) {
        for (auto i = 0u; i < QE_COUNT; i++) {
            AllocRelation(query_.qe_eidx_[i].first, csr_gpu[query_.qe_eidx_[i].first], 
                          query_.qe_list_[i].first, dvlabels, 
                          data_graph_gpu, 
                          global_index_gpu_all_versions.GetDataGraphByIdx(version_idx), 
                          capacity_prefix_sum,
                          data_graph_all_nbrs_[query_.qe_eidx_[i].first],
                          index_all_nbrs_[query_.qe_eidx_[i].first],
                          query_.first_NL_[query_.qe_eidx_[i].first] == query_.qe_eidx_[i].first ||
                          query_.last_NL_[query_.qe_eidx_[i].first] == query_.qe_eidx_[i].first);
            AllocRelation(query_.qe_eidx_[i].second, csr_gpu[query_.qe_eidx_[i].second], 
                          query_.qe_list_[i].second, dvlabels, 
                          data_graph_gpu, 
                          global_index_gpu_all_versions.GetDataGraphByIdx(version_idx), 
                          capacity_prefix_sum,
                          data_graph_all_nbrs_[query_.qe_eidx_[i].second],
                          index_all_nbrs_[query_.qe_eidx_[i].second],
                          query_.first_NL_[query_.qe_eidx_[i].second] == query_.qe_eidx_[i].second ||
                          query_.last_NL_[query_.qe_eidx_[i].second] == query_.qe_eidx_[i].second);
        }
    }

    cudaErrorCheck(cudaFree(dvlabels));
    cudaErrorCheck(cudaFree(capacity_prefix_sum));

    // allocate the candidate bits arrays
    for (auto version_idx = 0u; version_idx < QE_COUNT; version_idx++) {
        for (auto i = 0u; i < QV_COUNT; i++) {
            cudaErrorCheck(cudaMalloc(&(global_bitmap_gpu_all_versions.GetByIdx(version_idx).candidate_bits_[i]), 
                                        sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
            cudaErrorCheck(cudaMemset(global_bitmap_gpu_all_versions.GetByIdx(version_idx).candidate_bits_[i],
                                      0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
        }
    }
}

void MatchGPU::OursDeallocRelations(RelationsGPU& data_graph_gpu,
                                    RelationsGPUAllVersions& global_index_gpu_all_versions, 
                                    CandidatesGPUAllVersions& global_bitmap_gpu_all_version) {
    for (auto version_idx = 0u; version_idx < QE_COUNT; version_idx++) {
        for (auto i = 0u; i < QV_COUNT; i++) {
            cudaErrorCheck(cudaFree(global_bitmap_gpu_all_version.GetByIdx(version_idx).candidate_bits_[i]));
        }
    }
    for (auto i = 0u; i < QE_COUNT; i++) {
        for (const auto& idx: {query_.qe_eidx_[i].first, query_.qe_eidx_[i].second}) {
            if (query_.first_NL_[idx] == idx || query_.last_NL_[idx] == idx) {
                cudaErrorCheck(cudaFree(data_graph_all_nbrs_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.nbrs_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.capacity_[idx]));
                cudaErrorCheck(cudaFree(data_graph_gpu.sizes_[idx]));
            }
            cudaErrorCheck(cudaFree(index_all_nbrs_[idx]));
        }
    }

    for (auto version_idx = 0u; version_idx < QE_COUNT; version_idx++) {
        for (auto i = 0u; i < QE_COUNT; i++) {
            for (const auto& idx: {query_.qe_eidx_[i].first, query_.qe_eidx_[i].second}) {
                cudaErrorCheck(cudaFree(global_index_gpu_all_versions.GetDataGraphByIdx(version_idx).nbrs_[idx]));
                cudaErrorCheck(cudaFree(global_index_gpu_all_versions.GetDataGraphByIdx(version_idx).capacity_[idx]));
                cudaErrorCheck(cudaFree(global_index_gpu_all_versions.GetDataGraphByIdx(version_idx).sizes_[idx]));
            }
        }
    }
}


void MatchGPU::AllocOnline(
    RelationsGPU& local_index_base_gpu, CandidatesGPU& local_bitmap_gpu
) {
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        for (const auto& idx: {query_.qe_eidx_[i].first, query_.qe_eidx_[i].second})
        {
            cudaErrorCheck(cudaMalloc(&local_index_base_gpu.sizes_[idx], sizeof(uint32_t) * (DV_COUNT + 1u)));
            cudaErrorCheck(cudaMalloc(&local_index_base_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1u)));
            cudaErrorCheck(cudaMalloc(&local_index_base_gpu.nbrs_[idx], sizeof(uint32_t*) * (DV_COUNT)));
        }
    }
    for (auto i = 0u; i < QV_COUNT; i++)
    {
        cudaErrorCheck(cudaMalloc(&local_bitmap_gpu.candidate_bits_[i], sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
        cudaErrorCheck(cudaMemset(local_bitmap_gpu.candidate_bits_[i], 0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
    }

    cudaErrorCheck(cudaMalloc(&cum_bn_, sizeof(uint32_t) * DV_COUNT));
    cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));

    cudaErrorCheck(cudaMalloc(&res_size_cartesian_product_, SIZE_SPACE * sizeof(unsigned long)));
    res_queue_.Alloc(RES_SPACE);
    cudaErrorCheck(cudaMemcpyToSymbol(C_RES_QUEUE, &res_queue_, sizeof(CyclicQueue<uint32_t>)));
}

void MatchGPU::DeallocOnline(
    RelationsGPU& local_index_base_gpu, CandidatesGPU& local_bitmap_gpu
) {
    cudaErrorCheck(cudaFree(res_size_cartesian_product_));
    res_queue_.Free();
    cudaErrorCheck(cudaFree(cum_bn_));
    for (auto i = 0u; i < QV_COUNT; i++)
    {
        cudaErrorCheck(cudaFree(local_bitmap_gpu.candidate_bits_[i]));
    }
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        for (const auto& idx: {query_.qe_eidx_[i].first, query_.qe_eidx_[i].second})
        {
            cudaErrorCheck(cudaFree(local_index_base_gpu.sizes_[idx]));
            cudaErrorCheck(cudaFree(local_index_base_gpu.capacity_[idx]));
            cudaErrorCheck(cudaFree(local_index_base_gpu.nbrs_[idx]));
        }
    }
}

void MatchGPU::OursAllocOnline() {
    cudaErrorCheck(cudaMalloc(&cum_bn_, sizeof(uint32_t) * DV_COUNT));
    cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));

    cudaErrorCheck(cudaMalloc(&res_size_cartesian_product_, SIZE_SPACE * sizeof(unsigned long)));
    res_queue_.Alloc(RES_SPACE);
    cudaErrorCheck(cudaMemcpyToSymbol(C_RES_QUEUE, &res_queue_, sizeof(CyclicQueue<uint32_t>)));
}

void MatchGPU::OursDeallocOnline() {
    cudaErrorCheck(cudaFree(res_size_cartesian_product_));
    res_queue_.Free();
    cudaErrorCheck(cudaFree(cum_bn_));
}

void MatchGPU::SetGraphPtrs(RelationsGPU& data_graph_gpu)
{
    for (auto i = 0u; i < QE_COUNT; i++)
    {
        for (auto j = 0u; j < 2u; j++)
        {
            const auto& idx = j == 0u ? query_.qe_eidx_[i].first : query_.qe_eidx_[i].second;
            if (idx != query_.first_NL_[idx])
            {
                data_graph_gpu.nbrs_[idx] = data_graph_gpu.nbrs_[query_.last_NL_[idx]];
                data_graph_gpu.sizes_[idx] = data_graph_gpu.sizes_[query_.last_NL_[idx]];
            }
            else
            {
                // Doubt: Seems no-op, as idx == query_.first_NL_[idx].
                data_graph_gpu.nbrs_[idx] = data_graph_gpu.nbrs_[query_.first_NL_[idx]];
                data_graph_gpu.sizes_[idx] = data_graph_gpu.sizes_[query_.first_NL_[idx]];
            }
        }
    }
}

// void MatchGPU::GammaAllocOnline() {
//     cudaErrorCheck(cudaMalloc(&cum_bn_, sizeof(uint32_t) * DV_COUNT));
//     cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));

//     cudaErrorCheck(cudaMalloc(&res_size_cartesian_product_, SIZE_SPACE * sizeof(unsigned long)));
//     res_queue_.Alloc(RES_SPACE);
//     cudaErrorCheck(cudaMemcpyToSymbol(C_RES_QUEUE, &res_queue_, sizeof(CyclicQueue<uint32_t>)));
// }

// void MatchGPU::GammaDeallocOnline() {
//     cudaErrorCheck(cudaFree(res_size_cartesian_product_));
//     res_queue_.Free();
//     cudaErrorCheck(cudaFree(cum_bn_));
// }

void MatchGPU::GetSummary(
    const RelationsGPU& global_index_gpu, uint32_t *cardinalities, float *degrees
) {
    uint32_t h_temp[2], *d_temp;
    // cout << "before cudaMalloc" << endl;
    cudaErrorCheck(cudaMalloc(&d_temp, sizeof(uint32_t) * 2u));
    // cout << "before for-loop" << endl;
    for (auto i = 0u; i < QE_COUNT * 2; i++)
    {
        cudaErrorCheck(cudaMemset(d_temp, 0u, sizeof(uint32_t) * 2u));
        // i is the current edge_idx
        // *d_temp <- edge_count; *(d_temp + 1) <- num_source_vertices in global_index_gpu.xxx[i]
        statisticIndex<<<GRID_DIM, BLOCK_DIM>>>(
            global_index_gpu, i, d_temp, d_temp + 1
        );
        cudaErrorCheck(cudaMemcpy(h_temp, d_temp, sizeof(uint32_t) * 2u, cudaMemcpyDeviceToHost));
        // cout << "inside" << i << "-th loop" << "before cardinalities assignment" << endl;
        cardinalities[i] = h_temp[0];
        degrees[i] = h_temp[0] / (float) h_temp[1];
    }
}

void MatchGPU::UpdateGlobalIndex(
    RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
    CandidatesGPU& global_bitmap_gpu, const CSR_GPU csr_gpu[], const uint8_t i
) {
    // update the candidate edges of the query edges adjacent u or uu
    for (auto j = 0u; j < 2u; j++)
    {
        // cout << "inside UpdateGlobalIndex - first for loop, j: " << j << endl;  // 250216
        const auto& idx = j == 0u ? query_.qe_eidx_[i].first : query_.qe_eidx_[i].second;
        const auto& u = j == 0u ? query_.qe_list_[i].first : query_.qe_list_[i].second;
        const auto& uu = j == 0u ? query_.qe_list_[i].second : query_.qe_list_[i].first;

        // find new candidates of u (satisfies NLF after the update), and write to the trie
        ReAlloc(cand_flag_, csr_gpu[idx].vs_size_, cand_flag_capacity_, bool);
        cudaErrorCheck(cudaMemset(cand_flag_, false, sizeof(bool) * cand_flag_capacity_));
        cudaErrorCheck(cudaDeviceSynchronize());

        if (query_.NLF_[idx] == 0) continue;
        // Apply NLFs (considering the initial graph data_graph_gpu and new edges in csr_gpu[idx]). 
        // Write values to global_bitmap_gpu.candidate_bits_[u] and cand_flag_. (Add NLF-valid data vertices into candidates.)
        getGlobalCandidates<<<GRID_DIM, BLOCK_DIM>>>(
            data_graph_gpu, idx, csr_gpu[idx], global_bitmap_gpu.candidate_bits_[u], cand_flag_, u
        );
        cudaErrorCheck(cudaDeviceSynchronize());

        // cout << "inside UpdateGlobalIndex - after getGlobalCandidates, d_new_cand_count_[0]: " << d_new_cand_count_[0] << endl;  // 250216
        // cout << "inside UpdateGlobalIndex - after getGlobalCandidates, d_new_cand_count_[1]: " << d_new_cand_count_[1] << endl;  // 250216
        // uint32_t temp2;  // 250216
        // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
        // cout << "temp2: " << temp2 << endl;  // 250216
        // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
        // cout << "temp2: " << temp2 << endl;  // 250216

        // allocate temp_tries.vs_ and temp_tries.offs_
        ReAlloc(temp_tries_[0].vs_, csr_gpu[idx].vs_size_, temp_tries_capacity_[0].vs_capacity_, uint32_t);
        // cudaErrorCheck(cudaDeviceSynchronize());  // 250216

        // cout << "inside UpdateGlobalIndex - before cub::DeviceSelect::Flagged" << endl;  // 250216
        // cout << "csr_gpu[idx].vs_size_: " << csr_gpu[idx].vs_size_ << endl;  // 250216
        // cout << "cand_flag_capacity_: " << cand_flag_capacity_ << endl;  // 250216
        // cout << "bool* cand_flag_: " << cand_flag_ << endl;  // 250216
        // cout << "uint32_t* (temp_tries_[0].vs_): " << temp_tries_[0].vs_ << endl;  // 250216
        // cout << "idx: " << idx << endl;  // 250216
        // cout << "csr_gpu[idx].vs_: " << csr_gpu[idx].vs_ << endl;  // 250216
        // cout << "d_temp_storage_: " << d_temp_storage_ << endl;  // 250216

        uint32_t *temp_dest = new uint32_t[csr_gpu[idx].vs_size_];
        cudaErrorCheck(cudaMemcpy(temp_dest, csr_gpu[idx].vs_, sizeof(uint32_t) * csr_gpu[idx].vs_size_, cudaMemcpyDeviceToHost));
        bool *temp_bool_dest = new bool[csr_gpu[idx].vs_size_];
        cudaErrorCheck(cudaMemcpy(temp_bool_dest, cand_flag_, sizeof(bool) * csr_gpu[idx].vs_size_, cudaMemcpyDeviceToHost));
        uint32_t *temp_dest2 = new uint32_t[csr_gpu[idx].vs_size_];
        cudaErrorCheck(cudaMemcpy(temp_dest2, temp_tries_[0].vs_, sizeof(uint32_t) * csr_gpu[idx].vs_size_, cudaMemcpyDeviceToHost));
        // for (int i = 0; i < csr_gpu[idx].vs_size_; i++) {
        //     if (temp_bool_dest[i] == false) {
        //         cout << "temp_dest[" << i << "]: " << temp_dest[i] << ", temp_bool_dest[" << i << "]: " << temp_bool_dest[i] << ", temp_dest2[" << i << "]: " << temp_dest2[i] << endl;
        //     }
        //     // cout << "temp_dest[" << i << "]: " << temp_dest[i] << ", temp_bool_dest[" << i << "]: " << temp_bool_dest[i] << endl;
        // }

        // masked_select: csr_gpu[idx].vs_[cand_flag_] -> temp_tries_[0].vs_, 
        // *d_new_cand_count_[0] is the result item count.
        CUB(cub::DeviceSelect::Flagged(d_temp_storage_, temp_storage_bytes_, csr_gpu[idx].vs_, cand_flag_, temp_tries_[0].vs_, d_new_cand_count_[0], csr_gpu[idx].vs_size_));

        // // cout << "before void *pre_d_temp_storage_ = d_temp_storage_, d_temp_storage_: " << d_temp_storage_ << endl;  // 250216
        // void *pre_d_temp_storage_ = d_temp_storage_;  // 250216
        // d_temp_storage_ = NULL;    // 250216
        // cub::DeviceSelect::Flagged(d_temp_storage_, temp_storage_bytes_, csr_gpu[idx].vs_, cand_flag_, temp_tries_[0].vs_, d_new_cand_count_[0], csr_gpu[idx].vs_size_);    // 250216
        // cudaErrorCheck(cudaDeviceSynchronize());  // 250216
        // // cout << "before if, temp_storage_bytes_: " << temp_storage_bytes_ << endl;  // 250216
        // if (temp_storage_bytes_ > temp_storage_capacity_)    // 250216
        // {
        //     if (temp_storage_capacity_ != 0ul)    // 250216
        //     {
        //         cudaErrorCheck(cudaFree(pre_d_temp_storage_));    // 250216
        //     }
        //     cout << "inside if" << endl;  // 250216
        //     temp_storage_bytes_ = temp_storage_capacity_ = 
        //         (size_t)exp2(ceil(log2(temp_storage_bytes_)));    // 250216
        //     cudaErrorCheck(cudaMalloc(
        //         &d_temp_storage_, temp_storage_bytes_));    // 250216
        //     cub::DeviceSelect::Flagged(d_temp_storage_, temp_storage_bytes_, csr_gpu[idx].vs_, cand_flag_, temp_tries_[0].vs_, d_new_cand_count_[0], csr_gpu[idx].vs_size_);    // 250216
        //     cudaErrorCheck(cudaDeviceSynchronize());  // 250216
        
        // }
        // else
        // {
        //     cout << "inside else" << endl;  // 250216
        //     d_temp_storage_ = pre_d_temp_storage_;    // 250216
        // }
        // // cout << "temp_storage_bytes_: " << temp_storage_bytes_ << endl;  // 250216
        // // cout << "temp_storage_capacity_: " << temp_storage_capacity_ << endl;  // 250216
        // // cout << "d_temp_storage_: " << d_temp_storage_ << endl;  // 250216
        
        // size_t freeMem, totalMem;  // 250216
        // cudaErrorCheck(cudaMemGetInfo(&freeMem, &totalMem));  // 250216
        // std::cout << "Total GPU memory: " << totalMem / 1024 / 1024 << " MB" << std::endl;  // 250216
        // std::cout << "Free GPU memory: " << freeMem / 1024 / 1024 << " MB" << std::endl;  // 250216

        // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
        // cout << "temp2: " << temp2 << endl;  // 250216
        // cudaErrorCheck(cudaMemcpy(&temp2, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));  // 250216
        // cout << "temp2: " << temp2 << endl;  // 250216

        // temp_tries_[0].vs_size_, i.e. CSR_GPU::vs_size_ is in CPU memory.
        cudaErrorCheck(cudaMemcpy(&temp_tries_[0].vs_size_, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (temp_tries_[0].vs_size_ == 0)
        {
            continue;
        }
        ReAlloc(temp_tries_[0].offs_, csr_gpu[idx].vs_size_ + 1, temp_tries_capacity_[0].off_capacity_, uint32_t);

        // check each relation adjacent to u
        for (auto k = query_.qv_offs_[u]; k < query_.qv_offs_[u + 1]; k++)
        {
            // k is the current edge_idx
            const auto& u_other = query_.qv_nbrs_[k];

            // allocate temp_tries.es_
            // Calculate the number of valid data graph edges for each data graph source vertex in temp_tries_[0].vs_ and store the numbers in temp_tries_[0].offs_
            // A valid data graph edge means that the destination vertex is in `global_bitmap_gpu.candidate_bits_[u_other]`
            // Doubt: What if `global_bitmap_gpu.candidate_bits_[u_other]` has not been written (This may happen when invoked at initialization steps).
            getGlobalCandidateEdgesCount<<<GRID_DIM, BLOCK_DIM>>>(
                data_graph_gpu, k, temp_tries_[0].vs_, temp_tries_[0].vs_size_,
                global_bitmap_gpu.candidate_bits_[u_other], temp_tries_[0].offs_
            );
            cudaErrorCheck(cudaDeviceSynchronize());

            // exclusive_sum(temp_tries_[0].offs_0)
            CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, temp_tries_[0].offs_, temp_tries_[0].offs_, temp_tries_[0].vs_size_ + 1));
            cudaErrorCheck(cudaMemcpy(&temp_tries_[0].es_size_, &temp_tries_[0].offs_[temp_tries_[0].vs_size_], sizeof(uint32_t), cudaMemcpyDeviceToHost));

            if (temp_tries_[0].es_size_ == 0)
            {
                continue;
            }

            temp_tries_[1].es_size_ = temp_tries_[0].es_size_;

            ReAlloc(temp_tries_[0].nbrs_, temp_tries_[0].es_size_, temp_tries_capacity_[0].es_capacity_, uint32_t);
            ReAlloc(helper_relation_[0], temp_tries_[0].es_size_, helper_relation_capacity_[0], uint32_t);
            ReAlloc(temp_tries_[1].nbrs_, temp_tries_[0].es_size_, temp_tries_capacity_[1].es_capacity_, uint32_t);
            ReAlloc(helper_relation_[1], temp_tries_[0].es_size_, helper_relation_capacity_[1], uint32_t);

            // Seems that the size of temp_tries_[1].vs_ and temp_tries_[1].offs_ can be more than needed.
            ReAlloc(temp_tries_[1].vs_, temp_tries_[0].es_size_, temp_tries_capacity_[1].vs_capacity_, uint32_t);
            ReAlloc(temp_tries_[1].offs_, temp_tries_[0].es_size_ + 1, temp_tries_capacity_[1].off_capacity_, uint32_t);

            // fill in temp_tries.es_
            // Fill the source and destination vertices of valid data graph edges into helper_relation_[0] and temp_tries_[0].nbrs_, respectively
            // Discussion: Checking the bitmap again here. (getGlobalCandidateEdgesCount has checked the bitmap)
            getGlobalCandidateEdgesWrite<<<GRID_DIM, BLOCK_DIM>>>(
                data_graph_gpu, k, temp_tries_[0].vs_, temp_tries_[0].vs_size_,
                global_bitmap_gpu.candidate_bits_[u_other], temp_tries_[0].offs_,
                helper_relation_[0], temp_tries_[0].nbrs_
            );
            cudaErrorCheck(cudaDeviceSynchronize());

            // add the temp_tries to the index
            // Question (Solved): Why add the edges in `temp_tries_[0]` to `global_index_gpu`? Answer: temp_tries[0] contains updated edges.
            addTriesToGraph<<<GRID_DIM, BLOCK_DIM>>>(temp_tries_[0], global_index_gpu, k, nbr_mem_pool_);
            cudaErrorCheck(cudaDeviceSynchronize());
            if (nbr_mem_pool_.OutOfMemory())
            {
                exit(-1);
            }

            // reverse temp_tries_[0] into temp_tries_[1]
            CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
                temp_tries_[0].nbrs_, helper_relation_[1], helper_relation_[0], temp_tries_[1].nbrs_, temp_tries_[0].es_size_));

            CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, helper_relation_[1], temp_tries_[1].vs_, temp_tries_[1].offs_, d_new_cand_count_[1], temp_tries_[0].es_size_));
            cudaErrorCheck(cudaMemcpy(&temp_tries_[1].vs_size_, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));

            CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, temp_tries_[1].offs_, temp_tries_[1].offs_, temp_tries_[1].vs_size_ + 1));

            // add the temp_rcsr_gpu to the index
            // Question (Solved): Why add the edges in `temp_tries_[1]` to `global_index_gpu`? Answer: temp_tries[1] contains (reversed) udpated edges.
            addTriesToGraph<<<GRID_DIM, BLOCK_DIM>>>(temp_tries_[1], global_index_gpu, query_.eidx_[u_other * QV_COUNT + u], nbr_mem_pool_);
            cudaErrorCheck(cudaDeviceSynchronize());
            if (nbr_mem_pool_.OutOfMemory())
            {
                exit(-1);
            }
        }
    }
    for (auto j = 0u; j < 2u; j++)
    {
        const auto& idx = j == 0u ? query_.qe_eidx_[i].first : query_.qe_eidx_[i].second;
        const auto& u = j == 0u ? query_.qe_list_[i].first : query_.qe_list_[i].second;
        const auto& uu = j == 0u ? query_.qe_list_[i].second : query_.qe_list_[i].first;
        
        // cout << "inside UpdateGlobalIndex - second for loop, j: " << j << ", idx: " << idx << ", u: " << u << ", uu: " << uu << endl;  // 250216
        
        if (query_.first_NL_[idx] == idx || query_.last_NL_[idx] == idx)
        {
            // add all edges mapped to the current query edge to the data graph
            addTriesToGraph<<<GRID_DIM, BLOCK_DIM>>>(csr_gpu[idx], data_graph_gpu, idx, nbr_mem_pool_);
            cudaErrorCheck(cudaDeviceSynchronize());
            if (nbr_mem_pool_.OutOfMemory())
            {
                exit(-1);
            }
        }
        else
        {
            data_graph_gpu.nbrs_[idx] = data_graph_gpu.nbrs_[query_.first_NL_[idx]];
            data_graph_gpu.sizes_[idx] = data_graph_gpu.sizes_[query_.first_NL_[idx]];
        }

        // add all edges with both endpoints being candidate to the index
        ReAlloc(temp_tries_[0].vs_, csr_gpu[idx].vs_size_, temp_tries_capacity_[0].vs_capacity_, uint32_t);
        ReAlloc(temp_tries_[0].offs_, csr_gpu[idx].vs_size_ + 1, temp_tries_capacity_[0].off_capacity_, uint32_t);
        ReAlloc(temp_tries_[0].nbrs_, csr_gpu[idx].es_size_, temp_tries_capacity_[0].es_capacity_, uint32_t);

        SelectEdgesFromTrie(csr_gpu[idx], temp_tries_[0], global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]);

        addTriesToGraph<<<GRID_DIM, BLOCK_DIM>>>(temp_tries_[0], global_index_gpu, idx, nbr_mem_pool_);
        cudaErrorCheck(cudaDeviceSynchronize());
        if (nbr_mem_pool_.OutOfMemory())
        {
            exit(-1);
        }
    }
}

bool MatchGPU::BuildLocalIndex(
    const RelationsGPU& global_index_gpu, CandidatesGPU& global_bitmap_gpu,
    RelationsGPU& local_index_base_gpu, RelationsGPU& local_index,
    CSR_GPU csr_gpu[], const uint8_t cur_i, const float *avg_degrees
) {
    for (auto i = 1u; i < QV_COUNT; i++)
    {
        if (i == 1)
        {
            const auto& off = plan_.indexing_orders_[cur_i].bni_offs_[i];

            const auto& u0 = plan_.indexing_orders_[cur_i].vs_[1];
            const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];
            for (auto j = 0u; j < 2u; j++)
            {
                const auto& u = j == 0u ? u0 : u1;
                const auto& uu = j == 0u ? u1 : u0;
                const auto& index = query_.eidx_[u * QV_COUNT + uu];

                // build the relation from trie
                cudaErrorCheck(cudaMemset(local_index_base_gpu.sizes_[index], 0u, sizeof(uint32_t) * DV_COUNT));
                cudaErrorCheck(cudaDeviceSynchronize());
                edgeList2RelationCount<<<GRID_DIM, BLOCK_DIM>>>(
                    csr_gpu[index], local_index_base_gpu, index,
                    global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]);
                cudaErrorCheck(cudaDeviceSynchronize());

                CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index_base_gpu.sizes_[index], local_index_base_gpu.capacity_[index], DV_COUNT + 1));
                auto total_size = 0u;
                cudaErrorCheck(cudaMemcpy(&total_size, local_index_base_gpu.capacity_[index] + DV_COUNT, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                if (total_size == 0u) return false;

                ReAlloc(local_nbr_[index], total_size, local_nbr_capacity_[index], uint32_t*);
                setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[index], local_index_base_gpu.capacity_[index], DV_COUNT, local_index_base_gpu.nbrs_[index]);
                cudaErrorCheck(cudaDeviceSynchronize());

                edgeList2RelationWrite<<<GRID_DIM, BLOCK_DIM>>>(
                    csr_gpu[index], local_index_base_gpu, index,
                    global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]);
                cudaErrorCheck(cudaDeviceSynchronize());

                local_index.nbrs_[index] = local_index_base_gpu.nbrs_[index];
                local_index.sizes_[index] = local_index_base_gpu.sizes_[index];
            }
        }
        else
        {
            const auto& u0 = plan_.indexing_orders_[cur_i].vs_[i];

            if (plan_.rebuild_v_flags_[cur_i][u0])
            {
                // 1. find all candidates with at least one neighbor to match each backward neighbor
                cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));

                for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i]; off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

                    const auto& depth = plan_.indexing_orders_[cur_i].bni_[off];
                    const auto& first_bn_of_uu = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[
                        plan_.indexing_orders_[cur_i].bni_offs_[depth]
                    ]];
                    // lgh: edge_idx of (u1, first_backward_neighbor_of_u1)
                    const auto& first_bn_of_uu_index = query_.eidx_[u1 * QV_COUNT + first_bn_of_uu];

                    const auto& backward_index = query_.eidx_[u0 * QV_COUNT + u1];
                    const auto& forward_index = query_.eidx_[u1 * QV_COUNT + u0];
                    if (avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        getLocalCandidatesBackward<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, backward_index, first_bn_of_uu_index,
                            cum_bn_, off - plan_.indexing_orders_[cur_i].bni_offs_[i]
                        );
                    }
                    else
                    {
                        getLocalCandidatesForward<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, forward_index, backward_index, first_bn_of_uu_index,
                            cum_bn_, off - plan_.indexing_orders_[cur_i].bni_offs_[i]
                        );
                    }
                    cudaErrorCheck(cudaDeviceSynchronize());
                }

                // 2. build relations from these candidates (only for vertices v if cum_bn_[v] == # backward neighbors of u)
                // 3. build reversed relations
                for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i]; off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

                    const auto& depth = plan_.indexing_orders_[cur_i].bni_[off];
                    const auto& first_bn_of_uu = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[
                        plan_.indexing_orders_[cur_i].bni_offs_[depth]
                    ]];
                    const auto& first_bn_of_uu_index = query_.eidx_[u1 * QV_COUNT + first_bn_of_uu];

                    const auto& backward_index = query_.eidx_[u0 * QV_COUNT + u1];
                    const auto& forward_index = query_.eidx_[u1 * QV_COUNT + u0];
                    uint8_t work_index, reversed_index;

                    cudaErrorCheck(cudaMemset(local_index_base_gpu.sizes_[backward_index], 0u, sizeof(uint32_t) * DV_COUNT));
                    cudaErrorCheck(cudaMemset(local_index_base_gpu.sizes_[forward_index], 0u, sizeof(uint32_t) * DV_COUNT));
                    cudaErrorCheck(cudaDeviceSynchronize());

                    if (avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        // build a trie u0 -> u1
                        buildLocalRelationNew2OldCount<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, backward_index, first_bn_of_uu_index,
                            cum_bn_, plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i]);
                        work_index = backward_index;
                        reversed_index = forward_index;
                    }
                    else
                    {
                        // build a trie u1 -> u0
                        buildLocalRelationOld2NewCount<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, forward_index, first_bn_of_uu_index,
                            cum_bn_, plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i]);
                        work_index = forward_index;
                        reversed_index = backward_index;
                    }
                    cudaErrorCheck(cudaDeviceSynchronize());

                    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index_base_gpu.sizes_[work_index], local_index_base_gpu.capacity_[work_index], DV_COUNT + 1));
                    auto total_size = 0u;
                    cudaErrorCheck(cudaMemcpy(&total_size, local_index_base_gpu.capacity_[work_index] + DV_COUNT, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                    if (total_size == 0u) return false;

                    ReAlloc(helper_relation_[0], total_size, helper_relation_capacity_[0], uint32_t);
                    ReAlloc(helper_relation_[1], total_size, helper_relation_capacity_[1], uint32_t);
                    ReAlloc(temp_tries_[0].vs_, total_size, temp_tries_capacity_[0].vs_capacity_, uint32_t);
                    ReAlloc(temp_tries_[0].offs_, total_size, temp_tries_capacity_[0].off_capacity_, uint32_t);

                    ReAlloc(local_nbr_[work_index], total_size, local_nbr_capacity_[work_index], uint32_t*);
                    ReAlloc(local_nbr_[reversed_index], total_size, local_nbr_capacity_[reversed_index], uint32_t*);
                    setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[work_index], local_index_base_gpu.capacity_[work_index], DV_COUNT, local_index_base_gpu.nbrs_[work_index]);
                    cudaErrorCheck(cudaDeviceSynchronize());

                    if (avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        // build a trie u0 -> u1
                        buildLocalRelationNew2OldWrite<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, backward_index, first_bn_of_uu_index,
                            cum_bn_, plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i], helper_relation_[0]);
                    }
                    else
                    {
                        // build a trie u1 -> u0
                        buildLocalRelationOld2NewWrite<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, forward_index, first_bn_of_uu_index,
                            cum_bn_, plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i], helper_relation_[0]);
                    }
                    cudaErrorCheck(cudaDeviceSynchronize());

                    // reverse temp_tries_[0] into temp_tries_[1]
                    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
                        local_nbr_[work_index], helper_relation_[1], helper_relation_[0], local_nbr_[reversed_index], total_size));

                    CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, helper_relation_[1], temp_tries_[0].vs_, temp_tries_[0].offs_, d_new_cand_count_[0], total_size));
                    cudaErrorCheck(cudaDeviceSynchronize());
                    cudaErrorCheck(cudaMemcpy(&temp_tries_[0].vs_size_, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));

                    // map trie to relation
                    mapTrieToRelation<<<GRID_DIM, BLOCK_DIM>>>(local_index_base_gpu, reversed_index, temp_tries_[0].vs_, temp_tries_[0].offs_, temp_tries_[0].vs_size_);
                    cudaErrorCheck(cudaDeviceSynchronize());

                    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index_base_gpu.sizes_[reversed_index], local_index_base_gpu.capacity_[reversed_index], DV_COUNT + 1));
                    setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[reversed_index], local_index_base_gpu.capacity_[reversed_index], DV_COUNT, local_index_base_gpu.nbrs_[reversed_index]);
                    cudaErrorCheck(cudaDeviceSynchronize());

                    local_index.nbrs_[backward_index] = local_index_base_gpu.nbrs_[backward_index];
                    local_index.sizes_[backward_index] = local_index_base_gpu.sizes_[backward_index];
                    local_index.nbrs_[forward_index] = local_index_base_gpu.nbrs_[forward_index];
                    local_index.sizes_[forward_index] = local_index_base_gpu.sizes_[forward_index];
                }
            }
            else
            {
                for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i]; off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

                    //std::cout << "skip " << static_cast<uint32_t>(u0) << ' ' << static_cast<uint32_t>(u1) << '\n';
                    for (auto j = 0u; j < 2u; j++)
                    {
                        const auto& u = j == 0u ? u0 : u1;
                        const auto& uu = j == 0u ? u1 : u0;

                        // copy the relation from the global index
                        local_index.nbrs_[query_.eidx_[u * QV_COUNT + uu]] = global_index_gpu.nbrs_[query_.eidx_[u * QV_COUNT + uu]];
                        local_index.sizes_[query_.eidx_[u * QV_COUNT + uu]] = global_index_gpu.sizes_[query_.eidx_[u * QV_COUNT + uu]];
                    }
                }
            }
        }
    }
    return true;
}

bool MatchGPU::GammaBuildLocalIndex(const RelationsGPU& global_index_gpu, CandidatesGPU& global_bitmap_gpu,
                                    RelationsGPU& local_index_base_gpu, CandidatesGPU& local_bitmap_gpu,
                                    RelationsGPU& local_index, CSR_GPU csr_gpu[], 
                                    const uint8_t edge_list_idx, const float *avg_degrees) {
    // lgh: cur_i is current idx_in_qe_list_
    // Visit the query vertices in plan_.indexing_orders_[cur_i] one by one.

    std::pair<uint32_t, uint32_t> cur_edge = query_.getEdgeByEdgeListIdx(edge_list_idx);
    const uint32_t u0 = cur_edge.first;
    const uint32_t u1 = cur_edge.second;
    for (auto j = 0u; j < 2u; j++)
    {
        const auto& u = j == 0u ? u0 : u1;
        const auto& uu = j == 0u ? u1 : u0;
        // lgh: index is edge_idx
        const auto& edge_idx = query_.eidx_[u * QV_COUNT + uu];

        // build the relation from trie
        cudaErrorCheck(cudaMemset(local_bitmap_gpu.candidate_bits_[u], 0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
        cudaErrorCheck(cudaMemset(local_index_base_gpu.sizes_[edge_idx], 0u, sizeof(uint32_t) * DV_COUNT));
        cudaErrorCheck(cudaDeviceSynchronize());
        // index is the current edge_idx, inputs are csr_gpu[index], global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]
        // the output is local_index_base_gpu. Only fill values in output.sizes_[idx] (an array of length DV_COUNT)
        edgeList2RelationCount_v2<<<GRID_DIM, BLOCK_DIM>>>(
            csr_gpu[edge_idx], local_index_base_gpu, edge_idx,
            global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]);

        // Input: local_index_base_gpu.sizes_[index], Output: local_bitmap_gpu.candidate_bits_[u] (a bitmap).
        // Directly copy the information in local_index_base_gpu.sizes_[index] to the bitmap (size > 0 --> is a candidate)
        setLocalBitmap<<<GRID_DIM, BLOCK_DIM>>>(
            local_index_base_gpu, edge_idx, local_bitmap_gpu.candidate_bits_[u]);
        cudaErrorCheck(cudaDeviceSynchronize());

        // Fill values in local_index_base_gpu.capacity_[index]
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index_base_gpu.sizes_[edge_idx], local_index_base_gpu.capacity_[edge_idx], DV_COUNT + 1));
        
        auto total_size = 0u;
        cudaErrorCheck(cudaMemcpy(&total_size, local_index_base_gpu.capacity_[edge_idx] + DV_COUNT, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (total_size == 0u) {
            return false;
        }

        ReAlloc(local_nbr_[edge_idx], total_size, local_nbr_capacity_[edge_idx], uint32_t*);
        // Fill values (neighbor pointers) in local_index_base_gpu.nbrs_[index].
        // After this function,
        // segments of local_nbr_[edge_index] (an continuous array) are pointed to by pointers in local_index_base_gpu.nbrs_[edge_index] (an array of pointers).
        setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[edge_idx], local_index_base_gpu.capacity_[edge_idx], DV_COUNT, local_index_base_gpu.nbrs_[edge_idx]);
        cudaErrorCheck(cudaDeviceSynchronize());

        // Fill the neighbor IDs.
        edgeList2RelationWrite_v2<<<GRID_DIM, BLOCK_DIM>>>(
            csr_gpu[edge_idx], local_index_base_gpu, edge_idx,
            global_bitmap_gpu.candidate_bits_[uu],
            local_bitmap_gpu.candidate_bits_[u]);
        cudaErrorCheck(cudaDeviceSynchronize());

        local_index.nbrs_[edge_idx] = local_index_base_gpu.nbrs_[edge_idx];
        local_index.sizes_[edge_idx] = local_index_base_gpu.sizes_[edge_idx];
    }
    return true;
}

bool MatchGPU::BuildLocalIndex_v2(const RelationsGPU& global_index_gpu, CandidatesGPU& global_bitmap_gpu,
                                  RelationsGPU& local_index_base_gpu, CandidatesGPU& local_bitmap_gpu,
                                  RelationsGPU& local_index, CSR_GPU csr_gpu[],
                                  const uint8_t cur_edge_list_idx, const float *avg_degrees,
                                  const bool enable_heavy_vertex_avoidance) {
    // lgh: cur_i is current idx_in_qe_list_
    // Visit the query vertices in plan_.indexing_orders_[cur_i] one by one.
    for (auto i = 1u; i < QV_COUNT; i++)
    {
        if (i == 1)
        {
            const auto& off = plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i];

            const auto& u0 = plan_.indexing_orders_[cur_edge_list_idx].vs_[1];
            const auto& u1 = plan_.indexing_orders_[cur_edge_list_idx].vs_[plan_.indexing_orders_[cur_edge_list_idx].bni_[off]];
            for (auto j = 0u; j < 2u; j++)
            {
                const auto& u = j == 0u ? u0 : u1;
                const auto& uu = j == 0u ? u1 : u0;
                // lgh: index is edge_idx
                const auto& index = query_.eidx_[u * QV_COUNT + uu];

                // build the relation from trie
                cudaErrorCheck(cudaMemset(local_bitmap_gpu.candidate_bits_[u], 0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));
                cudaErrorCheck(cudaMemset(local_index_base_gpu.sizes_[index], 0u, sizeof(uint32_t) * DV_COUNT));
                cudaErrorCheck(cudaDeviceSynchronize());
                // index is the current edge_idx, inputs are csr_gpu[index], global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]
                // the output is local_index_base_gpu. Only fill values in output.sizes_[idx] (an array of length DV_COUNT)
                edgeList2RelationCount_v2<<<GRID_DIM, BLOCK_DIM>>>(
                    csr_gpu[index], local_index_base_gpu, index,
                    global_bitmap_gpu.candidate_bits_[u], global_bitmap_gpu.candidate_bits_[uu]);

                // Input: local_index_base_gpu.sizes_[index], Output: local_bitmap_gpu.candidate_bits_[u] (a bitmap).
                // Directly copy the information in local_index_base_gpu.sizes_[index] to the bitmap (size > 0 --> is a candidate)
                setLocalBitmap<<<GRID_DIM, BLOCK_DIM>>>(
                    local_index_base_gpu, index, local_bitmap_gpu.candidate_bits_[u]);
                cudaErrorCheck(cudaDeviceSynchronize());

                // Fill values in local_index_base_gpu.capacity_[index]
                CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index_base_gpu.sizes_[index], local_index_base_gpu.capacity_[index], DV_COUNT + 1));
                auto total_size = 0u;
                cudaErrorCheck(cudaMemcpy(&total_size, local_index_base_gpu.capacity_[index] + DV_COUNT, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                if (total_size == 0u) return false;

                ReAlloc(local_nbr_[index], total_size, local_nbr_capacity_[index], uint32_t*);
                // Fill values (neighbor pointers) in local_index_base_gpu.nbrs_[index].
                // After this function,
                // segments of local_nbr_[edge_index] (an continuous array) are pointed to by pointers in local_index_base_gpu.nbrs_[edge_index] (an array of pointers).
                setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[index], local_index_base_gpu.capacity_[index], DV_COUNT, local_index_base_gpu.nbrs_[index]);
                cudaErrorCheck(cudaDeviceSynchronize());

                // Fill the neighbor IDs.
                edgeList2RelationWrite_v2<<<GRID_DIM, BLOCK_DIM>>>(
                    csr_gpu[index], local_index_base_gpu, index,
                    global_bitmap_gpu.candidate_bits_[uu],
                    local_bitmap_gpu.candidate_bits_[u]);
                cudaErrorCheck(cudaDeviceSynchronize());

                local_index.nbrs_[index] = local_index_base_gpu.nbrs_[index];
                local_index.sizes_[index] = local_index_base_gpu.sizes_[index];
            }
        }
        else  // For vertices after the first two vertices in plan_.indexing_orders_.
        {
            const auto& u0 = plan_.indexing_orders_[cur_edge_list_idx].vs_[i];

            if (plan_.rebuild_B_flags_[cur_edge_list_idx][u0])  // If rebuilding the bitmap for the current query vertex.
            {
                // 1. find all candidates with at least one neighbor to match each backward neighbor
                cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));
                cudaErrorCheck(cudaMemset(local_bitmap_gpu.candidate_bits_[u0], 0u, sizeof(uint32_t) * DIV_CEIL(DV_COUNT, 32u)));

                for (auto off = plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i]; off < plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_edge_list_idx].vs_[plan_.indexing_orders_[cur_edge_list_idx].bni_[off]];

                    // lgh: depth is the index of `u1` in `plan_.indexing_orders_[cur_i].vs_`.
                    const auto& depth = plan_.indexing_orders_[cur_edge_list_idx].bni_[off];
                    // lgh: `first_bn_of_uu` is the first backward neighbor of `u1`.
                    const auto& first_bn_of_uu = plan_.indexing_orders_[cur_edge_list_idx].vs_[plan_.indexing_orders_[cur_edge_list_idx].bni_[
                        plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[depth]
                    ]];
                    const auto& first_bn_of_uu_index = query_.eidx_[u1 * QV_COUNT + first_bn_of_uu];

                    const auto& backward_index = query_.eidx_[u0 * QV_COUNT + u1];
                    const auto& forward_index = query_.eidx_[u1 * QV_COUNT + u0];
                    cudaErrorCheck(cudaDeviceSynchronize());
                    if (!enable_heavy_vertex_avoidance || avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        getLocalCandidatesBackward_v2<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_bitmap_gpu.candidate_bits_[u1], backward_index,
                            cum_bn_, off - plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i]
                        );
                    }
                    else
                    {
                        getLocalCandidatesForward_v2<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_bitmap_gpu.candidate_bits_[u1], forward_index, backward_index,
                            cum_bn_, off - plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i]
                        );
                    }
                }

                // set local_bitmap_gpu
                setLocalBitmap<<<GRID_DIM, BLOCK_DIM>>>(
                    cum_bn_, local_bitmap_gpu.candidate_bits_[u0], 
                    plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i + 1] - plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i]
                );
                cudaErrorCheck(cudaDeviceSynchronize());
            }

            if (plan_.rebuild_R_flags_[cur_edge_list_idx][u0])  // If rebuilding a `RelationsGPU` for the current query vertex.
            {
                // 2. build relations from these candidates (only for vertices v if cum_bn_[v] == # backward neighbors of u)
                // 3. build reversed relations
                for (auto off = plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i]; off < plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_edge_list_idx].vs_[plan_.indexing_orders_[cur_edge_list_idx].bni_[off]];

                    const auto& depth = plan_.indexing_orders_[cur_edge_list_idx].bni_[off];
                    const auto& first_bn_of_uu = plan_.indexing_orders_[cur_edge_list_idx].vs_[plan_.indexing_orders_[cur_edge_list_idx].bni_[
                        plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[depth]
                    ]];
                    const auto& first_bn_of_uu_index = query_.eidx_[u1 * QV_COUNT + first_bn_of_uu];

                    const auto& backward_index = query_.eidx_[u0 * QV_COUNT + u1];
                    const auto& forward_index = query_.eidx_[u1 * QV_COUNT + u0];
                    uint8_t work_index, reversed_index;

                    cudaErrorCheck(cudaMemset(local_index_base_gpu.sizes_[backward_index], 0u, sizeof(uint32_t) * DV_COUNT));
                    cudaErrorCheck(cudaMemset(local_index_base_gpu.sizes_[forward_index], 0u, sizeof(uint32_t) * DV_COUNT));
                    cudaErrorCheck(cudaDeviceSynchronize());

                    // According to `local_bitmap_gpu.candidate_bits_[u0]` and `local_bitmap_gpu.candidate_bits_[u1]`,
                    // fill values into local_index_base_gpu.sizes_[work_index].
                    if (!enable_heavy_vertex_avoidance || avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        // build a trie u0 -> u1
                        // work_index <- backward_index
                        buildLocalRelationNew2OldCount_v2<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, backward_index,
                            local_bitmap_gpu.candidate_bits_[u0], local_bitmap_gpu.candidate_bits_[u1]);
                        work_index = backward_index;
                        reversed_index = forward_index;
                    }
                    else
                    {
                        // build a trie u1 -> u0
                        buildLocalRelationOld2NewCount_v2<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, forward_index,
                            local_bitmap_gpu.candidate_bits_[u0], local_bitmap_gpu.candidate_bits_[u1]);
                        work_index = forward_index;
                        reversed_index = backward_index;
                    }
                    cudaErrorCheck(cudaDeviceSynchronize());

                    // Fill values into local_index_base_gpu.capacity_[work_index]
                    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index_base_gpu.sizes_[work_index], local_index_base_gpu.capacity_[work_index], DV_COUNT + 1));
                    auto total_size = 0u;
                    cudaErrorCheck(cudaMemcpy(&total_size, local_index_base_gpu.capacity_[work_index] + DV_COUNT, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                    if (total_size == 0u) return false;

                    ReAlloc(helper_relation_[0], total_size, helper_relation_capacity_[0], uint32_t);
                    ReAlloc(helper_relation_[1], total_size, helper_relation_capacity_[1], uint32_t);
                    ReAlloc(temp_tries_[0].vs_, total_size, temp_tries_capacity_[0].vs_capacity_, uint32_t);
                    ReAlloc(temp_tries_[0].offs_, total_size, temp_tries_capacity_[0].off_capacity_, uint32_t);

                    ReAlloc(local_nbr_[work_index], total_size, local_nbr_capacity_[work_index], uint32_t*);
                    ReAlloc(local_nbr_[reversed_index], total_size, local_nbr_capacity_[reversed_index], uint32_t*);
                    // Fill values (neighbor pointers) in local_index_base_gpu.nbrs_[work_index].
                    // After this function,
                    // segments of local_nbr_[work_index] (an continuous array) are pointed to by pointers in local_index_base_gpu.nbrs_[work_index] (an array of pointers).
                    setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[work_index], local_index_base_gpu.capacity_[work_index], DV_COUNT, local_index_base_gpu.nbrs_[work_index]);
                    cudaErrorCheck(cudaDeviceSynchronize());

                    // Fill dv IDs into helper_relation[0]
                    // Fill neighbor IDs into each local_index_base_gpu.nbrs_[work_index][dv] (resident at local_nbr_[work_index]).
                    if (!enable_heavy_vertex_avoidance || avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        // build a trie u0 -> u1, lgh: not a trie
                        buildLocalRelationNew2OldWrite_v2<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, backward_index,
                            local_bitmap_gpu.candidate_bits_[u0], local_bitmap_gpu.candidate_bits_[u1],
                            helper_relation_[0]);
                    }
                    else
                    {
                        // build a trie u1 -> u0, lgh: not a trie
                        buildLocalRelationOld2NewWrite_v2<<<GRID_DIM, BLOCK_DIM>>>(
                            global_index_gpu, local_index_base_gpu, forward_index,
                            local_bitmap_gpu.candidate_bits_[u0], local_bitmap_gpu.candidate_bits_[u1], helper_relation_[0]);
                    }
                    cudaErrorCheck(cudaDeviceSynchronize());

                    // reverse temp_tries_[0] into temp_tries_[1] lgh: No. Store the reverse trie into temp_tries_[0]
                    // (Solved) Question: Seems there are no values in local_nbr_[work_index] yet. 
                    // [local_nbr_ values are filled in the function `buildLocalRelationNew2OldWrite_v2` or `buildLocalRelationOld2NewWrite_v2`]
                    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
                        local_nbr_[work_index], helper_relation_[1], helper_relation_[0], local_nbr_[reversed_index], total_size));

                    // lgh: temp_tries_[0].offs_ does not store offsets but stores neighbor sizes.
                    CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, helper_relation_[1], temp_tries_[0].vs_, temp_tries_[0].offs_, d_new_cand_count_[0], total_size));
                    cudaErrorCheck(cudaDeviceSynchronize());
                    cudaErrorCheck(cudaMemcpy(&temp_tries_[0].vs_size_, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));

                    // map trie to relation
                    // Copy the information in temp_tries_[0].offs_ (neighbor sizes of the reverse trie) to local_index_base_gpu[reversed_index].sizes_
                    mapTrieToRelation<<<GRID_DIM, BLOCK_DIM>>>(local_index_base_gpu, reversed_index, temp_tries_[0].vs_, temp_tries_[0].offs_, temp_tries_[0].vs_size_);
                    cudaErrorCheck(cudaDeviceSynchronize());

                    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index_base_gpu.sizes_[reversed_index], local_index_base_gpu.capacity_[reversed_index], DV_COUNT + 1));
                    // Fill values (neighbor pointers) in local_index_base_gpu.nbrs_[reverse_index].
                    // After this function,
                    // segments of local_nbr_[revierse_index] (an continuous array) are pointed to by pointers in local_index_base_gpu.nbrs_[reverse_index] (an array of pointers).
                    // Neighbor IDs have been fiiled into local_nbr_[reversed_index] using `cub::DeviceRadixSort::SortPairs`
                    setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[reversed_index], local_index_base_gpu.capacity_[reversed_index], DV_COUNT, local_index_base_gpu.nbrs_[reversed_index]);
                    cudaErrorCheck(cudaDeviceSynchronize());

                    local_index.nbrs_[backward_index] = local_index_base_gpu.nbrs_[backward_index];
                    local_index.sizes_[backward_index] = local_index_base_gpu.sizes_[backward_index];
                    local_index.nbrs_[forward_index] = local_index_base_gpu.nbrs_[forward_index];
                    local_index.sizes_[forward_index] = local_index_base_gpu.sizes_[forward_index];
                }
            }
            else  // If not rebuilding a `RelationsGPU` for the current query vertex, directly copied global_index_gpu.xx[edge_idx] to local_index.xx[edge_idx].
            {
                for (auto off = plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i]; off < plan_.indexing_orders_[cur_edge_list_idx].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_edge_list_idx].vs_[plan_.indexing_orders_[cur_edge_list_idx].bni_[off]];

                    //std::cout << "skip " << static_cast<uint32_t>(u0) << ' ' << static_cast<uint32_t>(u1) << '\n';
                    for (auto j = 0u; j < 2u; j++)
                    {
                        const auto& u = j == 0u ? u0 : u1;
                        const auto& uu = j == 0u ? u1 : u0;

                        // copy the relation from the global index
                        local_index.nbrs_[query_.eidx_[u * QV_COUNT + uu]] = global_index_gpu.nbrs_[query_.eidx_[u * QV_COUNT + uu]];
                        local_index.sizes_[query_.eidx_[u * QV_COUNT + uu]] = global_index_gpu.sizes_[query_.eidx_[u * QV_COUNT + uu]];
                    }
                }
            }
        }
    }
    return true;
}  // void MatchGPU::BuildLocalIndex_v2

// void MatchGPU::OursMatching(RelationsGPUAllVersions &local_index_all_versions, unsigned long long int& num_matches)
// {
//     uint8_t num_query_vertices = query_.getVerticesCount();

//     // lgh: i is current idx_in_qe_list_
//     res_queue_.Reset();
//     bool oom = false;

//     unsigned long long int *effective_num_results_dptr;
//     cudaErrorCheck(cudaMalloc(&effective_num_results_dptr, sizeof(unsigned long long int)));
//     cudaErrorCheck(cudaMemset(effective_num_results_dptr, 0u, sizeof(unsigned long long int)));

//     /************************************ Step 1 ************************************/
//     // initialize the partial results as all data edges mapping to the first two query vertices
//     cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
//     new_res_ = res_queue_.TryMax();  // TryMax(): return available_start_
//     // new_depth_ = 2u;
//     h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
//     cudaErrorCheck(cudaDeviceSynchronize());

//     // std::cout << "query_: " << &query_ << ". Before uint8_t num_query_edges = query_.getEdgeCount();" << endl;

//     uint8_t num_query_edges = query_.getEdgeCount();

//     // std::cout << "Before bool enumerate_cartesian_product = false;" << endl;
    
//     bool enumerate_cartesian_product = false;
//     // bool &write_results_to_global_mem = enumerate_cartesian_product;
//     for (uint8_t i = 0; i < num_query_edges; i++) {
//         uint8_t cur_non_tail_leaf_count = plan_.getNonTailLeafDepthByEdgeListIdx(i);
//         std::cout << "edge_list_idx: " << (uint32_t)i << " cur_non_tail_leaf_count: " << (uint32_t)cur_non_tail_leaf_count << endl;
//         if (cur_non_tail_leaf_count < num_query_vertices) {
//             enumerate_cartesian_product = true;
//             break;
//         }
//     }

//     std::cout << "enumerate_cartesian_product: " << (enumerate_cartesian_product ? "true" : "false") << endl;
    
//     h_new_res_size_ = 0;
//     unsigned long long int h_new_res_size_old = 0;
//     for (uint8_t cur_edge_list_idx = 0; cur_edge_list_idx < num_query_edges; cur_edge_list_idx++) {
//         ours_write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(
//             local_index_all_versions.GetDataGraphByIdx((size_t)cur_edge_list_idx), cur_edge_list_idx, num_query_vertices,
//             new_res_, new_res_size_, h_max_new_res_size_);
//         // ours_write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(
//         //     local_index_all_versions.GetDataGraphByIdx((size_t)cur_edge_list_idx), cur_edge_list_idx, num_query_vertices,
//         //     new_res_, new_res_size_, h_max_new_res_size_);
//         cudaErrorCheck(cudaDeviceSynchronize());
//         cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

// #ifdef IS_DEBUGGING_MATCH_GPU
//         std::cout << "after gamma_write_initial_partial_results for the representative edge " << (uint32_t)cur_edge_list_idx << endl;
//         std::cout << "h_new_res_size_: " << h_new_res_size_ << endl;
//         std::cout << "newly written res_size: " << h_new_res_size_ - h_new_res_size_old << endl;
//         // std::cout << "num_query_vertices: " << (uint32_t)num_query_vertices << endl;
//         // std::cout << "new_res_: " << new_res_ << endl;
// #endif  // IS_DEBUGGING_MATCH_GPU

//         h_new_res_size_old = h_new_res_size_;
//     }
//     cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

//     //std::cout << "Num matches of size 2 by initialization: " << h_new_res_size_ << '\n';

//     res_queue_.Push(h_new_res_size_ * (num_query_vertices + 1));
//     res_ = new_res_;  // preserve the old value of `new_res_`, which is the starting point of existing results.
//     res_size_ = h_new_res_size_;
//     if (res_size_ == 0ul) {
//         cudaErrorCheck(cudaFree(effective_num_results_dptr));
//         return;
//     }
//     // cur_depth_ = new_depth_;

//     /************************************ Step 2 ************************************/

//     cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(*new_res_size_)));
//     // new_depth_ = cur_depth_ + 1;
//     new_res_ = res_queue_.TryMax();
//     h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
//     cudaErrorCheck(cudaDeviceSynchronize());

// #ifdef IS_DEBUGGING_MATCH_GPU
//     std::cout << "res_: " << res_ << ", new_res_: " << new_res_ << ", before ours_enumerate." << endl;
//     std::cout << "# initial results: " << res_size_ << endl;
// #endif  // IS_DEBUGGING_MATCH_GPU

//     uint32_t grid_dim = DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK);
//     uint32_t num_threads = grid_dim * BLOCK_DIM;

//     std::cout << "launching " << grid_dim << " blocks, " << num_threads << " threads, " << num_threads / 32 << " warps." << endl;

//     cur_depth_ = 2;
//     ours_enumerate<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
//         res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, local_index_all_versions, cur_depth_, num_query_vertices, true, enumerate_cartesian_product, enumerate_cartesian_product,
//         effective_num_results_dptr
//     );

//     cudaErrorCheck(cudaDeviceSynchronize());
//     cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(*new_res_size_), cudaMemcpyDeviceToHost));

//     unsigned long long int effective_num_results_v0 = 0;
//     cudaErrorCheck(cudaMemcpy(&effective_num_results_v0, effective_num_results_dptr, sizeof(*effective_num_results_dptr), cudaMemcpyDeviceToHost));

// #ifdef IS_DEBUGGING_MATCH_GPU
//     std::cout << "ours_enumerate finished. (after synchronization), current h_new_res_size_: " << h_new_res_size_ << ", effective_num_results_v0: " << effective_num_results_v0 << endl;
// #endif  // IS_DEBUGGING_MATCH_GPU

//     if (h_new_res_size_ >= h_max_new_res_size_)
//     {
//         //std::cout << "Stop when trying to extend by BFS from depth " << static_cast<uint32_t>(cur_depth_) << " to " << static_cast<uint32_t>(new_depth_) << '\n';
//         std::cout << "OOM: after ours_enumerate, h_new_res_size_ >= h_max_new_res_size_ !" << endl;
//         oom = true;
//         // break;
//     }

//     res_queue_.Push(h_new_res_size_ * (num_query_vertices + 1));
//     res_queue_.Pop(res_size_ * (num_query_vertices + 1));

//     res_ = new_res_;
//     res_size_ = h_new_res_size_;
//     if (res_size_ == 0ul) {
//         cudaErrorCheck(cudaFree(effective_num_results_dptr));
//         return;
//     }

//     /************************************ Step 4 ************************************/
//     // if the workload is imbalanced, perform cartesian product
    
//     // bool enumerate_cartesian_product = plan_.cartesian_product_info_[i][cur_depth_] == Plan::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;
    
//     if (enumerate_cartesian_product)
//     {

// #ifdef IS_DEBUGGING_MATCH_GPU
//         std::cout << "inside if(enemerate_cartesian_product)." << endl;
// #endif  // IS_DEBUGGING_MATCH_GPU

//         // generate max_num_matches array
//         OursGetNumTree<<<GRID_DIM, BLOCK_DIM>>>(
//             res_, res_size_, res_size_cartesian_product_, local_index_all_versions, num_query_vertices
//         );
//         cudaErrorCheck(cudaDeviceSynchronize());
        
//         // prefix sum
//         CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
//             res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
//         cudaErrorCheck(cudaDeviceSynchronize());

//         cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(*new_res_size_)));
//         cudaErrorCheck(cudaMemset(effective_num_results_dptr, 0u, sizeof(*effective_num_results_dptr)));
//         new_res_ = res_queue_.TryMax();
//         cudaErrorCheck(cudaDeviceSynchronize());

//         unsigned long max_result_size = 123u;
//         cudaMemcpy(&max_result_size, res_size_cartesian_product_ + res_size_, sizeof(*res_size_cartesian_product_), cudaMemcpyDeviceToHost);

// #ifdef IS_DEBUGGING_MATCH_GPU
//         std::cout << "max_result_size: " << max_result_size << ", oursEnumerateCartesianProductTree." << endl;
// #endif  // IS_DEBUGGING_MATCH_GPU

//         oursEnumerateCartesianProductTree<<<GRID_DIM, BLOCK_DIM>>>(
//             res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_, local_index_all_versions, num_query_vertices, new_res_size_,
//             effective_num_results_dptr
//         );

//         cudaErrorCheck(cudaDeviceSynchronize());
//         cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(*new_res_size_), cudaMemcpyDeviceToHost));
//     }

//     unsigned long long int effective_num_results = 0;
//     cudaErrorCheck(cudaMemcpy(&effective_num_results, effective_num_results_dptr, sizeof(*effective_num_results_dptr), cudaMemcpyDeviceToHost));

// #ifdef IS_DEBUGGING_MATCH_GPU
//     std::cout << "Matching finished, h_new_res_size_: " << h_new_res_size_ << ", effective_num_results: " << effective_num_results << "." << endl;
// #endif  // IS_DEBUGGING_MATCH_GPU

//     cudaErrorCheck(cudaFree(effective_num_results_dptr));
//     num_matches += effective_num_results;
//     // num_matches += h_new_res_size_;
// }  // void MatchGPU::OursMatching

void MatchGPU::GammaMatching(RelationsGPU& global_index, RelationsGPU& local_index, unsigned long long int& num_matches,
                             const bool materialize_results, uint32_t ** result_ptr_ptr, uint32_t * result_size_ptr) {
    uint8_t num_query_vertices = query_.getVerticesCount();

    // lgh: i is current idx_in_qe_list_
    res_queue_.Reset();
    bool oom = false;

    unsigned long long int *effective_num_results_dptr;
    cudaErrorCheck(cudaMalloc(&effective_num_results_dptr, sizeof(unsigned long long int)));
    cudaErrorCheck(cudaMemset(effective_num_results_dptr, 0u, sizeof(unsigned long long int)));

    /************************************ Step 1 ************************************/
    // initialize the partial results as all data edges mapping to the first two query vertices
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();  // TryMax(): return available_start_
    // new_depth_ = 2u;
    h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
    cudaErrorCheck(cudaDeviceSynchronize());

    // uint8_t num_groups = am_ptr_->getNumEquivGroups();
    std::vector<uint8_t> rep_edge_list_idxs = am_ptr_->getRepEdgeListIdxVector();
    uint8_t num_query_edges = query_.getEdgeCount();
    
    bool enumerate_cartesian_product = false;
    // bool &write_results_to_global_mem = enumerate_cartesian_product;
    for (uint8_t i = 0; i < num_query_edges; i++) {
        uint8_t cur_non_tail_leaf_count = plan_.getNonTailLeafDepthByEdgeListIdx(i);

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "edge_list_idx: " << (uint32_t)i << " cur_non_tail_leaf_count: " << (uint32_t)cur_non_tail_leaf_count << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        if (cur_non_tail_leaf_count < num_query_vertices) {
            enumerate_cartesian_product = true;
            break;
        }
    }

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "enumerate_cartesian_product: " << (enumerate_cartesian_product ? "true" : "false") << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    h_new_res_size_ = 0;
    for (uint8_t cur_edge_list_idx : rep_edge_list_idxs) {
        uint8_t cur_edge_idx = query_.getEdgeIdxByEdgeListIdx(cur_edge_list_idx);
        gamma_write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(
            local_index, cur_edge_idx, cur_edge_list_idx, num_query_vertices,
            new_res_, new_res_size_, h_max_new_res_size_);
        cudaErrorCheck(cudaDeviceSynchronize());
// #ifdef IS_DEBUGGING_MATCH_GPU
        // cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        // std::cout << "after gamma_write_initial_partial_results for the representative edge " << (uint32_t)cur_edge_list_idx << endl;
        // std::cout << "h_new_res_sizew_: " << h_new_res_size_ << endl;
        // std::cout << "num_query_vertices: " << (uint32_t)num_query_vertices << endl;
        // std::cout << "new_res_: " << new_res_ << endl;
// #endif  // IS_DEBUGGING_MATCH_GPU
    }
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

// #ifdef IS_DEBUGGING_MATCH_GPU
    //std::cout << "Num matches of size 2 by initialization: " << h_new_res_size_ << '\n';
// #endif  // IS_DEBUGGING_MATCH_GPU

    res_queue_.Push(h_new_res_size_ * (num_query_vertices + 1));
    res_ = new_res_;  // preserve the old value of `new_res_`, which is the starting point of existing results.
    res_size_ = h_new_res_size_;
    if (res_size_ == 0ul) {
        cudaErrorCheck(cudaFree(effective_num_results_dptr));
        return;
    }
    // cur_depth_ = new_depth_;

    /************************************ Step 2 ************************************/

    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(*new_res_size_)));
    // new_depth_ = cur_depth_ + 1;
    new_res_ = res_queue_.TryMax();
    h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
    cudaErrorCheck(cudaDeviceSynchronize());

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "res_: " << res_ << ", new_res_: " << new_res_ << ", before gamma_enumerate." << endl;
    std::cout << "# initial results: " << res_size_ << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    uint32_t grid_dim = DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK);
    uint32_t num_threads = grid_dim * BLOCK_DIM;

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "launching " << grid_dim << " blocks, " << num_threads << " threads, " << num_threads / 32 << " warps." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cur_depth_ = 2;
    bool write_results_to_global_memory = enumerate_cartesian_product || materialize_results;
    gamma_enumerate<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
        res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, global_index, cur_depth_, num_query_vertices, 
        /*enale_work_stealing=*/true, enumerate_cartesian_product, write_results_to_global_memory,
        effective_num_results_dptr
    );

    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(*new_res_size_), cudaMemcpyDeviceToHost));

    unsigned long long int effective_num_results_v0 = 0;
    cudaErrorCheck(cudaMemcpy(&effective_num_results_v0, effective_num_results_dptr, sizeof(*effective_num_results_dptr), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "gamma_enumerate finished. (after synchronization), current h_new_res_size_: " << h_new_res_size_ << ", effective_num_results_v0: " << effective_num_results_v0 << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    if (h_new_res_size_ >= h_max_new_res_size_)
    {
        //std::cout << "Stop when trying to extend by BFS from depth " << static_cast<uint32_t>(cur_depth_) << " to " << static_cast<uint32_t>(new_depth_) << '\n';
        std::cout << "OOM: after gamma_enumerate, h_new_res_size_ >= h_max_new_res_size_ !" << endl;
        oom = true;
        // break;
    }

    res_queue_.Push(h_new_res_size_ * (num_query_vertices + 1));
    res_queue_.Pop(res_size_ * (num_query_vertices + 1));

    res_ = new_res_;
    res_size_ = h_new_res_size_;
    if (res_size_ == 0ul) {
        cudaErrorCheck(cudaFree(effective_num_results_dptr));
        return;
    }

    /************************************ Step 4 ************************************/
    // if the workload is imbalanced, perform cartesian product
    
    // bool enumerate_cartesian_product = plan_.cartesian_product_info_[i][cur_depth_] == Plan::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;
    
    if (enumerate_cartesian_product)
    {

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "inside if(enemerate_cartesian_product)." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        // generate max_num_matches array
        GammaGetNumTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, global_index, num_query_vertices
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        
        // prefix sum
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
        cudaErrorCheck(cudaDeviceSynchronize());

        cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(*new_res_size_)));
        cudaErrorCheck(cudaMemset(effective_num_results_dptr, 0u, sizeof(*effective_num_results_dptr)));
        new_res_ = res_queue_.TryMax();
        h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
        cudaErrorCheck(cudaDeviceSynchronize());

        unsigned long max_result_size = 123u;
        cudaMemcpy(&max_result_size, res_size_cartesian_product_ + res_size_, sizeof(*res_size_cartesian_product_), cudaMemcpyDeviceToHost);

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "max_result_size: " << max_result_size << ", gammaEnumerateCartesianProductTree." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        gammaEnumerateCartesianProductTree<<<GRID_DIM, BLOCK_DIM>>>(
                res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_, 
                global_index, num_query_vertices, new_res_size_, effective_num_results_dptr, materialize_results);

        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(*new_res_size_), cudaMemcpyDeviceToHost));
    }

    if (materialize_results) {
        if (h_new_res_size_ < h_max_new_res_size_) {
            *result_ptr_ptr = res_queue_.CopyToHost(new_res_, h_new_res_size_ * (num_query_vertices + 1));
            *result_size_ptr = h_new_res_size_;
        } else {
            oom = true;
            std::cout << "OOM: h_new_res_size_(" << h_new_res_size_ << ") >= h_max_new_res_size_.(" << h_max_new_res_size_ << ")" << endl;
        }
    }

    unsigned long long int effective_num_results = 0;
    cudaErrorCheck(cudaMemcpy(&effective_num_results, effective_num_results_dptr, sizeof(*effective_num_results_dptr), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "Matching finished, h_new_res_size_: " << h_new_res_size_ << ", effective_num_results: " << effective_num_results << "." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaFree(effective_num_results_dptr));
    num_matches += effective_num_results;
}  // void MatchGPU::GammaMatching

void MatchGPU::QOGammaMatching(RelationsGPU& data_graph_index, RelationsGPU& trival_local_index, 
                                    const uint8_t edge_list_idx, unsigned long long int& num_matches, 
                                    const bool enable_work_stealing, const bool enable_cartesian_product,
                                    const bool materialize_results) {
    uint8_t num_query_vertices = query_.getVerticesCount();

    // lgh: i is current idx_in_qe_list_
    res_queue_.Reset();
    bool oom = false;

    unsigned long long int *effective_num_results_dptr;
    cudaErrorCheck(cudaMalloc(&effective_num_results_dptr, sizeof(unsigned long long int)));
    cudaErrorCheck(cudaMemset(effective_num_results_dptr, 0u, sizeof(unsigned long long int)));

    /************************************ Step 1 ************************************/
    // initialize the partial results as all data edges mapping to the first two query vertices
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();  // TryMax(): return available_start_
    // new_depth_ = 2u;
    h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
    cudaErrorCheck(cudaDeviceSynchronize());

    uint8_t num_query_edges = query_.getEdgeCount();
    bool enumerate_cartesian_product = false;
    // bool &write_results_to_global_mem = enumerate_cartesian_product;
    if (enable_cartesian_product) {
        uint8_t cur_non_tail_leaf_count = plan_.getNonTailLeafDepthByEdgeListIdx(edge_list_idx);
#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "edge_list_idx: " << static_cast<uint32_t>(edge_list_idx)
                  << " cur_non_tail_leaf_count: " << static_cast<uint32_t>(cur_non_tail_leaf_count) << endl;
#endif  // IS_DEBUGGING_MATCH_GPU
        if (cur_non_tail_leaf_count < num_query_vertices) {
            enumerate_cartesian_product = true;
        }
    }

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "enumerate_cartesian_product: " << (enumerate_cartesian_product ? "true" : "false") << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    h_new_res_size_ = 0;
    for (uint8_t cur_edge_list_idx : {edge_list_idx}) {
        // uint8_t cur_edge_idx = query_.getEdgeIdxByEdgeListIdx(cur_edge_list_idx);  // Wrong!
        
        uint8_t first_query_vertex = plan_.orders_[cur_edge_list_idx].vs_[0];
        uint8_t second_query_vertex = plan_.orders_[cur_edge_list_idx].vs_[1];
        uint8_t cur_edge_idx = query_.getEdgeIdxByVertex(first_query_vertex, second_query_vertex);
        
        gamma_write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(
            trival_local_index, cur_edge_idx, cur_edge_list_idx, num_query_vertices,
            new_res_, new_res_size_, h_max_new_res_size_);
        
        cudaErrorCheck(cudaDeviceSynchronize());
// #ifdef IS_DEBUGGING_MATCH_GPU
        // cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        // std::cout << "after gamma_write_initial_partial_results for the representative edge " << (uint32_t)cur_edge_list_idx << endl;
        // std::cout << "h_new_res_sizew_: " << h_new_res_size_ << endl;
        // std::cout << "num_query_vertices: " << (uint32_t)num_query_vertices << endl;
        // std::cout << "new_res_: " << new_res_ << endl;
// #endif  // IS_DEBUGGING_MATCH_GPU
    }
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

// #ifdef IS_DEBUGGING_MATCH_GPU
    //std::cout << "Num matches of size 2 by initialization: " << h_new_res_size_ << '\n';
// #endif  // IS_DEBUGGING_MATCH_GPU

    res_queue_.Push(h_new_res_size_ * (num_query_vertices + 1));
    res_ = new_res_;  // preserve the old value of `new_res_`, which is the starting point of existing results.
    res_size_ = h_new_res_size_;
    if (res_size_ == 0ul) {
        cudaErrorCheck(cudaFree(effective_num_results_dptr));
        std::cout << "res_size_ == 0, after gamma_write_initial_partial_results, return early. " << endl;
        return;
    }
    // cur_depth_ = new_depth_;

    /************************************ Step 2 ************************************/

    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(*new_res_size_)));
    // new_depth_ = cur_depth_ + 1;
    new_res_ = res_queue_.TryMax();
    h_max_new_res_size_ = res_queue_.GetFree() / (num_query_vertices + 1);
    cudaErrorCheck(cudaDeviceSynchronize());

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "res_: " << res_ << ", new_res_: " << new_res_ << ", before gamma_enumerate." << endl;
    std::cout << "# initial results: " << res_size_ << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    uint32_t grid_dim = DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK);
    uint32_t num_threads = grid_dim * BLOCK_DIM;

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "launching " << grid_dim << " blocks, " << num_threads << " threads, " << num_threads / 32 << " warps." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cur_depth_ = 2;
    gamma_enumerate<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
        res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, 
        data_graph_index, cur_depth_, num_query_vertices, 
        enable_work_stealing, enumerate_cartesian_product, /*write_results=*/enumerate_cartesian_product,
        effective_num_results_dptr
    );

    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(*new_res_size_), cudaMemcpyDeviceToHost));

    unsigned long long int effective_num_results_v0 = 0;
    cudaErrorCheck(cudaMemcpy(&effective_num_results_v0, effective_num_results_dptr, sizeof(*effective_num_results_dptr), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "gamma_enumerate finished. (after synchronization), current h_new_res_size_: " << h_new_res_size_ << ", effective_num_results_v0: " << effective_num_results_v0 << endl;
    
    size_t freeMem, totalMem;  // 250217
    cudaErrorCheck(cudaMemGetInfo(&freeMem, &totalMem));  // 250217
    std::cout << "Total GPU memory: " << totalMem / 1024 / 1024 << " MB" << std::endl;  // 250217
    std::cout << "Free GPU memory: " << freeMem / 1024 / 1024 << " MB" << std::endl;  // 250217

#endif  // IS_DEBUGGING_MATCH_GPU
    
    if (h_new_res_size_ >= h_max_new_res_size_)
    {
        // std::cout << "Stop when trying to extend by BFS from depth " << static_cast<uint32_t>(cur_depth_) << " to " << static_cast<uint32_t>(new_depth_) << '\n';
        std::cout << "OOM: after gamma_enumerate, h_new_res_size_ >= h_max_new_res_size_ !" << endl;
        oom = true;
        // break;
    }

    res_queue_.Push(h_new_res_size_ * (num_query_vertices + 1));
    res_queue_.Pop(res_size_ * (num_query_vertices + 1));

    res_ = new_res_;
    res_size_ = h_new_res_size_;
    if (res_size_ == 0ul) {
        cudaErrorCheck(cudaFree(effective_num_results_dptr));
        std::cout << "res_size_ == 0, after gamma_enumerate, return early. " << endl;
        return;
    }

    /************************************ Step 4 ************************************/
    // if the workload is imbalanced, perform cartesian product
    
    // bool enumerate_cartesian_product = plan_.cartesian_product_info_[i][cur_depth_] == Plan::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;
    
    if (enumerate_cartesian_product)
    {

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "inside if(enemerate_cartesian_product)." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        // generate max_num_matches array
        GammaGetNumTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, data_graph_index, num_query_vertices
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        
        // prefix sum
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
        cudaErrorCheck(cudaDeviceSynchronize());

        cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(*new_res_size_)));
        cudaErrorCheck(cudaMemset(effective_num_results_dptr, 0u, sizeof(*effective_num_results_dptr)));
        new_res_ = res_queue_.TryMax();
        cudaErrorCheck(cudaDeviceSynchronize());

        unsigned long max_result_size = 123u;
        cudaMemcpy(&max_result_size, res_size_cartesian_product_ + res_size_, sizeof(*res_size_cartesian_product_), cudaMemcpyDeviceToHost);

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "max_result_size: " << max_result_size << ", gammaEnumerateCartesianProductTree." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        gammaEnumerateCartesianProductTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_, data_graph_index, num_query_vertices, new_res_size_,
            effective_num_results_dptr
        );

        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(*new_res_size_), cudaMemcpyDeviceToHost));
    }

    unsigned long long int effective_num_results = 0;
    cudaErrorCheck(cudaMemcpy(&effective_num_results, effective_num_results_dptr, sizeof(*effective_num_results_dptr), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "CorrectGammaMatching finished, h_new_res_size_: " << h_new_res_size_ << ", effective_num_results: " << effective_num_results << "." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaFree(effective_num_results_dptr));
    num_matches += h_new_res_size_;
}  // void MatchGPU::CorrectGammaMatching

void MatchGPU::Matching(RelationsGPU& data_graph_index, const uint8_t i, unsigned long long int& num_matches, 
                        const bool enable_cartesian_product, RelationsGPU *ptr_trivial_local_index,
                        const bool materialize_results) {
    // lgh: i is current idx_in_qe_list_
    res_queue_.Reset();
    bool oom = false;

    /************************************ Step 1 ************************************/
    // initialize the partial results as all data edges mapping to the first two query vertices
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();  // TryMax(): return available_start_
    new_depth_ = 2u;
    h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
    cudaErrorCheck(cudaDeviceSynchronize());

    RelationsGPU *ptr_index_for_initial = ptr_trivial_local_index ? ptr_trivial_local_index : &data_graph_index;

#ifdef IS_DEBUGGING_MATCH_GPU
    printf("ptr_trivial_local_index: %p, ptr_index_for_initial: %p\n", ptr_trivial_local_index, ptr_index_for_initial);    
    //std::cout << "Num matches of size 2 by initialization: " << h_new_res_size_ << '\n';
#endif  // IS_DEBUGGING_MATCH_GPU

    uint8_t edge_idx = query_.eidx_[plan_.orders_[i].vs_[0] * QV_COUNT + plan_.orders_[i].vs_[1]];
    write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(*ptr_index_for_initial, edge_idx,
                                                           new_res_, new_res_size_, h_max_new_res_size_);

    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "after write_initial_partial_results for the edge " << (uint32_t)i << endl;
    std::cout << "h_new_res_sizew_: " << h_new_res_size_ << endl;
    std::cout << "num_query_vertices: " << QV_COUNT << endl;
    std::cout << "new_res_: " << new_res_ << endl;
    //std::cout << "Num matches of size 2 by initialization: " << h_new_res_size_ << '\n';
#endif  // IS_DEBUGGING_MATCH_GPU

    res_queue_.Push(h_new_res_size_ * new_depth_);
    res_ = new_res_;
    res_size_ = h_new_res_size_;
    if (res_size_ == 0ul) return;
    cur_depth_ = new_depth_;

    /************************************ Step 2 ************************************/
    // if there are not enough partial results, extend one vertex at a time
    while (res_size_ < MIN_NUM_RESULTS_TO_GPU && QV_COUNT - cur_depth_ > 1)
    {
        cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
        new_depth_ = cur_depth_ + 1;
        new_res_ = res_queue_.TryMax();
        h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
        cudaErrorCheck(cudaDeviceSynchronize());

        extendBFSDFSRegTwo<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, data_graph_index, i, cur_depth_, new_depth_, true
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        if (h_new_res_size_ >= h_max_new_res_size_)
        {
            std::cout << "Stop when trying to extend by BFS from depth " << static_cast<uint32_t>(cur_depth_) << " to " << static_cast<uint32_t>(new_depth_) << '\n';
            oom = true;
            break;
        }
        else
        {

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "Num matches of size " << static_cast<uint32_t>(new_depth_) << " by BFS: " << h_new_res_size_ << '\n';
#endif  // IS_DEBUGGING_MATCH_GPU

            res_queue_.Push(h_new_res_size_ * new_depth_);
            res_queue_.Pop(res_size_ * cur_depth_);
            res_ = new_res_;
            res_size_ = h_new_res_size_;
            if (res_size_ == 0ul) return;
            cur_depth_ = new_depth_;
        }
    }

    /************************************ Step 3 ************************************/
    // match all remaining query vertices except for the last several ones for cartesian product
    if (!oom && plan_.cartesian_product_info_[i][cur_depth_] != PlanManager::CartesianProductType::TreeCartesianProduct && QV_COUNT - cur_depth_ > 2)
    {
        new_depth_ = cur_depth_;
        while (
            new_depth_ < QV_COUNT && 
            plan_.cartesian_product_info_[i][new_depth_] != PlanManager::CartesianProductType::TreeCartesianProduct &&
            plan_.cartesian_product_info_[i][new_depth_] != PlanManager::CartesianProductType::TreeSingle
        ) {
            new_depth_++;
        }
        if (QV_COUNT - new_depth_ > 0)
        {
            cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
            new_res_ = res_queue_.TryMax();
            h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
            cudaErrorCheck(cudaDeviceSynchronize());

            extendBFSDFSRegTwo<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
                res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, data_graph_index, i, cur_depth_, new_depth_, true
            );
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
            if (h_new_res_size_ >= h_max_new_res_size_)
            {

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "Stop when trying to extend by BFS-DFS from depth " << static_cast<uint32_t>(cur_depth_) << " to " << static_cast<uint32_t>(new_depth_) << '\n';
#endif  // IS_DEBUGGING_MATCH_GPU

                oom = true;
            }
            else
            {

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "Num matches of size " << static_cast<uint32_t>(new_depth_) << " by BFS-DFS: " << h_new_res_size_ << '\n';
#endif  // IS_DEBUGGING_MATCH_GPU

                res_queue_.Push(h_new_res_size_ * new_depth_);
                res_queue_.Pop(res_size_ * cur_depth_);
                res_ = new_res_;
                res_size_ = h_new_res_size_;
                if (res_size_ == 0ul) return;
                cur_depth_ = new_depth_;
            }
        }
    }

    /************************************ Step 4 ************************************/
    // if the workload is imbalanced, perform cartesian product
    bool enumerate_cartesian_product = plan_.cartesian_product_info_[i][cur_depth_] == PlanManager::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;
    if (!enable_cartesian_product) {
        enumerate_cartesian_product = false;
    }
    if (enumerate_cartesian_product)
    {
        // generate max_num_matches array
        GetNumTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, data_graph_index, i, QV_COUNT - cur_depth_
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        // get max
        CUB(cub::DeviceReduce::Max(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, max_res_size_cartesian_product_, res_size_));
        cudaErrorCheck(cudaDeviceSynchronize());
        // prefix sum
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
        cudaErrorCheck(cudaDeviceSynchronize());

        unsigned long h_max, h_total;
        cudaErrorCheck(cudaMemcpy(&h_max, max_res_size_cartesian_product_, sizeof(unsigned long), cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(&h_total, res_size_cartesian_product_ + res_size_, sizeof(unsigned long), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "max_result_size: " << h_max << ", after GetNumTree." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        const float avg_res_size = (float)h_total / res_size_;
        const float ratio = h_max / avg_res_size;
        //std::cout << "Avg Num matches: " << avg_res_size << '\n';
        //std::cout << "Ratio of matches (Max / Avg): " << ratio << '\n';
        // Doubt: Why avg_res_size >= MIN_NRESULTS_TO_GPU? (Solved) BFSDFS  does not materialize the results for the last query vertex.
        if (ratio < 50.f && avg_res_size >= MIN_NUM_RESULTS_TO_GPU)
        {
            enumerate_cartesian_product = false;
        }
    }

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "enumerate_cartesian_product: " << (enumerate_cartesian_product ? "true" : "false") << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();
    cudaErrorCheck(cudaDeviceSynchronize());
    
    if (enumerate_cartesian_product)
    {
        enumerateCartesianProductTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_, data_graph_index, i, new_res_size_, QV_COUNT - cur_depth_
        );
    }
    else
    {
        extendBFSDFSRegTwo<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, 0, data_graph_index, i, cur_depth_, QV_COUNT, false
        );
    }
    cudaErrorCheck(cudaDeviceSynchronize());

    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    if (enumerate_cartesian_product)
    {
        std::cout << "Num matches of size " << QV_COUNT << " by cartesian product: " << h_new_res_size_ << '\n';
    }
    else
    {
        std::cout << "Num matches of size " << QV_COUNT << " by BFS-DFS: " << h_new_res_size_ << '\n';
    }
#endif  // IS_DEBUGGING_MATCH_GPU

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "Matching finished, h_new_res_size_: " << h_new_res_size_ << "." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    num_matches += h_new_res_size_;
}

void MatchGPU::BfsMatching(RelationsGPU& data_graph_index, const uint8_t edge_list_idx, unsigned long long int& num_matches, 
                           const bool enable_cartesian_product, RelationsGPU *ptr_trivial_local_index, const bool materialize_results) {
    uint32_t num_query_vertices = query_.getVerticesCount();
    // lgh: i is current idx_in_qe_list_
    res_queue_.Reset();
    bool oom = false;

    /************************************ Step 1 ************************************/
    // initialize the partial results as all data edges mapping to the first two query vertices
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();  // TryMax(): return available_start_
    new_depth_ = 2u;
    h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
    cudaErrorCheck(cudaDeviceSynchronize());

    RelationsGPU *ptr_index_for_initial = ptr_trivial_local_index ? ptr_trivial_local_index : &data_graph_index;
    uint8_t edge_idx = query_.eidx_[plan_.orders_[edge_list_idx].vs_[0] * QV_COUNT + plan_.orders_[edge_list_idx].vs_[1]];
    write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(*ptr_index_for_initial, edge_idx,
                                                           new_res_, new_res_size_, h_max_new_res_size_);

    // write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(
    //     data_graph_index, query_.eidx_[plan_.orders_[edge_list_idx].vs_[0] * num_query_vertices + plan_.orders_[edge_list_idx].vs_[1]],
    //     new_res_, new_res_size_, h_max_new_res_size_);

    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "after write_initial_partial_results for the edge " << (uint32_t)edge_list_idx << endl;
    std::cout << "h_new_res_sizew_: " << h_new_res_size_ << endl;
    std::cout << "num_query_vertices: " << num_query_vertices << endl;
    std::cout << "new_res_: " << new_res_ << endl;
    //std::cout << "Num matches of size 2 by initialization: " << h_new_res_size_ << '\n';
#endif  // IS_DEBUGGING_MATCH_GPU

    res_queue_.Push(h_new_res_size_ * new_depth_);
    res_ = new_res_;
    res_size_ = h_new_res_size_;
    if (res_size_ == 0ul) {
        std::cout << "res_size_ == 0, after write_initial_partial_results, return early. " << endl;
        return;
    }
    cur_depth_ = new_depth_;

    uint8_t bfs_depth = static_cast<uint8_t>(num_query_vertices);
    if (enable_cartesian_product) {
        bfs_depth = plan_.getNonTailLeafDepthByEdgeListIdx(edge_list_idx);
    }
    
    /************************************ Step 2 ************************************/
    // if there are not enough partial results, extend one vertex at a time
    // while (res_size_ < MIN_NUM_RESULTS_TO_GPU && QV_COUNT - cur_depth_ > 1)
    while (cur_depth_ < bfs_depth && cur_depth_ < num_query_vertices - 1) {
        cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
        new_depth_ = cur_depth_ + 1;
        new_res_ = res_queue_.TryMax();
        h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
        cudaErrorCheck(cudaDeviceSynchronize());

        extendBFSDFSRegTwo<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, 
            data_graph_index, edge_list_idx, cur_depth_, new_depth_, /*write_results=*/true);

        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        
        if (h_new_res_size_ >= h_max_new_res_size_) {
            std::cout << "Stop when trying to extend by BFS from depth " << static_cast<uint32_t>(cur_depth_) 
                      << " to " << static_cast<uint32_t>(new_depth_) << '\n';
            oom = true;
            break;
        }

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "Num matches of size " << static_cast<uint32_t>(new_depth_) << " by BFS: " << h_new_res_size_ << '\n';
#endif  // IS_DEBUGGING_MATCH_GPU

        res_queue_.Push(h_new_res_size_ * new_depth_);
        res_queue_.Pop(res_size_ * cur_depth_);
        res_ = new_res_;
        res_size_ = h_new_res_size_;
        if (res_size_ == 0ul) {
            std::cout << "res_size_ == 0, after extendBFSDFSRegTwo, return early. cur_depth_ == " << static_cast<uint32_t>(cur_depth_) << endl;
            return;
        }
        cur_depth_ = new_depth_;    
    }

    /************************************ Step 3 ************************************/
    // if the workload is imbalanced, perform cartesian product
    // bool enumerate_cartesian_product = plan_.cartesian_product_info_[edge_list_idx][cur_depth_] == 
    //                                    PlanManager::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;

    // bool enumerate_cartesian_product = enable_cartesian_product && (bfs_depth >= num_query_vertices);  // Wrong!
    bool enumerate_cartesian_product = enable_cartesian_product && (bfs_depth < num_query_vertices);
    if (enumerate_cartesian_product) {
        // generate max_num_matches array
        GetNumTree<<<GRID_DIM, BLOCK_DIM>>>(res_, res_size_, res_size_cartesian_product_, 
                                            data_graph_index, edge_list_idx, num_query_vertices - cur_depth_);
        cudaErrorCheck(cudaDeviceSynchronize());
        // get max
        CUB(cub::DeviceReduce::Max(d_temp_storage_, temp_storage_bytes_,
                                   res_size_cartesian_product_, 
                                   max_res_size_cartesian_product_, res_size_));
        cudaErrorCheck(cudaDeviceSynchronize());
        // prefix sum
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
                                          res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
        cudaErrorCheck(cudaDeviceSynchronize());

        unsigned long h_max, h_total;
        cudaErrorCheck(cudaMemcpy(&h_max, max_res_size_cartesian_product_, sizeof(unsigned long), cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(&h_total, res_size_cartesian_product_ + res_size_, sizeof(unsigned long), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
        std::cout << "max_total_result_size: " << h_total << ", after GetNumTree." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

        const float avg_res_size = (float)h_total / res_size_;
        const float ratio = h_max / avg_res_size;
        //std::cout << "Avg Num matches: " << avg_res_size << '\n';
        //std::cout << "Ratio of matches (Max / Avg): " << ratio << '\n';
        // Doubt: Why avg_res_size >= MIN_NRESULTS_TO_GPU? (Solved) BFSDFS  does not materialize the results for the last query vertex.
        if (ratio < 50.f && avg_res_size >= MIN_NUM_RESULTS_TO_GPU) {
            enumerate_cartesian_product = false;
        }
    }

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "enumerate_cartesian_product: " << (enumerate_cartesian_product ? "true" : "false") << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();
    cudaErrorCheck(cudaDeviceSynchronize());

    if (enumerate_cartesian_product) {
        enumerateCartesianProductTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_, 
            data_graph_index, edge_list_idx, new_res_size_, num_query_vertices - cur_depth_);
    }
    else {
        extendBFSDFSRegTwo<<<DIV_CEIL(res_size_, NUM_WARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, 
            /*h_max_new_res_size_=*/0,  // As write_results is false, the limit of h_max_new_res_size_ will not be applied.
            data_graph_index, edge_list_idx, cur_depth_, num_query_vertices, /*write_results=*/false);
    }
    cudaErrorCheck(cudaDeviceSynchronize());

    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

#ifdef IS_DEBUGGING_MATCH_GPU
    if (enumerate_cartesian_product) {
        std::cout << "Num matches of size " << num_query_vertices << " by cartesian product: " << h_new_res_size_ << '\n';
    }
    else {
        std::cout << "Num matches of size " << num_query_vertices << " by BFS-DFS: " << h_new_res_size_ << '\n';
    }
#endif  // IS_DEBUGGING_MATCH_GPU

#ifdef IS_DEBUGGING_MATCH_GPU
    std::cout << "BfsMatching finished, h_new_res_size_: " << h_new_res_size_ << "." << endl;
#endif  // IS_DEBUGGING_MATCH_GPU

    num_matches += h_new_res_size_;
}

void MatchGPU::BuildTriesFromEdgeList(
    const EdgeList& edge_list, 
    CSR_GPU& csr_gpu, CSR_GPU& rcsr_gpu
) {
    const auto& ecount = edge_list.first.size();

    // 1. allocate csr_gpu
    cudaErrorCheck(cudaMalloc(&csr_gpu.vs_, sizeof(uint32_t) * ecount));
    cudaErrorCheck(cudaMalloc(&csr_gpu.offs_, sizeof(uint32_t) * (ecount + 1)));
    cudaErrorCheck(cudaMalloc(&csr_gpu.nbrs_, sizeof(uint32_t) * ecount));
    ReAlloc(helper_relation_[0], ecount, helper_relation_capacity_[0], uint32_t);

    cudaErrorCheck(cudaMalloc(&rcsr_gpu.vs_, sizeof(uint32_t) * ecount));
    cudaErrorCheck(cudaMalloc(&rcsr_gpu.offs_, sizeof(uint32_t) * (ecount + 1)));
    cudaErrorCheck(cudaMalloc(&rcsr_gpu.nbrs_, sizeof(uint32_t) * ecount));
    ReAlloc(helper_relation_[1], ecount, helper_relation_capacity_[1], uint32_t);

    cudaErrorCheck(cudaMemcpy(helper_relation_[0], edge_list.first.data(), sizeof(uint32_t) * ecount, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(csr_gpu.nbrs_, edge_list.second.data(), sizeof(uint32_t) * ecount, cudaMemcpyHostToDevice));

    // 2. sort the neighbor list
    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
        csr_gpu.nbrs_, helper_relation_[1], helper_relation_[0], rcsr_gpu.nbrs_, ecount));

    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
        rcsr_gpu.nbrs_, helper_relation_[0], helper_relation_[1], csr_gpu.nbrs_, ecount));

    // Doubt: This step seems redundant.
    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
        csr_gpu.nbrs_, helper_relation_[1], helper_relation_[0], rcsr_gpu.nbrs_, ecount));

    csr_gpu.es_size_ = edge_list.first.size();
    rcsr_gpu.es_size_ = edge_list.first.size();

    // 3. run length encoding of the neighbor list
    CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, 
        helper_relation_[0], csr_gpu.vs_, csr_gpu.offs_, d_new_cand_count_[0], ecount));
    cudaErrorCheck(cudaMemcpy(&csr_gpu.vs_size_, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, 
        helper_relation_[1], rcsr_gpu.vs_, rcsr_gpu.offs_, d_new_cand_count_[1], ecount));
    cudaErrorCheck(cudaMemcpy(&rcsr_gpu.vs_size_, d_new_cand_count_[1], sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 4. exclusive sum
    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
        csr_gpu.offs_, csr_gpu.offs_, csr_gpu.vs_size_ + 1));
    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
        rcsr_gpu.offs_, rcsr_gpu.offs_, rcsr_gpu.vs_size_ + 1));
}

void MatchGPU::AllocRelation(
    const uint32_t idx, const CSR_GPU& csr_gpu,
    const uint32_t u, const uint32_t *dvlabels, 
    RelationsGPU& data_graph_gpu, RelationsGPU& global_index_gpu, 
    uint32_t *capacity_prefix_sum,
    uint32_t *data_graph_all_nbrs,
    uint32_t *index_all_nbrs,
    bool build_graph
) {
    // lgh: idx is edge_idx
    // a. malloc
    // cout << "DV_COUNT: " << DV_COUNT << endl;
    if (build_graph)
    {
        cudaErrorCheck(cudaMalloc(&data_graph_gpu.nbrs_[idx], sizeof(uint32_t*) * (DV_COUNT + 1)));
        cudaErrorCheck(cudaMalloc(&data_graph_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1)));
        cudaErrorCheck(cudaMalloc(&data_graph_gpu.sizes_[idx], sizeof(uint32_t) * (DV_COUNT + 1)));
    }
    cudaErrorCheck(cudaMalloc(&global_index_gpu.nbrs_[idx], sizeof(uint32_t*) * (DV_COUNT + 1)));
    cudaErrorCheck(cudaMalloc(&global_index_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1)));
    cudaErrorCheck(cudaMalloc(&global_index_gpu.sizes_[idx], sizeof(uint32_t) * (DV_COUNT + 1)));

    // b. set sizes
    if (build_graph)
    {
        cudaErrorCheck(cudaMemset(data_graph_gpu.sizes_[idx], 0u, sizeof(uint32_t) * (DV_COUNT + 1)));
    }
    cudaErrorCheck(cudaMemset(global_index_gpu.sizes_[idx], 0u, sizeof(uint32_t) * (DV_COUNT + 1)));

    // c. set capacities
    if (build_graph)
    {
        cudaErrorCheck(cudaMemset(data_graph_gpu.capacity_[idx], 0u, sizeof(uint32_t) * (DV_COUNT + 1)));
    }
    // auto cur_capacity = global_index_gpu.capacity_[idx][0];  // 250216
    // cudaErrorCheck(cudaDeviceSynchronize());  // 250216
    cudaErrorCheck(cudaMemset(global_index_gpu.capacity_[idx], 0u, sizeof(uint32_t) * (DV_COUNT + 1)));

    // Set global_index_gpu.capacity_[idx] according to csr_gpu.offs_
    setCapacities<<<GRID_DIM, BLOCK_DIM>>>(csr_gpu, global_index_gpu.capacity_[idx]);
    cudaErrorCheck(cudaDeviceSynchronize());
    // Round each capacity value to the smallest 2's power that is larger than the current capacity value.
    // Set the capacity value of `s` to 0 if the label of the source vertex `s` (a data graph vertex) does not match 
    // the label of `u` (the source vertex of current query edge, `idx` is the edge_idx of current query edge).
    // Doubt: Explicitly setting the capacity value of label-unmatched source vertex seems redundant, 
    // as all edges in `csr_gpu` should be label-matched (which implies the label-matching of source vertices, edge labels and destination vertices).
    roundCapacities<<<GRID_DIM, BLOCK_DIM>>>(global_index_gpu.capacity_[idx], DV_COUNT, dvlabels, query_.vlabels_[u]);
    cudaErrorCheck(cudaDeviceSynchronize());

    if (build_graph)
    {
        cudaErrorCheck(cudaMemcpy(data_graph_gpu.capacity_[idx], global_index_gpu.capacity_[idx], sizeof(uint32_t) * (DV_COUNT + 1), cudaMemcpyDeviceToDevice));
    }

    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, global_index_gpu.capacity_[idx], capacity_prefix_sum, DV_COUNT + 1));
    
    // d. set nbrs
    uint32_t total_size;
    cudaErrorCheck(cudaMemcpy(&total_size, &capacity_prefix_sum[DV_COUNT], sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // cout << "edge_idx: " << idx << ", total_size: " << total_size << endl;
    if (build_graph)
    {
        cudaErrorCheck(cudaMalloc(&data_graph_all_nbrs, sizeof(uint32_t) * total_size));
        cudaErrorCheck(cudaMemset(data_graph_all_nbrs, 0u, sizeof(uint32_t) * total_size));
        cudaErrorCheck(cudaDeviceSynchronize());
        setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(data_graph_all_nbrs, capacity_prefix_sum, DV_COUNT, data_graph_gpu.nbrs_[idx]);
    }
    cudaErrorCheck(cudaMalloc(&index_all_nbrs, sizeof(uint32_t) * total_size));
    cudaErrorCheck(cudaMemset(index_all_nbrs, 0u, sizeof(uint32_t) * total_size));
    cudaErrorCheck(cudaDeviceSynchronize());
    setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(index_all_nbrs, capacity_prefix_sum, DV_COUNT, global_index_gpu.nbrs_[idx]);
}

void MatchGPU::SelectEdgesFromTrie(
    const CSR_GPU& input,
    CSR_GPU& output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    output.vs_size_ = input.vs_size_;
    cudaErrorCheck(cudaMemcpy(output.vs_, input.vs_, sizeof(uint32_t) * input.vs_size_, cudaMemcpyDeviceToDevice));

    filterRelevantCount<<<GRID_DIM, BLOCK_DIM>>>(
        input, output, first_candidates, second_candidates
    );
    cudaErrorCheck(cudaDeviceSynchronize());

    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, output.offs_, output.offs_, output.vs_size_ + 1));

    filterRelevantWrite<<<GRID_DIM, BLOCK_DIM>>>(
        input, output, first_candidates, second_candidates
    );
    cudaErrorCheck(cudaDeviceSynchronize());
}

void MatchGPU::Match(const TechniqueOption option, RelationsGPU& global_index, RelationsGPU& local_index,
                     const uint8_t edge_list_idx, unsigned long long int& num_matches, const bool materialize_results) {
    
    RelationsGPU *ptr_data_graph_index = nullptr;
    if (option.local_index_option == LocalIndexOption::kUseLocalIndex) {
        ptr_data_graph_index = &local_index;
    } else {
        ptr_data_graph_index = &global_index;
    }

    switch (option.search_strategy) {
        case SearchStrategy::kDfsWithWorkStealing:
            QOGammaMatching(*ptr_data_graph_index, /*trival_local_index=*/local_index, 
                                 edge_list_idx, num_matches, /*enable_work_stealing=*/true,
                                 option.use_cartesian_product, materialize_results);
            break;
        case SearchStrategy::kDfsWithoutWorkStealing:
            QOGammaMatching(*ptr_data_graph_index, /*trival_local_index=*/local_index, 
                                 edge_list_idx, num_matches, /*enable_work_stealing=*/false,
                                 option.use_cartesian_product, materialize_results);
            break;
        case SearchStrategy::kDynamicBfsDfs:
            Matching(*ptr_data_graph_index, edge_list_idx, num_matches, option.use_cartesian_product,
                     /*ptr_trival_local_index=*/&local_index, materialize_results);
            break;
        case SearchStrategy::kBfs:
            BfsMatching(*ptr_data_graph_index, edge_list_idx, num_matches, option.use_cartesian_product,
                        /*ptr_trival_local_index=*/&local_index, materialize_results);
            break;
    }
}