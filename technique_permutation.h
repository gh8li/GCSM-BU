#ifndef TECHNIQUE_PERMUTATION_H
#define TECHNIQUE_PERMUTATION_H

#include <cstring>
#include <string>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cuda_runtime.h>

#include "utils/automorphism.h"
#include "utils/constants.h"
#include "utils/cuda_helpers.h"
#include "utils/technique_option.h"
#include "graph/graph.h"
#include "graph/match_gpu.h"
#include "graph/plan.h"


namespace {
void print_gpu_memory() {
    size_t freeMem, totalMem;
    cudaErrorCheck(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Total GPU memory: " << totalMem / 1024 / 1024 << " MiB" << std::endl;
    std::cout << "Free GPU memory: " << freeMem / 1024 / 1024 << " MiB" << std::endl;
}
}  // namespace

void TechniquePermutationProcess(const TechniqueOption &option, const std::string &query_path, const std::string &data_path, 
                                 const std::string &update_path, uint32_t batch_size, const bool show_gpu_memory, 
                                 const bool print_indexing_time) {
    std::cout << "----------- Read Graphs from Files ------------\n";

    TIME_INIT();
    LTIME_INIT();

    QueryGraph query_graph;
    DataGraphManager data_graph_manager;
    PlanManager plan_manager(query_graph);

    TIME_START();

    GraphFileIoManager graph_file_io_manager(query_path, query_graph);
    graph_file_io_manager.SetQueryMeta();  // Set the metadata and set NLF.
    graph_file_io_manager.LoadInitial(data_path, data_graph_manager);
    graph_file_io_manager.LoadUpdate(update_path, data_graph_manager, batch_size);
    
    plan_manager.GenerateIndexingOrders();

    TIME_END();
    PRINT_LOCAL_TIME("Read Graphs from Files");

    std::cout << "----------- Preprocessing ------------\n";

    MatchGPU match_gpu(graph_file_io_manager, query_graph, plan_manager);
    RelationsGPU data_graph_gpu;
    RelationsGPU global_index_gpu;
    CandidatesGPU global_bitmap_gpu;
    CSR_GPU csr_gpu[MAX_QE_COUNT * 2];  // edge_idx -> CSR_GPU

    TIME_START();

    match_gpu.LoadQuery();
    for (uint8_t i = 0u; i < QE_COUNT; i++) {
        match_gpu.BuildTries(data_graph_manager.initial_edges_, csr_gpu, i);
    }

    TIME_END();
    PRINT_LOCAL_TIME("Load Initial Graph");

    TIME_START();

    match_gpu.AllocRelations(data_graph_manager, csr_gpu, data_graph_gpu, 
                             global_index_gpu, global_bitmap_gpu);
    match_gpu.SetGraphPtrs(data_graph_gpu);

    TIME_END();
    PRINT_LOCAL_TIME("Allocate Relations");

    TIME_START();
    
    for (uint8_t edge_list_idx = 0u; edge_list_idx < QE_COUNT; edge_list_idx++) {
        // cout << "UpdateGlobalIndex for edge_list_idx: " << (uint32_t)edge_list_idx << endl;  // 250216
        match_gpu.UpdateGlobalIndex(data_graph_gpu, global_index_gpu, global_bitmap_gpu, 
                                    csr_gpu, edge_list_idx);  // Add edges in csr_gpu into global_index_gpu.
    }

    TIME_END();
    PRINT_LOCAL_TIME("Build Global Index Offline");

    TIME_START();

    // std::cout << "before cardinalities" << endl;
    uint32_t cardinalities[QE_COUNT * 2];  // edge_idx -> edge_count in data graph (global_index_gpu)
    float avg_degrees[QE_COUNT * 2];  // edge_idx -> edge_count / src_vertex_count in data graph (global_index_gpu)
    
    // std::cout << "before GetSummary" << endl;
    match_gpu.GetSummary(global_index_gpu, cardinalities, avg_degrees);
    
    // std::cout << "before GenerateGammaMatchingOrder" << endl;
    plan_manager.GenerateMatchingOrders(option, cardinalities, avg_degrees);
    
    // std::cout << "before CorrectGammaLoadPlan" << endl;
    match_gpu.CorrectGammaLoadPlan();
    
    TIME_END();
    PRINT_LOCAL_TIME("Build Matching Order");

    plan_manager.PrintOrders();

    std::cout << "--------- Incremental Matching --------\n";
    
    RelationsGPU local_index_base_gpu;
    RelationsGPU local_index;
    CandidatesGPU local_bitmap_gpu;
    unsigned long long int num_positive_matches = 0ull;
    
    match_gpu.AllocOnline(local_index_base_gpu, local_bitmap_gpu);

    // std::cout << "Batch size = " << batch_size << ", " 
    //           << data_graph_manager.updated_edges_.size() << " batches\n";

    TIME_START();

    auto batch_num = 0u;
    
    for (const auto &batch : data_graph_manager.updated_edges_) {  // Larger batch size is better for us.
        
        std::cout << std::endl;
        std::cout << "Batch #" << batch_num << " --------" << std::endl;
        match_gpu.SetGraphPtrs(data_graph_gpu);
        
        for (uint8_t edge_list_idx = 0u; edge_list_idx < QE_COUNT; edge_list_idx++) {            
            // std::cout << "Query Edge #" + std::to_string(edge_list_idx) << '\n';
            
            cudaEvent_t index_cuda_start;
            float index_kernel_time;
            std::chrono::system_clock::time_point index_start_clock;
            if (print_indexing_time) {
                index_start_clock = std::chrono::high_resolution_clock::now();
                cudaEventCreate(&index_cuda_start);
                cudaEventRecord(index_cuda_start);
            }

            match_gpu.BuildTries(batch, csr_gpu, edge_list_idx);
            // std::cout << "before UpdateGlobalIndex" << '\n';
            match_gpu.UpdateGlobalIndex(data_graph_gpu, global_index_gpu, global_bitmap_gpu, 
                                        csr_gpu, edge_list_idx);



            cudaEvent_t local_cuda_start, local_cuda_end;
            float local_kernel_time;                                
            std::chrono::system_clock::time_point local_start_clock;
            if (print_indexing_time) {
                local_start_clock = std::chrono::high_resolution_clock::now();
                cudaEventCreate(&local_cuda_start);
                cudaEventCreate(&local_cuda_end);
                cudaEventRecord(local_cuda_start);
            }

            bool local_index_nonempty = false;
            if (option.local_index_option == LocalIndexOption::kUseLocalIndex) {
                // std::cout << "Before BuildLocalIndex_v2" << '\n';
                local_index_nonempty = match_gpu.BuildLocalIndex_v2(global_index_gpu, global_bitmap_gpu, local_index_base_gpu,
                                                                    local_bitmap_gpu, local_index, csr_gpu, edge_list_idx, avg_degrees);
            } else {  // LocalIndexOption::kDontUseLocalIndex, generate trival local index.
                // std::cout << "Before GammaBuildLocalIndex" << '\n';
                local_index_nonempty = match_gpu.GammaBuildLocalIndex(global_index_gpu, global_bitmap_gpu, local_index_base_gpu, 
                                               local_bitmap_gpu, local_index, csr_gpu, edge_list_idx, avg_degrees);
            }



            std::chrono::system_clock::time_point local_end_clock;
            if (print_indexing_time) {
                local_end_clock = std::chrono::high_resolution_clock::now();
                cudaEventRecord(local_cuda_end);
                cudaEventSynchronize(index_cuda_start);
                cudaEventSynchronize(local_cuda_start);
                cudaEventSynchronize(local_cuda_end);
                cudaEventElapsedTime(&local_kernel_time, local_cuda_start, local_cuda_end);
                std::chrono::duration<double> local_diff = local_end_clock - local_start_clock;
                
                cudaEventElapsedTime(&index_kernel_time, index_cuda_start, local_cuda_end);
                std::chrono::duration<double> index_diff = local_end_clock - index_start_clock;
        
                std::cout << "Finish Index Maintenance" << ", time (ms): " <<             \
                        static_cast<unsigned long>(index_diff.count() * 1000) <<          \
                        "(host), " << static_cast<unsigned long>(index_kernel_time) <<    \
                        "(kernel)" <<", Local Index time (ms): " <<                       \
                        static_cast<unsigned long>(local_diff.count() * 1000) <<          \
                        "(host), " << static_cast<unsigned long>(local_kernel_time) <<    \
                        "(kernel)\n";;
            }



            if (show_gpu_memory) {
                print_gpu_memory();
            }

            if (local_index_nonempty) {
                // std::cout << "Before CorrectGammaMatching" << '\n';
                match_gpu.Match(option, global_index_gpu, local_index, edge_list_idx, num_positive_matches);
            } else {
                std::cout << "Local index is empty, skip matching." << endl;
            }
            
            std::cout << endl;
        }

        batch_num++;
        match_gpu.DeallocTries(csr_gpu);
    }
    TIME_END();
    PRINT_LOCAL_TIME("Incremental Matching");

    match_gpu.DeallocOnline(local_index_base_gpu, local_bitmap_gpu);
    match_gpu.DeallocRelations(data_graph_gpu, global_index_gpu, global_bitmap_gpu);

    // std::cout << "---------------------------------------\n";
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "Num Positive Matches: " << num_positive_matches << '\n';
}

#endif  // TECHNIQUE_PERMUTATION_H