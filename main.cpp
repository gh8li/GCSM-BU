#include <cstring>
#include <string>
#include <cuda_runtime.h>


#include "technique_permutation.h"
#include "utils/simple_command_parser.h"

namespace {
void PrintMacros(){
    std::cout << "Macros (Items after the third one are for GPU configuration.):" << '\n';
    std::cout << "MAX_QV_COUNT: " << MAX_QV_COUNT << '\n';
    std::cout << "MAX_QE_COUNT: " << MAX_QE_COUNT << '\n';
    std::cout << "MIN_NBR_SIZE: " << MIN_NBR_SIZE << '\n';
    std::cout << "GRID_DIM: " << GRID_DIM << '\n';
    std::cout << "BLOCK_DIM: " << BLOCK_DIM << '\n';
    std::cout << "WARP_SIZE: " << WARP_SIZE << '\n';
    std::cout << "NUM_WARP_PER_BLOCK: " << NUM_WARP_PER_BLOCK << '\n';
    std::cout << "NBR_SPACE: " << NBR_SPACE * sizeof(uint32_t) / 1024 / 1024 << " MiB of uint32_t" << '\n';
    std::cout << "RES_SPACE: " << RES_SPACE * sizeof(uint32_t) / 1024 / 1024 << " MiB of uint32_t" << '\n';
    std::cout << "SIZE_SPACE: " << SIZE_SPACE * sizeof(long) / 1024 / 1024 << " MiB of long" << '\n';
    std::cout << "MIN_NUM_RESULTS_TO_GPU: " << MIN_NUM_RESULTS_TO_GPU << '\n';
}
void PrintInfo(std::string query_path, std::string data_path, std::string update_path,
               const int32_t device_id, const uint32_t batch_size, const bool show_gpu_memory,
               const TechniqueOption &option) {
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    std::cout << "Command Line:" << '\n';
    std::cout << "\tQuery Graph: " << query_path << '\n';
    std::cout << "\tData Graph: " << data_path << '\n';
    std::cout << "\tData Graph Update: " << update_path << '\n';
    std::cout << "\tDevice ID: " << device_id << '\n';
    std::cout << "\tBatch Size: " << batch_size << '\n';
    std::cout << "\tShow GPU Memory?: " << (show_gpu_memory?"yes":"no") << '\n';

    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    std::string str_matching_order;
    std::string str_search_strategy;
    std::string str_local_index_option;
    std::string str_use_cartesian_product;
    std::string str_use_heavy_vertex_avoidance;
    std::string str_use_relation_switch;

    switch (option.matching_order) {
        case MatchingOrder::kGammaOrder:
            str_matching_order = "Gamma Order";
            break;
        case MatchingOrder::kIndexingOrder:
            str_matching_order = "BFS Order";
            break;
        case MatchingOrder::kGcsmOrder:
            str_matching_order = "RapidCSM Order";
            break;
        case MatchingOrder::kRiOrder:
            str_matching_order = "RI Order";
            break;
    }

    switch (option.search_strategy) {
        case SearchStrategy::kBfs:
            str_search_strategy = "BFS";
            break;
        case SearchStrategy::kDfsWithoutWorkStealing:
            str_search_strategy = "DFS w/o Work Stealing";
            break;
        case SearchStrategy::kDfsWithWorkStealing:
            str_search_strategy = "DFS w/ Work Stealing";
            break;
        case SearchStrategy::kDynamicBfsDfs:
            str_search_strategy = "Dynamic BFS DFS";
    }

    switch (option.local_index_option) {
        case LocalIndexOption::kDontUseLocalIndex:
            str_local_index_option = "w/o Local Index";
            break;
        case LocalIndexOption::kUseLocalIndex:
            str_local_index_option = "w/ Local Index";
    }

    if (option.use_cartesian_product) {
        str_use_cartesian_product = "w/ Cartesian Product";
    } else {
        str_use_cartesian_product = "w/o Use Cartesian Product";
    }

    if (option.use_heavy_vertex_avoidance) {
        str_use_heavy_vertex_avoidance = "w/ Heavy Vertex Avoidance";
    } else {
        str_use_heavy_vertex_avoidance = "w/o Use Heavy Vertex Avoidance";
    }

    if (option.use_relation_switch) {
        str_use_relation_switch = "w/ Relation Switch";
    } else {
        str_use_relation_switch = "w/o Use Relation Switch";
    }

    std::cout << "Technique Option:" << '\n';
    std::cout << "\tMatching Order: " << str_matching_order << '\n';
    std::cout << "\tSearch Strategy: " << str_search_strategy << '\n';
    std::cout << "\tUse Local Index?: " << str_local_index_option << '\n';
    std::cout << "\tUse Cartesian Product?: " << str_use_cartesian_product << '\n';
    std::cout << "\tUse Heavy Vertex Avoidance?: " << str_use_heavy_vertex_avoidance << '\n';
    std::cout << "\tUse Relation Switch?: " << str_use_relation_switch << '\n';

    std::cout << "--------------------------------------------------------------------" << std::endl;
    PrintMacros();
    std::cout << "--------------------------------------------------------------------" << std::endl;
}
void PrintGammaInfo(std::string query_path, std::string data_path, std::string update_path,
                    const int32_t device_id, const uint32_t batch_size, const bool print_results) {
    std::cout << "--------------------------------------------------------------------" << std::endl;

    std::cout << "Command Line:" << '\n';
    std::cout << "\tGamma: yes" << '\n';
    std::cout << "\tQuery Graph: " << query_path << '\n';
    std::cout << "\tData Graph: " << data_path << '\n';
    std::cout << "\tData Graph Update: " << update_path << '\n';
    std::cout << "\tDevice ID: " << device_id << '\n';
    std::cout << "\tBatch Size: " << batch_size << '\n';
    std::cout << "\tPrint Results?: " << (print_results?"yes":"no") << '\n';

    std::cout << "--------------------------------------------------------------------" << std::endl;
    PrintMacros();
    std::cout << "--------------------------------------------------------------------" << std::endl;
}
void PrintCpuVersionInfo(std::string query_path, std::string data_path, std::string update_path,
                         const uint32_t batch_size, const int32_t num_threads, const Algorithm algorithm) {
    string str_algorithm;
    switch (algorithm) {
        case Algorithm::kQOGamma:
            str_algorithm = "CorrectGamma";
            break;
        case Algorithm::kGamma:
            str_algorithm = "Gamma";
            break;
        default:
            str_algorithm = "RapidCSM";
            break;
    }

    std::cout << "--------------------------------------------------------------------" << std::endl;

    std::cout << "Command Line:" << '\n';
    std::cout << "\tAlgorithm: " << str_algorithm << '\n';
    std::cout << "\tOn CPU?: yes" << '\n';
    std::cout << "\tQuery Graph: " << query_path << '\n';
    std::cout << "\tData Graph: " << data_path << '\n';
    std::cout << "\tData Graph Update: " << update_path << '\n';
    std::cout << "\tBatch Size: " << batch_size << '\n';
    std::cout << "\t#Threads: " << num_threads << '\n';

    std::cout << "--------------------------------------------------------------------" << std::endl;
    PrintMacros();
    std::cout << "--------------------------------------------------------------------" << std::endl;
}
}  // namespace

int main(int argc, char *argv[]) {
    InputParser cmd_parser(argc, argv);

    std::cout << cmd_parser.get_cmd() << endl;

    const std::string input_query_path = cmd_parser.get_cmd_option("--query");
    const std::string input_data_path = cmd_parser.get_cmd_option("--data");
    const std::string input_update_path = cmd_parser.get_cmd_option("--update");

    const int32_t input_device_id = cmd_parser.get_int32_cmd_option("--device", /*default_value=*/0);
    const uint32_t input_batch_size = cmd_parser.get_uint32_cmd_option("--batch_size", /*default_value=*/UINT32_MAX);

    int32_t algorithm_id = cmd_parser.get_int32_cmd_option("--algorithm", 
                                                           /*default_value=*/static_cast<int32_t>(Algorithm::kTechniquePermutation));
    
    const Algorithm input_algorithm = static_cast<Algorithm>(algorithm_id);

    cudaSetDevice(input_device_id);


    const bool show_gpu_memory = cmd_parser.get_bool_cmd_option("--show_gpu_memory", /*default_value=*/false);
    const bool print_indexing_time = cmd_parser.get_bool_cmd_option("--print_indexing_time", /*default_value=*/false);
    TechniqueOption option = TechniqueOptionBuilder::build(cmd_parser, input_algorithm);

    PrintInfo(input_query_path, input_data_path, input_update_path, input_device_id, input_batch_size, show_gpu_memory, option);

    TechniquePermutationProcess(option, input_query_path, input_data_path, input_update_path, input_batch_size, 
                                show_gpu_memory, print_indexing_time);
}