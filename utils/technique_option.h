#ifndef UTILS_TECHNIQUE_OPTION_H
#define UTILS_TECHNIQUE_OPTION_H

#include "utils/simple_command_parser.h"

// RapidCsm is the alias for GCSM-BU

enum class Algorithm {
    kTechniquePermutation = 0,
    kGcsm = 1,  // GCSM-BU in the paper
    kQOGamma = 2,  // QO-GAMMA in the paper
    kGamma = 3,  // Not used anymore. We have a separate project for Gamma implementation.
    kGcsmForDense = 4,
    kGcsmForDenseGammaOrder = 5  // New option for GCSM-BU with dense graphs in gamma order
};

enum class MatchingOrder {
    kIndexingOrder = 0,
    kGammaOrder = 1,
    kGcsmOrder = 2,
    kRiOrder = 3,
};

enum class SearchStrategy {
    kDfsWithWorkStealing = 0,
    kDfsWithoutWorkStealing = 1,
    kDynamicBfsDfs = 2,
    kBfs = 3,
};

enum class LocalIndexOption {
    kUseLocalIndex = 0,
    kDontUseLocalIndex = 1,
};

class TechniqueOption {
  public:
    const MatchingOrder matching_order;
    const SearchStrategy search_strategy;
    const LocalIndexOption local_index_option;
    const bool use_cartesian_product;
    const bool use_heavy_vertex_avoidance;
    const bool use_relation_switch;
    
    TechniqueOption(MatchingOrder order, SearchStrategy strategy, 
                    LocalIndexOption local_index_option,
                    bool cartesian_product, bool heavy_vertex_avoidance, 
                    bool relation_switch):
                        matching_order(order),
                        search_strategy(strategy),
                        local_index_option(local_index_option),
                        use_cartesian_product(cartesian_product),
                        use_heavy_vertex_avoidance(heavy_vertex_avoidance),
                        use_relation_switch(relation_switch) {}
};

namespace TechniqueOptionBuilder {
static TechniqueOption build(const InputParser &parser, const Algorithm algorithm=Algorithm::kTechniquePermutation) {
    switch (algorithm) {
        case Algorithm::kGcsm:
            return TechniqueOption(MatchingOrder::kGcsmOrder, SearchStrategy::kDynamicBfsDfs,LocalIndexOption::kUseLocalIndex, 
                                   /*cartesian_product=*/true, /*heavy_vertex_avoidance=*/true, /*relation_switch=*/true);
        case Algorithm::kQOGamma:
            return TechniqueOption(MatchingOrder::kGammaOrder, SearchStrategy::kDfsWithWorkStealing,LocalIndexOption::kDontUseLocalIndex, 
                                   /*cartesian_product=*/true, /*heavy_vertex_avoidance=*/false, /*relation_switch=*/false);
        case Algorithm::kGcsmForDense:
            return TechniqueOption(MatchingOrder::kIndexingOrder, SearchStrategy::kDynamicBfsDfs, LocalIndexOption::kDontUseLocalIndex, 
                                   /*cartesian_product=*/true, /*heavy_vertex_avoidance=*/true, /*relation_switch=*/true);
        case Algorithm::kGcsmForDenseGammaOrder:
            return TechniqueOption(MatchingOrder::kGammaOrder, SearchStrategy::kDynamicBfsDfs, LocalIndexOption::kDontUseLocalIndex, 
                                   /*cartesian_product=*/true, /*heavy_vertex_avoidance=*/true, /*relation_switch=*/true);
        default:
            int32_t order_id = parser.get_int32_cmd_option("--matching_order", 
                                                           /*default_value=*/static_cast<int32_t>(MatchingOrder::kGcsmOrder));
            int32_t search_strategy_id = parser.get_int32_cmd_option("--search_strategy", 
                                                                     /*default_value=*/static_cast<int32_t>(SearchStrategy::kDynamicBfsDfs));
            bool use_local_index = parser.get_bool_cmd_option("--local_index", /*default_value=*/true);
            LocalIndexOption local_index_option = use_local_index ? LocalIndexOption::kUseLocalIndex : LocalIndexOption::kDontUseLocalIndex;
            bool use_cartesian_product = parser.get_bool_cmd_option("--cartesian_product", /*default_value=*/true);
            bool use_heavy_vertex_avoidance = parser.get_bool_cmd_option("--heavy_vertex_avoidance", /*default_value=*/true);
            bool use_relation_switch = parser.get_bool_cmd_option("--relation_switch", /*default_value=*/true);
            return TechniqueOption(static_cast<MatchingOrder>(order_id), static_cast<SearchStrategy>(search_strategy_id),
                                   static_cast<LocalIndexOption>(local_index_option), use_cartesian_product,
                                   use_heavy_vertex_avoidance, use_relation_switch);
    }
}
}  // namespace TechniqueOptionBuilder

#endif  // UTILS_TECHNIQUE_OPTION_H