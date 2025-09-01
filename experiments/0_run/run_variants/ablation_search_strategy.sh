#!/usr/bin/env bash

#
# Created by gli945 on 2025/1/18.
#

# set -eu
set -u
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../..

input_device=${1:-"0"}

# 0 for IndexingOrder, 1 for GammaOrder, 2 for RapidCsmOrder, 3 for RiOrder
matching_order_list=("0" "1" "2" "3")

# 0 for DfsWithWorkStealing, 1 for DfsWithoutWorkStealing, 2 for DynamicBfsDfs, 3 for Bfs
search_strategy_list=("0" "1" "2" "3")

use_local_index_list=("True")
use_cartesian_product_list=("True")
use_heavy_vertex_avoidance_list=("True")
use_relation_switch_list=("True")

mkdir -p log
log_dir="./log"
log_path=${log_dir}/log_$(date +%m%d).txt

for MATCHING_ORDER in ${matching_order_list[@]}; do
  for SEARCH_STRATEGY in ${search_strategy_list[@]}; do
    for USE_LOCAL_INDEX in ${use_local_index_list[@]}; do
      for USE_CARTESIAN_PRODUCT in ${use_cartesian_product_list[@]}; do
        for USE_HEAVY_VERTEX_AVOIDANCE in ${use_heavy_vertex_avoidance_list[@]}; do
          for USE_RELATION_SWITCH in ${use_relation_switch_list[@]}; do
            if [[ ${MATCHING_ORDER} == "2" && ${USE_LOCAL_INDEX} == "False" ]]; then  # Use RapidCsmOrder and do not use local index.
              echo "Skip: Matching order is 2 (RapidCsmOrder) and use_local_index is False." >> ${log_path}
              continue
            fi
            if [[ ${MATCHING_ORDER} == "3" && ${USE_LOCAL_INDEX} == "False" ]]; then  # Use RiOrder and do not use local index.
              echo "Skip: Matching order is 3 (RiOrder) and use_local_index is False." >> ${log_path}
              continue
            fi
            # if [[ ${USE_LOCAL_INDEX} == "False" && ${USE_HEAVY_VERTEX_AVOIDANCE} == "True" ]]; then
            #   echo "Skip: use_local_index is False and use_heavy_vertex_avoidance is True." >> ${log_path}
            #   continue
            # fi
            # if [[ ${USE_LOCAL_INDEX} == "False" && ${USE_RELATION_SWITCH} == "True" ]]; then
            #   echo "Skip: use_local_index is False and use_relation_switch is True." >> ${log_path}
            #   continue
            # fi

            if [[ ${USE_LOCAL_INDEX} == "False" ]]; then
              USE_HEAVY_VERTEX_AVOIDANCE="False"
            fi
            if [[ ${USE_LOCAL_INDEX} == "False" ]]; then
              USE_RELATION_SWITCH="False"
            fi

            log_entry="$(date) python -u run_csm.py --matching_order ${MATCHING_ORDER} \
                                                    --search_strategy ${SEARCH_STRATEGY} \
                                                    --use_local_index ${USE_LOCAL_INDEX} \
                                                    --use_cartesian_product ${USE_CARTESIAN_PRODUCT} \
                                                    --use_heavy_vertex_avoidance ${USE_HEAVY_VERTEX_AVOIDANCE} \
                                                    --use_relation_switch ${USE_RELATION_SWITCH} \
                                                    --device ${input_device} >> ${log_path} 2>&1"
            echo "${log_entry//                                                    /}" >> ${log_path}
            echo "--------------------------------------------------------------------" >> ${log_path}
            python -u ./experiments/0_run/run_csm.py --matching_order ${MATCHING_ORDER} \
                                 --search_strategy ${SEARCH_STRATEGY} \
                                 --use_local_index ${USE_LOCAL_INDEX} \
                                 --use_cartesian_product ${USE_CARTESIAN_PRODUCT} \
                                 --use_heavy_vertex_avoidance ${USE_HEAVY_VERTEX_AVOIDANCE} \
                                 --use_relation_switch ${USE_RELATION_SWITCH} \
                                 --device ${input_device} >> ${log_path} 2>&1
            echo "--------------------------------------------------------------------" >> ${log_path}
            echo "end." >> ${log_path}
            echo >> ${log_path}
          done
        done
      done
    done
  done
done