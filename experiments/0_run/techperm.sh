#!/usr/bin/env bash

#
# Created by gli945.
#

# set -eu
set -u
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../..

input_device=${1:-"6"}
input_num_queries=${2:-"100"}
input_time_limit=${3:-"10"}
input_suffix=${4:-"run_0126_v1_timeLimit10"}
input_dataset=${5:-"lsbench/x1"}


# 0 for IndexingOrder, 1 for GammaOrder, 2 for RapidCsmOrder, 3 for RiOrder
# matching_order_list=("0" "1" "2" "3")
matching_order_list=("0" "1" "2")  # w/o RiOrder
# matching_order_list=("2")  # w/o RiOrder

# 0 for DfsWithWorkStealing, 1 for DfsWithoutWorkStealing, 2 for DynamicBfsDfs, 3 for Bfs
search_strategy_list=("0" "1" "2" "3")
# search_strategy_list=("2")
# search_strategy_list=("0" "3")  # w/o DfsWithoutWorkStealing and DynamicBfsDsf

# use_local_index_list=("True" "False")
use_local_index_list=("True")
# use_cartesian_product_list=("True" "False")
use_cartesian_product_list=("True")
# use_heavy_vertex_avoidance_list=("True" "False")
use_heavy_vertex_avoidance_list=("True")
# use_relation_switch_list=("True" "False")
use_relation_switch_list=("True")

mkdir -p log
log_dir="./log"
log_path=${log_dir}/log_${input_dataset//\//}_$(date +%m%d).txt

for MATCHING_ORDER in ${matching_order_list[@]}; do
  for SEARCH_STRATEGY in ${search_strategy_list[@]}; do
    for USE_LOCAL_INDEX in ${use_local_index_list[@]}; do
      for USE_CARTESIAN_PRODUCT in ${use_cartesian_product_list[@]}; do
        for LOOP_USE_HEAVY_VERTEX_AVOIDANCE in ${use_heavy_vertex_avoidance_list[@]}; do
          for LOOP_USE_RELATION_SWITCH in ${use_relation_switch_list[@]}; do
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

            # USE_LOCAL_INDEX="True"
            # if [[ ${MATCHING_ORDER} == "0" ]]; then
            #   USE_LOCAL_INDEX="False"
            # elif [[ ${MATCHING_ORDER} == "1" ]]; then
            #   USE_LOCAL_INDEX="False"
            # elif [[ ${MATCHING_ORDER} == "2" ]]; then
            #   USE_LOCAL_INDEX="True"
            # elif [[ ${MATCHING_ORDER} == "3" ]]; then
            #   USE_LOCAL_INDEX="True"
            # fi

            if [[ ${USE_LOCAL_INDEX} == "False" ]]; then
              USE_HEAVY_VERTEX_AVOIDANCE="False"
              USE_RELATION_SWITCH="False"
            else
              USE_HEAVY_VERTEX_AVOIDANCE=${LOOP_USE_HEAVY_VERTEX_AVOIDANCE}
              USE_RELATION_SWITCH=${LOOP_USE_RELATION_SWITCH}
            fi

            # if [[ ${USE_LOCAL_INDEX} == "False" ]]; then
            #   USE_RELATION_SWITCH="False"
            # else
            #   USE_RELATION_SWITCH=${LOOP_USE_RELATION_SWITCH}
            # fi

            log_entry="$(date) python -u run_csm.py --matching_order ${MATCHING_ORDER} \
                                                       --search_strategy ${SEARCH_STRATEGY} \
                                                       --use_local_index ${USE_LOCAL_INDEX} \
                                                       --use_cartesian_product ${USE_CARTESIAN_PRODUCT} \
                                                       --use_heavy_vertex_avoidance ${USE_HEAVY_VERTEX_AVOIDANCE} \
                                                       --use_relation_switch ${USE_RELATION_SWITCH} \
                                                       --device ${input_device} \
                                                       --num_queries ${input_num_queries} \
                                                       --time_limit ${input_time_limit} \
                                                       --suffix ${input_suffix} \
                                                       --dataset_list ${input_dataset} >> ${log_path} 2>&1"
            echo "${log_entry//                                                       /}" >> ${log_path}
            echo "--------------------------------------------------------------------" >> ${log_path}
            python -u ./experiments/0_run/run_csm.py --matching_order ${MATCHING_ORDER} \
                                                        --search_strategy ${SEARCH_STRATEGY} \
                                                        --use_local_index ${USE_LOCAL_INDEX} \
                                                        --use_cartesian_product ${USE_CARTESIAN_PRODUCT} \
                                                        --use_heavy_vertex_avoidance ${USE_HEAVY_VERTEX_AVOIDANCE} \
                                                        --use_relation_switch ${USE_RELATION_SWITCH} \
                                                        --device ${input_device} \
                                                        --num_queries ${input_num_queries} \
                                                        --time_limit ${input_time_limit} \
                                                        --suffix ${input_suffix} \
                                                        --dataset_list ${input_dataset} >> ${log_path} 2>&1
            echo "--------------------------------------------------------------------" >> ${log_path}
            echo "end." >> ${log_path}
            echo >> ${log_path}
          done
        done
      done
    done
  done
done