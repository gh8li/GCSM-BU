#! /usr/bin/env python3

#
# Created by gli945.
# Modified from the Python script written by xsunax.
# Run our GPU-based CSM algorithms and some baseline methods on all combinations of queries and datasets,
# according to the given parameters.
#

from datetime import datetime
from pathlib import Path
from subprocess import Popen, PIPE, TimeoutExpired
import argparse
import os
import shutil
import time

def get_args():
    parser = argparse.ArgumentParser(
        description="Run our GPU-based CSM algorithm on all combinations of queries and datasets, according to the given parameters.")
    parser.add_argument("--device", help="which GPU to use", type=int, default=0)

    # Algorithm configuration
    parser.add_argument("--algorithm", help="the algorithm to run, such as [\"GCSM\", \"GAMMA\", \"SymBi\", \"RapidFlow\", "
                                            "\"CaliG\", \"NewSP\", \"GCSM_CPU\", \"CorrectGamma_CPU\", \"GAMMA_CPU\", ]", default="GCSM")
    parser.add_argument("--batch_size", help="batch size", type=int, default=4294967295)
    parser.add_argument("--matching_order", help="matching order index, "
                                                 "0 for Indexing Order (BFS Order),"
                                                 "1 for Gamma Order, "
                                                 "2 for GCSM-BU Order, "
                                                 "3 for RI Order", type=int, default=2)
    parser.add_argument("--search_strategy", help="search strategy index, "
                                                  "0 for DFS with Work Stealing, "
                                                  "1 for DFS without Work Stealing, "
                                                  "2 for Dynamic BFS-DFS, "
                                                  "3 for BFS", type=int, default=2)
    parser.add_argument("--use_local_index", help="use local index or not", choices=["True", "False"], default="True")
    parser.add_argument("--use_cartesian_product", help="use cartesian product or not", choices=["True", "False"], default="True")
    parser.add_argument("--use_heavy_vertex_avoidance", help="use heavy vertex avoidance or not", choices=["True", "False"], default="True")
    parser.add_argument("--use_relation_switch", help="use relation switch or not", choices=["True", "False"], default="True")
    
    # Workload configuration
    parser.add_argument("--dataset_list", help="the list of datasets to run on, "
                                          "default: [\"amazon/6\", \"netflow\", \"livejournal/30\", \"lsbench/x1\", \"ldbc-3\"]", 
                                          nargs="*")
    parser.add_argument("--query_shape_list", help="the list of query graph shapes, default: [\"tree_6\", \"sparse_6\", \"dense_6\"]", 
                                              nargs="*")
    parser.add_argument("--query_id_list", help="the list of query ids to run", nargs="*")
    parser.add_argument("--query_id_range", help="the range of query ids to run, e.g., 0 100, invalid if `--query_id_list` is not empty",
                                            nargs=2, type=int)
    parser.add_argument("--num_queries", help="number of queries to run, invalid if `--query_id_list` or `--query_id_range` is not empty",
                                         type=int, default=100)
    parser.add_argument("--skip_query_id_list", help="the list of query ids to skip", nargs="*")
    parser.add_argument("--rerun", help="rerun the queries, if the some query has already been run, backup the query result file", 
                                   action="store_true")
    parser.add_argument("--skip_previous_error", help="skip the queries that have been run and have error", action="store_true")
    
    # Others
    parser.add_argument("--time_limit", help="the limit of execution time (seconds)", type=int, default=3600)
    parser.add_argument("--suffix", help="the beginning part of the suffix of the result file names (followed by technique options)", 
                                    type=str, default=f'run_{time.strftime("%y%m%d", time.localtime())}') 
    parser.add_argument("--show_gpu_memory", help="show GPU memory usage or not", choices=["True", "False"], default="False")
    parser.add_argument("--print_indexing_time", help="print the indexing time or not", choices=["True", "False"], default="False")
    parser.add_argument("--num_threads", help="number of threads to use, only valid when `--algorithm` is \"GCSM_CPU\", "
                                              "\"CorrectGamma_CPU\" or \"GAMMA_CPU\" ", type=int, default=16)
    return parser.parse_args()

def interpret_matching_order(matching_order):
    if matching_order == 0:
        return "IndexingOrder"
    elif matching_order == 1:
        return "GammaOrder"
    elif matching_order == 2:
        return "GcsmOrder"
    elif matching_order == 3:
        return "RiOrder"
    else:
        return "UnknownMatchingOrder"
    
def interpret_search_strategy(search_strategy):
    if search_strategy == 0:
        return "DfsWithWorkStealing"
    elif search_strategy == 1:
        return "DfsWithoutWorkStealing"
    elif search_strategy == 2:
        return "DynamicBfsDfs"
    elif search_strategy == 3:
        return "Bfs"
    else:
        return "UnknownSearchStrategy"

def encode_algorithm(algorithm):
    if algorithm.lower() == "gcsm" or algorithm.lower() == "gcsm_cpu":
        return 1
    elif algorithm.lower() == "correctgamma" or algorithm.lower() == "correctgamma_cpu":
        return 2
    elif algorithm.lower() == "gamma" or algorithm.lower() == "gamma_cpu":
        return 3
    elif algorithm.lower() == "gcsmfordense":
        return 4
    elif algorithm.lower() == "gcsmfordensegammaorder":
        return 5
    else:
        return 0

def create_process(args, query_path, data_path, update_path, query_shape):
    if "dense" in query_shape:
        gpu_csmg_path="./build/csmg_nbrSize1-5gb_resSpace2gb_sizeSpace1gb"
    else:
        gpu_csmg_path="./build/csmg_nbrSize1-5gb_resSpace3gb"  # 2GiB of size space
    if args.algorithm.lower() == "gcsm":
        return Popen([gpu_csmg_path,
                      "--query", query_path,
                      "--data", data_path,
                      "--update", update_path,
                      "--device", f"{args.device}", 
                      "--batch_size", f"{args.batch_size}",
                      "--matching_order", f"{args.matching_order}",
                      "--search_strategy", f"{args.search_strategy}",
                      "--local_index", f"{args.use_local_index}",
                      "--cartesian_product", f"{args.use_cartesian_product}",
                      "--heavy_vertex_avoidance", f"{args.use_heavy_vertex_avoidance}",
                      "--relation_switch", f"{args.use_relation_switch}",
                      "--show_gpu_memory", f"{args.show_gpu_memory}"], 
                     stdout=PIPE, stderr=PIPE)
    elif (args.algorithm.lower() == "gcsm" or
          args.algorithm.lower() == "correctgamma" or
          args.algorithm.lower() == "gcsmfordense" or
          args.algorithm.lower() == "gcsmfordensegammaorder"):
        return Popen([gpu_csmg_path,
                      "--algorithm", f"{encode_algorithm(args.algorithm)}",
                      "--query", query_path,
                      "--data", data_path,
                      "--update", update_path,
                      "--device", f"{args.device}", 
                      "--batch_size", f"{args.batch_size}",
                      "--show_gpu_memory", f"{args.show_gpu_memory}",
                      "--print_indexing_time", f"{args.print_indexing_time}"], 
                     stdout=PIPE, stderr=PIPE)
    elif args.algorithm.lower() == "gamma":
        return Popen(["./other_methods/gamma",
                      "--query", query_path,
                      "--data", data_path,
                      "--update", update_path,
                      "--device", f"{args.device}", 
                      "--batch_size", f"{args.batch_size}",
                      "--print_indexing_time", f"{args.print_indexing_time}"], 
                     stdout=PIPE, stderr=PIPE)
    elif (args.algorithm.lower() == "gcsm_cpu" or
          args.algorithm.lower() == "correctgamma_cpu" or
          args.algorithm.lower() == "gamma_cpu"):
        # return Popen(["./other_methods/gcsm_cpu",
        return Popen(["./build/csmg",
                      "--cpu", "True",
                      "--algorithm", f"{encode_algorithm(args.algorithm)}",
                      "--query", query_path,
                      "--data", data_path,
                      "--update", update_path,
                      "--num_threads", f"{args.num_threads}",
                      "--batch_size", f"{args.batch_size}"], 
                     stdout=PIPE, stderr=PIPE)
    elif args.algorithm.lower() == "symbi":  # csm -q <query-graph-path> -d <data-graph-path> -u <update-stream-path> -a <algorithm>
        return Popen(["./other_methods/csm",
                      "-a", "symbi",
                      "-q", query_path,
                      "-d", data_path,
                      "-u", update_path,
                      "--time-limit", "4294967295",
                      "--report-initial", "off",
                      "--print-prep", "off"], 
                     stdout=PIPE, stderr=PIPE)
    elif args.algorithm.lower() == "rapidflow":
        return Popen(["./other_methods//RapidFlow.out",
                      "-q", query_path,
                      "-d", data_path,
                      "-u", update_path,
                      "-num", "429496729",
                      "-time-limit", "4294967295"], 
                     stdout=PIPE, stderr=PIPE)
    elif args.algorithm.lower() == "calig":
        return Popen(["./other_methods/calig_modified",
                      "-q", query_path,
                      "-d", data_path,
                      "-s", update_path], 
                     stdout=PIPE, stderr=PIPE)
                    # "-c", "2000000000",  # 2^31 = 2147483647
    elif args.algorithm.lower() == "newsp":
        return Popen(["./other_methods/newsp",
                      "-q", query_path,
                      "-d", data_path,
                      "-u", update_path,
                      "--report-initial", "off",
                      "--time-limit", "4294967295"], 
                     stdout=PIPE, stderr=PIPE)

def backup_output(output):
    print(f"{datetime.now()} {output} already exists, backing up ...")
    bak = Path(f"bak")
    bak.mkdir(parents=True, exist_ok=True)
    create_time_ts = os.stat(output).st_ctime
    time_prefix = datetime.fromtimestamp(create_time_ts).strftime('%y%m%d_%H%M%S-')
    backup_path = shutil.move(str(output), str(bak / (time_prefix + os.path.basename(output))))
    print(f"{datetime.now()} backed up to {backup_path}")

def run_with_time_limit(args, dataset, query_shape, query_id, suffix=""):
    
    if args.algorithm.lower() == "gcsm":
        # output_suffix = (f"{args.algorithm}_{suffix}_"
        output_suffix = (f"{suffix}_"
                         f"timeLimit{args.time_limit}_"
                         f"batchSize{args.batch_size}_{interpret_matching_order(args.matching_order)}_"
                         f"{interpret_search_strategy(args.search_strategy)}_"
                         f"LocalIndex{args.use_local_index}_"
                         f"CartesianProduct{args.use_cartesian_product}_"
                         f"HeavyVertexAvoidance{args.use_heavy_vertex_avoidance}_"
                         f"RelationSwitch{args.use_relation_switch}")
    else:
        output_suffix = (f"{args.algorithm}_{suffix}_"
                        f"timeLimit{args.time_limit}_"
                        f"batchSize{args.batch_size}")

    base_path = f"datasets/{dataset}"
    data_path = f"{base_path}/data_graph/data.graph"
    update_graph = f"{base_path}/data_graph/insertion.graph"
    query_path = f"{base_path}/query_graph/{query_shape}/Q_{query_id}"

    result_dir = f"{base_path}/results/{query_shape}"
    result_file_base_name = f"{query_id}_{output_suffix}"
    normal_result_path = f"{result_dir}/{result_file_base_name}.txt"
    error_result_path = f"{result_dir}/{result_file_base_name}.err"

    # print(f"args.use_local_index {args.use_local_index}")

    print(f"{datetime.now()} Running {result_file_base_name}...")

    os.makedirs(result_dir, exist_ok=True)
    
    if not os.path.exists(query_path):
        print(f"{datetime.now()} {query_path} not found! Return early!")
        return
    if os.path.exists(normal_result_path):
        if args.rerun:
            backup_output(normal_result_path)
        else:
            print(f"{datetime.now()} {normal_result_path} already exists! Return early!")
            return
    if os.path.exists(error_result_path):
        if args.rerun:
            if args.skip_previous_error:
                print(f"{datetime.now()} {error_result_path} already exists! Skip the query!")
                return
            else:
                backup_output(error_result_path)
        else:
            print(f"{datetime.now()} {error_result_path} already exists! Return early!")
            return

    p = create_process(args, query_path, data_path, update_graph, query_shape)
    
    try:
        output, err = p.communicate(timeout=args.time_limit)
    except TimeoutExpired:
        p.kill()
        output, err = p.communicate()

    if not err and p.returncode == 0:
        print(f"{datetime.now()} {result_file_base_name} finished!")
        with open(normal_result_path, 'w') as f:
            f.write(output.decode())
    else:
        print(f"{datetime.now()} {result_file_base_name} **ERROR**")
        with open(error_result_path, 'w') as f:
            f.write(output.decode())
            f.write("\n")
            f.write(f"Return Code: {p.returncode}")
            f.write("\n")
            f.write(err.decode())

def main():
    args = get_args()

    if args.dataset_list:
        dataset_list = args.dataset_list
    else:
        dataset_list = ["lsbench/x1", "amazon/6", "netflow", "livejournal/30"]
    
    if args.query_shape_list:
        query_shape_list = args.query_shape_list
    else:
        query_shape_list = ["tree_6", "sparse_6", "dense_6"]

    if args.query_id_list:
        query_id_list = [int(item) for item in args.query_id_list]
    elif args.query_id_range:
        query_id_list = list(range(args.query_id_range[0], args.query_id_range[1]))
    else:
        query_id_list = list(range(args.num_queries))
    
    if args.skip_query_id_list:
        query_id_list = [query_id for query_id in query_id_list if query_id not in args.skip_query_id_list]

    for dataset in dataset_list:
        for query_shape in query_shape_list:
            for query_id in query_id_list:
                run_with_time_limit(args, dataset, query_shape, query_id, suffix=args.suffix)  # "debug_0119_v20"

if __name__ == "__main__":
    python_file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(f"{python_file_dir}/../../")
    main()
