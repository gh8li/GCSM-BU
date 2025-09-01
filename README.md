# Efficient GPU-Based Continuous Subgraph Matching on Batch Updates

<!-- ## Introduction -->

> This repository contains the implementation of a GPU-based continuous subgraph matching algorithm: GCSM-BU.

## Directory Structure

```
GCSM-BU/
├── build/                   # Compiled binaries and build files
├── datasets/                # Graph datasets and querysets
├── experiments/   
│   └── 0_run/   
│       ├── run_csm.py       # Python script for running experiments
│       └── scripts.sh       # Example usage scripts
├── graph/                   # Graph and SM related data structures
├── kernels/                 # CUDA kernels for graph update and matching
├── utils/                   
├── technique_permutation.h  # Code for ablation study.
├── main.cpp
├── CMakeLists.txt           # CMake build configuration
└── README.md                # Project documentation
```

\* SM is short for Subgraph Matching.


## Compile

Our program requires cmake (Version >= 3.10), Make (Version >= 3.82), GCC (Version >= 11.2.0), and nvcc (Version >= 11.1). You can compile the code by executing the following commands from the root directory of the project. 

```shell
mkdir build
cd build
cmake ..
make
cd ..
```

## Execute

After a successful compilation, the binary file is created under the `build/` directory. You can execute GCSM-BU using the following command.

```shell
./build/csmg --algorithm <algorithm ID> --query <query-graph-path> --data <data-graph-path>  --update <update-path>
```

### Commandline Parameters
Commandline parameters supported by the program are listed in the following table.

| Parameters  | Description                               | Valid Value     | Default Value |
|-------------|-------------------------------------------|-----------------|---------------|
| --algorithm | Algorithm ID.                             | 0-5             | 0       |
| --query     | Path to the query graph file.             | n/a             | n/a           |
| --data      | Path to the data graph file.              | n/a             | n/a           |
| --update    | Path to the update file.                  | n/a             | n/a           |
| --batch_size| Number of update edges in a batch.        | 1-4294967295    | 4294967295 (all in one batch)            |
| --show_gpu_memory| Print GPU memory usage or not.       | True/False      | False            |
| --print_indexing_time| Print indexing time or not.      | True/False      | False            |
| --device    | GPU ID for execution.                     | 0-7             | 0             |


| Algorithm ID  | Algorithm                               |Note |
|---------------|-----------------------------------------|---|
| 0  | Defult ID, for ablation study      ||
| 1  | GCSM-BU                            ||
| 2  | QO-GAMMA*                           ||
| 3  | GAMMA* (Not used)                 |Use [this repository]() instead.|
| 4  | GCSM-BU, with indexing order as matching order.        ||
| 5  | GCSM-BU, using GAMMA*'s matching order.            ||


`experiments/0_run/scripts.sh` shows examples of running GCSM-BU with commandline.

## Input File Format
Both the input query graph and data graph are vertex-labeled and edge-labeled. Each vertex is represented by a distinct unsigned integer (from 0 to 4294967295). There is at most one edge between two arbitrary vertices. A vertex and an edge are formatted as `v <vertex-id> <vertex-label>` and `e <vertex-id-1> <vertex-id-2> <edge-label>`, respectively. The two endpoints of an edge must appear before the edge. For example, 

```
v 0 3
v 1 0
v 2 6
v 3 1
v 4 5
v 5 1
v 6 0
v 7 3
e 0 1 0
e 0 2 0
e 0 5 0
e 1 3 0
e 2 3 0
e 2 5 0
e 3 4 0
e 3 6 0
e 6 7 0
```

The update file contains only update edges, represented in the same format as in the data graph.

## Ablation Study

With `--algorithm 0`, we perform ablation study, and some other parameters are available:

| Parameters             | Default Value | Description                                      |
|------------------------|--------------|--------------------------------------------------|
| --matching_order       | 2            | Specifies the matching order to use.    |
| --search_strategy      | 2            | Specifies the search strategy to use.             |
| --local_index          | True         | Enable local index or not.            |
| --cartesian_product    | True         | Enable backtracking flattening or not.      |
| --heavy_vertex_avoidance | True       | Enable heavy vertex avoidance or not.       |
| --relation_switch      | True         | Enable relation switch or not. |


| Matching Order ID  | Descrition          |
|---------------|--------------------------|
| 0  | Use indexing order                ||
| 1  | Use GAMMA*'s matching order       ||
| 2  | Use GCSM-BU's matchin order       ||


| Search Strategy ID  | Descrition |
|---------------|------------------|
| 0  | DFS (w/ work stealing)      ||
| 1  | DFS (w/o work stealing)     ||
| 2  | Dynamic Semi-BFS            ||
| 3  | BFS                         ||

## Scripts

Use the python script `experiments/0_run/run_csm.py` to run the experiments in our paper. The script parameters are listed in the `run_csm.py` file. `experiments/0_run/scripts.sh` also shows the usage of `run_csm.py`.

## Datasets and Querysets

The graph datasets and their corresponding querysets used in our paper can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xsunax_connect_ust_hk/Et-cxVY7l5FCoZoKeDyMzmQBaCBn8ffbPFFQfIFOqGIodA?e=4vT3OI). The dataset zip files should be uncompressed inside the `datasets/` directory.
