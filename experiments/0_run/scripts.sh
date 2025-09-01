
# Run the following command at the root directory of the project to execute the program after compilation
# Please change the paths of query graph and data graph accordingly

# --algorithm 1 means GCSM-BU
./build/csmg --algorithm 2 --query $(pwd)/datasets/amazon/6/query_graph/dense_6/Q_0 --data $(pwd)/datasets/amazon/6/data_graph/data.graph --update $(pwd)/datasets/amazon/6/data_graph/insertion.graph --device 0

# --algorithm 2 means QO-GAMMA
./build/csmg --algorithm 2 --query $(pwd)/datasets/amazon/6/query_graph/dense_6/Q_0 --data $(pwd)/datasets/amazon/6/data_graph/data.graph --update $(pwd)/datasets/amazon/6/data_graph/insertion.graph --device 0

# run the algorithms with python scripts
python -u experiments/0_run/run_csm.py --device 0 --time_limit 1800 --suffix 0820 --algorithm Gcsm --dataset_list amazon/6 livejournal/30 lsbench/x1 netflow  --query_shape_list dense_6  --num_queries 100 --rerun --print_indexing_time True >> ./log/0820_log_Gcsm.txt 2>&1 &
python -u experiments/0_run/run_csm.py --device 0 --time_limit 1800 --suffix 0820 --algorithm GcsmForDense --dataset_list amazon/6 livejournal/30 lsbench/x1 netflow  --query_shape_list dense_6  --num_queries 100 --rerun --print_indexing_time True >> ./log/0820_log_GcsmForDense.txt 2>&1 &

# run our implementation of GAMMA
./other_methods/gamma --query $(pwd)/datasets/amazon/6/query_graph/tree_6/Q_0 --data $(pwd)/datasets/amazon/6/data_graph/data.graph --update $(pwd)/datasets/amazon/6/data_graph/insertion.graph --device 0