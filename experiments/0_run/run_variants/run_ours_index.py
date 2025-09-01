import os
from subprocess import Popen, PIPE, TimeoutExpired
import multiprocessing

def run_execution_time(dataset, label, query_shape, i):
    base_path = f'datasets/{dataset}/{label}'
    
    data_path = f'{base_path}/data_graph/data.graph'
    update_graph = f'{base_path}/data_graph/insertion.graph'
    
    query_base_path = f'{base_path}/query_graph/{query_shape}'
    result_path_base = f'{base_path}/results/{query_shape}'

    os.makedirs(result_path_base, exist_ok=True)
    
    for b in ['v10_gg', 'v10_ggl', 'v10_noavd', 'v10_noswt']:
        if os.path.exists(f'{result_path_base}/{i}_ours_{b}.txt') or \
            os.path.exists(f'{result_path_base}/{i}_ours_{b}.err') or \
            not os.path.exists(f'{query_base_path}/Q_{i}'):
            continue
        p = Popen([f'./bin/csmg_{b}',
            f'{query_base_path}/Q_{i}',
            data_path,
            update_graph,
            str({0:3,1:4}[int(multiprocessing.current_process()._identity[0]) - 1])],
            stdout=PIPE, stderr=PIPE)
        try:
            output, err = p.communicate(timeout=3600)
        except TimeoutExpired:
            p.kill()
            output, err = p.communicate()

        if not err and p.returncode == 0:
            print(f'finish-{i}-ours_{b}')
            with open(f'{result_path_base}/{i}_ours_{b}.txt', 'w') as f:
                f.write(output.decode())
        else:
            with open(f'{result_path_base}/{i}_ours_{b}.err', 'w') as f:
                f.write(output.decode() + '\n')
                f.write(f'return code: {p.returncode}\n')
                f.write(err.decode())
            print(f'{i}-ours_{b} **ERROR**')

if __name__ == '__main__':
    pool = multiprocessing.Pool(2)
    for dataset, label in [['lsbench', 'x1'], ['netflow', ''], ['livejournal', '30'], ['amazon', '6']]:
        for query_shape in ['tree_6', 'sparse_6', 'dense_6']:
            for i in range(100):
                pool.apply_async(run_execution_time, (dataset, label, query_shape, i))
    pool.close()
    pool.join()