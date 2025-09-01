import os
from subprocess import Popen, PIPE, TimeoutExpired

def run_execution_time(dataset, label, query_shape, algorithm, i):
    base_path = f'datasets/{dataset}/{label}'
    
    data_path = f'{base_path}/data_graph/data.graph'
    update_graph = f'{base_path}/data_graph/insertion.graph'
    
    query_base_path = f'{base_path}/query_graph/{query_shape}'
    result_path_base = f'{base_path}/results/{query_shape}'

    os.makedirs(result_path_base, exist_ok=True)
    
    if os.path.exists(f'{result_path_base}/{i}_{algorithm}.txt') or \
        os.path.exists(f'{result_path_base}/{i}_{algorithm}.err') or \
        not os.path.exists(f'{query_base_path}/Q_{i}'):
        return
    p = Popen(['./bin/csmg_v8_release_2_graph',
        f'{query_base_path}/Q_{i}',
        data_path,
        update_graph,
        '4'], 
        stdout=PIPE, stderr=PIPE)
    try:
        output, err = p.communicate(timeout=3600)
    except TimeoutExpired:
        p.kill()
        output, err = p.communicate()

    if not err and p.returncode == 0:
        print(f'finish-{i}-{algorithm}')
        with open(f'{result_path_base}/{i}_{algorithm}.txt', 'w') as f:
            f.write(output.decode())
    else:
        with open(f'{result_path_base}/{i}_{algorithm}.err', 'w') as f:
            f.write(output.decode() + '\n')
            f.write(f'return code: {p.returncode}\n')
            f.write(err.decode())
        print(f'{i}-{algorithm} **ERROR**')

if __name__ == '__main__':
    for dataset, label in [['lsbench', 'x1'], ['netflow', ''], ['livejournal', '30'], ['amazon', '6']]:
        for query_shape in ['sparse_6', 'dense_6']:
            for i in range(100):
                run_execution_time(dataset, label, query_shape, 'ours_v8_2_graph', i)
