import os, re
import numpy as np

if __name__ == '__main__':
    out_path = 'results/csv-v1/local'
    os.makedirs(out_path, exist_ok=True)

    for dataset, version in [['lsbench', 'x1'], ['netflow', ''], ['amazon', '6'], ['livejournal', '30']]:
        for query in ['tree_6', 'sparse_6', 'dense_6']:
            algorithms = ['ours_v10_gg', 'ours_v10_noswt', 'ours_v10_noavd', 'ours_v10_ggl']
            time_array = np.zeros((100, len(algorithms)), dtype=float)

            for i, algorithm in enumerate(algorithms):
                for j in range(100):
                    err_file = f'datasets/{dataset}/{version}/results/{query}/{j}_{algorithm}.err'
                    log_file = f'datasets/{dataset}/{version}/results/{query}/{j}_{algorithm}.txt'

                    if not os.path.exists(log_file):
                        time_array[j, i] = 3600000
                    else:
                        with open(log_file) as f:
                            full_txt = ''.join(f.readlines())
                        match = re.search('Incremental Matching, time \(ms\)\: \d*\(host\), (\d*)\(kernel\)', full_txt)
                        if match:
                            time_array[j, i] = float(match.group(1))

            np.savetxt(f'{out_path}/{dataset}_{version}_{query}_ablation_new.csv', np.maximum(time_array - time_array[:,0:1], 0.1), fmt='%12.1f', delimiter=',')