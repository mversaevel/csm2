""" Convert pickle results to more portable formats like CSV and PNG. """

import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

# load CSM index data
data_path = 'data/index_data/'
file_name = 'run4-mimicking' # w/o extension

output_path = f'rmarkdown/{file_name}/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(data_path + file_name + '.pkl', 'rb') as f:
    results = pickle.load(f)

results['correlations_grouped_by_cluster'] = pd.DataFrame(results['correlations_grouped_by_cluster'])
results['correlations_grouped_by_region'] = pd.DataFrame(results['correlations_grouped_by_region'])
results['correlations_no_grouping'] = pd.DataFrame(results['correlations_no_grouping'])

### convert output to csv's for easier use in R markdown

# write dataframes in nested dicts to csv
for k, v in results.items():
    if isinstance(v, pd.DataFrame):
        v.to_csv(f'{output_path}/{k}.csv')
    elif isinstance(v, dict):
        for k2, v2 in v.items():
            if isinstance(v2, pd.DataFrame):
                v2.to_csv(f'{output_path}/{k}_{k2}.csv')            
            elif isinstance(v2, dict):
                for k3, v3 in v2.items():
                    if isinstance(v3, pd.DataFrame):
                        v3.to_csv(f'{output_path}/{k}_{k2}_{k3}.csv')
                    else:
                        print(f'warning! incomplete export for {k}, {k2}, {k3}')
            else:
                print(f'warning! incomplete export for {k}, {k2}')
    else:
        print(f'warning! incomplete export for {k}')