#!/usr/bin/python3

import pandas as pd
import sys
import matplotlib.pyplot as plt

# Load CSV into DataFrame
csv_file = sys.argv[1]
data = pd.read_csv(csv_file, sep='\t', usecols=['time', 'node', 'gradientCount'])

# Make sure node gradient recording exists from creation to end of simulation
max_time = data['time'].max()
group_by_node = data.groupby('node')
for node_name, table in group_by_node:
    max_node_time = table['time'].max()
    last_row = table.loc[table['time']==max_node_time]
    while max_node_time < max_time:
        new_row = last_row.copy()
        new_row['time'] = max_node_time + 1
        data = data.append(new_row, ignore_index=True)
        max_node_time += 1

# Sum the number of gradients at each timestamp
cumulative_gradients = data.groupby('time').sum()
cumulative_gradients.plot()
plt.legend(loc='upper left')
plt.show()
