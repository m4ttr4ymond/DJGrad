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
    print(node_name)
    max_node_time = table['time'].max()
    last_row = table.loc[table['time']==max_node_time]
    while max_node_time < max_time:
        new_row = last_row.copy()
        new_row['time'] = max_node_time = max_node_time + 1
        data = data.append(new_row, ignore_index=True)

# Determine cumulative number of vehicles
group_by_time = data.groupby('time')
num_vehicles = [len(table) for _, table in group_by_time]

# Sum the number of gradients at each timestamp
cumulative_gradients = data.groupby('time').sum()

# Calculate percentage of vehicles with gradients
percent_gradients = [cum/num*100 if cum > 0 else 0 for num, cum in zip(num_vehicles, cumulative_gradients['gradientCount'])]

# Plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
plt.xlim(0, max_time*1.01)
cumulative_gradients.plot(ax=ax1, legend=None)
ax1.plot(num_vehicles)
ax1.set_ylabel('Vehicle Count')
ax1.set_ylim(0, max(num_vehicles)*1.1)
ax1.legend(['Vehicles with Gradients', 'Total Vehicles', 'Percent Vehicles with Gradients'], loc='upper left')
ax2.plot(percent_gradients, 'g')
ax2.set_ylabel('Percentage')
ax2.set_ylim(0, 100)
ax2.legend(['Percent Vehicles with Gradients'], loc='upper right')
plt.show()
