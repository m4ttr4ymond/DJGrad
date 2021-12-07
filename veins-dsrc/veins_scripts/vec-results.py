#!/usr/bin/python3

import pandas as pd
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load CSV into DataFrame
csv_file = sys.argv[1]
title = sys.argv[2]
data = pd.read_csv(csv_file, sep='\t', usecols=['time', 'node', 'gradientCount'])

# Make sure node gradient recording exists from creation to end of simulation
max_time = data['time'].max()
group_by_node = data.groupby('node')
for node_name, table in tqdm(group_by_node):
    max_node_time = table['time'].max()
    num_new_rows = max_time - max_node_time
    if num_new_rows > 0:
        last_row = table.loc[table['time']==max_node_time]
        temp_dataframe = pd.concat([last_row]*num_new_rows)
        temp_dataframe['time'] = [i for i in range(max_node_time+1, max_time+1)]
        data = pd.concat([data, temp_dataframe])
    # break

# Determine cumulative number of vehicles
group_by_time = data.groupby('time')
num_vehicles = [len(table) for _, table in group_by_time]

# Sum the number of gradients at each timestamp
cumulative_gradients = group_by_time.sum()

# Calculate percentage of vehicles with gradients
percent_gradients = [cum/num*100 if cum > 0 else 0 for num, cum in zip(num_vehicles, cumulative_gradients['gradientCount'])]

# Plot
fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
plt.xlim(0, max_time*1.01)
# cumulative_gradients.plot(ax=ax1, legend=None, color='b')
# ax1.plot(num_vehicles, color='r')
# ax1.set_ylabel('Vehicle Count')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylim(0, max(num_vehicles)*1.1)
# ax1.legend(['Vehicles with Gradients', 'Total Vehicles', 'Percent Vehicles with Gradients'], loc='upper left')
ax1.plot(percent_gradients, 'g')
ax1.set_ylabel('Percentage')
ax1.set_ylim(0, 100)
ax1.legend(['Percent Vehicles with Gradients'], loc='upper left')
plt.title(title)
# ax1.text(1025, num_vehicles[-1], f'{num_vehicles[-1]}', color='r')
# ax1.text(1025, cumulative_gradients.iloc[-1], f'{int(cumulative_gradients.iloc[-1])}\n({percent_gradients[-1]:.2f}%)', color='b')
ax1.text(900, percent_gradients[-1]+2.5, f'{percent_gradients[-1]:.2f}%', color='g')
plt.show()
