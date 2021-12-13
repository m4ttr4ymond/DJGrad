#!/usr/bin/python3

import pandas as pd
import sys

data = pd.read_csv(sys.argv[1], sep='\t')
gradients=data['gradientCount']
tot_vehs = len(gradients)
sum_grads = gradients.sum()
print(f'Total vehicles: {tot_vehs}')
print(f'Total vehicles with gradients: {sum_grads}')
print(f'Percent with gradients: {sum_grads/tot_vehs*100:.2f}%')
