#!/usr/bin/python3

import pandas as pd
import sys

data = pd.read_csv(sys.argv[1], sep='\t')
gradients=data['gradientCount']
print(f'Total vehicles with gradients: {gradients.sum()}')
