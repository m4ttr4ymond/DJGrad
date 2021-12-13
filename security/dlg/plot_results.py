import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3]])

eps = [0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135]
sns.set(style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
counts = []
eps_list = []
for e in eps:
    counts.append(len(os.listdir('outputs-{}'.format(e))) / 200)
    eps_list.append(e)
df = pd.DataFrame({'Success Rate': counts, 'Epsilon': eps_list})
g = sns.relplot(data=df, x='Epsilon', y='Success Rate', kind='line', legend='auto', palette=PAL)

# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('{}.png'.format('data-reconstruction', bbox_inches='tight')))
