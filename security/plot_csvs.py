import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3]])
mode = 'djgrad'
attack = 'backdoor'

df = pd.read_csv('attack-{}-{}.csv'.format(mode, attack))
df = df.where(df['Sample Count'] < 3000)
sns.set(style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
if attack == 'backdoor':
    g = sns.relplot(data=df, col='Trigger', x='Sample Count', y='Accuracy', hue='Model Number', kind='line', legend='auto', palette=PAL)
else:
    g = sns.relplot(data=df, x='Sample Count', y='Accuracy', hue='Model Number', kind='line', legend='auto', palette=PAL)

# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('plots', '{}.png'.format('attack-{}-{}'.format(mode, attack))), bbox_inches='tight')
