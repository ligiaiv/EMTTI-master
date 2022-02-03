import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET = "fortuna_reduced"
HERE = os.getcwd()
DATASET = "{}/{}.csv".format(HERE,DATASET)

df = pd.read_csv(DATASET)
print(df)
df['count'] = df.apply(lambda row: len(row["text"].split()), axis = 1)
valores = df["count"].values
rate = df["class"].values.sum()/len(df.index)
plt.hist(valores,bins=np.arange(valores.min(),valores.max(),1))
plt.savefig('{}.png'.format(DATASET))
print(rate)

# plt.show() 