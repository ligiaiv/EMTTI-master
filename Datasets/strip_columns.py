import pandas as pd
import sys
name = sys.argv[1]
df = pd.read_csv(name)
df_final = df[["text","class"]]
df_final.to_csv(name,index=False)