import pandas as pd

df = pd.read_csv("fakeeng.csv")
df.rename(columns={"label": "class"})
df["text"] = df["text"].str.replace('\n', ' ')
# print(df)
# print(df["text"].apply(type))
problematic = df[df["text"].apply(type) != str ]
print("dasdada",problematic)
df = df.drop(problematic.index)
problematic = df[df["text"].apply(type) != str ]
print("dasdada",problematic)
df_final = df[["text","class"]]
df_final.to_csv("fakeeng.csv",index=False)
