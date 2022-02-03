from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import sys,os


if len(sys.argv) < 2:
    print(sys.argv)
    print("Please specify the dataset.")
    quit()
fileIn = sys.argv[-1]
fileOut_partial = fileIn.split('.')[-2]
boot = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
df = pd.read_csv(fileIn)
X = df["text"].values
Y = df["class"].values
k = 0
for train, test in boot.split(X,Y):
    print(test)
    k+=1
    print("K : ",k)

    # x_train = X[train]
    # y_train = Y[train]
    x_test = X[test]
    y_test = Y[test]
    df_partial = df.iloc[test]
    df_partial.to_csv("{}_{}.csv".format(fileOut_partial,k),index = False)

    # eda = EDA(NAME)
    # if AUG ==   "syn":
    #     eda.read_synonims("synonims.json")
    #     augmented = eda.synonim_replacement(2)
    # elif AUG == "swap":
    #     augmented = eda.random_swap()
    # elif AUG == "del":
    #     augmented = eda.random_deletion(2)
    # augmented["text"] = augmented["text"].str.join(' ')
    # augmented.to_csv(NEW_NAME,index = False)


