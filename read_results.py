import pandas as pd
import json
import numpy as np
import os,sys
from io import StringIO
from pprint import pprint


if __name__ == '__main__':

    if len(sys.argv) <2:
        print("Please specify the name of results file.")
        exit()
    file_name = sys.argv[1]
    try:
        with open(file_name) as infile:
            data_dict = json.load(infile)

    except IOError as err:
        print(err)
        exit()
    data_DF = pd.read_csv(StringIO(data_dict['results']))
    data_DF['correct'] = data_DF.iloc[:, 1]+data_DF.iloc[:, 4]
    print(data_DF.iloc[:,1:4])
    data_DF['total'] = data_DF.iloc[:,1:5].sum(axis = 1)

    data_DF['avg'] = (data_DF.iloc[:, 1]+data_DF.iloc[:, 4])/data_DF.iloc[:,1:5].sum(axis = 1)
    avg_acc = data_DF['avg'].mean()
    print("avg_acc", avg_acc)
    pprint(data_dict["config"])