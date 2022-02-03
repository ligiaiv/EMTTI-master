# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
import os

ENCODING = 'ISO-8859-15'
ENCODING = 'ASCII'
# ENCODING = 'latin-1'
# ENCODING = 'ISO-8859-1'
# ENCODING = 'cp1252'

def read_files(target,i):
    INVALID_BYTES = [b"\xe9\xe1"]
    text_file = "full_texts/{}/{}.txt".format(target,i)
    meta_file = "full_texts/{}-meta-information/{}-meta.txt".format(target,i)
    # print(text_file)
    # target = True
    try:

        with open(text_file,'rb') as file:
            raw = file.read()
            # print(repr(raw))
            # print(raw)
            try:
                text = raw.decode('utf8')
            except UnicodeDecodeError:
                text = raw.decode('latin-1',errors='backslashreplace')
        with open(meta_file,encoding='utf8') as file:
            meta_info = file.read().split('\n')

    except FileNotFoundError:
        text = None
        meta_info = [None]*len(meta_param)
    return text,meta_info

#
#   Read file containing information on metadata
#
with open("meta_info") as file:
    meta_param = file.readlines()
    meta_param = [line.strip() for line in meta_param]

# print(meta_param)
# quit()s
COLUMN_NAMES = ['text','class']+meta_param
print(len(COLUMN_NAMES))
dataset = pd.DataFrame(columns=COLUMN_NAMES)

#
#   Read True News
#
not_found = 0
print(COLUMN_NAMES)
# quit()
# raw_list = []

for i in tqdm(range(1,3602+1)):
# for i in tqdm(range(1,5)):

    text,meta_info = read_files("true",i)
    row = pd.DataFrame([[text,True]+meta_info], columns = COLUMN_NAMES)
    dataset = dataset.append(row,ignore_index=True)
    # raw_list.append(raw)

    text,meta_info= read_files("fake",i)
    row = pd.DataFrame([[text,False]+meta_info], columns = COLUMN_NAMES)
    dataset = dataset.append(row,ignore_index=True)

    # raw_list.append(raw)


    # print([text,target]+meta_info)
    # a = [text,target]+meta_info
    # print(len(a))
    # print(row.size)
    # print(dataset.size)
    # df2 = pd.DataFrame([[5, 6], [7, 8]], columns=COLUMN_NAMES, index=['x', 'y'])

print(dataset)
dataset['text'] = dataset['text'].str.replace('\n',' ').str.replace('\r','')

# .replace('\r\n','').replace('\t','')
print('\r' in dataset['text'][1])
print(dataset['text'][1])
HERE = os.getcwd()
dataset.to_csv("{}/fakebr.csv".format(HERE))
# print(raw_list[1])
# print((dataset["author"]=="None").sum())
# print(not_found)
quit()    
for filename in os.listdir(os.getcwd()):
    with open(filename,'r') as f_in:
        pass
