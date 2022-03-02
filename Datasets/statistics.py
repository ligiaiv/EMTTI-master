import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from string import punctuation

def remove_punct(text):
    for p in punctuation:
        text = text.replace(p," ")
    text = text.split()
    return text
def avg_word_len(text_list):
    word_lens = [len(w) for w in text_list]
    return np.average(word_lens)
def read_dataset(filename):
    df = pd.read_csv(filename)
    print(df.columns)
    df["words"] = df["text"].apply(remove_punct)
    df["word_count"] = df["words"].apply(len)
    df["avg_word_len"] = df["words"].apply(avg_word_len)
    all_words = remove_punct(' '.join(df["text"].to_list()).lower())
    different_words = set(all_words)
    variedade = len(different_words)/len(all_words)
    print(len(all_words))
    print(all_words[0:100])
    # print(df["word_count"].sum())
    # print(df["number of words without punctuation"])
    df_fake = df[df["class"] == False]
    df_real = df[df["class"] == True]
    sns.set(style="darkgrid")
    # print(df_fake["number of words without punctuation"])
    # print(df_real["number of words without punctuation"])
    sns.histplot(data=df_real, x="word_count",binrange = [0,5000])
    plt.savefig('hist_len_real.png')
    plt.clf()
    sns.histplot(data=df_fake, x="word_count",binrange = [0,1000])
    plt.savefig('hist_len_fake.png')
    plt.clf()


    sns.histplot(data=df_real, x="avg_word_len",binrange = [2,8])
    plt.savefig('hist_wordlen_real.png')
    plt.clf()
    sns.histplot(data=df_fake, x="avg_word_len",binrange = [2,8])
    plt.savefig('hist_wordlen_fake.png')
    plt.clf()

    print("Fake",df_fake["word_count"].max())
    print("True",df_real["word_count"].max())
    print("variedade", variedade)
    print("vocab_size: ",len(different_words))
    print("total_words",len(all_words))

if __name__ == '__main__':
    read_dataset("fakebr.csv")
    

