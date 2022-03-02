import json
import matplotlib.pyplot as plt
import os,sys
import numpy as np

def read_file(FILE):
    print(FILE)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path+"/"+FILE, "r") as read_file:
        in_js = json.load(read_file)
    return in_js

def plot_graph(in_js):
    eval_loss=[]
    loss=[]
    acc = []
    lr = []
    epochs =[] 
    for epoch in in_js.keys():
        n_epock = int(epoch)
        for bump in in_js[epoch]:
            if "loss" in bump.keys():
                loss.append(bump["loss"])
            if "learning_rate" in bump.keys():
                lr.append(bump["learning_rate"])
            if "eval_loss" in bump.keys():
                eval_loss.append(bump["eval_loss"])
            if "eval_accuracy" in bump.keys():
                acc.append(bump["eval_accuracy"])
            true_epoch = n_epock+bump["epoch"]
            if true_epoch not in epochs:
                epochs.append(true_epoch) 
            
    # quit()
    x = list(range(len(eval_loss)))
    plt.plot(x, eval_loss, label = "eval_loss")
    plt.plot(x, acc, label = "eval_acc")
    plt.plot(x, loss, label = "train_loss")
    # plt.plot(x, np.cos(x), label = "curve 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file = sys.argv[1]
    in_js = read_file(file)
    plot_graph(in_js)
