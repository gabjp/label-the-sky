import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def retrieve_values(metrics, words):
    for i in range(len(words)):

        if words[i] == "loss:":
            metrics["loss"].append(float(words[i+1]))
        elif words[i] == "accuracy:":
            metrics["acc"].append(float(words[i+1]))
        elif words[i] == "val_loss:":
            metrics["val_loss"].append(float(words[i+1]))
        elif words[i] == "val_accuracy:":
            metrics["val_acc"].append(float(words[i+1]))


def get_values(path):
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    metrics = {"loss":loss, "val_loss":val_loss, "acc":acc, "val_acc":val_acc}

    with open(path, mode="r") as file:
        for line in file:
            words = line.split()
            if len(words) >1 and words[1] == "-":
                retrieve_values(metrics, words)
            else: continue

    return metrics 

def main():
    path = sys.argv[1]
    metrics = get_values(path)

    df = pd.DataFrame(metrics)

    title = path.spli("/")[-1] 
    df[["loss", "val_loss"]].plot(title=title)
    plt.savefig(f"outs/new/figures/{title}.png")

    if metrics["acc"] != []:
        df[["acc", "val_acc"]].plot()
        plt.savefig(f"outs/new/figures/{title}.png")


    return

if __name__== "__main__":
    main()