import matplotlib.pyplot as plt
import numpy as np 

def plot_graph(filename): 
    try:
        file = open(filename,"r")
        content = file.readlines()
        train = []
        test  = []
        ax = [i for i in range(len(train))]
        for i in content: 
            tr_loss, t_loss = i.split(',')
            print(tr_loss)
            print(t_loss)
        plt.plot(ax,train)
        plt.savefig('train')
        plt.plot(ax,test)
        plt.savefig('test')
            
    finally:
        file.close()
        
plot_graph("/home/harsh.shukla/SRCNN/Codes/Generative_VAE/Loss_128b_withoutbn.txt")