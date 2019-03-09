import os
from glob import glob
import pickle 
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
    #    'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

directory = 'results'
#methods.append(methods.pop(2)) # Move TSFS to end
datasets = ['PCMAC','BASEHOCK','RELATHE' ,'Isolet','mnist_subset','Yale']#'COIL20', 'Prostate_GE']

TYPE = 0 # 1 2 3


x_label = "% of Selected Features"
if(TYPE==0):
    y_label = "Classification Accuracy"
elif(TYPE==1):
    y_label = "Clustering Accuracy"
elif(TYPE==2):
    y_label = "NMI"
elif(TYPE==3):
    y_label = "Mean Square Error"

fig = plt.figure(figsize=(3.5*3, 2.5*3))

axs = {}
#fig_class.suptitle('Classification Accuracy', fontsize=16)

with open('results/Sensitivity (copy).pkl','rb') as f:
    results = pickle.load(f)
for index, dataset in enumerate(datasets):
    axs[dataset] = {}
    axs[dataset] = fig.add_subplot(2, 3, index+1)
    axs[dataset].set_title(dataset)
    axs[dataset].set_xlabel(x_label)
    axs[dataset].set_ylabel(y_label)
    #plt.ylim([0,1.0])
 

ps = [2, 4, 6, 8, 10, 20 ,30 ,40 ,50 ,60 ,70, 80 ,100]
for l1 in [0.001,.01,.1]:#[0.001,0.01,0.1,1]:
   
    keys_datasets = results.keys()
    keys_datasets = list(set(keys_datasets).intersection(set(datasets)))
    for dataset in keys_datasets:
        acc = results[dataset][l1]['mean'][TYPE,:]
        axs[dataset].plot(ps, acc, label= '$\lambda=%.3f$'%l1)
        axs[dataset].legend()
        axs[dataset].grid(b=True, which='major', color='b', linestyle='-') 
    
for dataset in keys_datasets:
        axs[dataset].grid(b=True, which='major', color='b', linestyle='-') 
plt.subplots_adjust(wspace=.4, hspace=.4)
fig.show()
fig.savefig('results%d.png'%TYPE)
plt.show()
