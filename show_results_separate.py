import os
from glob import glob
import pickle 
import matplotlib.pyplot as plt



directory = 'results'
methods = glob('results/supervised/*.pkl')
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


for index, dataset in enumerate(datasets):
    axs[dataset] = {}
    axs[dataset] = fig.add_subplot(2, 3, index+1)
    axs[dataset].set_title(dataset)
    axs[dataset].set_xlabel(x_label)
    axs[dataset].set_ylabel(y_label)
    #plt.ylim([0,1.0])
 

ps = [2, 4, 6, 8, 10, 20 ,30 ,40 ,50 ,60 ,70, 80 ,100]
for method in methods:
    with open(method,'rb') as f:
        results = pickle.load(f)
    keys_datasets = results.keys()
    keys_datasets = list(set(keys_datasets).intersection(set(datasets)))
    method_name = method.split('/')[-1].split('.')[0]
    for dataset in keys_datasets:
        acc = results[dataset]['mean'][TYPE,:]
        axs[dataset].plot(ps, acc, label= method_name)
        axs[dataset].legend()
        axs[dataset].grid()
        
        
plt.subplots_adjust(wspace=.4, hspace=.4)
fig.show()
fig.savefig('results%d.png'%TYPE)
plt.show()
