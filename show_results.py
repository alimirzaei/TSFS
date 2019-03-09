import os
from glob import glob
import pickle 
import matplotlib.pyplot as plt



directory = 'results'
methods = glob('results/mine/*.pkl')

datasets = ['COIL20','PCMAC','BASEHOCK','RELATHE' ,'Isolet','Yale']#'COIL20', 'Prostate_GE']

fig = plt.figure(figsize=(len(datasets)*3, 4*3))

axs = {}
#fig_class.suptitle('Classification Accuracy', fontsize=16)

for index, dataset in enumerate(datasets):
    axs[dataset] = {}
    axs[dataset]['classification_acc'] = fig.add_subplot(4, len(datasets), index+1)
    axs[dataset]['classification_acc'].set_title(dataset)
    axs[dataset]['classification_acc'].set_xlabel("% of Selected Features")
    axs[dataset]['classification_acc'].set_ylabel("Classification Accuracy")
    plt.grid()
    axs[dataset]['mse'] = fig.add_subplot(4, len(datasets), len(datasets)+index+1)
    axs[dataset]['mse'].set_title(dataset)
    axs[dataset]['mse'].set_xlabel("% of Selected Features")
    axs[dataset]['mse'].set_ylabel("Mean Square Error")
    plt.grid()

    axs[dataset]['clustring_acc'] = fig.add_subplot(4, len(datasets), 2*len(datasets)+index+1)
    axs[dataset]['clustring_acc'].set_title(dataset)
    axs[dataset]['clustring_acc'].set_xlabel('% of Selected Features')
    axs[dataset]['clustring_acc'].set_ylabel('Clustering Accuracy')
    plt.grid()
    
    axs[dataset]['clustring_nmi'] = fig.add_subplot(4, len(datasets), 3*len(datasets)+index+1)
    axs[dataset]['clustring_nmi'].set_title(dataset)
    axs[dataset]['clustring_nmi'].set_xlabel('% of Selected Features')
    axs[dataset]['clustring_nmi'].set_ylabel('NMI')
    plt.grid()


ps = [2, 4, 6, 8, 10, 20 ,30 ,40 ,50 ,60 ,70, 80 ,100]
for method in methods:
    with open(method,'rb') as f:
        results = pickle.load(f)
    keys_datasets = results.keys()
    keys_datasets = list(set(keys_datasets).intersection(set(datasets)))
    method_name = method.split('/')[-1].split('.')[0]
    for dataset in keys_datasets:
        acc = results[dataset]['mean'][0,:]
        axs[dataset]['classification_acc'].plot(ps, acc, label= method_name)
        axs[dataset]['classification_acc'].legend()
        
        acc = results[dataset]['mean'][3,:]
        axs[dataset]['mse'].plot(ps, acc, label= method_name)
        axs[dataset]['mse'].legend()

        acc = results[dataset]['mean'][1,:]
        axs[dataset]['clustring_acc'].plot(ps, acc, label= method_name)
        axs[dataset]['clustring_acc'].legend()
        
        acc = results[dataset]['mean'][2,:]
        axs[dataset]['clustring_nmi'].plot(ps, acc, label= method_name)
        axs[dataset]['clustring_nmi'].legend()

        #axs[dataset]['clustring_nmi'].grid()
        
        
plt.subplots_adjust(wspace=.4, hspace=.4)
fig.show()
fig.savefig('results.png')
plt.show()
