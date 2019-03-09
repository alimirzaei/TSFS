import os
from glob import glob
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'normal',
    #    'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

directory = 'results'
methods = glob('results/supervised/*.pkl')
methods.append(methods.pop(1)) # Move TSFS to end
dataset = 'mnist'#['PCMAC','BASEHOCK','RELATHE' ,'Isolet','mnist_subset','Yale']#'COIL20', 'Prostate_GE']

TYPE = 0 # 1 2 3


x_label = "% of Selected Features"
y_label = "Classification Accuracy"

ps = [2, 4, 6, 8, 10, 20 ,30 ,40 ,50 ,60 ,70, 80 ,100]
accs = np.zeros((len(methods), len(ps)))
method_names = []
for i, method in enumerate(methods):
    with open(method,'rb') as f:
        results = pickle.load(f)
    method_names.append(method.split('/')[-1].split('.')[0])
    accs[i,:] = results[dataset]['mean'][TYPE,:]

pos = list(range(len(ps))) 
width = 0.2 
    
# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

plt.setp(ax.spines.values(), linewidth=3)

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos, 
        #using df['pre_score'] data,
        accs[0], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        #color='#EE3224', 
        # with label the first value in first_name
        label=method_names[0],linewidth=3) 

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using df['mid_score'] data,
        accs[1],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        #color='#F78F1E', 
        # with label the second value in first_name
        label=method_names[1]) 

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos], 
        #using df['post_score'] data,
        accs[2], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        #color='#FFC222', 
        # with label the third value in first_name
        label=method_names[2]) 

# Set the y axis label
ax.set_ylabel(y_label)
ax.set_xlabel(x_label)

# Set the chart's title
ax.set_title('Supervised Feature Selection on MNIST')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(ps)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0.6, np.max(accs)] )

# Adding the legend and showing the plot
plt.legend(method_names, loc='upper left')
plt.grid()

plt.show()