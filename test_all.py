from scipy import io
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.manifold import TSNE
from RRFS import RRFS
from utils import evaluate_classification, evaluate_clustering, evaluate_reconstruction
import numpy as np
import pickle
import os
import methods as method_functions

def main():
    directory = 'results/'    
    if not os.path.exists(directory):
        os.makedirs(directory)
    methods =['my_tsne']#'MCFS','aefs', 'laplacian_score']#['my_supervised_mnist','my_tsne']#['MCFS','aefs']# ['udfs_score',  'laplacian_score', 'my_tsne', 'my_isomap']
    datasets = ['mnist_subset']##['COIL20','Yale','PCMAC','BASEHOCK','RELATHE','Prostate_GE' ,'Isolet']
    for method in methods:
        result_file_path = '%s/%s.pkl'%(directory, method)
        if(os.path.exists(result_file_path)):
            with open(result_file_path, 'rb') as f:
                results = pickle.load(f)    
        else:
            results={}
        
        for dataset in datasets:
            # load data
            mat = io.loadmat('/home/ali/Datasets/fs/%s.mat'%dataset)
            X = mat['X']    # data
            X = X.astype(float)
            y = mat['Y']    # label
            y = y[:, 0]


            percents = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 100]
            if(dataset not in results.keys()):
                results[dataset] = {}
                idx = getattr(method_functions, method)(X, y, dataset=dataset) 
                results[dataset]['mean'] = np.zeros((4, len(percents)))
                results[dataset]['std'] = np.zeros((4, len(percents)))
                results[dataset]['feature_ranking'] = idx
            else:
                idx = results[dataset]['feature_ranking']
           

            
            # perform evaluation on clustering task
            
            
            
            

            for index,p in enumerate(percents):
                # obtain the dataset on the selected features
                if(results[dataset]['mean'][0,index]!=0):
                    print('load %s, %s, %d'%(method, dataset, p))
                    continue
                num_fea = int(p*X.shape[1]/100)    # number of selected features
                selected_features = idx[:num_fea]
                selected_X = X[:, selected_features]

                (classification_accuracy_mean, classification_accuracy_std) = evaluate_classification(selected_X,y)
                #(clustering_nmi_mean, clustering_nmi_std), (clustering_accuracy_mean, clustering_accuracy_std) = evaluate_clustering(selected_X, y)
                #(reconstruction_mean, reconstruction_std) = evaluate_reconstruction(X, selected_features)
                
                results[dataset]['mean'][0,index] = classification_accuracy_mean# [classification_accuracy_mean,clustering_accuracy_mean,clustering_nmi_mean, reconstruction_mean]
                results[dataset]['std'][0,index] = classification_accuracy_std#[classification_accuracy_std, clustering_accuracy_std, clustering_nmi_std, reconstruction_std]
                
                with open(result_file_path,'wb') as f:
                    pickle.dump(results, f)

                print(50*'=')
                print('Method = %s, Dataset = %s, Percent = %d'%(method, dataset, p))
                print(50*'-')
                #print('Clustring (NMI, ACC) = %.3f, %.3f'%(clustering_nmi_mean, clustering_accuracy_mean))
                print('Classification (ACC) = %.3f'%classification_accuracy_mean)
                #print('Reconstruction (MSE) = %.3f'%reconstruction_mean)
if __name__ == '__main__':
    main()
