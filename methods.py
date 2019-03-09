from RRFS import RRFS
from sklearn.manifold import Isomap,TSNE,LocallyLinearEmbedding,MDS, SpectralEmbedding
from skfeature.function.sparse_learning_based import UDFS
from skfeature.utility.sparse_learning import feature_ranking
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
import numpy as np
from mnist_model import getCodes
from skfeature.function.information_theoretical_based import MRMR


def supervised_mrmr(X, y=None, **kwargs):
    idx, _, _ =MRMR.mrmr(X,y)
    return idx

def my_supervised_mnist(X, y=None, l1=.1, **kwargs):
    if(X.shape[1] != 28*28):
        print("Error Dataset")
        return range(len(X.shape[1]))
    rrfs = RRFS(28*28, hidden=2)
    codes = getCodes(X.reshape(X.shape[0],28,28,1))
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    idx = np.argsort(score)[::-1]
    return idx

def my_isomap(X, y=None, l1 = .1, n_components=2, **kwargs):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = Isomap(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def my_tsne(X, y=None,l1 = .1, n_components=2, **kwargs):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = TSNE(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def my_lle(X, y=None,l1 = .1, n_components=2, **kwargs):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = LocallyLinearEmbedding(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def my_mds(X, y=None,l1 = .1, n_components=2, **kwargs):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = MDS(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx



def my_se(X, y=None,l1 = .1, n_components=2, **kwargs):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    model = SpectralEmbedding(n_components=n_components)
    codes = model.fit_transform(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def laplacian_score(X, y=None, **kwargs):
    # construct affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)
    
    # obtain the scores of features
    score = lap_score.lap_score(X, W=W)

    # sort the feature scores in an ascending order according to the feature scores
    idx = lap_score.feature_ranking(score)

    return idx

def udfs_score(X, y, gamma=.1, **kwargs):
    Weight = UDFS.udfs(X, gamma=gamma, n_clusters=len(np.unique(y)))
    idx = feature_ranking(Weight)
    return idx

from AEFS_final import AEFS
def aefs(X, y=None, **kwargs):
    aefs = AEFS(input_dim=X.shape[1], encoding_dim=10)
    aefs.train(X, batch_size=32, epochs=150)
    weights = aefs.getFeatureWeights()
    idx = np.argsort(weights)[::-1]
    return idx

from skfeature.function.sparse_learning_based import MCFS as MCFS_CLASS
def MCFS(X, y=None, **kwargs):
    # construct affinity matrix
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs)

    num_cluster = len(np.unique(y))  

    # obtain the feature weight matrix
    Weight = MCFS_CLASS.mcfs(X, n_selected_features=X.shape[1], W=W, n_clusters=num_cluster)

    # sort the feature scores in an ascending order according to the feature scores
    idx = MCFS_CLASS.feature_ranking(Weight)

    return idx
    


from keras.models import Sequential, Model
from keras.layers import Dense, Input

def my_autoencoder(X, y=None, l1 = .1, n_components=2, **kwargs):
    rrfs = RRFS(X.shape[1], hidden=n_components)
    input_tensor = Input(shape=(X.shape[1],))
    l1 = Dense(10, activation='relu')(input_tensor)
    l2 = Dense(n_components, activation='relu')(l1)
    l3 = Dense(10, activation='relu')(l2)
    output_tensor = Dense(X.shape[1])(l3)
    model = Model(input_tensor, output_tensor)
    encoder = Model(input_tensor, l2)
    model.compile(optimizer='adam', loss='mse')                                
    model.fit(X, X, epochs=300)
    codes = encoder.predict(X)
    codes = (codes-np.min(codes))/(np.max(codes)-np.min(codes))
    #rrfs.train_representation_network(x_train, name=dataset+'_rep.hd5', epochs=1000)
    score = rrfs.train_fs_network(X,rep=codes, l1=l1, epochs=300, loss='mse')
    # sort the feature scores in an ascending order according to the feature scores
    idx = np.argsort(score)[::-1]
    return idx


def sup_ttest():
    pass


def sup_fisher():
    pass

def sup_CIFE():
    pass

from scipy.io import loadmat
def GAFS(X, y=None, dataset=None, **kwargs):
    res = loadmat('GAFS_result.mat')
    return res[dataset][0]-1


