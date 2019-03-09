from scipy.io import loadmat
import numpy as np
from skfeature.utility import unsupervised_evaluation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, KFold
from keras.layers import Dense
from keras.models import Sequential
import keras.backend as K
import gc


def load_data(dir ='/home/ali/Datasets/Channel', SNR = 22):
    noisy = loadmat(dir + '/My_noisy_H_%d.mat'%SNR)['My_noisy_H']
    noisy_image = np.zeros((40000,72,14,1))
    noisy_image[:,:,:,0] = np.real(noisy)
    perfect = loadmat(dir + '/My_perfect_H_%d.mat'%SNR)['My_perfect_H']
    perfect_image = np.zeros((40000,72,14,1))
    perfect_image[:,:,:,0] = np.real(perfect)

    return (noisy_image, perfect_image)

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def getSyntheticDataset(N = 10000,type = 'linear', indep = 5, dep = 4):
    X = np.zeros((N, indep + indep*dep))
    index = 0
    for i in range(indep):
        X[:, index] = np.random.rand(N)
        index = index + 1
        for j in range(dep):
            if(type == 'linear'):
                X[:, index+j] = (j+2)*X[:, index-1] 
            else:
                X[:, index+j] = sigmoid(X[:, index-1]*(3*float(j)/dep+1))

        index = index + dep
    return X

if __name__ == '__main__':

    X = getSyntheticDataset()

    print(X[0,:])




def evaluate_clustering(selected_features,y):
    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = np.zeros(20)
    acc_total = np.zeros(20)
    for i in range(0, 20):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=len(np.unique(y)), y=y)
        nmi_total[i]= nmi
        acc_total[i]= acc

    # output the average NMI and average ACC
    return (np.mean(nmi_total), np.std(nmi_total)), (np.mean(acc_total),np.std(acc_total))

def evaluate_classification(selected_features, y):
    clf = MLPClassifier()
    scores = cross_validate(clf, selected_features, y, cv=5, n_jobs=4)
    return (np.mean(scores['test_score']), np.std(scores['test_score']))


def evaluate_reconstruction(X, selected_features):
    kf = KFold(n_splits=5, shuffle=True)
    losses = []
    for train_index, test_index in kf.split(X):
        model = Sequential()
        model.add(Dense(10, input_dim=len(selected_features), activation = 'relu'))
        model.add(Dense(X.shape[1]))
        model.compile(optimizer='Adam', loss='mse')
        X_train, X_test = X[train_index], X[test_index]
        model.fit(X_train[:,selected_features], X_train, epochs=500, verbose=0)
        losses.append(model.evaluate(X_test[:, selected_features], X_test))
        K.clear_session()
    gc.collect()
    return (np.mean(losses), np.std(losses))

