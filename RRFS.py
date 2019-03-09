# Representation Reconstruction Feature Selection
# Author : Ali Mirzaei
from utils import getSyntheticDataset, load_data
from keras.models import Model
from keras.layers import Dense,Input
import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping

def loss_mse(y, y_true):
    return K.tf.reduce_mean(K.tf.square(y-y_true))


class RRFS():
    def __init__(self, dim,l1=.01, hidden=10, loss='mse'):
        self.l1 = l1
        self.early_stopping = EarlyStopping(patience=3)
        x1 = Input((dim,))
        x2 = Dense(hidden, activation='sigmoid')(x1)
        x3 = Dense(dim)(x2)
        self.autoencoder = Model(x1, x3)
        self.encoder = Model(x1, x2)
        self.autoencoder.compile(optimizer='adam', loss=loss)

        

    def kernel_init(self,shape, dtype=None):
        return self.autoencoder.layers[1].get_weights()[0]
    
    def bias_init(self,shape, dtype=None):
        return self.autoencoder.layers[1].get_weights()[1]
    
    def train_representation_network(self, X, name = 'representation_nn.hd5',batch_size=32, epochs = 10):
        try:
            self.autoencoder.load_weights(name)
        except:
            
            self.autoencoder.fit(X,X, epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping],validation_split=.15, verbose=0)
            self.autoencoder.save_weights(name)

    def train_fs_network(self,X , rep=None,l1=.01, name='fs_nn.hd5',loss='mse', batch_size=32, epochs = 10):
        self.l1 = l1
        hidden =rep.shape[1]
        dim = X.shape[1]
        y1 = Input((dim,))
        y2 = Dense(10*hidden, activation='relu', kernel_regularizer =self.layer1_reg)(y1)
        y3 = Dense(hidden)(y2)

        self.fs_model = Model(y1, y3)
        
        self.fs_model.compile(optimizer='adam', loss=loss, metrics=[loss_mse])
        if(rep is None):
            L = self.encoder.predict(X) 
        else:
            L = rep
        self.fs_model.fit(X, L, epochs=epochs, batch_size=batch_size, validation_split=.15, verbose=1)
        #self.fs_model.save_weights(name)
        
        w = self.fs_model.layers[1].get_weights()[0]
        w = np.sum(np.square(w),1)
        return w


    def layer1_reg(self, weight_matrix):
        return self.l1*K.sum(K.sqrt(K.tf.reduce_sum(K.square(weight_matrix), axis=1)))
    


