#!/usr/bin/env python
# coding: utf-8

# Εισαγωγή χρήσιμων Βιβλιοθηκών

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from keras import backend as K
from keras.layers import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from keras.initializers import Initializer
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score,mean_squared_error
import math
from sklearn.metrics import pairwise
import tensorflow_addons as tfa
import keras_tuner as kt


# Εισαγωγή, κανονικοποίηση δεδομένων

# In[2]:


(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data(test_split=0.25)

#Convert to float32
x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)
y_train, y_test = np.array(y_train, dtype=np.float32), np.array(y_test, dtype=np.float32)

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)


# Δημιουργία RBF επιπέδου και initializer με k-means.
#             RBFLayer: Available at https://github.com/PetraVidnerova/rbf_keras 
#             (τροποποιημένο καθώς δεν υπήρχε ο συντελεστής σ)

# In[3]:


class InitCentersKMeans(Initializer):

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_        

class RBFLayer(Layer):

    def __init__(self, output_dim, initializer=None, sigma=1.0, **kwargs):
        self.init_sigma = sigma
        self.output_dim = output_dim
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.sigma = self.add_weight(name='sigmas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=self.init_sigma),
                                     trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-(1/(2*(self.sigma**2))) * K.sum(H**2, axis=1))
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Χρήσιμες επιλογές

# In[4]:


layers_neurons = [round(0.1*x_train.shape[0]), round(0.5*x_train.shape[0]), round(0.9*x_train.shape[0])]

optimizer = keras.optimizers.SGD(learning_rate=0.001)
loss = keras.losses.MeanSquaredError()
metrics = ['accuracy']
epochs = 100
out_n = 128
validation_split = 0.2


# Δημιουργία, εκπαίδευση μοντέλων και υπολογισμός καμπύλης εκμάθησης και μετρηκών R^2, RMSE

# In[5]:


for neuron in layers_neurons:
    kmeans = KMeans(n_clusters=neuron).fit(x_train)
    centers = kmeans.cluster_centers_
    dmax = np.amax(pdist(centers, 'euclidean'))
    sigma = dmax/tf.math.sqrt(2*float(centers.shape[0]))
    
    model = keras.models.Sequential() 
    model.add(RBFLayer(neuron, initializer=InitCentersKMeans(x_train), sigma=sigma, input_shape=(x_train.shape[1],)))
    model.add(keras.layers.Dense(out_n))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    fit = model.fit(x_train, y_train, epochs=epochs ,validation_split=validation_split, verbose=1)
    # Leaning Curve
    plt.figure()
    plt.plot(fit.history['loss'],label='Training')
    plt.plot(fit.history['val_loss'],label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss') 
    plt.legend()
    plt.title(f"Learning curves of training and validation sets for Model with {neuron} on RBFLayer.")
    plt.show()
    plt.close()
    
    y_pred = model.predict(x_test)
    
    print(f"From predict:\nR^2: {r2_score(y_test,y_pred)}\nRMSE: {math.sqrt(mean_squared_error(y_test,y_pred))}")


# Fine Tuning
# 
# Συνάρτηση η οποία δημιουργεί και επιστρέφει ένα Keras μοντέλο.

# In[6]:


def build_model(hp):
    
    n_h1 = hp.Choice('hidden1', 
                    [round(0.05*x_train.shape[0]), round(0.15*x_train.shape[0]), round(0.3*x_train.shape[0]), round(0.5*x_train.shape[0])])
    n_h2 = hp.Choice('hidden2', [32, 64, 128, 256])
    dp = hp.Choice('dp', [0.2, 0.35, 0.5])
    
    kmeans = KMeans(n_clusters=n_h1).fit(x_train)
    centers = kmeans.cluster_centers_
    dmax = np.amax(pdist(centers, 'euclidean'))
    sigma = dmax/tf.math.sqrt(2*float(centers.shape[0]))
    
    optimizer = keras.optimizers.SGD(learning_rate=0.001)
    metrics = ['accuracy',keras.metrics.RootMeanSquaredError()]

    model = keras.models.Sequential()
    model.add(RBFLayer(n_h1, initializer=InitCentersKMeans(x_train), sigma=sigma, input_shape=(13,)))
    model.add(keras.layers.Dense(n_h2))
    model.add(keras.layers.Dropout(dp))
    model.add(keras.layers.Dense(1))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


# Αρχικοποίηση ενός tuner

# In[7]:


epochs = 100
objective=kt.Objective('val_accuracy', direction = 'max')

tuner = kt.Hyperband(build_model, objective=objective, max_epochs=epochs)


# Αναζήτηση βέλτιστων παραμέτρων

# In[8]:


tuner.search(x_train, y_train, epochs=epochs, validation_split=validation_split)
best_params = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Optimal number of neurons in 1st (RBF) layer: {best_params.get('hidden1')}")
print(f"Optimal number of neurons in 2nd layer: {best_params.get('hidden2')}")
print(f"Optimal value of dropout probability: {best_params.get('dp')}")


# Εκπαίδευση του βέλτιστου Μοντέλου

# In[9]:


model = tuner.hypermodel.build(best_params)
fit = model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split)

plt.figure()
plt.plot(fit.history['loss'],label='Training')
plt.plot(fit.history['val_loss'],label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.legend()
plt.title("Best Model's Learning curves of training and validation sets")
plt.show()
plt.close()

y_pred = model.predict(x_test)
    
print(f"From predict:\nR^2: {r2_score(y_test,y_pred)}\nRMSE: {math.sqrt(mean_squared_error(y_test,y_pred))}")

