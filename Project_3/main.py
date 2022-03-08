#!/usr/bin/env python
# coding: utf-8

# Εισαγωγή χρήσιμων Βιβλιοθηκών

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os
from keras.regularizers import l2
from keras.regularizers import l1
import tensorflow_addons as tfa
import keras_tuner as kt
import numpy as np


# Εισαγωγή, κανονικοποίηση δεδομένων

# In[3]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
y_test_r = y_test

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

#One-hot encoding (to increase performance)
y_train, y_test = keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)


# Αρχικοποίηση μοντέλου με δύο κρυφά στρώματα 128 και 256 νευρώνων αντίστοιχα. Συνάρτηση ενεργοποίησης ReLu, batch_size = 256, συνάρτηση στρώματος εξόδου η softmax. Τέλος ως μετρική αξιολόγησης θεωρείται η ακρίβεια accuracy, ενώ αντικειμενική συνάρτηση επιλέγεται η categorical cross-entropy.

# In[5]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(256, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

loss = keras.losses.CategoricalCrossentropy()
metrics = ['accuracy']
batch = 256
epochs = 100
validation_split = 0.2

path = "/home/alepetpan/Desktop/Computational_Intelligence/Project_3/images"


# Μοντέλο με {1,256,Ν_train} - ((online, minibatch, batch))

# In[6]:


# Run default model first
model.compile(loss=loss, metrics=metrics)
batch_size = [1, 256, x_train.shape[0]]
for b in batch_size:
    start = time.time()
    fit = model.fit(x_train, y_train, batch_size=b, epochs=epochs, validation_split=validation_split, verbose=1)
    training_time = time.time()-start
    
    #1 ταχύτητα εκπαίδευσης για τις τρείς διαφορετικές μεθόδους
    print(f"Training time with minibatch = {b}: {training_time}") 
    #2 Καμπύλες ακρίβειας για training και validation set.
    plt.figure() 
    plt.plot(fit.history['accuracy'], label='Training')
    plt.plot(fit.history['val_accuracy'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f"Model's performance with minibatch = {b}")
    plt.show()
    name = "performance_batch_"+str(b)+".png"
    full_path = fullpath = os.path.join(path, name)
    plt.savefig(full_path)
    plt.close()
    #2 Καμπύλες κόστους για training και validation set.
    plt.figure() 
    plt.plot(fit.history['loss'], label='Training')
    plt.plot(fit.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"Model's learning curve with minibatch = {b}")
    plt.show()  
    name = "learnCurve_batch_"+str(b)+".png"
    full_path = fullpath = os.path.join(path, name)
    plt.savefig(full_path)    
    plt.close()    


# Μοντέλο με RMSProp optimizer με learning_rate = 0.001 και ρ = 0.01 ή 0.99

# In[7]:


# Run default model first
r = [0.01, 0.99]
for rho in r:
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho = rho)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    fit = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=validation_split, verbose=1)
    
    #2 Καμπύλες ακρίβειας για training και validation set.
    plt.figure() 
    plt.plot(fit.history['accuracy'], label='Training')
    plt.plot(fit.history['val_accuracy'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')    
    plt.legend()
    plt.title(f"Model's performance with rho = {rho} and RMSProp optimizer with learning rate = 0.001")
    plt.show()
    name = "performance_rho_"+str(rho)+"_RMSProp_learnRate_0_001.png"
    full_path = fullpath = os.path.join(path, name)
    plt.savefig(full_path)  
    plt.close()
    #2 Καμπύλες κόστους για training και validation set.
    plt.figure() 
    plt.plot(fit.history['loss'], label='Training')
    plt.plot(fit.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')      
    plt.legend()
    plt.title(f"Model's learning curve with rho = {rho} and RMSProp optimizer with learning rate = 0.001")
    plt.show() 
    name = "learnCurve_rho_"+str(rho)+"_RMSProp_learnRate_0_001.png"
    full_path = fullpath = os.path.join(path, name)
    plt.savefig(full_path) 
    plt.close()


# Μοντέλο με SGD optimizer με learning_rate = 0.01 και αρχικοποίηση των συναπτικών βαρών W κάθε στρώματος με βάση μια κανονική καταανομή με μ.ο. 10.

# In[8]:


optimizer = keras.optimizers.SGD(learning_rate=0.01)
initializer = keras.initializers.RandomNormal(mean=10)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer=initializer))
model.add(keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=initializer))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
fit = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=validation_split, verbose=1)

#2 Καμπύλες ακρίβειας για training και validation set.
plt.figure() 
plt.plot(fit.history['accuracy'], label='Training')
plt.plot(fit.history['val_accuracy'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')  
plt.legend()
plt.title(f"Model's performance with kernel initializer = (Random Normal with mean 10) and SGD optimizer with learning rate = 0.01")
plt.show()
name = "performance_kernelInit_RandomNorm_10_SGDOpt_learnRate_0_01.png"
full_path = fullpath = os.path.join(path, name)
plt.savefig(full_path) 
plt.close()
#2 Καμπύλες κόστους για training και validation set.
plt.figure() 
plt.plot(fit.history['loss'], label='Training')
plt.plot(fit.history['val_loss'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')  
plt.legend()
plt.title(f"Model's learning curve with kernel initializer = (Random Normal with mean 10) and SGD optimizer with learning rate = 0.01")
plt.show() 
name = "learnCurve_kernelInit_RandomNorm_10_SGDOpt_learnRate_0_01.png"
full_path = fullpath = os.path.join(path, name)
plt.savefig(full_path) 
plt.close()


# Ίδιες επιλογές με παραπάνω. Επιπλέον προσθήκη κανονικοποίησης με L2-νόρμα για τα συναπτικά βάρη κάθε στρώματος με παράμετρο κανονικοποίησης 
# α = 0.1 ή 0.01 ή 0.001
# Τρία μοντέλα με διαφορετικά batch μεγέθη, δύο μοντέλα με RMSProp optimizer και ένα με SGD optimizer.

# In[9]:


a = [0.1, 0.01, 0.001]
for a in a:
    # Models with different batch sizes
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=l2(a)))
    model.add(keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=l2(a)))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(loss=loss, metrics=metrics)
    for b in batch_size:
        start = time.time()
        fit = model.fit(x_train, y_train, batch_size=b, epochs=epochs, validation_split=validation_split, verbose=1)
        training_time = time.time()-start

        #1 ταχύτητα εκπαίδευσης για τις τρείς διαφορετικές μεθόδους
        print(f"Training time with minibatch = {b} and regularizer l2 with a = {a}: {training_time}") 
        #2 Καμπύλες ακρίβειας για training και validation set.
        plt.figure() 
        plt.plot(fit.history['accuracy'], label='Training')
        plt.plot(fit.history['val_accuracy'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')         
        plt.legend()
        plt.title(f"Model's performance with minibatch = {b} and regularizer l2 with a = {a}")
        plt.show()
        name = "performance_batch_"+str(b)+"_regularizer_l2_"+str(a)+".png"
        full_path = fullpath = os.path.join(path, name)
        plt.savefig(full_path) 
        plt.close()
        #2 Καμπύλες κόστους για training και validation set.
        plt.figure() 
        plt.plot(fit.history['loss'], label='Training')
        plt.plot(fit.history['val_loss'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')            
        plt.legend()
        plt.title(f"Model's learning curve with minibatch = {b} and regularizer l2 with a = {a}")
        plt.show() 
        name = "learnCurve_batch_"+str(b)+"_regularizer_l2_"+str(a)+".png"
        full_path = fullpath = os.path.join(path, name)
        plt.savefig(full_path)         
        plt.close()        
        
    # Models with RMSProp optimizers
    for rho in r:
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho = rho)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        fit = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=validation_split, verbose=1)

        #2 Καμπύλες ακρίβειας για training και validation set.
        plt.figure() 
        plt.plot(fit.history['accuracy'], label='Training')
        plt.plot(fit.history['val_accuracy'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')          
        plt.legend()
        plt.title(f"Model's performance with rho = {rho}, RMSProp optimizer with learning rate = 0.001 and regularizer l2 with a = {a}")
        plt.show()
        name = "performance_rho_"+str(rho)+"_RMSProp_learnRate_0_001_regulizer_l2_"+str(a)+".png"
        full_path = fullpath = os.path.join(path, name)
        plt.savefig(full_path)         
        plt.close()        
        #2 Καμπύλες κόστους για training και validation set.
        plt.figure() 
        plt.plot(fit.history['loss'], label='Training')
        plt.plot(fit.history['val_loss'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')          
        plt.legend()
        plt.title(f"Model's learning curve with rho = {rho}, RMSProp optimizer with learning rate = 0.001 and regularizer l2 with a = {a}")
        plt.show() 
        name = "learnCurve_rho_"+str(rho)+"_RMSProp_learnRate_0_001_regulizer_l2_"+str(a)+".png"
        full_path = fullpath = os.path.join(path, name)
        plt.savefig(full_path)         
        plt.close()         
    
    # Models with SGD optimizers
    optimizer = keras.optimizers.SGD(learning_rate=0.01)
    initializer = keras.initializers.RandomNormal(mean=10)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer=initializer, kernel_regularizer=l2(a)))
    model.add(keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=initializer, kernel_regularizer=l2(a)))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    fit = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=validation_split, verbose=1)

    #2 Καμπύλες ακρίβειας για training και validation set.
    plt.figure() 
    plt.plot(fit.history['accuracy'], label='Training')
    plt.plot(fit.history['val_accuracy'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')       
    plt.legend()
    plt.title(f"Model's performance with kernel initializer = (Random Normal with mean 10), SGD optimizer with learning rate = 0.01 and regularizer l2 with a = {a}")
    plt.show()
    name = "performance_kernelInit_RandomNorm_10_SGDOpt_learnRate_0_01_regulizer_l2_"+str(a)+".png"
    full_path = fullpath = os.path.join(path, name)
    plt.savefig(full_path)
    plt.close()
    #2 Καμπύλες κόστους για training και validation set.
    plt.figure() 
    plt.plot(fit.history['loss'], label='Training')
    plt.plot(fit.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')      
    plt.legend()
    plt.title(f"Model's learning curve with kernel initializer = (Random Normal with mean 10), SGD optimizer with learning rate = 0.01 and regularizer l2 with a = {a}")
    plt.show() 
    name = "learnCurve_kernelInit_RandomNorm_10_SGDOpt_learnRate_0_01_regulizer_l2_"+str(a)+".png"
    full_path = fullpath = os.path.join(path, name)
    plt.savefig(full_path)    
    plt.close()


# Κανονικοποίηση με L1-νόρμα για τα συναπτικά βάρη των στρωμάτων του δικτύου (α = 0.01) και ταυτόχρονη χρήση dropout με dropout probability = 0.3

# In[10]:


dropout_prob = 0.3
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=l1(0.01)))
model.add(keras.layers.Dropout(dropout_prob))
model.add(keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=l1(0.01)))
model.add(keras.layers.Dropout(dropout_prob))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(loss=loss, metrics=metrics)
fit = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=validation_split, verbose=1)

#2 Καμπύλες ακρίβειας για training και validation set.
plt.figure() 
plt.plot(fit.history['accuracy'], label='Training')
plt.plot(fit.history['val_accuracy'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy') 
plt.legend()
plt.title(f"Model's performance with regularizer l2 with a = 0.01 and dropout probability = 0.3")
plt.show()
name = "performance_regulizer_l1_0_01_drop_prob_0_3.png"
full_path = fullpath = os.path.join(path, name)
plt.savefig(full_path)    
#2 Καμπύλες κόστους για training και validation set.
plt.figure() 
plt.plot(fit.history['loss'], label='Training')
plt.plot(fit.history['val_loss'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.legend()
plt.title(f"Model's learning curve with regularizer l2 with a = 0.01 and dropout probability = 0.3")
plt.show() 
name = "learnCurve_regulizer_l1_0_01_drop_prob_0_3.png"
full_path = fullpath = os.path.join(path, name)
plt.savefig(full_path)  
plt.close()


# Fine Tuning

# Συνάρτηση η οποία δημιουργεί και επιστρέφει ένα Keras μοντέλο.

# In[11]:


# For F-measure
def build_model(hp):
    ''' Documentation on https://keras.io/keras_tuner/ '''
    n_h1 = hp.Choice('hidden1', [64, 128])
    n_h2 = hp.Choice('hidden2', [256, 512])
    a = hp.Choice('a', [0.1, 0.001, 0.000001])
    learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001])
    
    Heinitializer = keras.initializers.HeNormal()
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    metrics = [tfa.metrics.F1Score(10, 'macro')] # num_classes=10, ValueError=macro ## more on macro from https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(n_h1, activation=tf.nn.relu, kernel_regularizer=l2(a),
             kernel_initializer=Heinitializer))
    model.add(keras.layers.Dense(n_h2, activation=tf.nn.relu, kernel_regularizer=l2(a),
             kernel_initializer=Heinitializer))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model
    


# Αρχικοποίηση ενός tuner 

# In[12]:


epochs = 1000
objective=kt.Objective('val_f1_score', direction = 'max')

tuner = kt.Hyperband(build_model, objective=objective, max_epochs=epochs)


# Για επιτάχυνση της διαδικασίας χρησιμοποιείται η μέθοδος Early Stopping, χωρίς το βήμα της μετέπειτα επανεκπαίδευσης του δικτύου, με patience = 200 για κάθε ένα από τα εξεταζόμενα μοντέλα.

# In[13]:


early_stopping = [keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', patience=200, verbose=1)]


# Αναζήτηση βέλτιστων παραμέτρων

# In[14]:


tuner.search(x_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=early_stopping)
best_params = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Optimal number of neurons in 1st layer: {best_params.get('hidden1')}")
print(f"Optimal number of neurons in 2nd layer: {best_params.get('hidden2')}")
print(f"Optimal value of a for L2-regularization: {best_params.get('a')}")
print(f"Optimal value of learning rate: {best_params.get('learning_rate')}")


# Εκπαίδευση του βέλτιστου Μοντέλου

# In[15]:


model = tuner.hypermodel.build(best_params)
fit = model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=early_stopping)

plt.figure()
plt.plot(fit.history['loss'],label='Training')
plt.plot(fit.history['val_loss'],label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.legend()
plt.title("Best Model's Learning curves of training and validation sets")
plt.show()
name = "learnCurve_BestModel.png"
full_path = fullpath = os.path.join(path, name)
plt.savefig(full_path)  
plt.close()


# Τελική αξιολόγηση γίνεται στο testing υποσύνολο

# In[16]:


y_pred = model.predict(x_test, verbose=1)
y_pred_b = np.argmax(y_pred, axis=1)


conf_matrix = confusion_matrix(y_test_r, y_pred_b)
conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0,1,2,3,4,5,6,7,8,9])
conf_matrix_display.plot(cmap=plt.cm.Blues)
plt.show

print(classification_report(y_test_r, y_pred_b))

