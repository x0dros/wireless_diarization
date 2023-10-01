#!/usr/bin/env python
# coding: utf-8
# %%
###
# This script trains and evaluates UWDNet as described in the paper:
# Unsupervised Wireless Diarization: A Potential New Attack on Encrypted Wireless Networks, presented in ICC 2023
###

# %%
#import UWDNet training constants 
from UWDNet_config import seed_val, dataset_location

# %%
#evaluate for repeatable experiments [Script can be run with multiple seeds for the reported average in paper]
import os
os.environ['PYTHONHASHSEED']         =str(seed_val)
os.environ['TF_CUDNN_DETERMINISTIC']  ='1'
import numpy as np
np.random.seed(seed=seed_val)
import tensorflow as tf
try:
    tf.random.set_random_seed(seed_val)
except:
    tf.random.set_seed(seed_val)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random
random.seed(seed_val)

# %%
#import other standard libraries
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Bidirectional, Conv1D, MaxPool1D, Flatten, Lambda
from tensorflow.keras.activations import softmax, relu
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import f1_score, fbeta_score, precision_recall_curve, confusion_matrix, accuracy_score, adjusted_rand_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

# %%
#specialized functions for UWDNet training
from UWDNet_helper_functions import get_mgntd, scale_data, time_in_h_m_s, make_pairs, dist_output_shape, compute_accuracy,\
                                        get_npkt_nlbl, merge_labels, tune_threshold_valset, tune_hdbscan, return_optimal_indexing


# %%
def reset_seeds():
    '''
        Reset the random generator's seed to the chosen seed value.
        Note: UWDNet is evaluated for multiple random seeds 
    '''
    os.environ['PYTHONHASHSEED']=str(seed_val)
    np.random.seed(seed=seed_val)
    try:
        tf.random.set_random_seed(seed_val)
    except:
        tf.random.set_seed(seed_val)
    random.seed(seed_val)

def make_base_cnnet(input_shape):
    '''
        Create the base neural network model for the Siamese neural network that is a part of UWDNet.
        Input:  
                input_shape: tuple:  dimensions of the input data
                
        Output:
                model: Keras object: NN model 
    '''
    reset_seeds() 
    input        = keras.Input(shape=input_shape)
    cnn_1        = Conv1D(80,5,activation='relu')(input)
    drop1        = Dropout(rate=0.2)(cnn_1)
    cnn_2        = Conv1D(80,5,activation='relu')(drop1)
    max_pool_cnn = MaxPool1D()(cnn_2)
    flat_lyr     = Flatten()(max_pool_cnn)
    drop2        = Dropout(rate=0.2)(flat_lyr)
    dense1_cnn   = Dense(256, activation='relu')(drop2)
    drop3        = Dropout(rate=0.2)(dense1_cnn)
    dense3_cnn   = Dense(128, activation='relu', name='emb_128')(drop3)
    model        = keras.Model(inputs=input,outputs=dense3_cnn)
    return model  


def euclid_dis(vects):
    '''
        Calculate the Euclidean distance between two embeddings.
        Input:  
                vects: list of embeddings
                
        Output:
                float: distance between the embeddings
    '''
    x,y = vects
    sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def accuracy(y_true, y_pred):
    '''
    Compute classification accuracy for given true label and predicted label probabilities at fixed threshold of 0.5
    Input:
        y_true: tf array: array of true labels
        y_pred: tf array: array of predicted labels (probabilities)
    Output:
        accuracy: float: resulting accuracy 
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
    '''
        Calculate the constasitive loss based on how far inputs are from each other
        Input:  
                y_true: tf array: array of true labels
                y_pred: tf array: array of predicted labels (probabilities)
                
        Output:
                float: loss
    '''
    y_true        =tf.dtypes.cast(y_true, tf.float32)
    y_pred        =tf.dtypes.cast(y_pred, tf.float32)
    margin        = 1 #margin of 1 as is customarily used
    square_pred   = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square) 


# %%
# Flow of processing: load and prepare data, then evaluate in cross-validation

#get data
sim_data    = np.load(dataset_location, allow_pickle=True)
lst         = sim_data.files
data_mat    = sim_data['arr_0']
all_lab_str = sim_data['arr_2'] 
all_label   = sim_data['arr_1']
print("Shape of data = {}".format(data_mat.shape) )

#scale data for making packets independent of power
x_scaled = scale_data(data_mat)
print("Shape of x_scaled = {}".format(x_scaled.shape) )

#get magnitude, angle, i, q components of the data
x_magn = get_mgntd(x_scaled)
x_ang  = np.angle(x_scaled)
x_i    = np.real(x_scaled)
x_q    = np.imag(x_scaled)
print("Norm of x_scaled:{}".format(np.linalg.norm(x_scaled, axis=-1)))
x_magn = x_magn[:,:,None]
x_ang  = x_ang[:,:, None]
x_i    = x_i[:,:,None]
x_q    = x_q[:,:,None]

#pool features to data for training/evaluation
all_data      =   np.dstack((x_magn,x_ang))
all_label_int =   np.argmax(all_label,axis=1)
n_class       =  len(np.unique(all_label_int))
print("n_class = {}".format(n_class))
print("Shapes of all_label and all_data: {} and {}".format(all_label.shape, all_data.shape))


# %%
from UWDNet_config import sel_affinity, sel_linkage, num_group, n_val, batch_size, epochs

# %%
#create directory to save intermediate/checkpoint files for DL model, if one does not exists
save_dir = r'model_file'
if(not os.path.exists(save_dir)):
    os.mkdir(save_dir)

# %%
# ### Evaluate UWDNet with 5-fold cross-validation 
t_strt_all = time.time()

#to store cv results
cv_test_res = []

# AHC clustering definition.
clust_model = AgglomerativeClustering(n_clusters=None,affinity=sel_affinity,linkage=sel_linkage)

# first definition of the CNN and saving the weights
reset_seeds()
print("attempting to call base_net")
base_cnn_wt = make_base_cnnet(all_data.shape[1:])
base_cnn_wt.save_weights(os.path.join(save_dir,"base_cnn_wt.tf"),save_format='tf')

#group k-fold cross validation
gkfold = GroupKFold(n_splits=num_group)

#counter 
cv_idx = 0
for train_idx_, test_idx in gkfold.split(all_data,all_label_int,all_label_int):

    #get the train and test split. validation set will be obtained from a portion of train set
    #(reserved set not used for training)
    #******************* Start of train/test/val data prepare
    train_data_,  test_data           = all_data[train_idx_,:,:], all_data[test_idx,:,:]
    train_label_, test_label          = all_label[train_idx_,:], all_label[test_idx,:]
    train_label_str_, test_label_str  = all_lab_str[train_idx_], all_lab_str[test_idx]

    test_set        = np.unique(np.argmax(test_label,axis=1))   
    train_set_      = np.unique(np.argmax(train_label_,axis=1))
    test_idx_       = np.unique(test_label_str, return_index=True)[1]
    test_set_str    = np.array([test_label_str[idx] for idx in sorted(test_idx_)])
    train_idx_      = np.unique(train_label_str_, return_index=True)[1] 
    train_set_str_  = np.array([train_label_str_[idx] for idx in sorted(train_idx_)])

    val_set     = train_set_[-n_val:]
    val_set_str = train_set_str_[-n_val:]
    train_set   = train_set_[:-n_val]
    train_set_str = train_set_str_[:-n_val]
    print("Test set: {}".format(test_set))
    print("Shapes of sets: train_set= {}, val_set = {}, test_set = {}".format(
                                                           train_set.shape, val_set.shape, test_set.shape))
    
    val_idx                                  = np.where(pd.Series(all_label_int).isin(val_set))[0]
    train_idx                                = np.where(pd.Series(all_label_int).isin(train_set))[0]
    val_data, val_label, val_label_str       = all_data[val_idx,:], all_label[val_idx,:], all_lab_str[val_idx]
    train_data, train_label, train_label_str = all_data[train_idx,:], all_label[train_idx,:], all_lab_str[train_idx]

    train_lbl_i = np.argmax(train_label,axis=1)  
    val_lbl_i  = np.argmax(val_label,axis=1)
    test_lbl_i = np.argmax(test_label,axis=1)
    # call get_npkt_nlbl:
    n_pkt_trn, n_label_trn = get_npkt_nlbl(train_data, np.argmax(train_label, axis=1))
    n_pkt_val, n_label_val = get_npkt_nlbl(val_data, np.argmax(val_label, axis=1))
    n_pkt_tst, n_label_tst = get_npkt_nlbl(test_data, np.argmax(test_label, axis=1))
    print("n_pkt_trn= {}, n_label_trn={} , n_pkt_val= {}, n_label_val= {}, \
        n_pkt_tst= {}, n_label_tst= {}\n".format(n_pkt_trn, n_label_trn,
                                                  n_pkt_val, n_label_val,
                                                  n_pkt_tst, n_label_tst ))

     
    assert(len(np.intersect1d(train_idx,val_idx))==0)
    assert(len(np.intersect1d(test_idx,val_idx))==0)
    assert(len(np.intersect1d(train_idx,test_idx))==0)
    print("Shapes of training data: {}, {}".format(train_data.shape, train_label.shape))
    print("Shapes of validation data: {}, {}".format(val_data.shape, val_label.shape))
    print("Shapes of test data: {}, {}".format(test_data.shape, test_label.shape))
    
    #******************* End of train/test/val data prepare

    #### Create pair of data for training contrastive models
    reset_seeds()
    tr_pairs, tr_y    = make_pairs(train_data, train_label)
    reset_seeds()
    val_pairs, val_y  = make_pairs(val_data, val_label)
    reset_seeds()
    te_pairs, te_y    = make_pairs(test_data, test_label)

    print("Shapes of training pairs: {}, {}".format(tr_pairs.shape, tr_y.shape))
    print("Shapes of validation pairs: {}, {}".format(val_pairs.shape, val_y.shape))
    print("Shapes of test pairs: {}, {}".format(te_pairs.shape, te_y.shape))

    print("Shapes of training pairs[:,0/1]: {}, {}".format(tr_pairs[:,0].shape, tr_pairs[:,1].shape))
    print("Shapes of validation pairs[:,0/1]: {}, {}".format(val_pairs[:,0].shape, val_pairs[:,1].shape))
    print("Shapes of test pairs[:,0/1]: {}, {}".format(te_pairs[:,0].shape, te_pairs[:,1].shape))

    # base CNN:
    reset_seeds()
    base_cnn = make_base_cnnet(train_data.shape[1:])
    base_cnn.load_weights(os.path.join(save_dir,'base_cnn_wt.tf'))
    base_cnn.summary()    

    #setup siamese training
    input_a = keras.Input(shape=train_data.shape[1:])
    input_b = keras.Input(shape=train_data.shape[1:])

    processed_a         = base_cnn(input_a)
    processed_b         = base_cnn(input_b)
    distance            = Lambda(euclid_dis, output_shape=dist_output_shape)([processed_a, processed_b])
    siamese_model       = keras.Model([input_a, input_b], distance)

    #train the siamese network
    optimizer = keras.optimizers.Adam(lr=1e-4)
    siamese_model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])
    cm_file   =  os.path.join(save_dir,'best_model_plda_cvfold_'+str(cv_idx)+'.tf')
    print(cm_file)
    checkpoint = ModelCheckpoint(cm_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
    es         = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    siamese_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=([val_pairs[:, 0], val_pairs[:, 1]],val_y),
                      callbacks =[checkpoint,es])

    #evaluate on the test set pairs
    siamese_model = load_model(cm_file, custom_objects={'contrastive_loss': contrastive_loss}) 
    print('== Evaluating the trained model in the test pairs ====')
    y_pred_te = siamese_model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc    = compute_accuracy(te_y, y_pred_te)
    print("Accuracy:Test set:{}".format(te_acc))

    ##### From the trained model, extrack embeddings based on which unsupervised clustering will be done
    ### cosine distance between the embeddings are considered for clustering
    
    ### **************** Start of embedding extraction
    # Embedding model:
    trned_model = siamese_model.layers[2]
    emb_model   = keras.Model(inputs=trned_model.input,
                              outputs=trned_model.get_layer('emb_128').output) 
    test_data_emb   = test_data[:,:,:]
    test_emb        = emb_model.predict(test_data_emb)
    test_emb_cosine = cosine_distances(test_emb) 
    #obtain the embedding for the training set and validation set
    train_emb        = emb_model.predict(train_data)
    train_emb_cosine = cosine_distances(train_emb) 

    val_emb        = emb_model.predict(val_data)
    val_emb_cosine = cosine_distances(val_emb) 

    print("Embeddings for Cosine distances.")
    train_emb_dist = train_emb_cosine
    test_emb_dist  = test_emb_cosine           
    val_emb_dist   = val_emb_cosine
    
    ### **************** End of embedding extraction    

    #### The threshold for clustering the inter-embedding distance is tuned in validation set
    
    #obtain best clustering thresholds in the validation set
    val_label_int      = np.argmax(val_label,axis=1)
    best_thr_val_accu  = tune_threshold_valset(val_emb_dist,val_label_int,sel_linkage)
    #we also test with hdbscan that works better but requires balanced classes
    best_minpt_val_acc = tune_hdbscan(n_pkt_val,n_label_val, val_emb_dist, val_label_int)
    
    #use the threshold on optimal error count to compute rand index in the test set

    #use the threshold on optimal accuracy to compute rand index in the test set
    print("AHC on test set (opt_accu)...")
    clust_model.distance_threshold = best_thr_val_accu
    test_clustering_optaccu = AgglomerativeClustering(n_clusters=None,affinity='precomputed',linkage=sel_linkage,
                         distance_threshold=best_thr_val_accu).fit_predict(test_emb_dist)
    print("HDBSCAN on test set (opt_accu)...")
    test_adjrand_index_optaccu = adjusted_rand_score(test_lbl_i,test_clustering_optaccu)    
    test_clstr_minpt_optacc =  hdbscan.HDBSCAN(min_cluster_size= int(best_minpt_val_acc),
                         metric='precomputed').fit_predict(test_emb_dist.astype(np.float64))
    test_adjrand_idx_hdbscan_optacc = adjusted_rand_score(test_lbl_i,test_clstr_minpt_optacc)
    #obtian optimal mapping
    test_clustering_optaccu_mapped = return_optimal_indexing(test_lbl_i,test_clustering_optaccu)
    clf_accu_optaccu               = accuracy_score(test_lbl_i,test_clustering_optaccu_mapped)
    test_adjrand_idx_hdbscan_optacc_mapped = return_optimal_indexing(test_lbl_i,test_clstr_minpt_optacc)
    clf_accu_hdbscan_optaccu               = accuracy_score(test_lbl_i,test_adjrand_idx_hdbscan_optacc_mapped) 

    conf_mat_AHC = confusion_matrix(test_lbl_i,
                                    test_clustering_optaccu_mapped, labels = test_set)
    conf_mat_HDBSCAN = confusion_matrix(test_lbl_i,
                                        test_adjrand_idx_hdbscan_optacc_mapped, labels = test_set)

    test_label_int             = np.argmax(test_label,axis=1)
    true_numnodes              = len(np.unique(test_label_int))
    detected_numnodes_optaccu  = len(np.unique(test_clustering_optaccu)) 

    cv_test_res.append({'cv_idx':cv_idx,'test_accu':te_acc,
               'test_set':test_set,
               'true_numnodes_test':true_numnodes,
                'det_numnodes_test': detected_numnodes_optaccu,
               'test_adjrand_idx_hdbscan_optacc':test_adjrand_idx_hdbscan_optacc,
               'test_adjrand_indx_optaccu':test_adjrand_index_optaccu,
               'clf_accu_hdbscan_optaccu':clf_accu_hdbscan_optaccu,
               'test_diar_accu_optaccu':clf_accu_optaccu,
               'best_minpt_val_acc':best_minpt_val_acc,
               'best_thr_val_accu':best_thr_val_accu})

    print(cv_test_res)
    cv_idx = cv_idx + 1
    
    break

# %%
cv_test_res_pd = pd.DataFrame(cv_test_res)


# %%
print(cv_test_res_pd)

# %%
print(cv_test_res_pd.mean())

# %%
t_end_all = time.time()
t_scrpt = time_in_h_m_s(t_end_all - t_strt_all)
print("Done! Script duration: {}".format(t_scrpt))


# %%
