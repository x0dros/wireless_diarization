# +
###
# This file contains configuration setting for UWDNet training
###
# -

dataset_location = '/home/cb56/data_sets/real_data/falcon_lts-qpsk_1000_frames_02_22.npz'

seed_val = 19
epochs   = 20
n_pair   = 40

#UWDNet training configuration 
sel_affinity = 'cosine'
sel_linkage  = 'average'
n_val        = 5
num_group    = 5
batch_size   = 64
