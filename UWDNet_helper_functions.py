# +
###
# This file contains helper functions for training and evaluate UWDNet model
###
# -

import numpy as np
import math
from UWDNet_config import seed_val, n_pair

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment as linear_assignment

import hdbscan


# +
def get_mgntd(x):
    '''
        Get the magnitute of a complex array
        Input:  
                x: numpy array
                
        Output:
                mgnt: float: magnitude 
    '''
    mgnt = np.sqrt(np.abs(x)**2)
    return mgnt


def scale_data(x):
    '''
        Scale the input array so that it's norm is the same as the number of elements
        of the input.
        Input:  
                x: numpy array        
        Output:
                x_scaled: numpy array: scaled input
    '''
    x_nrm = np.linalg.norm(x,axis=-1)
    x_scaled = np.sqrt(x.shape[1])*(x.T/x_nrm).T
    return x_scaled

def time_in_h_m_s(t):
    '''
        Get the current time in hours minutes and seconds
        Input:  
                t: time object
                
        Output:
                t_str: string: time in hms format 
    '''

    h = math.floor(t/3600)
    m = math.floor((t - h*3600)/60)
    s = t - (h*3600 + m*60)
    t_str = str(h) + 'h' + str(m) + 'm'+ str(s)+'s'
    return t_str


def make_pairs(x, one_hot_lbls):
    '''
        Create random pairs of the input array entries
        Input:  
                x: input data: numpy array
                one_hot_lbsl:  labels in one hot code format: numpy array 
                
        Output:
                pairs: numpy array
		labels: numpy array 
    '''
    np.random.seed(seed_val)
    pairs  = []
    labels = []
    # make indices:
    lbls        = np.argmax(one_hot_lbls,axis=1)
    num_classes = len(np.unique(lbls))
    idx = [np.where(lbls == i)[0] for i in np.sort(np.unique(lbls))]
    n        =min([len(idx[d]) for d in range(num_classes)])
    num_run = n_pair
    for i_run in range(num_run):
        for d in range(num_classes):
            for i in range(n-1):
                #generate pairs from same class
                z1, z2 = idx[d][np.random.choice(n-1)], idx[d][np.random.choice(n-1)]
                pairs += [[x[z1], x[z2]]]
                dn = np.random.choice(np.setdiff1d(range(num_classes),d))
                z1, z2 = idx[d][np.random.choice(n-1)], idx[dn][np.random.choice(n-1)]
                pairs += [[x[z1], x[z2]]]
                labels += [1,0]
    return np.array(pairs), np.array(labels)



# +
def dist_output_shape(shapes):
    '''
        Get the shape of the input
        Input:  
                shapes: list of shapes
                
        Output:
                array: shape of the first item in the input
    '''
    shape1, shape2 = shapes
    return (shape1[0], 1)


def compute_accuracy(y_true, y_pred):
    '''
        Compute classification accuracy with a fixed threshold on distances.
        Input:  
                y_true: tf array: true lables
		y_pred: tf array: predicted labels
                
        Output:
                float: accuracy
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def get_npkt_nlbl(d_set, labels):
    '''
        Get the number of packets per label
        Input:  
                d_set: numpy array: data
                labels: numpy array: labels
                
        Output:
                npkt: numpy aray: number of packets per label
		nlbl: int: number of unique labels
    '''
    nlbl = len(np.unique(labels))
    idx  = [np.where(labels == i)[0] for i in np.sort(np.unique(labels))]
    npkt = min([len(idx[d]) for d in range(nlbl)])
    return np.array(npkt), nlbl

# combine all the string labels and output new one-hot-code labels
def merge_labels(labels_lst, data_lst):
    '''
        Combine all the string labels from different data sets and output new one-hot-code labels
        Input:  
                labels_lst: list of numpy arrays: labels of different datasets
                data_lst: list of numpy arrays: different datasets
                
        Output:
                all_data: numpy array: combined dataset
		all_labels_str: numpy array: combined string labels
		lbl_ohc: numpy array: combined one hot code labels
    '''
    n_sets = len(labels_lst)
    all_labels_str = []
    all_data = data_lst[0]
    cntr = 0
    for il in range(n_sets):
        all_labels_str = all_labels_str + labels_lst[il]
        if 0 < cntr < n_sets:
            all_data = np.vstack((all_data, data_lst[il])) 
        cntr =  cntr +1
    all_labels_str = np.array(all_labels_str)
    
    indexes = np.unique(all_labels_str, return_index=True)[1]
    nodes_ = [all_labels_str[index] for index in sorted(indexes)]
    #nodes_ = np.unique(all_labels_str)
    n_node = len(nodes_)
    all_labels_str = np.array(all_labels_str, dtype=object)
    
    lbl_ohc = np.zeros([all_labels_str.shape[0],n_node])
    print("Nodes:\n{}".format(nodes_))
    for j in range(n_node):
        idx_j = np.argwhere(all_labels_str == nodes_[j])
        lbl_ohc[idx_j, j] = 1
        print("j = {}, nodes_[j]={}, idx_j[0]={}".format(j, nodes_[j], idx_j[0]))

    return all_data, all_labels_str,lbl_ohc



# -

def return_optimal_indexing(true_node,cluster_node):
    '''
    Given a set of true label and predicted cluster labels, find the optimal mapping from unsupervised predicted cluster
    label to true labels
    
    Input:
        true_node: numpy array: 
        cluster_node: numpy array: predicted label values
        
    Output:
        cluster_node_mapped: numpy array: mapped value of predicted cluster labels
    
    '''
    
    true_node_unique    = np.sort(np.unique(true_node))
    cluster_node_unique = np.sort(np.unique(cluster_node))

    true_node_dict    = dict(zip(true_node_unique,
                                    range(len(true_node_unique))))
    cluster_node_dict = dict(zip(cluster_node_unique,
                                    range(len(cluster_node_unique))))
    #reverse look-up dictionary
    true_node_dict_rev    = dict(zip(range(len(true_node_unique)),true_node_unique))
    cluster_node_dict_rev = dict(zip(range(len(cluster_node_unique)),cluster_node_unique))   

    true_node_indexed = np.array([true_node_dict[x] for x in true_node])
    cluster_node_indexed = np.array([cluster_node_dict[x] for x in cluster_node])

    confu_mat      = confusion_matrix(true_node_indexed,cluster_node_indexed)

    #linear assignment minimizes a cost. let's invert confusion matrix so that
    #minimization of it would lead to maximization of confusion matrix
    confu_mat_cost = np.max(confu_mat) - confu_mat
    assign_indices = np.array(list(zip(*linear_assignment(confu_mat_cost))))
    map_index      = assign_indices[:,1]
    true_spkr_index= assign_indices[:,0]

    cluster_node_map = {}
    for true_idx, map_idx in zip(true_spkr_index,map_index):
        #we need to map only for the nodes in cluster_node_unique
        if(map_idx < len(cluster_node_unique)):
            cluster_nd = cluster_node_dict_rev[map_idx]    
            if(true_idx < len(true_node_unique)):
                mapped_true_node = true_node_dict_rev[true_idx] 
            else:
                mapped_true_node = cluster_nd                
            cluster_node_map[cluster_nd] = mapped_true_node

    cluster_node_mapped = np.array([cluster_node_map[x] for x in cluster_node])
    return cluster_node_mapped


# +
def return_best_threshold(thr_sweep,val_emb_dist,val_label_int,sel_linkage):
    '''
    Find best threshold to cluster data in the validation set for the AHC model
    
        Input:  
                thr_sweep: numpy array :range of values to scan over for the best threshold
                val_emb_dist: numpy array: pairwise distances from each node in the validation set to every other node in the set 
        	val_label_int: numpy array: lables in integer (transformed from one hot codes) 
        	sel_linkage: string: kind of linkage to be used in AHC

        Output:
        	best_thr_train: float: threshold yielding the highest accuracy
        	thr_sweep_res_pd: dataframe: contains threshold values, yielded accuracy, error
    '''

    thr_sweep_res = []

    for each_thr in thr_sweep:
        clustering = AgglomerativeClustering(n_clusters=None,affinity='precomputed',
                                        linkage=sel_linkage,distance_threshold=each_thr).fit_predict(val_emb_dist)
        clustering_mapped = return_optimal_indexing(val_label_int,clustering)

        curr_accu = accuracy_score(val_label_int,clustering_mapped.squeeze())
        curr_num  = len(np.unique(clustering))
        true_num  = len(np.unique(val_label_int))
        err_count = np.abs(true_num-curr_num)

        thr_sweep_res.append({'thr':each_thr,'accu':curr_accu,'err_count':err_count})
        print('computed for threshold: {0}'.format(each_thr))

    thr_sweep_res_pd = pd.DataFrame(thr_sweep_res)
    min_error        = thr_sweep_res_pd['err_count'].min()
    max_acc          = thr_sweep_res_pd['accu'].max()

    #sel_thr_sweep_res_pd = thr_sweep_res_pd.sort_values(by='accu').iloc[-1,:].to_frame()
    sel_thr_sweep_res_pd = thr_sweep_res_pd.sort_values(by='accu').iloc[-1,:].to_frame().transpose()

    best_thr_train = sel_thr_sweep_res_pd.thr.values[0]

    return best_thr_train, thr_sweep_res_pd # end of function


def tune_threshold_valset(val_emb_dist,val_label_int,sel_linkage):
    # all these thrsholds 
    best_thr_val_coarse, _ = return_best_threshold(np.arange(0,2,0.1),val_emb_dist,val_label_int,sel_linkage)
    if(best_thr_val_coarse<0.1):
        best_thr_val_coarse = 0.1
    if(best_thr_val_coarse>0.9):
        best_thr_val_coarse = 0.9       
    best_thr_val, thr_sweep_res_pd  = return_best_threshold(np.arange(best_thr_val_coarse-0.1,
                                      best_thr_val_coarse+0.1,0.01),val_emb_dist,val_label_int)
    
    return best_thr_val


# +
def return_best_minpt(pt_range,n_lbl,val_emb_dist,val_label_int):
    ''' Returns value of best HDBSCAN min points to obtain the best accuracy used for paramter tuning in validation set.
	The fnc. turns the percentages into number of points.
   		Input:
			pt_range: numpy array: range of possible min points in percent
			n_lbl : integer: number of labels
			val_emb_dist: numpy array: pairwise distances of the embeddings in the validation set
			val_label_int: numpy array: validation set labels in iteger 
    		Outut:	bst_min_pts_val: integer: min_pts giving the gighest accuracy 
			bst_min_pts_pd: data frame: contains tried min_pts, yielded accuracies, and error count
			'''
    bst_min_pts = []

    for j_pt in pt_range:
        i_pt = int(j_pt)
        clusters_dbs = hdbscan.HDBSCAN(min_cluster_size=i_pt,
                     metric='precomputed').fit_predict(val_emb_dist.astype(np.float64))
        clstr_idx_mapped = return_optimal_indexing(val_label_int,clusters_dbs)

        curr_accu = accuracy_score(val_label_int, clstr_idx_mapped.squeeze())
        curr_num  = len(np.unique(clusters_dbs))
        true_num  = n_lbl #len(np.unique(np.argmax(val_l,axis=1) ))
        err_count = np.abs(true_num-curr_num)

        bst_min_pts.append({'mpt':i_pt,'accu':curr_accu,'err_count':err_count})
        print('computed for i_pt= {}, curr_num = {}, err_count = {}'.format(i_pt, curr_num, err_count))

    bst_min_pts_pd = pd.DataFrame(bst_min_pts)
    min_error        = bst_min_pts_pd['err_count'].min()

    sel_bst_min_pts_pd = bst_min_pts_pd[bst_min_pts_pd.err_count==min_error]

    if(sel_bst_min_pts_pd.shape[0]==1):
        bst_min_pts_val = sel_bst_min_pts_pd.mpt.values[0]
    else:
        bst_min_pts_val = sel_bst_min_pts_pd.loc[sel_bst_min_pts_pd.accu.idxmax()]['mpt']

    return int(bst_min_pts_val), bst_min_pts_pd # end of function



def tune_hdbscan(n_pkt_val,n_label_val,val_emb_dist,val_label_int):
    pt_range_coarse = (n_pkt_val * np.arange(50,91,10)/100).astype(int) # 50% - 90%
    print("pt_range_coarse:{}\n".format(pt_range_coarse))
    best_minpt_val_coarse,_ = return_best_minpt(pt_range_coarse, n_label_val,val_emb_dist,val_label_int)
    print("best_minpt_val_coarse = {}\n".format(best_minpt_val_coarse))
    delta_pt = int(best_minpt_val_coarse * 10/100) # 10 percent of the best minimum points
    incrmnt_pt = int(best_minpt_val_coarse * 1/100) # 1 percent of the minimum points
    if incrmnt_pt == 0:
        incrmnt_pt = 1
    pt_range_fine = (np.arange(best_minpt_val_coarse - delta_pt, best_minpt_val_coarse+delta_pt, incrmnt_pt)).astype(int)
    print("delta_pt = {}, incrmnt_pt = {}, pt_range_fine:{}\n".format(delta_pt, incrmnt_pt, pt_range_fine))


    best_minpt_val, bst_min_pts_pd = return_best_minpt(pt_range_fine, n_label_val,val_emb_dist,val_label_int)
    best_minpt_val_acc              = bst_min_pts_pd.loc[bst_min_pts_pd.accu.idxmax()]['mpt'].astype(int)
    print("best_minpt_val_acc = {}\n".format(best_minpt_val_acc))
    print("AHC on test set (opt_count)...")
    
    return best_minpt_val    
# -


