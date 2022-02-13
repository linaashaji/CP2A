import os
import argparse
import pickle
import glob
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def pad_sequence(inp_list, max_len):
    padded_sequence = []
    for source in inp_list:
        target = np.zeros((max_len, 4))
        source = np.array(source)
        target[:source.shape[0], :] = source
        
        padded_sequence.append(target)
        
    return np.array(padded_sequence)
    
def ex_name(path):
	p = path.replace('.pkl', '/pkl')
	p = p.replace('data_', '')
	p = p.split('/')
	return p[-2]


def balance_dataset(dataset, actions, flip = True):

	print('\n#####################################')
	print('Generating balanced raw data')
	print('#####################################')

	d = dataset[:].copy()
	gt_labels = actions[:].copy()
	num_pos_samples = np.count_nonzero(np.array(gt_labels))
	num_neg_samples = len(gt_labels) - num_pos_samples

	# finds the indices of the samples with larger quantity
	if num_neg_samples == num_pos_samples:
	    print('Positive and negative samples are already balanced')  
	    
	else:
	    print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
	    if num_neg_samples > num_pos_samples:
	        gt_augment = 1
	    else:
	        gt_augment = 0
	        
	    
	    img_width = 1920
	    num_samples = len(dataset)
	    
	    for i in range(num_samples):
	        if gt_labels[i] == gt_augment:
	            seq = d[i].copy()                
	            seq['center_x'] = (seq['bbox_right_x'] - seq['bbox_left_x']) / 2 + seq['bbox_left_x']
	            seq['center_y'] = (seq['bbox_up_y'] - seq['bbox_down_y']) / 2 + seq['bbox_down_y']
	            
	            d[i]['center_x'] = seq['center_x']
	            d[i]['center_y'] = seq['center_y']
	            d[i]['flip_flag'] = [False] * len(d[i])
	            
	            seq_flipped = seq.copy()
	            seq_flipped['center_x'] = img_width - seq['center_x']
	            seq_flipped['bbox_right_x'] = img_width - seq_flipped['bbox_right_x']
	            seq_flipped['bbox_left_x'] = img_width - seq_flipped['bbox_left_x']
	            
	            seq_flipped['flip_flag'] = [True] * len(seq_flipped)
	            d.append(seq_flipped)
	            gt_labels.append(gt_labels[i])
	        else:
	            seq = d[i].copy()
	            d[i]['center_x'] = (seq['bbox_right_x'] - seq['bbox_left_x']) / 2 + seq['bbox_left_x']
	            d[i]['center_y'] = (seq['bbox_up_y'] - seq['bbox_down_y']) / 2 + seq['bbox_down_y']
	            d[i]['flip_flag'] = [False] * len(d[i])
	            
	            
	            
	    num_pos_samples = np.count_nonzero(np.array(gt_labels))
	    num_neg_samples = len(gt_labels) - num_pos_samples
	    
	    if num_neg_samples > num_pos_samples:
	        rm_index = np.where(np.array(gt_labels) == 0)[0]
	    else:
	        rm_index = np.where(np.array(gt_labels) == 1)[0]
	        
	    # Calculate the difference of sample counts
	    dif_samples = abs(num_neg_samples - num_pos_samples)
	    
	    # shuffle the indices
	    np.random.seed(42)
	    np.random.shuffle(rm_index)
	    # reduce the number of indices to the difference
	    rm_index = rm_index[0:dif_samples]
	    
	    # update the data
	    for i in sorted(rm_index, reverse=True):
	        del d[i]
	        del gt_labels[i]
	        
	    num_pos_samples = np.count_nonzero(np.array(gt_labels))
	    print('Balanced:\t Positive: %d  \t Negative: %d\n'
	          % (num_pos_samples, len(d) - num_pos_samples))
	    print('Total Number of samples: %d\n'
	          % (len(d)))
	    
	return d, gt_labels



def tte_dataset(dataset, actions, time_to_event, overlap, obs_length):
	d = dataset[:].copy()
	gt_labels = actions[:].copy()
	to_drop = []
	new_dataset = []
	new_labels = []

	olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
	olap_res = 1 if olap_res < 1 else olap_res

	for ind, seq in enumerate(d):
	    start_idx = len(seq) - obs_length - time_to_event[1]
	    end_idx = len(seq) - obs_length - time_to_event[0]
	    
	    for i in range(start_idx, end_idx, olap_res):
	        new_dataset.append(seq.loc[i:i + obs_length].reset_index())
	        new_labels.append(gt_labels[ind])
	        
	for ind, seq in enumerate(new_dataset):
	    if(len(seq) != obs_length+1):
	        to_drop.append(ind)
	        
	for i in sorted(to_drop, reverse=True):
	    del new_dataset[i] 
	    del new_labels[i]
	    
	return new_dataset, new_labels




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='CP2A_dataset')
	parser.add_argument('--out_dir', default='CP2A_bb_data')
	args = parser.parse_args()

	path_list = glob.glob(f"{args.data_path}/*.pkl")
	if(path_list == []):
		print('path not found')
		exit()

	data = []
	for path in path_list:
	    f = open(path, "rb")
	    set_name = ex_name(path)
	    try:
	        local_data = pd.read_pickle(f)
	        for df in local_data:
	            set_id_cl = [set_name] * len(df)
	            df['set_id'] = set_id_cl
	        data += local_data
	    except:
	        print(f'Path <{path}> Not Found')


	all_data = data[:]
	LEN = 60 + 16
	to_pop = []
	total_len = 0
	for j, df in enumerate(all_data):
	    total_len += len(df)
	    if len(df) <= LEN:
	        to_pop.append(j)
	        continue
	to_pop.reverse()
	for p in to_pop:
	    _ = all_data.pop(p)
	print(f'Total number of pedestrians : {len(data)}')


	to_drop = []
	count = 0
	data_copy = all_data[:]
	for ind, seq in enumerate(data_copy):
	    if int(sum(seq['semantic_label'])) == 7 * len(seq['semantic_label']):
	        #The pedestrian is on the road all the time (There's nothing to predict)
	        to_drop.append(ind)
	    if int(sum(seq['semantic_label'])) == 6 * len(seq['semantic_label']):
	        #The pedestrian is on the road all the time (There's nothing to predict)
	        to_drop.append(ind)
	        
	for i in sorted(to_drop, reverse=True):
	    del data_copy[i]



	to_drop = []
	for ind, seq in enumerate(data_copy):    
	    if seq['semantic_label'][0] == 7 or seq['semantic_label'][0] == 6:
	        #If the observation seq of the ped started on the road, Find the first point on the side walk
	        try:
	            semantic_seq = list(seq['semantic_label'])
	            f_point = semantic_seq.index(next(filter(lambda x: x!=7 and x!=6, semantic_seq)))
	            data_copy[ind] = data_copy[ind].loc[f_point:]
	            data_copy[ind].reset_index()
	        except:
	            to_drop.append(ind)

	for i in sorted(to_drop, reverse=True):
	    del data_copy[i]  



	critical_point_list = list(np.zeros(len(data_copy), dtype=np.int32))
	actions = list(np.zeros(len(data_copy), dtype=np.int32))
	to_drop = []
	for ind, seq in enumerate(data_copy):        
	    #Find the first crossing point in the list
	    try:
	        semantic_seq = list(seq['semantic_label'])
	        cr_point = semantic_seq.index(next(filter(lambda x: x==7 or x==6 or x==16, semantic_seq)))
	        critical_point_list[ind] = cr_point
	        actions[ind] = 1
	        data_copy[ind] = data_copy[ind].loc[:cr_point]
	        data_copy[ind].reset_index()
	        if(len(data_copy[ind]) < 60 + 16):
	           to_drop.append(ind) 
	    except:
	        # The pedestrian have never crossed the road
	        cr_point = len(seq) - 1
	        critical_point_list[ind] = cr_point
	        actions[ind] = 0
	        data_copy[ind] = data_copy[ind].loc[:cr_point]
	        data_copy[ind].reset_index()
	        if(len(data_copy[ind]) < 60 + 16):
	           to_drop.append(ind)  
	        
	for i in sorted(to_drop, reverse=True):
	    del data_copy[i] 
	    del critical_point_list[i]
	    del actions[i]	   
	    

	assert len(data_copy) == len(critical_point_list) == len(actions), f'The length of the data <{len(data_copy)}> is different from the length of the labels <{len(actions)}>' 


	data_copy, actions = shuffle(data_copy, actions)

	train_data, test_data = data_copy[:int(len(data_copy)*0.78)], data_copy[int(len(data_copy)*0.78):] 
	train_actions, test_actions = actions[:int(len(actions)*0.78)], actions[int(len(actions)*0.78):] 

	test_data, val_data = test_data[:int(len(test_data)*0.5)], test_data[int(len(test_data)*0.5):] 
	test_actions, val_actions = test_actions[:int(len(test_actions)*0.5)], test_actions[int(len(test_actions)*0.5):]

	print('Balancing the Training dataset split')
	d_train, gt_train_labels = balance_dataset(train_data, train_actions)


	tte_dataset_train, tte_actions_train = tte_dataset(d_train, gt_train_labels, [30,60], 0.6, 15)
	tte_dataset_val, tte_actions_val = tte_dataset(val_data, val_actions, [30,60], 0, 15)
	tte_dataset_test, tte_actions_test = tte_dataset(test_data, test_actions, [30,60], 0, 15)


	assert len(tte_dataset_train) == len(tte_actions_train), f'The length of the data <{len(tte_dataset_train)}> is different from the length of the labels <{len(tte_actions_train)}>' 


	os.makedirs(args.out_dir, exist_ok=True)
	print(f'Saving the processed data to: {args.out_dir}')
	with open(f"{args.out_dir}/train_data.pkl",'wb') as f:
	    pickle.dump(tte_dataset_train,f)
	with open(f"{args.out_dir}/test_data.pkl",'wb') as f:
	    pickle.dump(tte_dataset_test,f)
	with open(f"{args.out_dir}/val_data.pkl",'wb') as f:
	    pickle.dump(tte_dataset_test,f)

	with open(f"{args.out_dir}/train_labels.pkl",'wb') as f:
	    pickle.dump(tte_actions_train,f)
	with open(f"{args.out_dir}/test_labels.pkl",'wb') as f:
	    pickle.dump(tte_actions_test,f)
	with open(f"{args.out_dir}/val_labels.pkl",'wb') as f:
	    pickle.dump(tte_actions_test,f)

	print(f' -- Done -- ')







