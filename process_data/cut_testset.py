import numpy as np 
import pandas as pd 
from utils import *

# Load data
data_test      = pd.read_csv('./data/test_m4C_window20_Mus_musculus.csv')
sequences_test = data_test['sequence']
labels_test    = data_test['label']

# Positive
pos_labels_test      = np.where(labels_test == 1)[0]
pos_sequences_test   = sequences_test[pos_labels_test].tolist()

#--------------------------------------------------------------------------------
# Path to save test cut dataset
path = './test_cut_dataset'

# test cut 1 element on the left 
pos_seq_left_cut_1                          = [pos_seq[1:] for pos_seq in pos_sequences_test]
seq_test_cut_left_1, label_test_cut_left_1  = cut_pos_seqs(pos_seq_left_cut_1, pos_idx = 19)
np.save(path + '/test_cut_left_1/sequence.npy', seq_test_cut_left_1)
np.save(path + '/test_cut_left_1/label.npy', label_test_cut_left_1)

# test cut 1 element on the right 
pos_seq_right_cut_1                          = [pos_seq[:len(pos_seq)-1] for pos_seq in pos_sequences_test]
seq_test_cut_right_1, label_test_cut_right_1 = cut_pos_seqs(pos_seq_right_cut_1, pos_idx = 20)
np.save(path + '/test_cut_right_1/sequence.npy', seq_test_cut_right_1)
np.save(path + '/test_cut_right_1/label.npy', label_test_cut_right_1)

# test cut 2 element on the left 
pos_seq_left_cut_2                           = [pos_seq[2:] for pos_seq in pos_sequences_test]
seq_test_cut_left_2, label_test_cut_left_2   = cut_pos_seqs(pos_seq_left_cut_2, pos_idx = 18)
np.save(path + '/test_cut_left_2/sequence.npy', seq_test_cut_left_2)
np.save(path + '/test_cut_left_2/label.npy', label_test_cut_left_2)

# test cut 2 element on the right 
pos_seq_right_cut_2                          = [pos_seq[:len(pos_seq)-2] for pos_seq in pos_sequences_test]
seq_test_cut_right_2, label_test_cut_right_2 = cut_pos_seqs(pos_seq_right_cut_2, pos_idx = 20)
np.save(path + '/test_cut_right_2/sequence.npy', seq_test_cut_right_2)
np.save(path + '/test_cut_right_2/label.npy', label_test_cut_right_2)

# test cut 3 element on the left 
pos_seq_left_cut_3                           = [pos_seq[3:] for pos_seq in pos_sequences_test]
seq_test_cut_left_3, label_test_cut_left_3   = cut_pos_seqs(pos_seq_left_cut_3, pos_idx = 17)
np.save(path + '/test_cut_left_3/sequence.npy', seq_test_cut_left_3)
np.save(path + '/test_cut_left_3/label.npy', label_test_cut_left_3)

# test cut 3 element on the right 
pos_seq_right_cut_3                          = [pos_seq[:len(pos_seq)-3] for pos_seq in pos_sequences_test]
seq_test_cut_right_3, label_test_cut_right_3 = cut_pos_seqs(pos_seq_right_cut_3, pos_idx = 20)
np.save(path + '/test_cut_right_3/sequence.npy', seq_test_cut_right_3)
np.save(path + '/test_cut_right_3/label.npy', label_test_cut_right_3)

# test cut 1 element on the right and on the left 
pos_seq_right_left_cut_1                               = [pos_seq[1:len(pos_seq)-1] for pos_seq in pos_sequences_test]
seq_test_right_left_cut_1, label_test_right_left_cut_1 = cut_pos_seqs(pos_seq_right_left_cut_1, pos_idx = 19)
np.save(path + '/test_cut_left_right_1/sequence.npy', seq_test_right_left_cut_1)
np.save(path + '/test_cut_left_right_1/label.npy', label_test_right_left_cut_1)