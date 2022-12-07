import numpy as np
import pandas as pd 
from utils import *
import pickle
import os

type_padding_data = 'w_weight' # no_weight, w_weight
path_save = './processed_data/'
# Load data
# Train
data_train      = pd.read_csv('./data/train_m4C_window20_Mus_musculus.csv')
sequences_train = data_train['sequence']
labels_train    = data_train['label']
# Val
data_val        = pd.read_csv('./data/val_m4C_window20_Mus_musculus.csv')
sequences_val   = data_val['sequence']
labels_val      = data_val['label']
# Test
data_test       = pd.read_csv('./data/test_m4C_window20_Mus_musculus.csv')
sequences_test  = data_test['sequence']
labels_test     = data_test['label']

#------------------------------------------------------------------------------------------------------------------
# Positive
# Train
pos_labels_train     = np.where(labels_train == 1)[0]
pos_sequences_train  = sequences_train[pos_labels_train].tolist()
neg_labels_train     = np.where(labels_train == 0)[0]
neg_sequences_train  = sequences_train[neg_labels_train].tolist()

# Validation
pos_labels_val       = np.where(labels_val == 1)[0]
pos_sequences_val    = sequences_val[pos_labels_val].tolist()
neg_labels_val       = np.where(labels_val == 0)[0]
neg_sequences_val    = sequences_val[neg_labels_val].tolist()

# Test
pos_labels_test      = np.where(labels_test == 1)[0]
pos_sequences_test   = sequences_test[pos_labels_test].tolist()
neg_labels_test      = np.where(labels_test == 0)[0]
neg_sequences_test   = sequences_test[neg_labels_test].tolist()

#------------------------------------------------------------------------------------------------------------------
# Find three left/right characters --> dictionaries 
left_dict, right_dict = get_dict(pos_sequences_train, list_key)

# Take out 5 triplet set with highest occuring frequencies
left_dict, left_weight, right_dict, right_weight = get_dict_and_weight(left_dict, right_dict, list_key)

# Save dict 
if not os.path.exists(path_save + 'save_dict'):
    os.mkdir(path_save + 'save_dict')
save_dict(left_dict, path_save + 'save_dict/left_dict.pkl')
save_dict(left_weight, path_save + 'save_dict/left_weight.pkl')
save_dict(right_dict, path_save + 'save_dict/right_dict.pkl')
save_dict(right_weight, path_save + 'save_dict/right_weight.pkl')

#------------------------------------------------------------------------------------------------------------------
# Padding negative sequence : padding_with_w, padding_without_w, padding_random
if type_padding_data == 'w_weight':
    pad_neg_sequences_train = [padding_with_w(seq, list_key, left_dict, left_weight, right_dict, right_weight, neg_sequences_train.index(seq)) for seq in neg_sequences_train]
    pad_neg_sequences_val   = [padding_with_w(seq, list_key, left_dict, left_weight, right_dict, right_weight, neg_sequences_val.index(seq)) for seq in neg_sequences_val]
    pad_neg_sequences_test  = [padding_with_w(seq, list_key, left_dict, left_weight, right_dict, right_weight, neg_sequences_test.index(seq)) for seq in neg_sequences_test]

if type_padding_data == 'no_weight':
    pad_neg_sequences_train = [padding_without_w(seq, list_key, left_dict, right_dict, neg_sequences_train.index(seq)) for seq in neg_sequences_train]
    pad_neg_sequences_val   = [padding_without_w(seq, list_key, left_dict, right_dict, neg_sequences_val.index(seq)) for seq in neg_sequences_val]
    pad_neg_sequences_test  = [padding_without_w(seq, list_key, left_dict, right_dict, neg_sequences_test.index(seq)) for seq in neg_sequences_test]

else:
    pad_neg_sequences_train = [padding_random(seq, list_key, list_values, neg_sequences_train.index(seq)) for seq in neg_sequences_train]
    pad_neg_sequences_val   = [padding_random(seq, list_key, list_values, neg_sequences_val.index(seq)) for seq in neg_sequences_val]
    pad_neg_sequences_test  = [padding_random(seq, list_key, list_values, neg_sequences_test.index(seq)) for seq in neg_sequences_test]

train_padded_seqs = np.concatenate((pos_sequences_train, pad_neg_sequences_train), axis =0)
val_padded_seqs   = np.concatenate((pos_sequences_val, pad_neg_sequences_val), axis =0)
test_padded_seqs  = np.concatenate((pos_sequences_test, pad_neg_sequences_test), axis =0)

if not os.path.exists(path_save + 'pad_{}'.format(type_padding_data)):
    os.mkdir(path_save + 'pad_{}'.format(type_padding_data))

np.save(path_save + 'pad_{}/sequence_train.npy'.format(type_padding_data), train_padded_seqs)
np.save(path_save + 'pad_{}/sequence_train.npy'.format(type_padding_data), train_padded_seqs)

np.save(path_save + 'pad_{}/sequence_val.npy'.format(type_padding_data), val_padded_seqs)
np.save(path_save + 'pad_{}/sequence_train.npy'.format(type_padding_data), train_padded_seqs)

np.save(path_save + 'pad_{}/sequence_test.npy'.format(type_padding_data), test_padded_seqs)
np.save(path_save + 'pad_{}/sequence_test.npy'.format(type_padding_data), test_padded_seqs)

#------------------------------------------------------------------------------------------------------------------
# # Encode data
encoded_data_train = encode_index(train_padded_seqs, list_values)
encoded_data_val   = encode_index(val_padded_seqs, list_values)
encoded_data_test  = encode_index(test_padded_seqs, list_values)

print(encoded_data_train.shape)
print(encoded_data_val.shape)
print(encoded_data_test.shape)

if not os.path.exists(path_save + '/encode_pad_{}'.format(type_padding_data)):
    os.mkdir(path_save + '/encode_pad_{}'.format(type_padding_data))

np.save(path_save + '/encode_pad_{}/data_train'.format(type_padding_data), encoded_data_train)
np.save(path_save + '/encode_pad_{}/label_train'.format(type_padding_data), labels_train)
np.save(path_save + '/encode_pad_{}/data_val'.format(type_padding_data), encoded_data_val)
np.save(path_save + '/encode_pad_{}/label_val'.format(type_padding_data), labels_val)
np.save(path_save + '/encode_pad_{}/data_test'.format(type_padding_data), encoded_data_test)
np.save(path_save + '/encode_pad_{}/label_test'.format(type_padding_data), labels_test)