import numpy as np
import random 
import pickle


char_list = ['T', 'A', 'G', 'C']
list_values = []
list_key    = []
for char1 in char_list:
    for char2 in char_list:
        list_key.append(char1 + char2)
        for char3 in char_list:
            list_values.append(char1 + char2 + char3)

#-------------------------------------------------------------------------------
def save_dict(dict_, path_save):
    a_file = open(path_save, "wb")
    pickle.dump(dict_, a_file)
    a_file.close()

#-------------------------------------------------------------------------------
# Get dict containing duplet key: triplet value for both left dict and right dict
def get_dict(pos_sequences, char_dict):
    # Init dictionary
    left_dict    = {}
    right_dict   = {}
    for key in list_key:
        left_dict[key] = []
        right_dict[key] = []

    for seq in  pos_sequences:
        for key in char_dict:
            if key == seq[3:5]:
                left_dict[key].append(seq[0:3])
            if key == seq[36:38]:
                right_dict[key].append(seq[38:])
    return left_dict, right_dict 

#-------------------------------------------------------------------------------
# Get 5 triplet sets with highest occuring frequencies and their frequencies
def extract_elements(list_sequence):

    list_char = np.array(list(set(list_sequence)))
    weight = []
    for char in list_char:
        weight.append(list_sequence.count(char))
    weight = np.array(weight)

    sorted_index  = sorted(range(len(weight)), key=lambda k: weight[k])
    sorted_char   = list_char[sorted_index]
    sorted_weight = weight[sorted_index]

    return sorted_char[len(sorted_char)-5:], sorted_weight[len(sorted_weight)-5:]

def get_dict_and_weight(left_dict, right_dict, char_dict):
    left_weight  = {}
    right_weight = {}
    for key in char_dict:
        # left dict
        left_list_seq     = left_dict[key]
        seq, weight       = extract_elements(left_list_seq)
        left_weight[key]  = weight
        left_dict[key]    = seq
        # right dict
        right_list_seq    = right_dict[key]
        seq, weight       = extract_elements(right_list_seq)
        right_weight[key] = weight
        right_dict[key]   = seq
    return left_dict, left_weight, right_dict, right_weight

#-------------------------------------------------------------------------------
# padding methods
# padding with weight
def padding_with_w(sequence, char_dict, left_dict, left_weight, right_dict, right_weight, seed =0):
    random.seed(seed)
    for key in char_dict:
        if sequence[0:2] == key:
            add_left_value = random.choices(left_dict[key], weights = left_weight[key], k=1)[0]
        if sequence[len(sequence)-2:] == key:
            add_right_value = random.choices(right_dict[key], weights = right_weight[key], k=1)[0]
    sequence = add_left_value + sequence + add_right_value
    return sequence

# padding without weight
def padding_without_w(sequence, char_dict, left_dict, right_dict, seed=0):
    for key in char_dict:
        random.seed(seed)
        if sequence[0:2] == key:
            add_left_value = random.choice(left_dict[key])     
        if sequence[len(sequence)-2:] == key:
            add_right_value = random.choice(right_dict[key])

    sequence = add_left_value + sequence + add_right_value
    return sequence

# padding random
def padding_random(sequence, char_dict, list_values, seed):
    for key in char_dict:
        if sequence[0:2] == key:
            random.seed(seed)
            add_left_value = random.choice(list_values)
        if sequence[len(sequence)-2:] == key:
            random.seed(seed + 1)
            add_right_value = random.choice(list_values)

    sequence = add_left_value + sequence + add_right_value
    return sequence

#-------------------------------------------------------------------------------
def encode_index(list_seqs, char_dict):
    list_matrix = []
    count = 0
    for DNA in list_seqs:   
        index_vector = np.zeros(len(DNA)-2, dtype=np.float32)
        for i in range(len(DNA)-2):    
            index_ = char_dict.index(DNA[i:i+3])
            index_vector[i] = index_
        list_matrix.append(index_vector)
    
    
    list_matrix = np.array(list_matrix)
    # print(list_matrix.shape)
    return list_matrix


# Cut positive sequences with 35-bp window
def cut_pos_seqs(list_pos_seq, pos_idx):
    pos_labels = []
    pos_seq    = []
    neg_seq    = []
    neg_labels = []
    for pos_ex in list_pos_seq:
        for idx in range(17, len(pos_ex)-17):
            if pos_ex[idx] == 'C' and idx == pos_idx:
                pos_seq.append(pos_ex[idx-17: idx+1+17])
                pos_labels.append(1)
            if pos_ex[idx] == 'C' and idx != pos_idx:
                neg_seq.append(pos_ex[idx-17: idx+1+17])
                neg_labels.append(0)
    list_sequence = pos_seq + neg_seq
    labels        = pos_labels + neg_labels
    return list_sequence, labels
