''' get paths and labels for MICCAI2018 dataset
'''


import os
import numpy as np
import pickle

#miccai2018==================
root_dir = 'path/to/dataset/'
tool_dir = os.path.join(root_dir, 'annotation_tool')
#miccai2018==================

# print(root_dir)
# print(img_dir)
# print(phase_dir)

seq_set_train = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
seq_set_val = [1, 5, 16]

#miccai2018==================
def get_dirs(root_dir, seq_list):
    file_paths = []
    file_names = []
    for seq_no in seq_list:
        temp_path = 'seq_' + str(seq_no) + '/left_frames/'
        path = os.path.join(root_dir, temp_path)
        if os.path.isdir(path):
            file_paths.append(path)
    # file_names.sort
    # file_paths.sort(key=lambda x:int(os.path.basename(x)))
    return file_paths

def get_files(root_dir, seq_list):
    file_paths = []
    file_names = []
    for seq_no in seq_list:
        temp_path = 'seq_' + str(seq_no) + '.txt'
        path = os.path.join(root_dir, temp_path)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    # file_names.sort()
    # file_paths.sort()
    return file_names, file_paths
#miccai2018==================


#miccai2018==================
img_dir_paths_train = get_dirs(root_dir, seq_set_train)   
img_dir_paths_val = get_dirs(root_dir, seq_set_val)

tool_file_names_train, tool_file_paths_train = get_files(os.path.join(tool_dir, 'train_dataset'), seq_set_train)  # get the directory of the tool annotation files
tool_file_names_val, tool_file_paths_val = get_files(os.path.join(tool_dir, 'test_dataset'), seq_set_val)
#miccai2018==================


#miccai2018==================
def get_info (img_dir_paths, tool_file_paths):
    info_path = []
    info_label = []
    info_num = []

    for j in range(len(tool_file_paths)):
        tool_file = open(tool_file_paths[j])
        
        first_line = True
        cnt = 0
        for tool_line in tool_file:
            tool_split = tool_line.split()   # tool label: [frame, tool1, tool2, ....]
            if first_line:
                first_line = False
                continue
            img_file_each_path = os.path.join(img_dir_paths[j], 'frame' + tool_split[0] + '.png')  # directory to every image 
            info_path.append(img_file_each_path)
            tool_label = tool_split[1:]
            tool_label.append('0')      # for 'hook' class at cholec80 dataset
            tool_label.append('0')      # for 'specimen bag' class at cholec80 dataset
            tool_label.append('1')      # add additional class for 'tissue'. Total class = 9 (8 tools & 1 tissue)
            info_label.append([tool_label])
            cnt += 1
        info_num.append(cnt)
    return info_path, info_label, info_num
#miccai2018==================

info_path_train, info_label_train, info_num_train = get_info(img_dir_paths_train, tool_file_paths_train)
info_path_val, info_label_val, info_num_val = get_info(img_dir_paths_val, tool_file_paths_val)
print('total train frame:', len(info_path_train))          # 1560
print('total val frame:', len(info_path_val))              # 447
print('total train class:', len(info_label_train[0][0]))   # 11
print('total val class:', len(info_label_val[0][0]))       # 11

print(info_num_train)
print(info_num_val)


train_val_test_paths_labels = []
train_val_test_paths_labels.append(info_path_train)
train_val_test_paths_labels.append(info_path_val)

train_val_test_paths_labels.append(info_label_train)
train_val_test_paths_labels.append(info_label_val)

train_val_test_paths_labels.append(info_num_train)
train_val_test_paths_labels.append(info_num_val)

with open('miccai2018_train_val_paths_labels_adjusted.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)

# miccai2018_train_val_paths_labels_1.pkl => label: 9 classes for 8 tools and 1 tissue
# miccai2018_train_val_paths_labels_adjusted.pkl => label: 11 classes

print('Done')
print()