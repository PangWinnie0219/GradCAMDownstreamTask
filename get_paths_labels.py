import os
import numpy as np
import pickle

root_dir2 = './cholec80/'
img_dir2 = os.path.join(root_dir2, 'cropped_image')
phase_dir2 = os.path.join(root_dir2, 'tool_annotations')

#train_video_num = 8
#val_video_num = 4
#test_video_num = 2

print(root_dir2)
print(img_dir2)
print(phase_dir2)


#cholec80==================
def get_dirs2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x:int(x))
    file_paths.sort(key=lambda x:int(os.path.basename(x)))
    return file_names, file_paths

def get_files2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths
#cholec80==================


#cholec80==================
img_dir_names2, img_dir_paths2 = get_dirs2(img_dir2)   # get the directory of the video files
tool_file_names2, tool_file_paths2 = get_files2(phase_dir2)  # get the directory of the tool annotation files

## we only use the first 40 videos
tool_file_names2 = tool_file_names2[0:40]  
tool_file_paths2 = tool_file_paths2[0:40]

# phase_dict = {}
# phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
#                   'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
# for i in range(len(phase_dict_key)):
#     phase_dict[phase_dict_key[i]] = i
# print(phase_dict)
#cholec80==================


#cholec80==================
all_info_all2 = []

for j in range(len(tool_file_names2)):
    downsample_rate = 25
    tool_file = open(tool_file_paths2[j])

    video_num_file = int(os.path.splitext(os.path.basename(tool_file_paths2[j]))[0][5:7])
    video_num_dir = int(os.path.basename(img_dir_paths2[j]))
    
    print("video_num_file:", video_num_file,"video_num_dir:", video_num_dir, "rate:", downsample_rate)

    info_all = []
    first_line = True
    for tool_line in tool_file:
        tool_split = tool_line.split()   # tool label: [frame, tool1, tool2, ....]
        if first_line:
            first_line = False
            continue
        # if int(tool_split[0]) % downsample_rate == 0:
        info_each = []
        img_file_each_path = os.path.join(img_dir_paths2[j], tool_split[0] + '.png')  # directory to every image 
        info_each.append(img_file_each_path)
        info_each.append(tool_split[1:])
        info_all.append(info_each)              

    # print(len(info_all))
    all_info_all2.append(info_all)
#cholec80==================

with open('./cholec80.pkl', 'wb') as f:
    pickle.dump(all_info_all2, f)

with open('./cholec80.pkl', 'rb') as f:
    all_info_40 = pickle.load(f)

#cholec80==================
train_file_paths_40 = []
test_file_paths_40 = []
val_file_paths_40 = []
val_labels_40 = []
train_labels_40 = []
test_labels_40 = []

train_num_each_40 = []
val_num_each_40 = []
test_num_each_40 = []

train_idx_file = open(os.path.join(root_dir2, 'train_index.txt'))
val_idx_file = open(os.path.join(root_dir2, 'val_index.txt'))

for train_idx in train_idx_file:
    idx = int(train_idx)            # video index: 1 => video01
    train_num_each_40.append(len(all_info_40[idx-1]))   # index in the all_info_40: 0 => video01
    for j in range(len(all_info_40[idx-1])):
        train_file_paths_40.append(all_info_40[idx-1][j][0])    # path to the frame
        
        # add additional class for 'tissue'. Total class = 8 (7 tools & 1 tissue)
        label_tool = all_info_40[idx-1][j][1:]
        label_tool[0].append('1')
        train_labels_40.append(all_info_40[idx-1][j][1:])       # label of the frame

print(len(train_file_paths_40))
print(len(train_labels_40))

for val_idx in val_idx_file:
    idx = int(val_idx)            # video index: 1 => video01
    val_num_each_40.append(len(all_info_40[idx-1]))   # index in the all_info_40: 0 => video01
    for j in range(len(all_info_40[idx-1])):
        val_file_paths_40.append(all_info_40[idx-1][j][0])    # path to the frame
        
        # add additional class for 'tissue'. Total class = 8 (7 tools & 1 tissue)
        label_tool = all_info_40[idx-1][j][1:]
        label_tool[0].append('1')
        val_labels_40.append(all_info_40[idx-1][j][1:])       # label of the frame

print(len(val_file_paths_40))
print(len(val_labels_40))


#cholec80==================


train_val_test_paths_labels = []
train_val_test_paths_labels.append(train_file_paths_40)
train_val_test_paths_labels.append(val_file_paths_40)

train_val_test_paths_labels.append(train_labels_40)
train_val_test_paths_labels.append(val_labels_40)

train_val_test_paths_labels.append(train_num_each_40)
train_val_test_paths_labels.append(val_num_each_40)

# train_val_paths_labels.pkl  => label: 7 classes for 7 tools
# train_val_paths_labels_1.pkl => label: 8 classes for 7 tools and 1 tissue
with open('train_val_paths_labels_1.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)


print('Done')
print()