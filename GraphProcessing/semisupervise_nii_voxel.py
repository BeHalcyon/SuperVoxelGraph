# -*- coding: utf-8 -*-

# After generating the npy array from semi-supervise GCN,
# this py file depart the volume to the corrsponding parts.


import argparse
import json
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Translate the predict result to segmented result.")

    parser.add_argument('--configure_file', default='../x64/Release/workspace/spheres_supervoxel.json',
                        help='configure json file')

    args = parser.parse_args()
    return args


import numpy as np


# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
#
# for i in range(len(labels)):
#     print(i, labels[i])
#
# dimension = [480, 720, 120] # x y z
# file_path = 'I:\\science data\\4 Combustion\\jet_0051\\'
# histogram_size = 256
#
# volume_array = np.fromfile(file_path + 'jet_mixfrac_0051.raw', dtype=np.float32)
# label_array = np.fromfile(file_path + 'jet_mixfrac_0051_label.raw', dtype=np.int)
# for i in range(-1, n_clusters_):
#
#     buf_volume_array = np.zeros(dtype=np.float32, shape=volume_array.shape)
#     buf_index_array = np.array(np.where(labels == i))
#     # print(buf_index_array)
#     for j in buf_index_array[0]:
#         buf = np.where(label_array == j)
#         # print(buf)
#         buf_volume_array[buf] = volume_array[buf]
#
#     buf_volume_array.tofile('jet_mixfrac_0051_super_voxles'+str(len(buf_index_array[0]))+'_part_'+str(i)+'.raw')
#     print("Cluster ", i , " in ", n_clusters_, " has been saved.")

def readVifo(file_name):
    with open(file_name, "r") as f:
        ls = f.readlines()
        for i in range(len(ls)):
            ls[i] = ls[i].strip('\n')

        volume_number = int(ls[0])
        volume_type = ls[1]
        dimension = [int(x) for x in ls[2].split(' ')]
        space = [float(x) for x in ls[3].split(' ')]
        raw_file_name = ls[4]
        return volume_number, volume_type, dimension, space, raw_file_name


def readVolumeRaw(file_name, dtype='uchar'):
    if dtype == 'uchar':
        return np.fromfile(file_name, dtype=np.uint8)
    elif dtype == 'float':
        return np.fromfile(file_name, dtype=np.float32)
    elif dtype == 'ushort':
        return np.fromfile(file_name, dtype=np.uint16)
    elif dtype == 'int':
        return np.fromfile(file_name, dtype=np.int)


if __name__ == "__main__":
    time_start = time.time()
    hp = parse_args()
    f = open(hp.configure_file)
    json_content = json.load(f)

    file_prefix = json_content["data_path"]["file_prefix"]
    label_raw_file = file_prefix + json_content["file_name"]["label_file"]
    labeled_gexf_file = file_prefix + json_content["file_name"]["labeled_graph_file"]

    # read vifo file
    vifo_file = json_content["data_path"]["vifo_file"]
    volume_number, volume_type, dimension, space, raw_file_name = readVifo(vifo_file)

    # read origin raw file
    vifo_file_path = vifo_file[:vifo_file.rfind('/') + 1]
    volume_raw_data = readVolumeRaw(vifo_file_path + raw_file_name, volume_type)


    # read semi-supervise file TODO: integrate the labeled file to json file.
    # semisupervise_result_file = '../VolumeGCN/labeled.npy'
    semisupervise_result_file = file_prefix + json_content['file_name']['predict_labeled_voxel_file']
    labels = np.load(semisupervise_result_file)


    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    print("Number of labeled type : {}".format(n_clusters_))
    print("Max labeled type id : {}".format(max(set(labels))))

    # Extension 20200713
    import SimpleITK

    itk_snap_labeled_nii_file = json_content["data_path"]["vifo_file"]
    itk_snap_labeled_nii_file = itk_snap_labeled_nii_file[:-4] + "nii.gz"

    file_prefix = json_content["data_path"]["file_prefix"]
    itk_snap_labeled_voxel_file = file_prefix + json_content["file_name"]["itk_snap_labeled_voxel_file"]

    itk_img = SimpleITK.ReadImage(itk_snap_labeled_nii_file)
    img_array = SimpleITK.GetArrayFromImage(itk_img)

    origin = itk_img.GetOrigin()
    direction = itk_img.GetDirection()
    space = itk_img.GetSpacing()

    buf_volume_array = np.zeros(dtype=np.int16, shape=volume_raw_data.shape)
    for i in range(0, max(set(labels)) + 1):
        buf = np.where(labels == i)
        label_number = len(np.array(buf)[0])
        buf_volume_array[buf] = i + 1
        print("Cluster {} in {} has {} labels, and it has been saved.".format(i, n_clusters_,
                                                                              label_number))

    buf_volume_array = buf_volume_array.reshape(img_array.shape)

    save_nii_img = SimpleITK.GetImageFromArray(buf_volume_array)
    save_nii_img.SetOrigin(origin)
    save_nii_img.SetDirection(direction)
    save_nii_img.SetSpacing(space)

    nii_gz_file_name = file_prefix + json_content['file_name']['semi_supervise_nii_gz_file']
    SimpleITK.WriteImage(save_nii_img, nii_gz_file_name)

    print("Segmentation results for voxels have been saved in nii.gz file.")

    time_end = time.time()
    all_time = int(time_end - time_start)

    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')