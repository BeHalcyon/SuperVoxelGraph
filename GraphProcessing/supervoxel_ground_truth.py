

import numpy as np

def readVolumeRaw(file_name, dtype='uchar'):
    if dtype == 'uchar':
        return np.fromfile(file_name, dtype=np.uint8)
    elif dtype == 'float':
        return np.fromfile(file_name, dtype=np.float)
    elif dtype == 'ushort':
        return np.fromfile(file_name, dtype=np.uint16)
    elif dtype == 'int':
        return np.fromfile(file_name, dtype=np.int)

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

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Caculate groundtruth supervoxel labels")

    parser.add_argument('--configure_file', default='../x64/Release/workspace/spheres_supervoxel.json',
                        help='configure json file')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import time

    import json
    start = time.clock()
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
    ground_truth_voxel_data = readVolumeRaw(vifo_file_path + raw_file_name[:-4] + "_ground_truth_voxel_label_int.raw",
                                    'int')

    # read label int file
    label_int_data = readVolumeRaw(label_raw_file, 'int')

    # create a numpy array to store the supervoxel ids and their type
    # labeled_supervoxel_dict = {}
    labeled_supervoxel_array = np.zeros(shape=max(label_int_data)+1, dtype=np.int)

    print('Number of supervoxels(nodes) : {}'.format(labeled_supervoxel_array.shape[0]))

    # update the numpy array
    for i in range(label_int_data.shape[0]):
        # labeled_supervoxel_dict[label_int_data[i]] = ground_truth_voxel_data[i]

        if labeled_supervoxel_array[label_int_data[i]] == 0:
            labeled_supervoxel_array[label_int_data[i]] = ground_truth_voxel_data[i]

    # save the ground truth file
    np.save(file_prefix+json_content['file_name']['ground_truth_labeled_supervoxel_file'], labeled_supervoxel_array)
    # labeled_supervoxel_array.tofile(file_prefix+json_content['file_name']['ground_truth_labeled_supervoxel_file'])

    elapsed = (time.clock() - start)
    print("Time for calculating the ground truth supervoxel graph: ", elapsed, "s.")