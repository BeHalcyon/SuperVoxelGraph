# encoding: utf-8
import argparse
import json
import numpy as np
def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)

    # Debug 20200522 hxy : add configure_file

    parser.add_argument('--configure_file',
                        default=r'D:\project\science_project\SLIC3DSuperVoxel\x64\Release\workspace\lung_groundtruth_supervoxel.json',
                        help='configure json file')
    parser.add_argument('--csv_file', nargs='*', help='csv file export by gephi')

    args = parser.parse_args()
    configure_json_file = args.configure_file
    f = open(configure_json_file)
    json_content = json.load(f)


    args.workspace = json_content["workspace"]

    # volumes
    args.file_path = json_content["volumes"]["file_path"]
    args.dtype = json_content["volumes"]["dtype"]
    # args.space = json_content["volumes"]["space"]
    args.dimension = json_content["volumes"]["dimension"]
    args.data_byte_order = json_content["volumes"]["data_byte_order"]
    args.file_names = json_content["volumes"]["file_names"]
    args.downsample = json_content["volumes"]["downsample"]

    #volume2supervoxel
    args.supervoxel_id_file = json_content["volume2supervoxel"]["supervoxel_id_file"]

    # supervoxelnodefeature
    args.histogram_size = json_content["supervoxelnodefeature"]["histogram_size"]
    args.is_histogram_stored = json_content["supervoxelnodefeature"]["is_histogram_stored"]
    args.is_gradient_stored = json_content["supervoxelnodefeature"]["is_gradient_stored"]
    args.is_entropy_stored = json_content["supervoxelnodefeature"]["is_entropy_stored"]
    args.is_barycenter_stored = json_content["supervoxelnodefeature"]["is_barycenter_stored"]
    args.label_histogram_file = json_content["supervoxelnodefeature"]["label_histogram_file"]
    args.edge_weight_file = json_content["supervoxelnodefeature"]["edge_weight_file"]

    # supervoxelgraph
    args.histogram_size = json_content["supervoxelgraph"]["histogram_size"]
    args.graph_file = json_content["supervoxelgraph"]["graph_file"]
    args.graph_node_feature_file = json_content["supervoxelgraph"]["graph_node_feature_file"]
    args.labeled_graph_file = json_content["supervoxelgraph"]["labeled_graph_file"]
    args.labeled_type = json_content["supervoxelgraph"]["label_type"]
    args.labeled_file = json_content["supervoxelgraph"]["labeled_file"]
    args.label_mask_file = json_content["supervoxelgraph"]["label_mask_file"]

    # model
    args.vector_dimension = json_content["model"]["vector_dimension"]
    args.epochs = json_content["model"]["epochs"]
    args.warmup_steps = json_content["model"]["warmup_steps"]
    args.label_type_number = json_content["model"]["label_type_number"]
    args.ratio = json_content["model"]["ratio"]
    args.node_embedding_file = json_content["model"]["node_embedding_file"]
    args.groundtruth_label_supervoxel_file = json_content['model']['groundtruth_label_supervoxel_file']
    args.groundtruth_label_voxel_file = json_content['model']['groundtruth_label_voxel_file']
    args.predict_label_supervoxel_file = json_content['model']['predict_label_supervoxel_file']
    args.predict_label_voxel_file = json_content['model']['predict_label_voxel_file']
    args.predict_label_nii_file = json_content['model']['predict_label_nii_file']

    args.type = args.labeled_type
    return args


def readVolumeRaw(file_name, dtype='uchar'):
    if dtype == 'uchar' or dtype == 'unsigned char' or dtype == 'uint8':
        return np.fromfile(file_name, dtype=np.uint8)
    elif dtype == 'float':
        return np.fromfile(file_name, dtype=np.float32)
    elif dtype == 'ushort' or dtype == 'unsigned short' or dtype == 'uint16':
        return np.fromfile(file_name, dtype=np.uint16)
    elif dtype == 'int' or dtype == 'unsigned int' or dtype == 'uint32':
        return np.fromfile(file_name, dtype=np.int)


from numba import jit


@jit
def volumeSegmentation(label_supervoxel_array, supervoxel_id_array):
    volume_voxel_based_array = np.zeros_like(supervoxel_id_array, dtype=np.int)

    for i in range(supervoxel_id_array.shape[0]):
        volume_voxel_based_array[i] = label_supervoxel_array[supervoxel_id_array[i]]
        if i % 10000000 == 0:
            print("Process supervoxel_id file to volume segmentation form : {:.2f}%".
                  format(i * 100 / (len(supervoxel_id_array))))

    return volume_voxel_based_array

def labelVoxel2Nii(hp, label_voxel_array, save_file_name):
    import SimpleITK
    if hp.type == 2:
        itk_img = SimpleITK.ReadImage(hp.workspace + hp.label_mask_file)
    else:
        itk_img = SimpleITK.ReadImage(hp.workspace + hp.labeled_file)
    img_array = SimpleITK.GetArrayFromImage(itk_img)  # the array is arranged with [z, y, x]
    # print(img_array.dtype, img_array.shape)
    dimension = itk_img.GetSize()  # the dimension is arranged with [x, y, z]
    origin = itk_img.GetOrigin()
    direction = itk_img.GetDirection()
    space = itk_img.GetSpacing()

    save_nii_img = SimpleITK.GetImageFromArray(label_voxel_array.copy().
                                               reshape(dimension[::-1]).astype(img_array.dtype))
    save_nii_img.SetOrigin(origin)
    save_nii_img.SetDirection(direction)
    save_nii_img.SetSpacing(space)

    nii_gz_file_name = hp.workspace + save_file_name
    SimpleITK.WriteImage(save_nii_img, nii_gz_file_name)
    print("predicted voxel array has been saved in nii form")

def logger_config(log_path,logging_name="supervoxelgraph"):
    import logging
    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger