# encoding: utf-8
import argparse
import json
from numba import jit
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Volume-GCN")

    # Debug 20200522 hxy : add configure_file

    parser.add_argument('--configure_file', default='../x64/Release/workspace/spheres_supervoxel.json',
                        help='configure json file')

    # # training scheme
    # parser.add_argument('--epochs', default=10, type=int)
    # # parser.add_argument('--batch_size_con', default=10000, type=int)
    # # parser.add_argument('--batch_size', default=5000, type=int)
    # # parser.add_argument('--eval_size', default=3000, type=int)
    #
    # parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
    # parser.add_argument('--warmup_steps', default=100, type=int)
    #
    # # model
    # parser.add_argument('--dim', type=int, default=256,
    #                     help='Number of dimensions. Default is 256  (0-255).')
    # parser.add_argument('--vec_dim', type=int, default=260,
    #                     help='Length of node features. Default is 261.')
    # # parser.add_argument('--hidden1', type=int, default=128,
    # #                     help='Dimension of hidden layer 1.')
    # # parser.add_argument('--hidden2', type=int, default=64,
    # #                     help='Dimension of hidden layer 2.')
    # parser.add_argument('--hidden1', type=int, default=256,
    #                     help='Dimension of hidden layer 1.')
    # parser.add_argument('--hidden2', type=int, default=128,
    #                     help='Dimension of hidden layer 2.')
    # parser.add_argument('--dropout', type=float, default=0.5,
    #                     help='Dropout.')
    # parser.add_argument('--labeled_node', type=int, default=100,
    #                     help='Number of labeled nodes.')
    # parser.add_argument('--ratio', type=float, default=0.8,
    #                     help='Ratio of training data.')
    # parser.add_argument('--label', type=int, default=1,
    #                     help='Number of label types.')
    # # parser.add_argument('--k', type=int, default=5,
    # #                     help='Subgraph size. Default is 5.')
    # parser.add_argument('--dataset', default='datasets/jet_mixfrac_0051_supervoxels.gexf',
    #                     help='Name of dataset')
    # parser.add_argument('--node_num', type=int, default=None,
    #                     help='Number of nodes.')


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
    args.gradient_file = json_content["volume2supervoxel"]["gradient_file"]

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

    # model
    # args.vector_dimension = json_content["model"]["vector_dimension"]
    # args.dimension = json_content["model"]["dimension"]
    # args.epochs = json_content["model"]["epochs"]
    # args.warmup_steps = json_content["model"]["warmup_steps"]
    # args.label_type_number = json_content["model"]["label_type_number"]
    # args.ratio = json_content["model"]["ratio"]
    # args.node_embedding_file = json_content["model"]["node_embedding_file"]

    args.type = args.labeled_type


    # supervoxelgraph
    args.graph_node_feature_file = json_content["supervoxelgraph"]["graph_node_feature_file"]
    args.labeled_graph_file = json_content["supervoxelgraph"]["labeled_graph_file"]
    args.label_mask_file = json_content["supervoxelgraph"]["label_mask_file"]

    # model
    args.dataset = args.workspace + args.labeled_graph_file
    args.epochs = json_content['model']['epochs']
    # args.dim = json_content['model']['dimension']
    args.vec_dim = json_content['model']['vector_dimension']
    args.learning_rate = json_content['model']['learning_rate']
    args.warmup_steps = json_content['model']['warmup_steps']
    args.lr = args.learning_rate
    args.ratio = json_content['model']['ratio']
    args.node_embedding = args.workspace + json_content['model']['node_embedding_file']
    args.predict_labeled_supervoxel_file = args.workspace + json_content['model']['predict_label_supervoxel_file']
    args.predict_labeled_voxel_file = args.workspace + json_content['model']['predict_label_voxel_file']
    args.predict_label_nii_file = json_content['model']['predict_label_nii_file']
    args.groundtruth_label_supervoxel_file = args.workspace + json_content['model']['groundtruth_label_supervoxel_file']
    args.dropout = json_content['model']['dropout']
    args.hidden_layers = json_content['model']['hidden_layers']
    args.groundtruth_label_voxel_file = json_content['model']['groundtruth_label_voxel_file']

    # comparedmodel
    args.voxel_based_svm_predict_file = json_content['comparedmodel']['voxel_based_svm_predict_file']
    args.voxel_based_nn_predict_file = json_content['comparedmodel']['voxel_based_nn_predict_file']
    args.voxel_based_rf_predict_file = json_content['comparedmodel']['voxel_based_rf_predict_file']
    args.voxel_based_knn_predict_file = json_content['comparedmodel']['voxel_based_knn_predict_file']
    args.supervoxel_based_svm_predict_file = json_content['comparedmodel']['supervoxel_based_svm_predict_file']
    args.supervoxel_based_nn_predict_file = json_content['comparedmodel']['supervoxel_based_nn_predict_file']
    args.supervoxel_based_knn_predict_file = json_content['comparedmodel']['supervoxel_based_knn_predict_file']
    args.supervoxel_based_rf_predict_file = json_content['comparedmodel']['supervoxel_based_rf_predict_file']



    # file_prefix = json_content["data_path"]["file_prefix"]
    # gexf_file = json_content["file_name"]["labeled_graph_file"]
    #
    # args.dataset = file_prefix + gexf_file
    # args.epochs = json_content['gcn']['epochs']
    # args.dim = json_content['gcn']['dimension']
    # args.vec_dim = json_content['gcn']['vector_dimension']
    # args.node_embedding = file_prefix + json_content['file_name']['node_embedding_file']
    # args.lr = json_content['gcn']['learning_rate']
    # args.ratio = json_content['gcn']['ratio']
    # args.label = json_content['gcn']['label_type_number']  # abandon
    # args.ground_truth_labeled_supervoxel_file = file_prefix + json_content['file_name']['ground_truth_labeled_supervoxel_file']
    # args.predict_labeled_supervoxel_file = file_prefix + json_content['file_name']['predict_labeled_supervoxel_file']
    # args.predict_labeled_voxel_file = file_prefix + json_content['file_name']['predict_labeled_voxel_file']
    return args

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

def evaluationForVoxels(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return precision, recall, f1, acc


@jit
def volumeSegmentation(label_supervoxel_array, supervoxel_id_array):
    volume_voxel_based_array = np.zeros_like(supervoxel_id_array, dtype=np.int32)

    for i in range(supervoxel_id_array.shape[0]):
        volume_voxel_based_array[i] = label_supervoxel_array[supervoxel_id_array[i]]
        if i % 10000000 == 0:
            print("Process supervoxel_id file to volume segmentation form : {:.2f}%".
                  format(i * 100 / (len(supervoxel_id_array))))

    return volume_voxel_based_array


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
