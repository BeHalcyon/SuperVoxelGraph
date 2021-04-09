import argparse
import json

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

    # supervoxelgraph
    args.graph_node_feature_file = json_content["supervoxelgraph"]["graph_node_feature_file"]
    args.labeled_graph_file = json_content["supervoxelgraph"]["labeled_graph_file"]

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

    args.dropout = json_content['model']['dropout']
    args.hidden_layers = json_content['model']['hidden_layers']

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